"""The replay gate's harness (G0 task 9), with fakes and no models."""

from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import anthropic
import httpx2
import numpy as np
import pytest
import soundfile as sf

from claudio.config import PROJECT_ROOT, Settings, load_settings
from claudio.pc.tools import App
from tests.conftest import FakeVad
from tests.unit.test_app import FakeStt
from tests.unit.test_claude_brain import (
    OVERLOADED_MID_REPLY,
    REQUEST,
    USAGE,
    ScriptedClient,
    ScriptedStream,
)
from tests.unit.test_pc_tools import FakeDesktop

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import replay_gate as rg

SR = 16000


class CountingWake:
    """Never fires: keeps the frames it was given."""

    confirmed: frozenset[str] = frozenset()
    names: ClassVar[list[str]] = ["claudio"]

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def process(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def pop_near_miss(self) -> None:
        return None

    def reset(self) -> None:
        pass


@pytest.fixture
def settings() -> Settings:
    base = load_settings()
    conversation = base.conversation.model_copy(update={"barge_in": False})  # needs Silero
    return base.model_copy(update={"conversation": conversation})


@pytest.fixture
def harness(settings: Settings, tmp_path: Path) -> rg.Harness:
    return rg.build_harness(
        settings,
        tmp_path,
        wake=CountingWake(),
        stt=FakeStt(),
        vad=FakeVad(),  # type: ignore[arg-type]
    )


def test_harness_refuses_real_actions(harness: rg.Harness) -> None:
    ctx = harness.responder.actions.ctx
    assert ctx.run == harness.recorder.ran.append
    assert ctx.press_key == harness.recorder.keys.append
    assert ctx.open_path == harness.recorder.opened.append
    assert "ActionContext.real" not in Path(rg.__file__).read_text(encoding="utf-8")
    # No brain, no WindowsDesktop: a Claude clip stops at the router.
    assert not harness.settings.claude.enabled
    assert not harness.settings.pc.enabled
    assert harness.responder.fallback is None


def test_harness_uses_production_wiring(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    import claudio.__main__ as entry

    calls: list[dict[str, Any]] = []
    sentinel = SimpleNamespace(history=[])

    def fake_build(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(entry, "build_assistant", fake_build)
    harness = rg.build_harness(
        settings, tmp_path, wake=CountingWake(), stt=FakeStt(), vad=FakeVad()
    )  # type: ignore[arg-type]
    assert harness.assistant is sentinel
    assert calls and calls[0]["responder"] is harness.responder
    assert calls[0]["speaker"] is harness.speaker


def test_tripwire_blocks_a_real_key_press(monkeypatch: pytest.MonkeyPatch) -> None:
    rg.install_tripwire(setter=monkeypatch.setattr)
    with pytest.raises(rg.RealActionBlocked):
        ctypes.windll.user32.keybd_event(0x10, 0, 0, 0)
    with pytest.raises(rg.RealActionBlocked):
        os.startfile("nothing")
    with pytest.raises(rg.RealActionBlocked):
        subprocess.Popen(["nothing"])
    import pywinauto.keyboard

    with pytest.raises(rg.RealActionBlocked):
        pywinauto.keyboard.send_keys("a")


def test_harness_keeps_its_data_in_the_work_dir(harness: rg.Harness, tmp_path: Path) -> None:
    s = harness.settings
    for path in (s.data_dir, s.logs_dir, s.cost.db_file, s.memory.db_file):
        assert s.resolve(path).is_relative_to(tmp_path), path


def test_replay_feeds_the_whole_wav_in_blocks(harness: rg.Harness, tmp_path: Path) -> None:
    block = harness.settings.audio.block_size
    samples = (np.arange(10 * block) % 20000 + 1).astype(np.int16)  # never zero
    wav = tmp_path / "clip.wav"
    sf.write(wav, samples, SR, subtype="PCM_16")
    outcome = rg.replay_clip(harness, rg.load_clip(wav))
    fed = np.concatenate(harness.wake.frames)  # type: ignore[attr-defined]
    silence = np.zeros(rg.SILENCE_S * SR, np.int16)
    tail = math.ceil(rg.tail_s(harness.settings) * SR)
    assert all(len(f) == block and f.dtype == np.int16 for f in harness.wake.frames)  # type: ignore[attr-defined]
    assert np.array_equal(fed[: len(silence)], silence)
    assert np.array_equal(fed[len(silence) : len(silence) + len(samples)], samples)
    assert not fed[len(silence) + len(samples) :].any()
    assert len(fed) >= len(silence) + len(samples) + tail
    assert len(fed) - (len(silence) + len(samples) + tail) < block
    assert tail > harness.settings.vad.no_speech_timeout_s * SR  # a wake with nothing said closes
    assert outcome.woke is False


def test_a_clip_that_never_settles_is_an_error_not_a_hang(
    monkeypatch: pytest.MonkeyPatch, harness: rg.Harness, tmp_path: Path
) -> None:
    monkeypatch.setattr(rg, "SETTLE_LIMIT_S", 0.05)
    harness.wake.frames.clear()  # type: ignore[attr-defined]
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.ones(1280, np.int16), SR, subtype="PCM_16")
    real_fresh = harness.fresh

    def stuck() -> None:
        real_fresh()
        harness.assistant._set_state(rg.State.TRANSCRIBING)  # nothing will ever end it

    harness.fresh = stuck  # type: ignore[method-assign]
    outcome = rg.replay_clip(harness, rg.load_clip(wav))
    assert outcome.outcome is not None and outcome.outcome.startswith("gate_error")


def test_gate_key_leaves_the_text_out() -> None:
    a = rg.Outcome(True, True, "claudio", "uno", "local", "x", (), (), None, "ok", False)
    b = rg.Outcome(True, True, "claudio", "due", "local", "x", (), (), None, "ok", False)
    assert a.gate_key() == b.gate_key()
    assert rg.determinism_report([a, b]) == {
        "fired": 1, "woke": 1, "wake_word": 1, "text": 2, "route": 1, "command": 1,
        "tools": 1, "actions": 1, "safety": 1, "outcome": 1, "error": 1, "args": 1,
        "tool_errors": 1, "claude_error": 1,
    }  # fmt: skip


def test_report_names_the_device() -> None:
    stt = SimpleNamespace(device="cuda", model_name="large-v3-turbo")
    line = rg.whisper_line(stt)
    assert "cuda" in line and "large-v3-turbo" in line


def test_clip_reads_its_note_and_ignores_the_old_language_key(tmp_path: Path) -> None:
    wav = tmp_path / "20261105-170000-000_ok.wav"
    sf.write(wav, np.zeros(160, np.int16), SR, subtype="PCM_16")
    assert rg.load_clip(wav).prompt is None
    wav.with_suffix(".json").write_text(
        '{"whisper_prompt": "P", "language": "it", "gain_db": 6.0}', encoding="utf-8"
    )
    clip = rg.load_clip(wav)
    assert (clip.prompt, clip.gain_db) == ("P", 6.0)


def test_build_assistant_passes_the_settings(settings: Settings) -> None:
    import claudio.__main__ as entry
    from claudio.router.matcher import CommandMatcher, load_commands
    from claudio.system import SystemControl

    conversation = settings.conversation.model_copy(update={"window_s": 7.5})
    vad = settings.vad.model_copy(update={"continue_s": 0.75})
    custom = settings.model_copy(update={"conversation": conversation, "vad": vad})
    matcher = CommandMatcher(load_commands(settings.resolve(settings.router.commands_file)), 85)
    responder = SimpleNamespace(fallback=None, fallback_guess=None, matcher=matcher, heard=None)
    clock = lambda: 1.0  # noqa: E731
    assistant = entry.build_assistant(
        custom,
        wake=CountingWake(),  # type: ignore[arg-type]
        vad=FakeVad(),  # type: ignore[arg-type]
        wake_vad=FakeVad(),  # type: ignore[arg-type]
        transcriber=FakeStt(),  # type: ignore[arg-type]
        responder=responder,  # type: ignore[arg-type]
        speaker=SimpleNamespace(),  # type: ignore[arg-type]
        system=SystemControl(),
        clock=clock,
    )
    assert assistant.conversation_s == 7.5
    assert assistant.continue_s == 0.75
    assert assistant.clock is clock


def test_build_assistant_without_a_clock_keeps_the_assistants_own(settings: Settings) -> None:
    import time

    import claudio.__main__ as entry
    from claudio.router.matcher import CommandMatcher, load_commands
    from claudio.system import SystemControl

    matcher = CommandMatcher(load_commands(settings.resolve(settings.router.commands_file)), 85)
    responder = SimpleNamespace(fallback=None, fallback_guess=None, matcher=matcher, heard=None)
    assistant = entry.build_assistant(
        settings,
        wake=CountingWake(),  # type: ignore[arg-type]
        vad=FakeVad(),  # type: ignore[arg-type]
        wake_vad=FakeVad(),  # type: ignore[arg-type]
        transcriber=FakeStt(),  # type: ignore[arg-type]
        responder=responder,  # type: ignore[arg-type]
        speaker=SimpleNamespace(),  # type: ignore[arg-type]
        system=SystemControl(),
    )
    assert assistant.clock is time.perf_counter


class GateStt(FakeStt):
    last_segments: ClassVar[list[object]] = []
    device = "cpu"
    model_name = "fake"

    def candidates(self, audio: np.ndarray, n: int = 5) -> list[str]:
        return []

    def transcribe(self, audio: np.ndarray, quick: bool = False) -> str:
        return super().transcribe(audio)


class FiringWake(CountingWake):
    """Fires `claudio` (score 0.9) on the given frames (counted from its creation)."""

    def __init__(self, fire_at: int | range | None) -> None:
        super().__init__()
        self.fire_at: range | set[int] = (
            {fire_at} if isinstance(fire_at, int) else (fire_at or set())
        )
        self.n = 0
        self.harness: rg.Harness | None = None
        self.checking_during: list[bool] = []  # per frame: was a second chance pending?

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        super().process(frame)
        self.n += 1
        if self.harness is not None:
            self.checking_during.append(bool(self.harness.assistant._checks))
        return ("claudio", 0.9) if self.n in self.fire_at else None


def loud_wav(path: Path, blocks: int) -> Path:
    t = np.arange(blocks * 1280) / SR
    sf.write(path, (np.sin(2 * np.pi * 220 * t) * 16000).astype(np.int16), SR, subtype="PCM_16")
    return path


def firing_harness(settings: Settings, tmp_path: Path, text: str = "apri chrome") -> rg.Harness:
    # The WAV starts at frame 38 (3 s of lead); the wake fires two frames into it.
    return rg.build_harness(
        settings,
        tmp_path,
        wake=FiringWake(40),
        stt=GateStt(text),
        vad=FakeVad(),  # type: ignore[arg-type]
    )


def test_a_firing_wake_reaches_the_router_and_the_actions(
    settings: Settings, tmp_path: Path
) -> None:
    harness = firing_harness(settings, tmp_path)
    outcome = rg.replay_clip(harness, rg.load_clip(loud_wav(tmp_path / "c.wav", 30)))
    assert outcome.fired and outcome.woke
    assert outcome.wake_word == "claudio"
    assert outcome.route == "local" and outcome.command == "open_chrome"
    assert outcome.actions == ("cmd /c start  chrome",)
    assert outcome.outcome == "ok" and not outcome.error


def test_a_wake_with_nothing_said_closes_its_row(settings: Settings, tmp_path: Path) -> None:
    no_voice_check = settings.wakeword.model_copy(update={"min_voice_ms": 0})  # FakeVad hears none
    harness = firing_harness(settings.model_copy(update={"wakeword": no_voice_check}), tmp_path)
    silent = tmp_path / "s.wav"
    sf.write(silent, np.zeros(10 * 1280, np.int16), SR, subtype="PCM_16")
    outcome = rg.replay_clip(harness, rg.load_clip(silent))
    assert outcome.fired and outcome.woke
    assert outcome.outcome == "timeout"


def test_a_detection_whisper_refuses_is_fired_but_not_woke(
    settings: Settings, tmp_path: Path
) -> None:
    weak = settings.wakeword.model_copy(
        update={"confirm_below": 0.95}
    )  # 0.9 is weak: Whisper checks
    harness = rg.build_harness(
        settings.model_copy(update={"wakeword": weak}),
        tmp_path,
        wake=FiringWake(40),
        stt=GateStt("niente di che"),  # not the name
        vad=FakeVad(),  # type: ignore[arg-type]
    )
    outcome = rg.replay_clip(harness, rg.load_clip(loud_wav(tmp_path / "c.wav", 30)))
    assert outcome.fired and not outcome.woke


def test_a_failing_action_is_an_error(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    real_ctx = rg.Recorder.ctx

    def blocked(argv: list[str]) -> None:
        raise rg.RealActionBlocked("tripped")

    monkeypatch.setattr(
        rg.Recorder, "ctx", lambda self: dataclasses.replace(real_ctx(self), run=blocked)
    )
    harness = firing_harness(settings, tmp_path)
    outcome = rg.replay_clip(harness, rg.load_clip(loud_wav(tmp_path / "c.wav", 30)))
    assert outcome.error


def test_a_bad_wav_is_an_error_in_its_outcome(harness: rg.Harness, tmp_path: Path) -> None:
    outcome = rg.replay_clip(harness, rg.load_clip(tmp_path / "missing.wav"))
    assert outcome.error and (outcome.outcome or "").startswith("gate_error")


def test_paced_mode_uses_the_real_clock(settings: Settings, tmp_path: Path) -> None:
    import time

    harness = rg.build_harness(
        settings,
        tmp_path,
        wake=CountingWake(),
        stt=FakeStt(),
        vad=FakeVad(),
        paced=True,  # type: ignore[arg-type]
    )
    assert harness.assistant.clock is time.perf_counter


def test_determinism_must_be_positive() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        rg.positive("0")
    assert rg.positive("20") == 20


def test_a_non_firing_clip_is_not_fired(harness: rg.Harness, tmp_path: Path) -> None:
    outcome = rg.replay_clip(harness, rg.load_clip(loud_wav(tmp_path / "c.wav", 5)))
    assert outcome.fired is False and outcome.woke is False


def test_nothing_sticks_from_one_clip_to_the_next(settings: Settings, tmp_path: Path) -> None:
    harness = firing_harness(settings, tmp_path)
    clip = rg.load_clip(loud_wav(tmp_path / "c.wav", 30))
    first = rg.replay_clip(harness, clip)
    harness.wake.fire_at = set()  # type: ignore[attr-defined]  # the same harness, now silent
    second = rg.replay_clip(harness, clip)
    assert first.fired and first.woke
    assert second.fired is False and second.woke is False and second.actions == ()


def test_a_wake_landing_with_the_row_still_open_counts(settings: Settings, tmp_path: Path) -> None:
    no_voice_check = settings.wakeword.model_copy(update={"min_voice_ms": 0})
    wav = tmp_path / "s.wav"
    sf.write(wav, np.zeros(2 * 1280, np.int16), SR, subtype="PCM_16")
    plain = settings.model_copy(update={"wakeword": no_voice_check})
    probe = rg.build_harness(plain, tmp_path, wake=CountingWake(), stt=GateStt(), vad=FakeVad())  # type: ignore[arg-type]
    rg.replay_clip(probe, rg.load_clip(wav))
    last = len(probe.wake.frames)  # type: ignore[attr-defined]
    harness = rg.build_harness(
        plain,
        tmp_path,
        wake=FiringWake(last - 1),
        stt=GateStt(),
        vad=FakeVad(),  # type: ignore[arg-type]
    )
    outcome = rg.replay_clip(harness, rg.load_clip(wav))
    assert outcome.fired and outcome.woke
    assert outcome.wake_word == "claudio" and outcome.outcome == "open"


def test_the_harness_keeps_the_memory_on(harness: rg.Harness) -> None:
    assert harness.settings.memory.enabled


def test_whisper_failing_on_the_request_is_an_error(settings: Settings, tmp_path: Path) -> None:
    harness = rg.build_harness(
        settings,
        tmp_path,
        wake=FiringWake(40),
        stt=GateStt(error=True),
        vad=FakeVad(),  # type: ignore[arg-type]
    )
    outcome = rg.replay_clip(harness, rg.load_clip(loud_wav(tmp_path / "c.wav", 30)))
    assert outcome.fired and outcome.woke
    assert outcome.outcome == "error" and outcome.error


def failing_harness(settings: Settings, tmp_path: Path) -> rg.Harness:
    return rg.build_harness(
        settings,
        tmp_path,
        wake=FiringWake(40),
        stt=GateStt(error=True),
        vad=FakeVad(),  # type: ignore[arg-type]
    )


def test_run_clips_prints_the_errors_and_says_failed(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wav = loud_wav(tmp_path / "c.wav", 30)
    assert rg.run_clips(failing_harness(settings, tmp_path), [wav], 1, False) is True
    assert "ERRORI (1/1)" in capsys.readouterr().out


def test_main_exits_1_on_an_error(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wav = loud_wav(tmp_path / "c.wav", 30)
    monkeypatch.setattr(rg, "install_tripwire", lambda: None)
    real_build = rg.build_harness

    def build_in_work_dir(_settings: Settings, work_dir: Path, **_kw: object) -> rg.Harness:
        # Built inside the work dir main hands it, so the ledger file there is really open.
        return real_build(
            settings,
            work_dir,
            wake=FiringWake(40),
            stt=GateStt(error=True),
            vad=FakeVad(),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(rg, "build_harness", build_in_work_dir)
    work = tmp_path / "gate-work"
    work.mkdir()
    monkeypatch.setattr(rg.tempfile, "mkdtemp", lambda prefix="": str(work))
    assert rg.main(["--clips", str(wav)]) == 1
    assert "ERRORI" in capsys.readouterr().out
    assert not work.exists()  # main removes its work dir (and closes the ledger first)


def test_main_rejects_a_zero_determinism(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        rg.main(["--determinism", "0", "--clips", "x.wav"])
    assert stop.value.code == 2


def test_a_bad_note_is_an_error_in_its_outcome(harness: rg.Harness, tmp_path: Path) -> None:
    wav = loud_wav(tmp_path / "c.wav", 3)
    wav.with_suffix(".json").write_text("{not json", encoding="utf-8")
    outcome = rg.replay_clip(harness, rg.load_clip(wav))
    assert outcome.error and (outcome.outcome or "").startswith("gate_error")


def test_a_bad_metrics_line_only_loses_the_comparison(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    wav = loud_wav(audio / "a_ok.wav", 3)
    (tmp_path / "metrics.jsonl").write_text(
        '{broken\n{"audio_file": "a_ok.wav", "route": "local"}\n', encoding="utf-8"
    )
    assert rg.live_row(rg.load_clip(wav)) == {"audio_file": "a_ok.wav", "route": "local"}
    (tmp_path / "metrics.jsonl").write_bytes(b"\xff\xfe not utf-8")
    assert rg.live_row(rg.load_clip(wav)) == {}


def paced_run(
    settings: Settings, tmp_path: Path, paced: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, int]:
    """Detections on frames 2-13 of a short clip, every Whisper call takes 0.2 s:
    (second chances Whisper was asked, frames fed while a check was pending)."""
    monkeypatch.setattr(rg, "SILENCE_S", 0)  # the clip starts at once: the test stays short
    weak = settings.wakeword.model_copy(
        update={"confirm_below": 0.95, "min_voice_ms": 0}
    )  # 0.9 is checked
    short_wait = settings.vad.model_copy(update={"no_speech_timeout_s": 0.5})
    custom = settings.model_copy(update={"wakeword": weak, "vad": short_wait})
    wake = FiringWake(range(2, 14))
    stt = GateStt("niente", delay=0.2)
    harness = rg.build_harness(custom, tmp_path, wake=wake, stt=stt, vad=FakeVad(), paced=paced)  # type: ignore[arg-type]
    wake.harness = harness
    wav = tmp_path / "z.wav"
    sf.write(wav, np.zeros(14 * 1280, np.int16), SR, subtype="PCM_16")
    started = time.monotonic()
    rg.replay_clip(harness, rg.load_clip(wav))
    paced_run.seconds = time.monotonic() - started  # type: ignore[attr-defined]
    return harness.detections.second_chances, sum(wake.checking_during)


def test_paced_keeps_feeding_during_a_check_and_gate_does_not(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    gate_calls, gate_fed = paced_run(settings, tmp_path / "g", False, monkeypatch)
    paced_calls, paced_fed = paced_run(settings, tmp_path / "p", True, monkeypatch)
    assert gate_fed == 0 and gate_calls >= 3  # a check whenever a detection finds none pending
    assert paced_fed >= 2  # the frames go on while Whisper listens
    frames = (
        len(rg.clip_audio(rg.load_clip(tmp_path / "p" / "z.wav"), harness_settings(settings)))
        // 1280
    )
    assert paced_run.seconds >= 0.9 * (frames - 1) * 1280 / SR  # one frame per block of time
    assert paced_calls < gate_calls  # and the guard drops the repeats


def harness_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"vad": settings.vad.model_copy(update={"no_speech_timeout_s": 0.5})}
    )


# --- the gate (G0 task 10): cassettes, manifest, scenes ---------------------------------------

PARAMS: dict[str, Any] = {
    "model": "claude-haiku-5-5",
    "system": [{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}],
    "tools": [{"name": "open_app", "input_schema": {"type": "object"}}],
    "messages": [{"role": "user", "content": "ciao"}],
    "max_tokens": 300,
}
SESSION = "2026-11-06_10-00"


async def drain(manager: Any) -> tuple[list[str], Any]:
    async with manager as stream:
        chunks = [chunk async for chunk in stream.text_stream]
        return chunks, await stream.get_final_message()


async def test_cassette_replays_what_was_recorded(tmp_path: Path) -> None:
    said = SimpleNamespace(type="text", text="Ciao mondo.")
    real = ScriptedClient((["Ciao ", "mondo."], "end_turn", [said]))
    chunks, final = await drain(rg.CassetteClient(tmp_path, real=real).stream(**PARAMS))
    replay = rg.CassetteClient(tmp_path)
    again, final2 = await drain(replay.stream(**PARAMS))
    assert again == chunks == ["Ciao ", "mondo."]
    assert final2.stop_reason == "end_turn" and final2.usage == final.usage
    assert final2.content[0].type == "text" and final2.content[0].text == "Ciao mondo."
    assert replay.misses == []
    assert [f.name for f in tmp_path.glob("*.json")] == [f"{rg.cassette_key(PARAMS)}.json"]
    assert (await replay.models.retrieve("claude-haiku-5-5")).id == "claude-haiku-5-5"
    assert replay.misses == []  # the warm-up is no miss and costs nothing


def test_cassette_key_changes_with_the_system_prompt_and_not_with_dict_order() -> None:
    key = rg.cassette_key(PARAMS)
    other_system = {**PARAMS, "system": [{"type": "text", "text": "OTHER"}]}
    assert rg.cassette_key(other_system) != key
    reordered = {
        "messages": [{"content": "ciao", "role": "user"}],
        "tools": [{"input_schema": {"type": "object"}, "name": "open_app"}],
        "system": [{"cache_control": {"type": "ephemeral"}, "text": "SYSTEM", "type": "text"}],
        "max_tokens": 1,  # not part of the key
        "model": "claude-haiku-5-5",
    }
    assert rg.cassette_key(reordered) == key


def make_corpus(root: Path, name: str, row: dict[str, Any] | None = None, note: bool = True) -> str:
    """A clip under <root>/<session>/audio with its metrics row (the live wake at 160 ms)."""
    audio = root / SESSION / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    loud_wav(audio / name, 30)
    if note:
        (audio / name).with_suffix(".json").write_text('{"whisper_prompt": "P"}', encoding="utf-8")
    live = {"audio_file": name, "prewake_ms": 160, "wake_word": "claudio", "text": "apri chrome"}
    with (root / SESSION / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**live, **(row or {})}) + "\n")
    return f"{SESSION}/audio/{name}"


def gate_harness(
    settings: Settings, work: Path, texts: str | list[str] = "apri chrome", **kw: Any
) -> rg.Harness:
    kw.setdefault("client", rg.CassetteClient(work / "cassettes"))
    return rg.build_harness(
        settings, work, wake=rg.GateWake(), stt=GateStt(texts), vad=FakeVad(), **kw
    )  # type: ignore[arg-type]


def local_gate(
    settings: Settings,
    tmp_path: Path,
    expect: dict[str, Any],
    texts: str = "apri chrome",
    note: bool = True,
    **entry: Any,
) -> tuple[rg.GateResult, rg.Harness]:
    clip = make_corpus(tmp_path / "corpus", "a.wav", note=note)
    harness = gate_harness(settings, tmp_path / "w", texts)
    result = rg.run_gate([rg.Entry(clip, expect, **entry)], harness, tmp_path / "corpus")
    return result, harness


def test_gate_compares_outcome_not_exact_text(settings: Settings, tmp_path: Path) -> None:
    expect = {"woke": True, "route": "local", "command": "open_chrome", "safety": 1}
    result, _ = local_gate(settings, tmp_path, expect, texts="Apri Chrome.")
    clip = f"{SESSION}/audio/a.wav"
    assert result.passed == [clip] and result.exit_code() == 0, result.reasons
    assert result.text_diffs == [(clip, "apri chrome", "Apri Chrome.")]


def test_known_bug_does_not_fail_and_a_fixed_one_does(settings: Settings, tmp_path: Path) -> None:
    known, _ = local_gate(
        settings,
        tmp_path / "k",
        {"command": "open_firefox"},
        known_bug="G2",
        known_diff=("command",),
    )
    assert known.known and not known.failed and known.exit_code() == 0
    fixed, _ = local_gate(
        settings,
        tmp_path / "f",
        {"command": "open_chrome"},
        known_bug="G2",
        known_diff=("command",),
    )
    assert fixed.fixed and not fixed.passed and fixed.exit_code() == 1
    failed, _ = local_gate(settings, tmp_path / "x", {"command": "open_firefox"})
    assert failed.failed and failed.exit_code() == 1


def test_clip_without_note_is_flagged(settings: Settings, tmp_path: Path) -> None:
    result, _ = local_gate(settings, tmp_path / "n", {"woke": True}, note=False)
    assert result.no_note == [f"{SESSION}/audio/a.wav"] and result.exit_code() == 0
    noted, _ = local_gate(settings, tmp_path / "y", {"woke": True})
    assert noted.no_note == []


def test_missing_clip_fails_with_its_name(settings: Settings, tmp_path: Path) -> None:
    harness = gate_harness(settings, tmp_path / "w")
    name = f"{SESSION}/audio/nothing.wav"
    result = rg.run_gate([rg.Entry(name, {"woke": True})], harness, tmp_path / "corpus")
    assert result.missing == [name] and result.exit_code() == 1


def test_gate_fails_if_it_is_not_on_the_recorder(settings: Settings, tmp_path: Path) -> None:
    harness = gate_harness(settings, tmp_path)
    ctx = harness.responder.actions.ctx
    harness.responder.actions.ctx = dataclasses.replace(ctx, run=lambda argv: None)
    scenes = harness.scenes
    with pytest.raises(rg.RealActionBlocked):
        rg.run_gate([rg.Entry("x.wav", {"woke": True})], harness, tmp_path)
    assert harness.scenes == scenes  # it refused before it started anything


def test_real_tools_are_never_built_in_the_gate(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    import claudio.pc.windows as windows

    def never(*args: object) -> None:
        raise AssertionError("WindowsDesktop was built")

    monkeypatch.setattr(windows, "WindowsDesktop", never)
    harness = gate_harness(settings, tmp_path)
    tools = rg.pc_tools(harness)
    assert tools.desktop is harness.desktop and isinstance(harness.desktop, FakeDesktop)
    brain = harness.responder.fallback.__self__  # type: ignore[union-attr]
    assert brain.client is harness.client and brain.settings.tell_time is False
    assert brain._tag() == ""  # no clock in the request, no tag
    assert tools.policy.home == harness.home.resolve()


def test_a_launched_app_shows_an_empty_window_in_the_gate() -> None:
    """A fresh app shows an empty document: the fixed text reads as an old one, and Haiku then
    refuses to write."""
    desktop = rg.GateDesktop()
    window = desktop.open_windows[0]
    assert desktop.read_window(window, 100, 1.0) == "Riga uno\nRiga due"
    desktop.launch(App("Blocco note", "notepad"))
    assert desktop.read_window(window, 100, 1.0) == ""


def test_a_new_scene_gets_its_window_text_back(settings: Settings, tmp_path: Path) -> None:
    harness = gate_harness(settings, tmp_path)
    desktop = harness.desktop
    window = desktop.open_windows[0]
    desktop.launch(App("Blocco note", "notepad"))
    desktop.paste("scritto")
    assert "scritto" in desktop.read_window(window, 100, 1.0)
    harness.fresh()
    assert desktop.read_window(window, 100, 1.0) == "Riga uno\nRiga due"
    assert desktop.typed == ""


def claude_script(*uses: SimpleNamespace) -> ScriptedClient:
    """Claude calls the tools, then answers in words."""
    return ScriptedClient(
        ([], "tool_use", list(uses)),
        (["Fatto."], "end_turn", [SimpleNamespace(type="text", text="Fatto.")]),
    )


def use(tool: str, **args: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id="toolu_1", name=tool, input=args)


def claude_gate(
    settings: Settings,
    tmp_path: Path,
    expect: dict[str, Any],
    real: ScriptedClient | None,
    work: str = "w",
) -> tuple[rg.GateResult, rg.Harness]:
    clip = make_corpus(tmp_path / "corpus", "c.wav") if work == "w" else f"{SESSION}/audio/c.wav"
    client = rg.CassetteClient(tmp_path / "cassettes", real=real)
    harness = gate_harness(
        settings, tmp_path / work, "Claudio, dimmi una cosa curiosa", client=client
    )
    return rg.run_gate([rg.Entry(clip, expect)], harness, tmp_path / "corpus"), harness


def test_gate_fails_on_a_wrong_tool_argument(settings: Settings, tmp_path: Path) -> None:
    expect = {
        "route": "claude", "tools": ["open_app"], "args": [["open_app", {"name": "Chrome"}]],
        "safety": 1,
    }  # fmt: skip
    script = claude_script(use("open_app", name="Chrome"))
    recorded, _ = claude_gate(settings, tmp_path, expect, script)
    assert recorded.passed and recorded.exit_code() == 0, recorded.reasons
    replay, _ = claude_gate(settings, tmp_path, expect, None, "w2")
    assert replay.passed and replay.misses == []  # the same cassettes, no key
    wrong = {**expect, "args": [["open_app", {"name": "Spotify"}]]}
    result, _ = claude_gate(settings, tmp_path, wrong, None, "w3")
    assert result.failed and "args" in result.reasons[result.failed[0]][0][0]


def test_a_missing_cassette_never_calls_the_api(settings: Settings, tmp_path: Path) -> None:
    client = rg.CassetteClient(tmp_path / "none")
    with pytest.raises(rg.CassetteMiss):
        client.stream(**PARAMS)
    assert client.misses == [rg.cassette_key(PARAMS)] and not (tmp_path / "none").exists()
    result, _ = claude_gate(settings, tmp_path, {"route": "claude"}, None)
    assert len(result.misses) == 1 and result.exit_code() == 1
    assert "cassetta mancante" in seen(result, f"{SESSION}/audio/c.wav")


def test_keys_do_not_depend_on_the_work_dir(settings: Settings, tmp_path: Path) -> None:
    """The fake home is in the work dir, a different one each run: no path of it may reach
    Claude, or every replay would be a miss."""
    real = claude_script(use("open_path", path="~\\Downloads"))

    def work(name: str) -> Path:
        (tmp_path / name / "home" / "Downloads").mkdir(parents=True)
        return tmp_path / name

    expect = {"tools": ["open_path"]}
    clip = make_corpus(tmp_path / "corpus", "c.wav")
    cassettes = tmp_path / "cas"
    ask = "Claudio, dimmi una cosa curiosa"
    first = gate_harness(settings, work("one"), ask, client=rg.CassetteClient(cassettes, real=real))
    assert rg.run_gate([rg.Entry(clip, expect)], first, tmp_path / "corpus").passed
    sent = json.dumps(real.calls[1]["messages"])
    assert "Opened ~" in sent and "claudio-gate" not in sent and str(tmp_path) not in sent
    second = gate_harness(settings, work("two"), ask, client=rg.CassetteClient(cassettes))
    replay = rg.run_gate([rg.Entry(clip, expect)], second, tmp_path / "corpus")
    assert replay.passed and replay.misses == []


def test_manifest_rejects_what_it_does_not_know(tmp_path: Path) -> None:
    def load(*lines: str) -> list[rg.Entry]:
        path = tmp_path / "m.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return rg.load_manifest(path)

    ok = (
        '{"clip": "a.wav", "expect": {"route": "local"}, "known_bug": "G2", '
        '"known_diff": ["route"], "why": "x"}'
    )
    assert load(ok)[0].known_bug == "G2"
    for bad in (
        '{"clip": "a.wav", "expect": {}, "knwon_bug": "G2"}',
        '{"clip": "a.wav", "expect": {"rout": "local"}}',
        '{"clip": "a.wav", "expect": {"args": ["open_app"]}}',
        '{"clip": "a.wav", "expect": {"woke": true}, "start": "off"}',  # not a field any more
        "not json",
    ):
        with pytest.raises(rg.ManifestError):
            load(bad)
    scene = '{"clip": "%s.wav", "expect": {"woke": true}, "scene": "%s"%s}'
    with pytest.raises(rg.ManifestError):  # a scene split by another line
        load(scene % ("a", "s", ""), scene % ("b", "t", ""), scene % ("c", "s", ""))


def test_a_scene_runs_its_clips_on_one_assistant(settings: Settings, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    first = make_corpus(corpus, "a.wav")
    second = make_corpus(corpus, "b.wav", {"prewake_ms": None, "wake_word": "conversation"})
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", "apri download"])
    manifest = [
        rg.Entry(first, {"woke": True, "command": "open_chrome"}, scene="s"),
        rg.Entry(second, {"woke": True, "command": "open_downloads"}, scene="s"),
    ]
    result = rg.run_gate(manifest, harness, corpus)
    assert result.passed == [first, second], result.reasons
    assert len(harness.recorder.ran) + len(harness.recorder.opened) == 2  # one Assistant ran both


def test_every_scene_gets_a_memory_store_of_its_own(settings: Settings, tmp_path: Path) -> None:
    harness = gate_harness(settings, tmp_path)
    harness.fresh()
    harness.fresh()
    names = sorted(p.name for p in (tmp_path / "data").glob("memory-*.db"))
    assert names == ["memory-1.db", "memory-2.db", "memory-3.db"]  # build_harness made the first
    # A memory's date reaches the prompt: its clock is fixed, so a cassette key never ages.
    assert harness.responder.memory.store.clock() == rg.GATE_TIME  # type: ignore[union-attr]


def test_a_clip_marked_real_wake_must_fire_the_real_model(
    settings: Settings, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "a.wav")
    entry = rg.Entry(clip, {"woke": True}, real_wake=True)
    silent = gate_harness(settings, tmp_path / "s", real_wake=CountingWake())  # never fires
    lost = rg.run_gate([entry], silent, corpus)
    assert lost.failed == [f"{clip} [real_wake]"] and lost.exit_code() == 1
    firing = gate_harness(settings, tmp_path / "f", real_wake=FiringWake(40))
    assert rg.run_gate([entry], firing, corpus).exit_code() == 0
    assert firing.wake is not firing.real_wake  # pass 1 went back to the replay wake


def test_the_report_names_counts_and_the_text_differences() -> None:
    result = rg.GateResult(passed=["a"], failed=["b"], text_diffs=[("a", "uno", "due")])
    result.reasons["b"] = [("route", "route: atteso 'local', trovato 'claude'")]
    text = rg.render_report(result, "Whisper: fake su cpu", "claude-haiku-5-5")
    assert "Whisper: fake su cpu" in text and "'uno'" in text and "b: route" in text
    assert rg.release_line(result) == (
        "Gate di replay: 1 superate, 1 fallite, 0 note, 0 corrette da togliere, "
        "0 cassette mancanti, 0 clip mancanti."
    )


class SdkClient:
    """Shaped as the SDK's client, answering with a real `anthropic.types.Message`."""

    def __init__(self, message: Any) -> None:
        self.message, self.messages = message, self

    def stream(self, **params: Any) -> Any:
        message = self.message

        class Stream:
            text_stream = ScriptedStream(["Apro."], "tool_use", [], {}).text_stream

            async def get_final_message(self) -> Any:
                return message

        class Manager:
            async def __aenter__(self) -> Any:
                return Stream()

            async def __aexit__(self, *exc: object) -> None:
                return None

        return Manager()


async def test_the_real_sdk_message_survives_record_and_replay(tmp_path: Path) -> None:
    from anthropic.types import Message, TextBlock, ToolUseBlock
    from anthropic.types import Usage as ApiUsage

    from claudio.brain.claude import _Leg, _Turn
    from claudio.cost.pricing import Usage

    message = Message(
        id="msg_1",
        content=[
            TextBlock(type="text", text="Apro."),
            ToolUseBlock(type="tool_use", id="toolu_9", name="open_app", input={"name": "Chrome"}),
        ],
        model="claude-haiku-5-5",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=ApiUsage(
            input_tokens=11,
            output_tokens=7,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=3,
        ),
    )
    _, live = await drain(rg.CassetteClient(tmp_path, real=SdkClient(message)).stream(**PARAMS))
    replay = rg.CassetteClient(tmp_path)
    _, saved = await drain(replay.stream(**PARAMS))
    assert replay.tool_uses == [("open_app", {"name": "Chrome"})]
    assert Usage.from_api(saved.usage, "5m") == Usage.from_api(live.usage, "5m")
    a, b = _Leg(_Turn("")), _Leg(_Turn(""))
    a.read(live)
    b.read(saved)
    assert a.content == b.content and a.calls == b.calls and a.calls
    follow = [*PARAMS["messages"], {"role": "assistant", "content": a.content}]
    again = [*PARAMS["messages"], {"role": "assistant", "content": b.content}]
    assert rg.cassette_key({**PARAMS, "messages": follow}) == rg.cassette_key(
        {**PARAMS, "messages": again}
    )  # the next leg finds its cassette
    saved_text = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json"))
    assert "api_key" not in saved_text.lower() and "x-api" not in saved_text.lower()


async def test_a_forced_retry_has_its_own_cassette(tmp_path: Path) -> None:
    forced = {**PARAMS, "tool_choice": {"type": "tool", "name": "type_text"}}
    assert rg.cassette_key(forced) != rg.cassette_key(PARAMS)
    loose = {**forced, "timeout": 10.0, "max_tokens": 2048}
    assert rg.cassette_key(loose) == rg.cassette_key(forced)
    cut = SimpleNamespace(type="text", text="Scrivo.")
    real = ScriptedClient((["Scrivo."], "max_tokens", [cut]))
    await drain(rg.CassetteClient(tmp_path, real=real).stream(**PARAMS))
    replay = rg.CassetteClient(tmp_path)
    with pytest.raises(rg.CassetteMiss):
        replay.stream(**forced)  # the retry is not answered with the cut again
    _, again = await drain(rg.CassetteClient(tmp_path).stream(**PARAMS, timeout=10.0))
    assert again.stop_reason == "max_tokens"  # `timeout=` is accepted and left out of the key


async def test_a_call_cut_at_max_tokens_survives_record_and_replay(tmp_path: Path) -> None:
    from anthropic.types import Message, TextBlock, ToolUseBlock
    from anthropic.types import Usage as ApiUsage

    message = Message(
        id="msg_cut",
        content=[
            TextBlock(type="text", text="Scrivo."),
            ToolUseBlock(type="tool_use", id="toolu_c", name="type_text", input={"text": "Pia"}),
        ],
        model="claude-haiku-5-5",
        role="assistant",
        stop_reason="max_tokens",
        stop_sequence=None,
        type="message",
        usage=ApiUsage(input_tokens=11, output_tokens=2048),
    )
    await drain(rg.CassetteClient(tmp_path, real=SdkClient(message)).stream(**PARAMS))
    _, saved = await drain(rg.CassetteClient(tmp_path).stream(**PARAMS))
    assert saved.stop_reason == "max_tokens"
    assert [(b.type, getattr(b, "name", None)) for b in saved.content] == [
        ("text", None),
        ("tool_use", "type_text"),
    ]
    assert saved.content[1].input == {"text": "Pia"}  # the partial block is still there


def test_a_cut_then_a_retry_records_and_replays_only_what_ran(
    settings: Settings, tmp_path: Path
) -> None:
    # The forced retry runs only on Haiku (Sonnet answers 400 to tool_choice): pin it.
    haiku = settings.claude.model_copy(update={"model": "claude-haiku-5-5"})
    settings = settings.model_copy(update={"claude": haiku})
    expect = {"route": "claude", "tools": ["open_app"], "args": [["open_app", {"name": "Chrome"}]]}
    say = SimpleNamespace(type="text", text="Apro.")
    script = ScriptedClient(
        (["Apro."], "max_tokens", [say, use("open_app", name="Chr")]),
        ([], "tool_use", [use("open_app", name="Chrome")]),
        (["Fatto."], "end_turn", [SimpleNamespace(type="text", text="Fatto.")]),
    )
    recorded, harness = claude_gate(settings, tmp_path, expect, script)
    assert recorded.passed and not recorded.failed, recorded.reasons
    assert rg.read_outcome(harness).args == ('open_app {"name": "Chrome"}',)
    assert len(list((tmp_path / "cassettes").glob("*.json"))) == 3
    replay, again = claude_gate(settings, tmp_path, expect, None, "w2")
    assert replay.passed and not replay.failed and replay.misses == [], replay.reasons
    row = rg.request_row(list(again.assistant.history))
    assert row["cut_retry"] == "ok"  # the brain saw a CUT_CALL through the cassette


def test_a_tool_that_never_ran_is_not_counted_as_run() -> None:
    row = {"tools": ["open_app(Chrome) ok", "type_text() not_run_cut", "run_command(x) error"]}
    harness = SimpleNamespace(
        assistant=SimpleNamespace(history=[row], _interaction=None),
        recorder=SimpleNamespace(ran=[], opened=[], keys=[], combos=[]),
        client=None,
        levels=[],
        command_levels={},
    )
    outcome = rg.read_outcome(harness, until=rg.Mark(rows=1))  # type: ignore[arg-type]
    assert outcome.tools == ("open_app", "run_command")
    assert outcome.tool_errors == ("run_command",)


def test_the_tripwire_blocks_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    import win32gui

    from claudio import gpu

    rg.install_tripwire(setter=monkeypatch.setattr)
    with pytest.raises(rg.RealActionBlocked):
        ctypes.windll.user32.SendInput(0, None, 0)
    with pytest.raises(rg.RealActionBlocked):
        os.system("echo nothing")
    with pytest.raises(rg.RealActionBlocked):
        gpu.read_gpu()  # its Popen was bound as a default, before the tripwire
    with pytest.raises(rg.RealActionBlocked):
        win32gui.PostMessage(0, 0x10, 0, 0)


def test_record_asks_before_spending_and_only_si_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clip = make_corpus(tmp_path / "corpus", "c.wav")
    harness = gate_harness(settings, tmp_path / "w", "Claudio, dimmi una cosa curiosa")
    manifest = [rg.Entry(clip, {"route": "claude"})]
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    assert (
        rg.confirm_recording(harness, manifest, tmp_path / "cassettes", tmp_path / "corpus", 1.0)
        is False
    )
    out = capsys.readouterr().out
    assert "almeno 1 cassette" in out and "$" in out and "dimmi" not in out
    monkeypatch.setattr("builtins.input", lambda _prompt: "si")
    assert (
        rg.confirm_recording(harness, manifest, tmp_path / "cassettes", tmp_path / "corpus", 1.0)
        is True
    )
    assert not (tmp_path / "cassettes").exists()  # asking recorded nothing


# --- fix round 1: what the review found ---------------------------------------------------------

ASK = "Claudio, dimmi una cosa curiosa"


def seen(result: rg.GateResult, clip: str) -> str:
    """All the reasons of a clip, public and detailed."""
    return " | ".join(f"{a} / {b}" for a, b in result.reasons.get(clip, []))


def test_known_bug_only_absorbs_the_listed_difference(settings: Settings, tmp_path: Path) -> None:
    clip = f"{SESSION}/audio/a.wav"
    # 1. an error is never a known bug (a tripped tripwire would look the same)
    bad_note = make_corpus(tmp_path / "e" / "corpus", "a.wav")
    (tmp_path / "e" / "corpus" / bad_note).with_suffix(".json").write_text("{x", encoding="utf-8")
    harness = gate_harness(settings, tmp_path / "e" / "w")
    entry = rg.Entry(bad_note, {"route": "claude"}, known_bug="G2", known_diff=("route",))
    errored = rg.run_gate([entry], harness, tmp_path / "e" / "corpus")
    assert errored.failed == [clip] and not errored.known and errored.exit_code() == 1
    # 2. a difference outside known_diff: the failure changed
    changed, _ = local_gate(
        settings,
        tmp_path / "c",
        {"command": "open_firefox", "route": "claude"},
        known_bug="G2",
        known_diff=("command",),
    )
    assert changed.failed == [clip] and "cambiato" in seen(changed, clip)
    # 3. the same difference as declared: known
    same, _ = local_gate(
        settings,
        tmp_path / "s",
        {"command": "open_firefox"},
        known_bug="G2",
        known_diff=("command",),
    )
    assert same.known == [clip] and same.exit_code() == 0


def test_known_diff_is_required_and_checked() -> None:
    expect = {"route": "local"}
    for kw in (
        {"known_bug": "G2"},  # a bug with no list of what is wrong
        {"known_bug": "G2", "known_diff": ()},
        {"known_bug": "G2", "known_diff": ("nope",)},
        {"known_diff": ("route",)},  # a list with no bug
    ):
        with pytest.raises(rg.ManifestError):
            rg.Entry("a.wav", expect, **kw)  # type: ignore[arg-type]


def test_silent_passes_are_errors(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(rg.ManifestError):
        rg.load_manifest(path)
    harness = gate_harness(settings, tmp_path / "w")
    with pytest.raises(rg.ManifestError):
        rg.run_gate([], harness, tmp_path)
    with pytest.raises(rg.ManifestError):  # expect {} passes on anything
        rg.Entry("a.wav", {})
    with pytest.raises(rg.ManifestError):  # args: null
        rg.parse_entry('{"clip": "a.wav", "expect": {"args": null}}', "m:1")


def test_a_clip_with_no_wake_point_proves_nothing_unless_it_says_so(
    settings: Settings, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "a.wav")
    (corpus / SESSION / "metrics.jsonl").unlink()  # no live row, no prewake_ms
    harness = gate_harness(settings, tmp_path / "w")
    silent = rg.run_gate([rg.Entry(clip, {"woke": False, "tools": []})], harness, corpus)
    assert silent.failed == [clip] and "no_wake" in seen(silent, clip)
    meant = rg.run_gate([rg.Entry(clip, {"woke": False}, no_wake=True)], harness, corpus)
    assert meant.passed == [clip] and meant.exit_code() == 0
    armed = make_corpus(tmp_path / "c2", "a.wav")  # a row with a wake point, but no_wake wins
    again = gate_harness(settings, tmp_path / "w2")
    unarmed = rg.run_gate([rg.Entry(armed, {"woke": False}, no_wake=True)], again, tmp_path / "c2")
    assert unarmed.passed == [armed]


def two_clip_scene(
    tmp_path: Path, second: dict[str, Any], silent_second: bool = False
) -> tuple[Path, str, str]:
    corpus = tmp_path / "corpus"
    first = make_corpus(corpus, "a.wav")
    last = make_corpus(corpus, "b.wav", {"prewake_ms": None, "wake_word": "conversation", **second})
    if silent_second:
        sf.write(corpus / last, np.zeros(1280 * 30, np.int16), SR, subtype="PCM_16")
    return corpus, first, last


def test_a_silent_last_clip_does_not_borrow_the_open_window(
    settings: Settings, tmp_path: Path
) -> None:
    corpus, first, last = two_clip_scene(tmp_path, {}, silent_second=True)
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", "apri download"])
    scene = [rg.Entry(first, {"woke": True}, scene="s"), rg.Entry(last, {"woke": True}, scene="s")]
    result = rg.run_gate(scene, harness, corpus)
    assert result.failed == [last] and "finestra" in seen(result, last)


def test_a_window_that_stays_shut_fails_the_clip(settings: Settings, tmp_path: Path) -> None:
    corpus, first, last = two_clip_scene(tmp_path, {})
    harness = gate_harness(settings, tmp_path / "w", ["", "apri download"])  # no reply, no window
    scene = [rg.Entry(first, {"woke": False}, scene="s"), rg.Entry(last, {"woke": True}, scene="s")]
    result = rg.run_gate(scene, harness, corpus)
    assert last in result.failed and "finestra" in seen(result, last)


def test_a_chiuditi_clip_does_not_crash_the_gate(settings: Settings, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "a.wav")
    harness = gate_harness(settings, tmp_path / "w", "Claudio, chiuditi")
    result = rg.run_gate([rg.Entry(clip, {"command": "shutdown"})], harness, corpus)
    assert result.passed == [clip], result.reasons
    corpus, first, last = two_clip_scene(tmp_path / "s", {})
    harness = gate_harness(settings, tmp_path / "w2", ["Claudio, chiuditi", "apri download"])
    scene = [rg.Entry(first, {"woke": True}, scene="s"), rg.Entry(last, {"woke": True}, scene="s")]
    cut = rg.run_gate(scene, harness, corpus)
    assert cut.passed == [first] and cut.failed == [last]  # it never got its turn
    assert "interrott" in seen(cut, last)


def test_the_safety_class_is_the_one_asked_at_the_time(settings: Settings, tmp_path: Path) -> None:
    # type_text into Notepad is level 1 (free typing): no "Confermi?", and the gate says 1.
    typing = claude_script(use("type_text", text="ciao", window="Blocco note"))
    free, _ = claude_gate(settings, tmp_path / "a", {"safety": 1, "tools": ["type_text"]}, typing)
    assert free.passed, free.reasons
    # close_window through Claude is level 2: asked, not run.
    closing = ScriptedClient(([], "tool_use", [use("close_window", window="Blocco note")]))
    asked, harness = claude_gate(settings, tmp_path / "b", {"safety": 2}, closing)
    assert asked.passed, asked.reasons
    assert any("onfermi" in s for s in harness.speaker.spoken)
    assert not [d for d in harness.desktop.done if d[0] == "close"]  # nothing was closed
    # a local command of level 2
    local, _ = local_gate(
        settings, tmp_path / "c", {"command": "close_window", "safety": 2}, "chiudi la finestra"
    )
    assert local.passed, local.reasons


def test_local_actions_can_be_expected(settings: Settings, tmp_path: Path) -> None:
    ok, _ = local_gate(settings, tmp_path / "a", {"actions": ["cmd /c start  chrome"]})
    assert ok.passed, ok.reasons
    wrong, _ = local_gate(settings, tmp_path / "b", {"actions": ["cmd /c start  firefox"]})
    assert wrong.failed and "actions" in seen(wrong, wrong.failed[0])


def test_the_console_names_what_the_constraints_require(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    quiet = make_corpus(corpus, "a.wav", note=False)
    gone = f"{SESSION}/audio/gone.wav"
    harness = gate_harness(settings, tmp_path / "w", ASK)
    manifest = [rg.Entry(quiet, {"route": "claude"}), rg.Entry(gone, {"woke": True})]
    result = rg.run_gate(manifest, harness, corpus)
    rg.print_result(result)
    out = capsys.readouterr().out
    assert gone in out and quiet in out  # the missing clip and the clip with no note
    assert "--record" in out  # the whole missing-cassette message, not its first word
    assert ASK not in out
    report = rg.render_report(result, "Whisper: x", "m")
    assert gone in report and quiet in report
    assert f"| fallita | {quiet} |" in report or f"| fallite | {quiet} |" in report


def test_the_estimate_prices_every_round_at_the_cap_the_brain_sends(
    settings: Settings, tmp_path: Path
) -> None:
    """Sonnet main with thinking and tools sends 2048 + 850 output tokens a round."""
    from claudio.cost.pricing import Usage

    pricing = gate_harness(settings, tmp_path / "p").tracker.pricing
    claude = settings.claude.model_copy(
        update={"model": "claude-sonnet-5-5", "thinking": True, "escalate": False}
    )
    pc = settings.pc.model_copy(update={"max_rounds": 1})
    shipped = settings.model_copy(update={"claude": claude, "pc": pc})
    tokens = math.ceil(10_000 / rg.TOKEN_CHARS_FLOOR)
    expected = pricing.cost(
        "claude-sonnet-5-5", Usage(cache_write_1h_tokens=tokens, output_tokens=2898)
    )
    assert rg.upper_bound_usd(1, 10_000, shipped, pricing) == pytest.approx(expected)


def test_the_estimate_is_a_real_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from claudio.cost.pricing import Usage

    pricing = gate_harness(settings, tmp_path / "p").tracker.pricing
    claude = settings.claude
    assert claude.smart_model
    sonnet = pricing.cost(
        claude.smart_model,
        Usage(cache_write_1h_tokens=5000, output_tokens=claude.smart_max_tokens),
    )
    haiku = pricing.cost(
        claude.model,
        Usage(
            cache_write_1h_tokens=5000, output_tokens=max(claude.max_tokens, settings.pc.max_tokens)
        ),
    )
    one = rg.upper_bound_usd(1, 10_000, settings, pricing)
    assert one >= sonnet * settings.pc.max_rounds > haiku  # the dearest model, every round
    assert rg.upper_bound_usd(3, 10_000, settings, pricing) == pytest.approx(3 * one)
    # a miss hides the later clips of its scene: the count says so
    corpus, first, last = two_clip_scene(tmp_path, {})
    harness = gate_harness(settings, tmp_path / "w", [ASK, ASK])
    scene = [rg.Entry(first, {"woke": True}, scene="s"), rg.Entry(last, {"woke": True}, scene="s")]
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "no")
    rg.confirm_recording(harness, scene, tmp_path / "cas", corpus, 1.0)
    out = capsys.readouterr().out
    assert "almeno 1 cassette" in out and "fino a 2 richieste" in out and "Non copre" in out
    assert "spend.db" in asked[0]  # the spend does not reach JenJo's tally


def test_spend_counts_only_the_replies_that_were_recorded(tmp_path: Path) -> None:
    from claudio.cost.pricing import load_pricing
    from tests.unit.test_cost import PRICES

    pricing = load_pricing(PRICES)
    real = ScriptedClient((["a"], "end_turn", []), (["b"], "end_turn", []))
    live = rg.CassetteClient(tmp_path, real=real, pricing=pricing)
    first = {**PARAMS, "messages": [{"role": "user", "content": "uno"}]}
    other = {**PARAMS, "messages": [{"role": "user", "content": "due"}]}
    asyncio.run(drain(live.stream(**first)))
    assert live.spent > 0
    one = live.spent
    replay = rg.CassetteClient(tmp_path, pricing=pricing)
    asyncio.run(drain(replay.stream(**first)))
    assert replay.spent == 0  # a replay costs nothing, whatever the tracker says
    again = rg.CassetteClient(tmp_path, real=real, pricing=pricing)
    asyncio.run(drain(again.stream(**first)))
    asyncio.run(drain(again.stream(**other)))
    assert again.spent == pytest.approx(one)  # only the new reply


def test_recording_stops_at_the_daily_budget_and_names_what_it_skipped(tmp_path: Path) -> None:
    from claudio.cost.pricing import load_pricing
    from tests.unit.test_cost import PRICES

    real = ScriptedClient((["a"], "end_turn", []), (["b"], "end_turn", []))
    live = rg.CassetteClient(tmp_path, real=real, pricing=load_pricing(PRICES), budget=1e-9)
    first = {**PARAMS, "messages": [{"role": "user", "content": "uno"}]}
    other = {**PARAMS, "messages": [{"role": "user", "content": "due"}]}
    asyncio.run(drain(live.stream(**first)))
    with pytest.raises(rg.CassetteMiss):
        live.stream(**other)
    assert live.braked == [rg.cassette_key(other)] and len(real.calls) == 1  # no second call
    assert live.misses == [rg.cassette_key(other)]


def test_a_failing_stream_saves_nothing(tmp_path: Path) -> None:
    class Broken(ScriptedClient):
        def stream(self, **params: Any) -> Any:
            manager = super().stream(**params)

            class Stream:
                @property
                def text_stream(self) -> Any:
                    async def boom() -> Any:
                        yield "a"
                        raise ConnectionError("down")

                    return boom()

            class Wrapped:
                async def __aenter__(self) -> Any:
                    await manager.__aenter__()
                    return Stream()

                async def __aexit__(self, *exc: object) -> None:
                    return None

            return Wrapped()

    live = rg.CassetteClient(tmp_path, real=Broken((["a"], "end_turn", [])))
    with pytest.raises(ConnectionError):
        asyncio.run(drain(live.stream(**PARAMS)))
    assert not list(tmp_path.glob("*"))


def test_the_wake_fires_on_the_recorded_frame(settings: Settings, tmp_path: Path) -> None:
    block = settings.audio.block_size
    wav = tmp_path / "w.wav"
    sf.write(wav, np.full(30 * block, 1000, np.int16), SR, subtype="PCM_16")  # never zero
    for prewake_ms in (160, 100, 1000):
        audio, fire_at = rg.clip_parts(rg.load_clip(wav), settings, prewake_ms)
        assert fire_at is not None
        starts = int(np.flatnonzero(audio)[0])  # where the WAV begins
        assert fire_at * block - starts == round(prewake_ms * SR / 1000)  # the live offset
    # and the wake object really sees it on that frame
    armed = rg.GateWake()
    armed.arm(3)
    assert [armed.process(np.zeros(block, np.int16)) is not None for _ in range(4)] == [
        False, False, True, False,
    ]  # fmt: skip


def test_the_home_has_the_folders_a_user_has(settings: Settings, tmp_path: Path) -> None:
    harness = gate_harness(settings, tmp_path)
    for sub in ("Desktop", "Documents", "Downloads"):
        assert (harness.home / sub).is_dir()
    assert harness.settings.pc.roots == ["~"]  # never the whole disk


def test_a_tool_that_failed_fails_the_clip_unless_it_is_expected(
    settings: Settings, tmp_path: Path
) -> None:
    broken = claude_script(use("open_app", name="Programma che non esiste"))
    failed, _ = claude_gate(settings, tmp_path, {"tools": ["open_app"]}, broken)
    clip = f"{SESSION}/audio/c.wav"
    assert failed.failed == [clip] and "tool_errors" in seen(failed, clip)
    meant = {"tools": ["open_app"], "tool_errors": ["open_app"]}
    replay, _ = claude_gate(settings, tmp_path, meant, None, "w2")
    assert replay.passed, replay.reasons  # expected, so it passes (same cassettes)
    opened = claude_script(use("open_path", path="~\\Downloads"))
    fine, _ = claude_gate(settings, tmp_path / "d", {"tools": ["open_path"]}, opened)
    assert fine.passed, fine.reasons  # the fake home has its Downloads


def test_a_second_scene_sees_only_its_own_tool_calls(settings: Settings, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    one, two = make_corpus(corpus, "a.wav"), make_corpus(corpus, "b.wav")
    real = claude_script(use("open_app", name="Chrome"))
    client = rg.CassetteClient(tmp_path / "cas", real=real)
    harness = gate_harness(settings, tmp_path / "w", ASK, client=client)
    args = [["open_app", {"name": "Chrome"}]]
    manifest = [rg.Entry(c, {"route": "claude", "args": args}) for c in (one, two)]
    result = rg.run_gate(manifest, harness, corpus)
    assert result.passed == [one, two], result.reasons  # the second one replays the first's


def test_a_missing_cassette_lands_on_the_clip_that_missed(
    settings: Settings, tmp_path: Path
) -> None:
    corpus, first, last = two_clip_scene(tmp_path, {})
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", ASK])
    scene = [rg.Entry(first, {"woke": True}, scene="s"), rg.Entry(last, {"woke": True}, scene="s")]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first] and result.failed == [last]
    assert "cassetta mancante" in seen(result, last)


def test_a_real_wake_clip_that_errors_says_why(settings: Settings, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "a.wav")
    (corpus / clip).with_suffix(".json").write_text("{x", encoding="utf-8")  # the clip is an error
    harness = gate_harness(settings, tmp_path / "w", real_wake=FiringWake(40))
    result = rg.run_gate([rg.Entry(clip, {"woke": True}, real_wake=True)], harness, corpus)
    assert "errore" in seen(result, f"{clip} [real_wake]")


def test_no_key_warning_when_a_client_is_injected(
    settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING")
    no_key = settings.model_copy(update={"anthropic_api_key": None})  # whatever .env holds
    gate_harness(no_key, tmp_path)
    assert "ANTHROPIC_API_KEY" not in caplog.text


def test_the_gate_refuses_to_create_the_personality(settings: Settings, tmp_path: Path) -> None:
    missing = settings.claude.model_copy(update={"personality_file": tmp_path / "CLAUDIO.md"})
    with pytest.raises(rg.GateRefused):
        rg.require_personality(settings.model_copy(update={"claude": missing}))
    assert not (tmp_path / "CLAUDIO.md").exists()
    rg.require_personality(settings)  # the real one is there


def test_gate_main_prints_a_bad_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"clip": "a.wav", "expect": {"woke": true}, "typo": 1}\n', encoding="utf-8"
    )
    assert rg.gate_main(False, False, manifest) == 1
    assert "manifest non valido" in capsys.readouterr().out


# --- fix round 2 ---------------------------------------------------------------------------------


class TrippingDesktop(FakeDesktop):
    """Starting an app goes through subprocess, which the tripwire blocks."""

    def launch(self, app: Any) -> None:
        subprocess.Popen(["notepad"])


def test_the_home_folders_carry_a_fixed_date(settings: Settings, tmp_path: Path) -> None:
    """search_files tells Claude `(modified <date>)`: a folder made today would put today's date
    in the next leg's messages, and the cassette would miss tomorrow."""
    harness = gate_harness(settings, tmp_path)
    folders = [harness.home, *(p for p in harness.home.rglob("*") if p.is_dir())]
    assert len(folders) > 3
    assert {round(p.stat().st_mtime) for p in folders} == {round(rg.GATE_TIME)}
    real = claude_script(use("search_files", query="Downloads"))
    result, _ = claude_gate(settings, tmp_path / "s", {"tools": ["search_files"]}, real)
    assert result.passed, result.reasons
    sent = json.dumps(real.calls[1]["messages"])
    today = time.strftime("%Y-%m-%d")
    assert "modified" in sent and today not in sent


def test_a_tripped_tripwire_in_a_tool_can_never_pass(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    rg.install_tripwire(setter=monkeypatch.setattr)
    clip = make_corpus(tmp_path / "corpus", "c.wav")
    cassettes = tmp_path / "cas"
    named = {"tools": ["open_app"], "tool_errors": ["open_app"]}
    for n, entry in enumerate(
        [
            rg.Entry(clip, named),  # naming the error does not absorb a trip
            rg.Entry(clip, {"tools": ["open_app"]}, known_bug="G9", known_diff=("tool_errors",)),
        ]
    ):
        real = claude_script(use("open_app", name="Chrome"))
        client = rg.CassetteClient(cassettes, real=real)
        harness = gate_harness(
            settings, tmp_path / f"w{n}", ASK, client=client, desktop=TrippingDesktop()
        )
        result = rg.run_gate([entry], harness, tmp_path / "corpus")
        assert result.failed == [clip] and result.exit_code() == 1, result.reasons
        assert "tripwire" in seen(result, clip)
        # recording: the failure must not be frozen as a good reply
        assert not list(cassettes.glob("*.json"))


def test_the_level_is_the_one_asked_before_the_tools_ran(
    settings: Settings, tmp_path: Path
) -> None:
    """list_windows brings outside text, so a `remember` after it would be level 2 *now*; at
    the time Claude's reply was judged it was level 1 and no yes was asked."""
    both = ScriptedClient(
        ([], "tool_use", [use("list_windows"), use("remember", text="mi piace il caffè")]),
        (["Fatto."], "end_turn", [SimpleNamespace(type="text", text="Fatto.")]),
    )
    result, harness = claude_gate(settings, tmp_path, {"safety": 1}, both)
    assert result.passed, result.reasons
    assert not any("onfermi" in s for s in harness.speaker.spoken)


def test_a_refused_tool_is_a_tool_error(settings: Settings, tmp_path: Path) -> None:
    refused = claude_script(use("run_command", command="dir", purpose=""))
    first, _ = claude_gate(settings, tmp_path, {"tools": ["run_command"]}, refused)
    clip = f"{SESSION}/audio/c.wav"
    assert first.failed == [clip] and "tool_errors" in seen(first, clip)
    named = {"tools": ["run_command"], "tool_errors": ["run_command"]}
    replay, _ = claude_gate(settings, tmp_path, named, None, "w2")
    assert replay.passed, replay.reasons


def test_known_diff_with_the_wrong_shape_is_a_manifest_error() -> None:
    for bad in ('[["woke"]]', '[1, "x"]', '"woke"'):
        line = '{"clip": "a.wav", "expect": {"woke": true}, "known_bug": "G", "known_diff": %s}'
        with pytest.raises(rg.ManifestError):
            rg.parse_entry(line % bad, "m:1")


def test_the_record_prompt_says_where_the_brake_stops_it(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clip = make_corpus(tmp_path / "corpus", "c.wav")
    harness = gate_harness(settings, tmp_path / "w", ASK)
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "no")
    rg.confirm_recording(
        harness,
        [rg.Entry(clip, {"woke": True})],
        tmp_path / "cas",
        tmp_path / "corpus",
        settings.cost.daily_budget,
    )
    out = capsys.readouterr().out + asked[0]
    assert f"{settings.cost.daily_budget:.2f}" in out and "per corsa" in out
    assert "domani" not in out and "spend.db" in out


# --- fix round 3: what the first real run found -------------------------------------------------


def long_window_scene(tmp_path: Path, lead_s: float) -> tuple[Path, str, str]:
    """A, then B whose WAV holds `lead_s` of silence before the voice: a conversation clip is
    saved from the first frame its window listened to, so its WAV starts with the wait."""
    corpus = tmp_path / "corpus"
    first = make_corpus(corpus, "a.wav")
    second = make_corpus(corpus, "b.wav", {"prewake_ms": None, "wake_word": "conversation"})
    voice = np.full(1280 * 12, 9000, np.int16)
    wait = np.zeros(round(lead_s * SR), np.int16)
    sf.write(corpus / second, np.concatenate((wait, voice)), SR, subtype="PCM_16")
    return corpus, first, second


def test_a_clip_that_holds_its_own_wait_is_heard_in_the_window(
    settings: Settings, tmp_path: Path
) -> None:
    """F1: the WAV of a conversation clip already holds the silence of the window (5 s live).
    The gate added its own gap on top, the 6 s window closed, and the clip was not heard."""
    corpus, first, second = long_window_scene(tmp_path, 5.8)
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", "apri download"])
    scene = [
        rg.Entry(first, {"woke": True, "command": "open_chrome"}, scene="s"),
        rg.Entry(second, {"woke": True, "command": "open_downloads"}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first, second], result.reasons


def test_the_same_holds_after_a_claude_reply(settings: Settings, tmp_path: Path) -> None:
    corpus, first, second = long_window_scene(tmp_path, 5.8)
    real = ScriptedClient((["Fatto."], "end_turn", [SimpleNamespace(type="text", text="Fatto.")]))
    client = rg.CassetteClient(tmp_path / "cas", real=real)
    harness = gate_harness(settings, tmp_path / "w", [ASK, "apri download"], client=client)
    scene = [
        rg.Entry(first, {"woke": True, "route": "claude"}, scene="s"),
        rg.Entry(second, {"woke": True, "command": "open_downloads"}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first, second], result.reasons


def test_a_discarded_phrase_must_stay_discarded_by_the_real_model(
    settings: Settings, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "a.wav")
    entry = rg.Entry(clip, {"woke": False}, no_wake=True, real_wake=False)
    quiet = gate_harness(settings, tmp_path / "q", real_wake=CountingWake())
    assert rg.run_gate([entry], quiet, corpus).exit_code() == 0
    waking = gate_harness(settings, tmp_path / "f", real_wake=FiringWake(40))
    result = rg.run_gate([entry], waking, corpus)
    name = f"{clip} [real_wake]"
    assert result.failed == [name] and result.exit_code() == 1
    assert "scartata" in seen(result, name)
    # absent means no pass 2 at all, and the loader keeps the three values apart
    absent = rg.parse_entry('{"clip": "a.wav", "expect": {"woke": true}}', "m:1")
    off = rg.parse_entry('{"clip": "a.wav", "expect": {"woke": true}, "real_wake": false}', "m:1")
    assert absent.real_wake is None and off.real_wake is False
    plain = rg.run_gate([dataclasses.replace(entry, real_wake=None)], waking, corpus)
    assert plain.exit_code() == 0  # the detector is not even asked


def test_the_tripwire_reason_does_not_claim_a_deletion() -> None:
    assert "scartate" in rg.TRIPPED[1]
    assert "--record" in rg.TRIPPED[1]


def test_known_diff_of_the_wrong_type_is_a_manifest_error() -> None:
    for bad in ("5", "true", "1.5"):
        line = '{"clip": "a.wav", "expect": {"woke": true}, "known_bug": "G", "known_diff": %s}'
        with pytest.raises(rg.ManifestError):
            rg.parse_entry(line % bad, "m:1")


# --- fix round 4: what the re-review measured ---------------------------------------------------


def scene_with_voice_at(
    tmp_path: Path, wait_blocks: int, voice_blocks: int, second: dict[str, Any] | None = None
) -> tuple[Path, str, str]:
    """A, then B whose WAV is `wait_blocks` of digital zero and `voice_blocks` of voice."""
    corpus, first, last = two_clip_scene(tmp_path, second or {})
    audio = np.zeros((wait_blocks + voice_blocks) * 1280, np.int16)
    audio[wait_blocks * 1280 :] = 9000
    sf.write(corpus / last, audio, SR, subtype="PCM_16")
    return corpus, first, last


def test_voice_in_the_first_moments_of_a_wav_is_heard(settings: Settings, tmp_path: Path) -> None:
    """A WAV starts at the first frame its window listened to. Fed during the deaf time, its
    first 0.4 s are lost: this voice lasts 0.32 s, so nothing of it would be left."""
    corpus, first, last = scene_with_voice_at(tmp_path, 0, 4)
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", "apri download"])
    scene = [
        rg.Entry(first, {"woke": True, "command": "open_chrome"}, scene="s"),
        rg.Entry(last, {"woke": True, "command": "open_downloads"}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first, last], result.reasons


def test_voice_in_the_first_moments_is_heard_in_a_follow_up_too(
    settings: Settings, tmp_path: Path
) -> None:
    """`_join_deaf` never gives back what a deaf follow-up heard: there the answer would be lost."""
    corpus, first, last = scene_with_voice_at(tmp_path, 0, 4)
    harness = gate_harness(settings, tmp_path / "w", ["chiudi la finestra", "sì"])
    scene = [
        rg.Entry(first, {"route": "confirm", "command": "close_window"}, scene="s"),
        rg.Entry(last, {"woke": True}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first, last], result.reasons
    assert harness.recorder.combos  # the yes was heard: the window was closed on the Recorder


def test_the_wait_in_a_wav_is_listened_for_as_live(settings: Settings, tmp_path: Path) -> None:
    """`listened_ms` of a row counts from the first frame the window listened to."""
    corpus, first, last = scene_with_voice_at(tmp_path, 13, 12)
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", "apri download"])
    scene = [
        rg.Entry(first, {"woke": True}, scene="s"),
        rg.Entry(last, {"woke": True}, scene="s"),
    ]
    rg.run_gate(scene, harness, corpus)
    # the 13 frames of wait and the one where the voice starts
    assert list(harness.assistant.history)[1]["listened_ms"] == 14 * 80


def test_a_phrase_left_halfway_ends_before_the_next_clip(
    settings: Settings, tmp_path: Path
) -> None:
    """It waits continue_s for its end: the next WAV must not fall into the open phrase."""
    corpus, first, last = two_clip_scene(tmp_path, {})
    real = ScriptedClient((["Fatto."], "end_turn", [SimpleNamespace(type="text", text="Fatto.")]))
    client = rg.CassetteClient(tmp_path / "cassettes", real=real)
    harness = gate_harness(
        settings, tmp_path / "w", ["Claudio, dimmi...", "apri download"], client=client
    )
    scene = [
        rg.Entry(first, {"woke": True, "route": "claude"}, scene="s"),
        rg.Entry(last, {"woke": True, "command": "open_downloads"}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first, last], result.reasons
    assert "continued_from" in harness.assistant.history[0]


class HangingVad(FakeVad):
    """After a very loud window, voice for `hang` more windows, whatever it is given."""

    def __init__(self, hang: int) -> None:
        super().__init__()
        self.hang, self.left = hang, 0

    def __call__(self, window: np.ndarray) -> float:
        if float(np.abs(window).max()) > 0.6:
            self.left = self.hang
        self.left -= 1
        return 0.9 if self.left >= 0 else super().__call__(window)


def test_a_phrase_that_outlasts_the_limit_fails_its_clip_and_gives_the_next_no_row(
    settings: Settings, tmp_path: Path
) -> None:
    """The second phrase would end during the third clip, which would read its row and pass."""
    corpus, first, second = two_clip_scene(tmp_path, {})
    third = make_corpus(corpus, "c.wav", {"prewake_ms": None, "wake_word": "conversation"})
    sf.write(corpus / second, np.full(1280 * 12, 30000, np.int16), SR, subtype="PCM_16")
    harness = rg.build_harness(
        settings,
        tmp_path / "w",
        wake=rg.GateWake(),
        stt=GateStt(["apri chrome", "apri download"]),
        vad=HangingVad(120),  # type: ignore[arg-type]  # 3.84 s of voice after the WAV's end
        client=rg.CassetteClient(tmp_path / "w" / "cassettes"),
    )
    scene = [
        rg.Entry(first, {"woke": True, "command": "open_chrome"}, scene="s"),
        rg.Entry(second, {"woke": True, "command": "open_downloads"}, scene="s"),
        rg.Entry(third, {"woke": True, "command": "open_downloads"}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first] and result.failed == [second, third], result.reasons
    assert "non è finita" in seen(result, second)
    assert "interrott" in seen(result, third)


def test_an_armed_clip_mid_scene_waits_for_the_window_to_close(
    settings: Settings, tmp_path: Path
) -> None:
    """Live, a wake fires only from idle: `wake.process` is not called while a window is open.
    The window's closing row is not the armed clip's: its wake word would be `conversation`."""
    corpus, first, last = two_clip_scene(tmp_path, {"prewake_ms": 160})
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", "apri download"])
    scene = [
        rg.Entry(first, {"woke": True, "command": "open_chrome"}, scene="s"),
        rg.Entry(last, {"woke": True, "command": "open_downloads"}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first, last], result.reasons
    rows = list(harness.assistant.history)
    assert [row.get("outcome") for row in rows] == ["ok", "timeout", "ok"]
    assert rows[2]["wake_word"] == "replay"


def test_an_armed_clip_waits_for_a_follow_up_with_a_tool_question_open(
    settings: Settings, tmp_path: Path
) -> None:
    """Claude asks before a terminal command; nobody answers, the follow-up closes by itself,
    then a phrase with the name."""
    corpus, first, last = two_clip_scene(tmp_path, {"prewake_ms": 160})
    real = claude_script(use("run_command", command="nvidia-smi", purpose="parametri"))
    client = rg.CassetteClient(tmp_path / "cassettes", real=real)
    harness = gate_harness(settings, tmp_path / "w", [ASK, ASK], client=client)
    scene = [
        rg.Entry(first, {"route": "claude", "tools": ["run_command"], "safety": 3}, scene="s"),
        rg.Entry(last, {"woke": True, "route": "claude"}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert first in result.passed, result.reasons
    assert "finestra" not in seen(result, last)
    assert list(harness.assistant.history)[-1]["wake_word"] == "replay"


def test_a_window_that_outlasts_the_limit_fails_the_armed_clip(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    monkeypatch.setattr(rg, "idle_limit_s", lambda _settings: 0.08)
    corpus, first, last = two_clip_scene(tmp_path, {"prewake_ms": 160})
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", "apri download"])
    scene = [
        rg.Entry(first, {"woke": True, "command": "open_chrome"}, scene="s"),
        rg.Entry(last, {"woke": True, "command": "open_downloads"}, scene="s"),
    ]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first] and result.failed == [last], result.reasons
    assert "non si è chiusa" in seen(result, last)


def test_a_detection_whisper_refuses_stays_discarded(settings: Settings, tmp_path: Path) -> None:
    """The real model fires, Whisper does not hear the name: not woken, so `real_wake: false`
    passes. The same detection with the name in Whisper's text fails: the test can tell."""
    weak = settings.wakeword.model_copy(update={"confirm_below": 0.95})
    weak_settings = settings.model_copy(update={"wakeword": weak})
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "a.wav")
    entry = rg.Entry(clip, {"woke": False}, no_wake=True, real_wake=False)

    def run(text: str, work: str) -> tuple[rg.GateResult, rg.Harness]:
        harness = gate_harness(weak_settings, tmp_path / work, text, real_wake=FiringWake(40))
        return rg.run_gate([entry], harness, corpus), harness

    refused, harness = run("niente di che", "r")
    assert refused.passed == [clip] and refused.exit_code() == 0, refused.reasons
    assert harness.detections.count == 1 and harness.detections.second_chances == 1
    heard, _ = run("Claudio, apri chrome", "h")
    assert heard.failed == [f"{clip} [real_wake]"]


# --- fix round 5: a failed recording must not read as a pass ------------------------------------


CONNECTION = anthropic.APIConnectionError(request=REQUEST)
SERVER = anthropic.InternalServerError(
    "x", response=httpx2.Response(500, request=REQUEST), body=None
)
BAD_KEY = anthropic.AuthenticationError(
    "x", response=httpx2.Response(401, request=REQUEST), body=None
)


class BrokenStream(ScriptedStream):
    """A reply that breaks after its first words."""

    def __init__(self, broken: Exception) -> None:
        super().__init__(["Fatto, ecco "], "end_turn", [], dict(USAGE))
        self.broken = broken

    @property
    def text_stream(self) -> Any:
        async def gen() -> Any:
            yield "Fatto, ecco "
            raise self.broken

        return gen()


class FailingClient(ScriptedClient):
    """The SDK's shape: a plain reply, except the calls in `failures`, by number from 1, which
    fail when the request is sent or, `midway`, after the first words of the reply."""

    def __init__(self, failures: dict[int, Exception], midway: bool = False) -> None:
        super().__init__()
        self.failures, self.midway = failures, midway

    def stream(self, **params: Any) -> ScriptedStream:
        self.calls.append(params)
        failure = self.failures.get(len(self.calls))
        if failure is not None and self.midway:
            return BrokenStream(failure)
        plain = [SimpleNamespace(type="text", text="Fatto.")]
        stream = ScriptedStream(["Fatto."], "end_turn", plain, dict(USAGE))
        stream.error = failure
        return stream


def three_clips(tmp_path: Path) -> tuple[Path, list[str], list[rg.Entry]]:
    """A scene of two clips, then a scene of one: all three go to Claude."""
    corpus = tmp_path / "corpus"
    first = make_corpus(corpus, "a.wav")
    second = make_corpus(corpus, "b.wav", {"prewake_ms": None, "wake_word": "conversation"})
    third = make_corpus(corpus, "c.wav")
    want = {"woke": True, "route": "claude"}
    entries = [
        rg.Entry(first, want, scene="s1"),
        rg.Entry(second, want, scene="s1"),
        rg.Entry(third, want, scene="s2"),
    ]
    return corpus, [first, second, third], entries


def recording(
    settings: Settings, tmp_path: Path, real: ScriptedClient, work: str = "w"
) -> tuple[rg.CassetteClient, rg.Harness]:
    client = rg.CassetteClient(tmp_path / "cassettes", real=real)
    texts = [ASK, "dimmi una cosa strana ora", "Claudio, dimmene una terza adesso"]
    return client, gate_harness(settings, tmp_path / work, texts, client=client)


@pytest.mark.parametrize(
    ("error", "midway", "named"),
    [
        (CONNECTION, False, "network"),
        (CONNECTION, True, "network"),
        (SERVER, False, "500"),
        (BAD_KEY, False, "auth"),
        (OVERLOADED_MID_REPLY, True, "overloaded_error"),
    ],
)
def test_a_claude_error_fails_its_clip_and_says_what_it_was(
    settings: Settings, tmp_path: Path, error: Exception, midway: bool, named: str
) -> None:
    """The brain catches it and speaks it: the row says `ok`, only `claude_error` tells."""
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "c.wav")
    client = rg.CassetteClient(
        tmp_path / "cassettes", real=FailingClient({1: error}, midway=midway)
    )
    harness = gate_harness(settings, tmp_path / "w", ASK, client=client)
    expect = {"woke": True, "route": "claude"}
    result = rg.run_gate([rg.Entry(clip, expect)], harness, corpus)
    assert result.failed == [clip] and result.exit_code() == 1
    assert f"errore di Claude: {named}" in seen(result, clip)
    assert "errore: ok" not in seen(result, clip)
    assert f"errore di Claude: {named}" in " ".join(p for p, _ in result.reasons[clip])  # printed


def test_a_known_bug_cannot_absorb_a_claude_error(settings: Settings, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "c.wav")
    client = rg.CassetteClient(tmp_path / "cassettes", real=FailingClient({1: SERVER}))
    harness = gate_harness(settings, tmp_path / "w", ASK, client=client)
    entry = rg.Entry(
        clip, {"route": "claude", "tools": ["open_app"]}, known_bug="G", known_diff=("tools",)
    )
    result = rg.run_gate([entry], harness, corpus)
    assert result.failed == [clip] and not result.known
    assert "errore di Claude: 500" in seen(result, clip)


def test_the_first_api_error_stops_the_recording(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus, (first, second, third), entries = three_clips(tmp_path)
    real = FailingClient({2: SERVER})
    client, harness = recording(settings, tmp_path, real)
    result = rg.run_gate(entries, harness, corpus)
    assert len(real.calls) == 2  # the third clip, in another scene, sent nothing
    assert result.passed == [first] and result.failed == [second, third]
    assert result.exit_code() == 1
    assert len(client.saved) == 1 and len(list((tmp_path / "cassettes").glob("*.json"))) == 1
    assert "errore di Claude: 500" in seen(result, second)
    assert "cassetta mancante" in seen(result, third)
    rg.print_spend(client)
    printed = capsys.readouterr().out
    assert "InternalServerError 500" in printed  # the error, by type and status
    assert all(key[:12] in printed for key in client.braked) and len(client.braked) == 2
    assert "Speso davvero" in printed
    assert "tagliata" in printed  # a cut request is not in the figure


class UnfinishedClient(ScriptedClient):
    """A body that ends cleanly with no `message_stop`: the SDK raises nothing and the final
    message has `stop_reason=None`."""

    def stream(self, **params: Any) -> ScriptedStream:
        self.calls.append(params)
        plain = [SimpleNamespace(type="text", text="Fatto, ecco ")]
        return ScriptedStream(["Fatto, ecco "], None, plain, dict(USAGE))  # type: ignore[arg-type]


def test_a_reply_with_no_stop_saves_nothing_and_fails_its_clip(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus, (first, second, third), entries = three_clips(tmp_path)
    real = UnfinishedClient()
    client, harness = recording(settings, tmp_path, real)
    result = rg.run_gate(entries, harness, corpus)
    assert len(real.calls) == 1  # the first API error stops the recording
    assert client.saved == [] and not list((tmp_path / "cassettes").glob("*.json"))
    assert client.spent == 0.0 and len(client.braked) == 3  # the cut one, then two refused
    assert result.passed == [] and result.failed == [first, second, third]
    assert result.exit_code() == 1
    assert "errore di Claude: network" in seen(result, first)
    rg.print_spend(client)
    assert "UnfinishedReply" in capsys.readouterr().out


def test_a_rerun_pays_only_for_what_is_missing(settings: Settings, tmp_path: Path) -> None:
    corpus, clips, entries = three_clips(tmp_path)
    _, harness = recording(settings, tmp_path, FailingClient({2: SERVER}))
    rg.run_gate(entries, harness, corpus)
    real = FailingClient({})
    _, again = recording(settings, tmp_path, real, "w2")
    result = rg.run_gate(entries, again, corpus)
    assert result.passed == clips, result.reasons
    assert len(real.calls) == 2  # the first reply came from its cassette


def test_a_reply_cut_midway_saves_nothing_and_stops_the_recording(
    settings: Settings, tmp_path: Path
) -> None:
    corpus, _clips, entries = three_clips(tmp_path)
    real = FailingClient({1: CONNECTION}, midway=True)
    client, harness = recording(settings, tmp_path, real)
    result = rg.run_gate(entries, harness, corpus)
    assert len(real.calls) == 1 and result.exit_code() == 1
    assert client.saved == [] and not list((tmp_path / "cassettes").glob("*.json"))
    assert client.failure == "APIConnectionError"


async def test_a_failed_guess_stops_the_recording_too(tmp_path: Path) -> None:
    class Guessing(ScriptedClient):
        async def create(self, **params: Any) -> Any:
            self.calls.append(params)
            raise SERVER

    real = Guessing()
    client = rg.CassetteClient(tmp_path, real=real)
    with pytest.raises(anthropic.InternalServerError):
        await client.create(**PARAMS)
    other = {**PARAMS, "messages": [{"role": "user", "content": "due"}]}
    with pytest.raises(rg.CassetteMiss):
        await client.create(**other)
    with pytest.raises(rg.CassetteMiss):
        client.stream(**other)
    assert len(real.calls) == 1 and client.failure == "InternalServerError 500"
    assert rg.cassette_key(PARAMS) in client.misses  # a clip's hard reason, even with no row error


def test_the_prompt_does_not_overclaim_the_brake(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "corpus"
    clip = make_corpus(corpus, "c.wav")
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    harness = gate_harness(settings, tmp_path / "w", ASK)
    rg.confirm_recording(
        harness, [rg.Entry(clip, {"route": "claude"})], tmp_path / "c", corpus, 2.58
    )
    printed = capsys.readouterr().out
    assert "si ferma comunque" not in printed and "prima di ogni invio" in printed
    assert "tagliata" in printed and "primo errore" in printed


def test_gate_main_names_a_work_dir_it_cannot_remove(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_harness(*_args: object, **_kw: object) -> None:
        raise rg.RealActionBlocked("stop")

    work = tmp_path / "gate-work"
    work.mkdir()
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"clip": "a.wav", "expect": {"woke": true}}\n', encoding="utf-8")
    monkeypatch.setattr(rg.tempfile, "mkdtemp", lambda prefix="": str(work))
    monkeypatch.setattr(rg, "arm_process", lambda: None)
    monkeypatch.setattr(rg, "require_personality", lambda _settings: None)
    monkeypatch.setattr(rg, "build_harness", no_harness)
    with (work / "heard.wav").open("wb"):  # open: Windows refuses to delete it
        assert rg.gate_main(False, False, manifest) == 1
    assert str(work) in capsys.readouterr().out


def test_main_names_a_work_dir_it_cannot_remove(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wav = loud_wav(tmp_path / "c.wav", 30)
    monkeypatch.setattr(rg, "install_tripwire", lambda: None)
    real_build = rg.build_harness

    def build_in_work_dir(_settings: Settings, work_dir: Path, **_kw: object) -> rg.Harness:
        return real_build(settings, work_dir, wake=CountingWake(), stt=GateStt(), vad=FakeVad())  # type: ignore[arg-type]

    monkeypatch.setattr(rg, "build_harness", build_in_work_dir)
    work = tmp_path / "gate-work"
    work.mkdir()
    monkeypatch.setattr(rg.tempfile, "mkdtemp", lambda prefix="": str(work))
    with (work / "heard.wav").open("wb"):  # open: Windows refuses to delete it
        rg.main(["--clips", str(wav)])
    printed = capsys.readouterr().out
    assert printed.count(str(work)) >= 2  # `cartella di lavoro: ...` and the warning
    assert "non riesco a cancellare" in printed


def test_an_armed_clip_whose_wake_does_not_fire_fails(settings: Settings, tmp_path: Path) -> None:
    """From idle, the voice meter stops a wake on silence: no row, and `woke: false` would pass."""
    corpus, first, last = two_clip_scene(tmp_path, {"prewake_ms": 160}, silent_second=True)
    harness = gate_harness(settings, tmp_path / "w", ["apri chrome", "apri download"])
    scene = [rg.Entry(first, {"woke": True}, scene="s"), rg.Entry(last, {"woke": False}, scene="s")]
    result = rg.run_gate(scene, harness, corpus)
    assert result.passed == [first] and result.failed == [last]
    assert "non è scattata" in seen(result, last)


def test_a_scene_stops_feeding_clips_once_the_caller_says_it_is_settled(
    settings: Settings, tmp_path: Path
) -> None:
    corpus, first, last = two_clip_scene(tmp_path, {})
    entries = [
        rg.Entry(first, {"woke": True}, scene="s"),
        rg.Entry(last, {"woke": True}, scene="s"),
    ]
    reached = {}
    for name, stop in (("never", lambda: False), ("at once", lambda: True)):
        harness = gate_harness(settings, tmp_path / name, ["apri chrome", "apri download"])
        harness.fresh()
        before = rg.take_mark(harness)
        planned = rg.plan_scene(harness, entries, corpus)
        rans = asyncio.run(rg._replay_scene(harness, planned, before, stop_when=stop))
        reached[name] = [r.reached for r in rans]
    assert reached == {"never": [True, True], "at once": [True, False]}
