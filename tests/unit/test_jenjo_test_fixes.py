"""Fixes from JenJo's test of v0.3.0 (23/10): weak wakes, Sonnet for detail, Jack's pace."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from claudio.app import Assistant, State
from claudio.audio.vad import Endpointer
from claudio.brain.prompts import OFFER
from claudio.metrics import MetricsWriter
from tests.unit.test_app import (
    AudioClock,
    FakeResponder,
    FakeSpeaker,
    FakeStt,
    FakeWake,
    SecondChance,
    run_frames,
)
from tests.unit.test_claude_brain import ScriptedClient, ask, brain, pricing  # noqa: F401


class WeakWake(FakeWake):
    """Fires once with a weak score, as on Claudio's own voice at 18:47:34."""

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        hit = super().process(frame)
        return ("hey_claudio", 0.34) if hit else None


class OnlyCommands(FakeResponder):
    def understands(self, text: str, language: str = "it") -> bool:
        return "chrome" in text.lower()


def weak(fake_vad, vad_settings, tmp_path, heard: str):
    speaker = FakeSpeaker()
    assistant = Assistant(
        WeakWake({20}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        OnlyCommands(),
        speaker,
        MetricsWriter(tmp_path / "metrics.jsonl"),
        clock=AudioClock(),
        second_chance=SecondChance(heard),
        confirm_below=0.45,
    )
    return assistant, speaker


async def test_weak_wake_without_the_name_stays_silent(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    assistant, speaker = weak(fake_vad, vad_settings, tmp_path, "")
    await run_frames(assistant, make_frames("silence", 2400))
    assert speaker.beeps == [] and assistant.state is State.IDLE
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "near_miss" and row["wake_score"] == 0.34


@pytest.mark.parametrize("heard", ["Hey Claudio.", "apri chrome"])
async def test_weak_wake_with_the_name_or_a_command_wakes(
    fake_vad, vad_settings, make_frames, tmp_path, heard: str
) -> None:
    """Said fast the name is often lost ("Apricrome" at 0.39): a command is enough."""
    assistant, speaker = weak(fake_vad, vad_settings, tmp_path, heard)
    await run_frames(assistant, make_frames("silence", 2400))
    assert speaker.beeps == ["wake"] and assistant.state is State.LISTENING
    assert assistant._interaction is not None
    assert assistant._interaction.info["second_chance"] == heard


async def test_a_strong_wake_needs_no_check(fake_vad, vad_settings, make_frames, tmp_path) -> None:
    check = SecondChance("")
    assistant = Assistant(
        FakeWake({20}),  # 0.9
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        FakeResponder(),
        FakeSpeaker(),
        clock=AudioClock(),
        second_chance=check,
        confirm_below=0.45,
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert assistant.state is State.LISTENING and check.heard == []


async def test_asking_for_detail_proposes_sonnet_without_haiku(pricing) -> None:  # noqa: F811
    client = ScriptedClient((["Il mutuo..."], "end_turn", []), (["Ecco."], "end_turn", []))
    b = brain(pricing, client)
    reply = await ask(b, "Spiegami in dettaglio come funziona un mutuo a tasso variabile.")
    assert reply == OFFER["it"] and client.calls == []
    assert b.last_call["escalated"] == "offered"
    await ask(b, "sì")
    assert [c["model"] for c in client.calls] == ["claude-sonnet-5-5"]


async def test_detail_phrases_need_sonnet_and_escalation(pricing) -> None:  # noqa: F811
    b = brain(pricing, ScriptedClient((["Ok."], "end_turn", [])), escalate=False)
    assert await ask(b, "spiegami in dettaglio il mutuo") == "Ok."
    assert not b.wants_offer("dettagli a parte, come stai", "it")
    b.smart_model = None
    assert not b.wants_offer("spiegami in dettaglio il mutuo", "it")


def test_jack_speed_plan() -> None:
    saved = sys.stdout
    try:
        from claudio.tts.jack_worker import speed_plan
    finally:
        sys.stdout = saved  # the worker sends its prints to stderr when imported
    assert speed_plan(1.0) == (2, 1, 0)
    up, down, semitones = speed_plan(0.9)  # 10% slower: stretched by 1/0.9, pitch +2
    assert up / down == pytest.approx(2 / 0.9) and semitones == 2
    assert speed_plan(1.12)[2] == -2


class ClaudioWake(FakeWake):
    """The "claudio" model: fires strong, but it is noisy, so Whisper always checks it."""

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        hit = super().process(frame)
        return ("claudio", 0.9) if hit else None


@pytest.mark.parametrize(("heard", "wakes"), [("Claudio.", True), ("il gatto", False)])
async def test_claudio_alone_is_always_checked(
    fake_vad, vad_settings, make_frames, tmp_path, heard: str, wakes: bool
) -> None:
    speaker = FakeSpeaker()
    assistant = Assistant(
        ClaudioWake({20}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        OnlyCommands(),
        speaker,
        MetricsWriter(tmp_path / "metrics.jsonl"),
        clock=AudioClock(),
        second_chance=SecondChance(heard),
        always_confirm=frozenset({"claudio"}),
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert (assistant.state is State.LISTENING) is wakes
    assert speaker.beeps == (["wake"] if wakes else [])
    if not wakes:  # a noisy model: no near miss row, no WAV
        assert not (tmp_path / "metrics.jsonl").exists() or not (
            tmp_path / "metrics.jsonl"
        ).read_text(encoding="utf-8")


async def test_a_burst_of_claudio_detections_is_checked_once(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    """One "Claudio" fires the model for many frames in a row (8 checks in 3 s on 24/10):
    while Whisper is listening the others are dropped, with no near miss row or WAV."""
    import threading

    gate = threading.Event()
    check = SecondChance("il gatto", gate=gate)
    assistant = Assistant(
        ClaudioWake(set(range(20, 30))),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        OnlyCommands(),
        FakeSpeaker(),
        MetricsWriter(tmp_path / "metrics.jsonl"),
        clock=AudioClock(),
        second_chance=check,
        always_confirm=frozenset({"claudio"}),
    )
    for frame in make_frames("silence", 2400)[:30]:
        await assistant.handle_frame(frame)
    gate.set()
    await assistant.wait_idle()
    assert len(check.heard) == 1
    rows = tmp_path / "metrics.jsonl"
    assert not rows.exists() or not rows.read_text(encoding="utf-8")
