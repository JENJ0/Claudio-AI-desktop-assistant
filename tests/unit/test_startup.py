"""Startup: the models load together, and what each one cost is logged."""

from __future__ import annotations

import logging
import time

import pytest

from claudio.__main__ import load_all, startup_notices


def sleeper(value: str, seconds: float = 0.3):
    def load() -> str:
        time.sleep(seconds)
        return value

    return load


def test_loaders_run_at_the_same_time(caplog) -> None:
    caplog.set_level(logging.INFO, logger="claudio")
    start = time.perf_counter()
    loaded = load_all({"whisper": sleeper("w"), "voce": sleeper("v"), "claude": sleeper("c")})
    assert time.perf_counter() - start < 0.8  # 3 x 0.3 s one after the other would be 0.9
    assert loaded == {"whisper": "w", "voce": "v", "claude": "c"}
    assert "whisper 0.3 s" in caplog.text and "voce 0.3 s" in caplog.text


def test_sequential_start_still_works() -> None:
    start = time.perf_counter()
    loaded = load_all({"a": sleeper("a", 0.1), "b": sleeper("b", 0.1)}, parallel=False)
    assert time.perf_counter() - start >= 0.2
    assert loaded == {"a": "a", "b": "b"}


@pytest.mark.parametrize("parallel", [True, False])
def test_a_failing_loader_raises_after_the_others_end(parallel: bool) -> None:
    ended: list[str] = []

    def other() -> str:
        time.sleep(0.2)
        ended.append("other")
        return "ok"

    def broken() -> str:
        raise FileNotFoundError("models/whisper")

    with pytest.raises(FileNotFoundError):
        load_all({"broken": broken, "other": other}, parallel=parallel)
    assert ended == (["other"] if parallel else [])


def test_spend_counter_works_after_a_threaded_load(tmp_path) -> None:
    """25/10 on JenJo's PC: the SQLite counter opened in a loader thread broke every Claude
    request ("SQLite objects created in a thread can only be used in that same thread")."""
    from claudio.__main__ import build_responder, build_tracker
    from claudio.config import load_settings
    from tests.unit.test_actions import Recorder

    settings = load_settings()
    settings = settings.model_copy(
        update={
            "cost": settings.cost.model_copy(update={"db_file": tmp_path / "spesa.db"}),
            "memory": settings.memory.model_copy(update={"enabled": False}),  # not JenJo's own
        }
    )
    tracker = build_tracker(settings)  # on this thread, as run_assistant does
    loaded = load_all(
        {"claude": lambda: build_responder(settings, Recorder().ctx(), tracker=tracker)}
    )
    assert loaded["claude"] is not None
    tracker.check()  # raised sqlite3.ProgrammingError when opened in the loader thread


async def test_run_text_acts_for_real(tmp_path, monkeypatch) -> None:
    """`run_text` is a place where Claudio really acts: it builds the real context."""
    from claudio import __main__ as entry
    from claudio.config import load_settings
    from claudio.router import actions

    class FakeSpeaker:
        def __init__(self) -> None:
            self.player = type("P", (), {"close": lambda self: None})()

        async def speak(self, replies, on_first_audio=None) -> None:
            async for _ in replies:
                pass

    ran: list[list[str]] = []
    monkeypatch.setattr(actions, "_run", ran.append)
    monkeypatch.setattr(entry, "build_speaker", lambda settings: FakeSpeaker())
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "data_dir": tmp_path,
            "logs_dir": tmp_path,
            "claude": settings.claude.model_copy(update={"enabled": False}),
            "memory": settings.memory.model_copy(update={"enabled": False}),
            "cost": settings.cost.model_copy(update={"db_file": tmp_path / "spend.db"}),
        }
    )
    await entry.run_text(settings, "apri il blocco note")
    assert ran


def _with_engine(engine: str):
    from claudio.config import Settings

    base = Settings()
    return base.model_copy(update={"tts": base.tts.model_copy(update={"engine": engine})})


def test_startup_notice_when_the_online_voice_did_not_start() -> None:
    notices = startup_notices(
        _with_engine("elevenlabs"), jack_running=True, stt_device="cuda", online_ok=False
    )
    assert notices == ["La voce online non è partita: uso quella locale.", "Il motivo è nel log."]


def test_jack_notice_also_with_the_elevenlabs_engine() -> None:
    notices = startup_notices(
        _with_engine("elevenlabs"), jack_running=False, stt_device="cuda", online_ok=True
    )
    assert "La voce di Jack non è partita: uso quella di riserva." in notices


@pytest.mark.parametrize("engine", ["jack", "kokoro", "piper"])
def test_no_online_notice_for_the_other_engines(engine: str) -> None:
    notices = startup_notices(
        _with_engine(engine), jack_running=True, stt_device="cuda", online_ok=False
    )
    assert notices == []


def test_no_notice_when_the_online_voice_and_jack_started() -> None:
    notices = startup_notices(
        _with_engine("elevenlabs"), jack_running=True, stt_device="cuda", online_ok=True
    )
    assert notices == []


def test_online_voice_ok() -> None:
    import httpx2

    from claudio.__main__ import online_voice_ok
    from claudio.tts.elevenlabs import ElevenLabsSynth

    def synth_answering(status: int) -> ElevenLabsSynth:
        transport = httpx2.MockTransport(lambda request: httpx2.Response(status))
        return ElevenLabsSynth("KEY", "VOICE", "m", object(), transport=transport)

    ok, refused = synth_answering(200), synth_answering(401)
    ok.probe()
    refused.probe()
    assert online_voice_ok(ok) is True
    assert online_voice_ok(refused) is False
    assert online_voice_ok(object()) is False
