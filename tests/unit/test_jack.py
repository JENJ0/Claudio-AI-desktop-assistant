import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from claudio.tts.jack import JackSynth, WorkerError

# A stand-in for jack_worker.py: same protocol, no GPU. MODE picks how it misbehaves.
FAKE_WORKER = textwrap.dedent(
    """
    import json, sys, time
    mode = sys.argv[1]
    out = sys.stdout.buffer
    def emit(header, payload=b""):
        out.write(json.dumps(header).encode() + b"\\n" + payload); out.flush()
    if mode == "no_start":
        sys.exit(1)
    if mode == "bad_ready":
        emit({"event": "error", "id": None, "message": "startup: no GPU"})
        sys.exit(1)
    if mode == "slow_start":
        time.sleep(30)
    emit({"event": "ready", "sample_rate": 24000, "load_s": 1.5, "vram_mb": 2900})
    cancelled = -1
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "quit":
            break
        if req["op"] == "interrupt":
            cancelled = 10**9
            continue
        rid = req["id"]
        if mode == "die":
            sys.exit(1)
        if mode == "hang":
            time.sleep(30)
        if mode == "error":
            emit({"event": "error", "id": rid, "message": "boom"})
            continue
        if mode == "cut":  # one piece of audio, then the worker dies
            emit({"event": "audio", "id": rid, "bytes": 2}, b"\\x01\\x00")
            sys.exit(1)
        if mode == "stale":  # a leftover of an earlier sentence arrives first
            emit({"event": "audio", "id": rid - 1, "bytes": 4}, b"\\x07\\x00\\x07\\x00")
        for k in range(3):
            pcm = (np_k := bytes([k + 1, 0])) * 100
            emit({"event": "audio", "id": rid, "bytes": len(pcm)}, pcm)
        emit({"event": "done", "id": rid, "cancelled": False, "first_ms": 5, "total_ms": 9})
    """
)


class FakeFallback:
    sample_rate = 22050

    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.full(50, 9, dtype=np.int16)


def make(tmp_path: Path, mode: str, **kwargs) -> tuple[JackSynth, FakeFallback]:
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER.replace("(np_k := bytes([k + 1, 0]))", "bytes([k + 1, 0])"))
    fallback = FakeFallback()
    return JackSynth([sys.executable, str(script), mode], fallback, **kwargs), fallback


@pytest.fixture
def synths():
    made: list[JackSynth] = []
    yield made
    for synth in made:
        synth.close()


def test_speaks_through_the_worker(tmp_path, synths) -> None:
    synth, fallback = make(tmp_path, "ok", startup_timeout_s=10)
    synths.append(synth)
    synth.start()
    assert (synth.sample_rate, synth.load_s, synth.vram_mb) == (24000, 1.5, 2900)
    audio = synth.synthesize("Ciao.")
    assert audio.dtype == np.int16
    assert len(audio) == 300
    assert audio[0] == 1 and audio[100] == 2 and audio[200] == 3  # chunks in order
    assert synth.sample_rate == 24000
    assert fallback.texts == []


def test_ignores_audio_of_an_earlier_sentence(tmp_path, synths) -> None:
    synth, _ = make(tmp_path, "stale", startup_timeout_s=10)
    synths.append(synth)
    synth.start()
    assert len(synth.synthesize("Ciao.")) == 300


@pytest.mark.parametrize("mode", ["no_start", "bad_ready"])
def test_start_fails_when_the_worker_does_not_come_up(tmp_path, synths, mode) -> None:
    synth, _ = make(tmp_path, mode, startup_timeout_s=10)
    synths.append(synth)
    with pytest.raises(WorkerError):
        synth.start()
    assert not synth.alive()


def test_start_times_out(tmp_path, synths) -> None:
    synth, _ = make(tmp_path, "slow_start", startup_timeout_s=0.5)
    synths.append(synth)
    with pytest.raises(WorkerError):
        synth.start()
    assert not synth.alive()


def test_missing_interpreter_is_a_worker_error() -> None:
    synth = JackSynth(["/nonexistent/python", "worker.py"], FakeFallback())
    with pytest.raises(WorkerError):
        synth.start()


@pytest.mark.parametrize("mode", ["die", "hang", "error"])
def test_falls_back_for_the_sentence_when_the_worker_fails(tmp_path, synths, mode) -> None:
    synth, fallback = make(tmp_path, mode, startup_timeout_s=10, sentence_timeout_s=0.5)
    synths.append(synth)
    synth.start()
    audio = synth.synthesize("Ciao.")
    assert fallback.texts == ["Ciao."]
    assert len(audio) == 50 and audio[0] == 9
    assert synth.sample_rate == 22050  # the rate of the voice that actually spoke
    time.sleep(0.3)  # the restart runs in the background and must not break anything


def test_restarts_the_worker_after_a_failure(tmp_path, synths) -> None:
    synth, fallback = make(tmp_path, "die", startup_timeout_s=10, sentence_timeout_s=2)
    synths.append(synth)
    synth.start()
    synth.synthesize("uno")
    assert fallback.texts == ["uno"]
    deadline = time.monotonic() + 10
    while not synth.alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert synth.alive()  # "die" mode dies on the next sentence, but it is back up


def test_cancel_without_a_worker_is_harmless(tmp_path) -> None:
    synth, _ = make(tmp_path, "ok")
    synth.cancel()


def test_cancel_reaches_a_running_worker(tmp_path, synths) -> None:
    synth, _ = make(tmp_path, "ok", startup_timeout_s=10)
    synths.append(synth)
    synth.start()
    synth.cancel()
    assert len(synth.synthesize("Ciao.")) == 300  # the fake only notes it; the protocol holds


def test_build_synth_falls_back_when_the_worker_is_missing(monkeypatch, tmp_path) -> None:
    from claudio import __main__ as entry
    from claudio.config import Settings

    fallback = FakeFallback()
    monkeypatch.setattr(entry, "build_base_synth", lambda settings: (fallback, "Fallback"))
    settings = Settings(models_dir=tmp_path, tts={"engine": "jack"})
    synth, name = entry.build_synth(settings)
    assert synth is fallback and name == "Fallback"


def test_build_synth_wraps_the_fallback_when_the_worker_starts(monkeypatch, tmp_path) -> None:
    from claudio import __main__ as entry
    from claudio.config import Settings
    from claudio.tts import jack

    fallback = FakeFallback()
    monkeypatch.setattr(entry, "build_base_synth", lambda settings: (fallback, "Fallback"))
    monkeypatch.setattr(jack.JackSynth, "start", lambda self: None)
    settings = Settings(models_dir=tmp_path, tts={"engine": "jack"})
    synth, name = entry.build_synth(settings)
    assert isinstance(synth, jack.JackSynth) and synth.fallback is fallback
    assert name == "Jack (fallback Fallback)"
    assert synth.command[2:4] == [
        "--checkpoint",
        str(tmp_path / "voices/jack/ft/0.6B/checkpoint-epoch-5"),
    ]


def test_stream_yields_the_chunks_as_they_come(tmp_path, synths) -> None:
    synth, fallback = make(tmp_path, "ok", startup_timeout_s=10)
    synths.append(synth)
    synth.start()
    pieces = list(synth.stream("Ciao."))
    assert [int(p[0]) for p in pieces] == [1, 2, 3]
    assert synth.last_first_ms == 5 and synth.last_total_ms == 9
    assert fallback.texts == []


def test_a_worker_that_fails_after_some_audio_is_not_repeated_by_the_fallback(
    tmp_path, synths
) -> None:
    synth, fallback = make(tmp_path, "cut", startup_timeout_s=10, sentence_timeout_s=1)
    synths.append(synth)
    synth.start()
    pieces = list(synth.stream("Ciao."))
    assert len(pieces) == 1 and fallback.texts == []


def test_digits_reach_the_worker_as_italian_words(tmp_path, synths) -> None:
    synth, _ = make(tmp_path, "ok", startup_timeout_s=10, sentence_timeout_s=2)
    synths.append(synth)
    synth.start()
    sent: list[dict] = []
    original = synth._send
    synth._send = lambda message, proc=None: (sent.append(message), original(message, proc))[1]
    synth.synthesize("Sono le 20 e 13.")
    assert [m["text"] for m in sent if m["op"] == "speak"] == ["Sono le venti e tredici."]
    assert [m["language"] for m in sent if m["op"] == "speak"] == ["Auto"]


def test_the_configured_language_reaches_the_worker(tmp_path, synths) -> None:
    """[tts] jack_language = "Italian": Qwen3 stops guessing the language per sentence."""
    synth, _ = make(tmp_path, "ok", startup_timeout_s=10, sentence_timeout_s=2, language="Italian")
    synths.append(synth)
    synth.start()
    sent: list[dict] = []
    original = synth._send
    synth._send = lambda message, proc=None: (sent.append(message), original(message, proc))[1]
    synth.synthesize("Ciao socio.")
    assert [m["language"] for m in sent if m["op"] == "speak"] == ["Italian"]


def test_no_module_next_to_the_worker_shadows_the_standard_library() -> None:
    """jack_worker.py runs as a script, so its folder comes first on sys.path: a tts/numbers.py
    once hid the standard `numbers` and the worker died at import (Claudio fell back to Kokoro)."""
    tts = Path(__file__).parents[2] / "src" / "claudio" / "tts"
    shadowing = sorted(
        p.name
        for p in tts.iterdir()
        if (p.suffix == ".py" or (p.is_dir() and (p / "__init__.py").exists()))
        and p.stem in sys.stdlib_module_names
    )
    assert shadowing == []
