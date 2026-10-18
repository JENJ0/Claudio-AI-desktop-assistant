"""Real Silero VAD and openWakeWord models on synthetic speech (espeak-ng).

Skipped unless the models are downloaded (scripts/download_models.py) and espeak-ng is
installed. Run with: uv run pytest -m slow
"""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from claudio.app import Assistant
from claudio.audio.player import resample
from claudio.audio.vad import Endpointer, EndpointEvent, SileroVad
from claudio.config import load_settings
from tests.unit.test_app import FakeResponder, FakeSpeaker, FakeStt

pytestmark = pytest.mark.slow

SETTINGS = load_settings()
MODELS = SETTINGS.resolve(SETTINGS.models_dir)
SR = 16000
BLOCK = 1280

if not shutil.which("espeak-ng"):
    pytest.skip("espeak-ng non installato", allow_module_level=True)
if not (MODELS / "vad" / "silero_vad.onnx").is_file():
    pytest.skip("modelli non scaricati", allow_module_level=True)


def speech(text: str, voice: str = "it") -> np.ndarray:
    wav = subprocess.run(
        ["espeak-ng", "-v", voice, "-s", "150", "--stdout", text], capture_output=True, check=True
    ).stdout
    audio, sr = sf.read(io.BytesIO(wav), dtype="int16")
    return resample(audio, sr, SR)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.int16)


def to_frames(audio: np.ndarray) -> list[np.ndarray]:
    pad = (-len(audio)) % BLOCK
    audio = np.concatenate((audio, np.zeros(pad, dtype=np.int16)))
    return [audio[i : i + BLOCK] for i in range(0, len(audio), BLOCK)]


def onset_offset(audio: np.ndarray) -> tuple[float, float]:
    loud = np.flatnonzero(np.abs(audio) > 500)
    return loud[0] / SR, loud[-1] / SR


@pytest.fixture(scope="module")
def detector():
    from claudio.wakeword.detector import OpenWakeWordDetector

    return OpenWakeWordDetector(SETTINGS.wakeword, MODELS / "wakeword", BLOCK)


def max_score(detector, audio: np.ndarray) -> float:
    detector.reset()
    best = 0.0
    for frame in to_frames(audio):
        if (hit := detector.process(frame)) is not None:
            best = max(best, hit[1])
    return best


def test_vad_finds_start_and_end_of_real_speech() -> None:
    phrase = speech("apri la cartella download")
    audio = np.concatenate((silence(0.6), phrase, silence(1.5)))
    true_start, true_end = onset_offset(audio)
    ep = Endpointer(SileroVad(MODELS / "vad" / "silero_vad.onnx"), SETTINGS.vad)
    start = end = None
    for i, frame in enumerate(to_frames(audio)):
        event = ep.process(frame)
        t = (i + 1) * BLOCK / SR
        if event is EndpointEvent.START:
            start = t
        elif event is EndpointEvent.END:
            end = t - ep.trailing_silence_s
            break
    assert start is not None and end is not None
    # START fires after min_speech_ms of speech; allow for that and one frame of granularity
    assert -0.1 <= start - true_start <= 0.1 + SETTINGS.vad.min_speech_ms / 1000 + 0.08
    assert abs(end - true_end) <= 0.15
    captured = len(ep.audio()) / SR
    assert captured >= true_end - true_start  # nothing cut at the start (pre-roll)


def test_vad_keeps_soft_first_words() -> None:
    # "apri la" 30 dB quieter than "calcolatrice", over a -60 dBFS noise floor
    soft = speech("apri la").astype(np.float32) * 10 ** (-30 / 20)
    loud = speech("calcolatrice").astype(np.float32)
    audio = np.concatenate((np.zeros(int(SR * 0.4)), soft, loud, np.zeros(SR)))
    audio += np.random.default_rng(1).standard_normal(len(audio)) * 32768 * 1e-3
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    ep = Endpointer(SileroVad(MODELS / "vad" / "silero_vad.onnx"), SETTINGS.vad)
    events = [e for f in to_frames(audio) if (e := ep.process(f)) is not EndpointEvent.NONE]
    assert events == [EndpointEvent.START, EndpointEvent.END]
    assert len(ep.audio()) >= len(soft) + len(loud)


def test_vad_ignores_silence_and_hum() -> None:
    t = np.arange(int(SR * (SETTINGS.vad.no_speech_timeout_s + 0.5))) / SR
    hum = (np.sin(2 * np.pi * 50 * t) * 800).astype(np.int16)
    ep = Endpointer(SileroVad(MODELS / "vad" / "silero_vad.onnx"), SETTINGS.vad)
    events = [ep.process(f) for f in to_frames(hum)]
    assert EndpointEvent.START not in events
    assert EndpointEvent.TIMEOUT in events


def test_wakeword_fires_on_phrase_only(detector) -> None:
    assert (
        max_score(detector, np.concatenate((silence(1), speech("hey jarvis", "en"), silence(1))))
        > 0.5
    )
    for text in ("apri la cartella download", "ciao come stai", "che ore sono"):
        assert max_score(detector, np.concatenate((silence(1), speech(text), silence(1)))) == 0


def test_reset_prevents_double_trigger(detector) -> None:
    detector.reset()
    frames = to_frames(np.concatenate((silence(1), speech("hey jarvis", "en"))))
    assert any(detector.process(f) is not None for f in frames)
    detector.reset()
    assert all(detector.process(f) is None for f in to_frames(silence(2)))


async def test_pipeline_with_real_wake_word_and_vad(detector) -> None:
    command = speech("apri chrome")
    audio = np.concatenate(
        (silence(1), speech("hey jarvis", "en"), silence(0.3), command, silence(2))
    )
    stt = FakeStt()
    speaker = FakeSpeaker()
    assistant = Assistant(
        detector,
        Endpointer(SileroVad(MODELS / "vad" / "silero_vad.onnx"), SETTINGS.vad),
        stt,
        FakeResponder(),
        speaker,
    )
    detector.reset()

    async def frames():
        for f in to_frames(audio):
            yield f
            await asyncio.sleep(0)

    await assistant.run(frames())
    assert speaker.beeps == ["wake"]
    assert speaker.spoken == ["Apro Chrome."]
    assert len(assistant.history) == 1 and assistant.history[0]["outcome"] == "ok"
    heard = len(stt.audio) / SR
    command_len = np.subtract(*onset_offset(command)[::-1])
    assert command_len <= heard <= command_len + 0.9  # pre-roll + ~100 ms tail


async def test_one_breath_request_reaches_the_transcriber_whole(detector) -> None:
    """ "hey jarvis che ore sono" without a pause: the words said before the wake word fired
    are kept too (the transcript drops the wake word itself)."""
    wake = speech("hey jarvis", "en")
    wake = wake[: int((onset_offset(wake)[1] + 0.02) * SR)]
    command = speech("che ore sono")
    command = command[max(int((onset_offset(command)[0] - 0.02) * SR), 0) :]
    audio = np.concatenate((silence(1), wake, command, silence(2)))
    stt = FakeStt()
    assistant = Assistant(
        detector,
        Endpointer(SileroVad(MODELS / "vad" / "silero_vad.onnx"), SETTINGS.vad),
        stt,
        FakeResponder(),
        FakeSpeaker(),
    )
    detector.reset()

    async def frames():
        for f in to_frames(audio):
            yield f
            await asyncio.sleep(0)

    await assistant.run(frames())
    assert stt.audio is not None
    spoken = np.subtract(*onset_offset(np.concatenate((wake, command)))[::-1])
    assert spoken <= len(stt.audio) / SR <= spoken + 0.5
