"""Piper speaks, Silero cuts, Whisper listens, the router understands.

Needs the Piper voice and (for the round trip) a Whisper model in models/:
    uv run python scripts/download_models.py --whisper small
    uv run pytest -m slow
CI runs this on Windows with Whisper `small` on CPU, the configured fallback model
(`base` is too weak: it heard "Alza il volume" as "A cell volume").
"""

from __future__ import annotations

import numpy as np
import pytest

from claudio.audio.player import resample
from claudio.audio.vad import Endpointer, EndpointEvent, SileroVad
from claudio.config import load_settings
from claudio.router.matcher import CommandMatcher, load_commands
from claudio.tts.speaker import PiperSynth

pytestmark = pytest.mark.slow

SETTINGS = load_settings()
MODELS = SETTINGS.resolve(SETTINGS.models_dir)
VOICE = MODELS / "piper" / f"{SETTINGS.tts.voice}.onnx"
WHISPER_DIR = MODELS / "whisper"

if not VOICE.is_file():
    pytest.skip("voce Piper non scaricata", allow_module_level=True)


def whisper_model() -> str | None:
    """Smallest downloaded Whisper model, if any."""
    for name, repo in (
        ("base", "faster-whisper-base"),
        ("small", "faster-whisper-small"),
        ("large-v3-turbo", "faster-whisper-large-v3-turbo"),
    ):
        if any(WHISPER_DIR.glob(f"models--*--{repo}")):
            return name
    return None


@pytest.fixture(scope="module")
def synth() -> PiperSynth:
    # No random variation: the same audio on every run, so results are reproducible.
    tts = SETTINGS.tts.model_copy(update={"noise_scale": 0.0, "noise_w_scale": 0.0})
    return PiperSynth(VOICE, tts)


def test_piper_speaks_italian(synth: PiperSynth) -> None:
    audio = synth.synthesize("Apro la cartella Download.")
    seconds = len(audio) / synth.sample_rate
    assert audio.dtype == np.int16 and 0.8 < seconds < 4
    assert np.abs(audio).max() > 3000


PHRASES = [
    ("Apri la cartella Download.", "open_downloads"),
    ("Alza il volume.", "volume_up"),
    ("Che ore sono?", "tell_time"),
    ("Apri la calcolatrice.", "open_calculator"),
    ("Apri il blocco note.", "open_notepad"),
]


@pytest.mark.skipif(whisper_model() is None, reason="nessun modello Whisper scaricato")
def test_roundtrip_speech_to_command(synth: PiperSynth) -> None:
    from claudio.stt.transcriber import WhisperTranscriber

    name = whisper_model()
    assert name is not None
    stt_settings = SETTINGS.stt.model_copy(update={"model": name, "fallback_model": name})
    stt = WhisperTranscriber(stt_settings, WHISPER_DIR)
    matcher = CommandMatcher(load_commands(SETTINGS.resolve(SETTINGS.router.commands_file)))
    vad = SileroVad(MODELS / "vad" / "silero_vad.onnx")
    endpointer = Endpointer(vad, SETTINGS.vad)

    hits = 0
    for phrase, expected in PHRASES:
        speech = resample(synth.synthesize(phrase), synth.sample_rate, 16000)
        audio = np.concatenate((np.zeros(8000, np.int16), speech, np.zeros(24000, np.int16)))
        endpointer.reset()
        event = EndpointEvent.NONE
        for i in range(0, len(audio) - 1279, 1280):
            event = endpointer.process(audio[i : i + 1280])
            if event is EndpointEvent.END:
                break
        assert event is EndpointEvent.END, phrase
        utterance = endpointer.audio()
        text = stt.transcribe(utterance)
        match = matcher.match(text)
        print(f"{phrase!r} -> {text!r} -> {match.command.id if match else None}")
        ok = bool(match and match.command.id == expected)
        hits += ok
        if not ok:  # diagnose: VAD cut, prompt bias, or the audio itself?
            raw = stt.transcribe(speech.astype(np.float32) / 32768)
            stt.settings = stt_settings.model_copy(update={"initial_prompt": ""})
            bare = stt.transcribe(utterance)
            stt.settings = stt_settings
            print(
                f"   speech {len(speech) / 16000:.2f} s, utterance {len(utterance) / 16000:.2f} s,"
                f" raw {raw!r}, no prompt {bare!r}"
            )
    assert hits >= len(PHRASES) - 1  # `small` may miss one; large-v3-turbo should get all
    assert not stt.transcribe(np.zeros(16000 * 2, dtype=np.float32))  # silence -> nothing
    hiss = np.random.default_rng(0).standard_normal(16000 * 2).astype(np.float32) * 0.003
    print(f"hiss -> {stt.transcribe(hiss)!r}")  # informative: v0.1.1 tunes no_speech filters
