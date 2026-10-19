"""The real Kokoro model speaks Italian. Skipped unless scripts/download_models.py ran."""

from __future__ import annotations

import time

import numpy as np
import pytest

from claudio.audio.player import resample
from claudio.audio.vad import Endpointer, EndpointEvent, SileroVad
from claudio.config import load_settings

pytestmark = pytest.mark.slow

SETTINGS = load_settings()
MODELS = SETTINGS.resolve(SETTINGS.models_dir)

try:
    from claudio.tts.kokoro import KokoroSynth, kokoro_files

    kokoro_files(MODELS / "kokoro")
except (FileNotFoundError, ImportError) as exc:
    pytest.skip(f"Kokoro non disponibile: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def synth() -> KokoroSynth:
    return KokoroSynth(MODELS / "kokoro", SETTINGS.tts)


def test_nicola_speaks_a_short_sentence(synth: KokoroSynth) -> None:
    synth.synthesize("Ciao.")
    start = time.perf_counter()
    audio = synth.synthesize("Apro la cartella Download.")
    elapsed = time.perf_counter() - start
    seconds = len(audio) / synth.sample_rate
    print(
        f"Kokoro {synth.voice} speed {synth.speed:.2f}: {elapsed * 1000:.0f} ms for {seconds:.2f} s"
    )
    assert audio.dtype == np.int16 and 0.6 < seconds < 3.5
    assert np.abs(audio).max() > 3000
    lead = np.abs(audio[: synth.sample_rate // 20]).max()  # first 50 ms: speech, not silence
    assert lead > 300


def test_phonemes_stay_right_next_to_piper(synth: KokoroSynth) -> None:
    """Kokoro and Piper (its fallback) in one process: setting up espeak-ng twice garbled
    every phoneme of both voices on Windows."""
    from claudio.tts.speaker import PiperSynth, voice_path

    try:
        path = voice_path(MODELS / "piper", SETTINGS.tts.voice)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    expected = synth.phonemes("Che ore sono?")
    PiperSynth(path, SETTINGS.tts).synthesize("Apro la calcolatrice.")
    assert synth.phonemes("Che ore sono?") == expected == "ke ˈore sˈono?"


def test_silero_hears_nicola_from_start_to_end(synth: KokoroSynth) -> None:
    speech = resample(synth.synthesize("Che ore sono?"), synth.sample_rate, 16000)
    audio = np.concatenate((np.zeros(8000, np.int16), speech, np.zeros(16000, np.int16)))
    endpointer = Endpointer(SileroVad(MODELS / "vad" / "silero_vad.onnx"), SETTINGS.vad)
    events = [
        e
        for i in range(0, len(audio) - 1279, 1280)
        if (e := endpointer.process(audio[i : i + 1280])) is not EndpointEvent.NONE
    ]
    assert events == [EndpointEvent.START, EndpointEvent.END]
    assert len(endpointer.audio()) >= len(speech)


def test_empty_text_gives_no_audio(synth: KokoroSynth) -> None:
    assert synth.synthesize("").size == 0
