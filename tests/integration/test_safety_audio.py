"""Whisper hears the recorded safety phrases; parse_answer must never turn a near-yes into a yes.

Needs JenJo's takes in samples/safety/<yes|no|unclear>/ (uv run python scripts/record_samples.py
safety --repeat 2) and the configured Whisper model in models/. Run: uv run pytest -m slow
Limit: this covers Whisper plus parse_answer only; the live path (responder's open-question
window, the distance gate) is tested on text elsewhere.
"""

from __future__ import annotations

import pytest
import soundfile as sf

from claudio.config import PROJECT_ROOT, load_settings
from claudio.safety.confirmation import Answer, parse_answer
from claudio.stt.transcriber import WhisperTranscriber

pytestmark = pytest.mark.slow

SETTINGS = load_settings()
SAFETY_DIR = PROJECT_ROOT / "samples" / "safety"
WAVS = sorted(SAFETY_DIR.glob("*/*.wav"))

if not WAVS:
    pytest.skip("nessuna registrazione in samples/safety/", allow_module_level=True)


def test_no_near_yes_phrase_becomes_a_yes(capsys: pytest.CaptureFixture[str]) -> None:
    n_negatives = sum(w.parent.name != "yes" for w in WAVS)
    n_yes = len(WAVS) - n_negatives
    # An early "q" in the recorder leaves only yes/: that must not pass as "nothing accepted".
    assert n_negatives or not n_yes, "samples/safety/ has yes takes but no no/ or unclear/ take"
    stt = WhisperTranscriber(SETTINGS.stt, SETTINGS.resolve(SETTINGS.models_dir) / "whisper")
    understood = empty = 0
    wrongly_accepted = []
    with capsys.disabled():
        for wav in WAVS:
            name = wav.relative_to(SAFETY_DIR).as_posix()
            audio, sr = sf.read(wav, dtype="float32")
            assert sr == 16000, f"{name}: serve 16 kHz"
            text = stt.transcribe(audio)
            verdict = parse_answer(text)
            empty += not text.strip()
            print(f"{name:<40} -> {verdict.value:<8} {text!r}")
            if wav.parent.name == "yes":
                understood += verdict is Answer.YES
            elif verdict is Answer.YES:
                wrongly_accepted.append(f"{name}: {text!r}")
        print(
            f"\nSì capiti: {understood}/{n_yes}; negativi ascoltati: {n_negatives}; "
            f"trascrizioni vuote: {empty}"
        )
    assert not wrongly_accepted, f"un NON-sì è stato accettato come sì: {wrongly_accepted}"
