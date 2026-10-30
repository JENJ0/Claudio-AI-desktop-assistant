"""The real wake word models right after a reset: quiet audio must not wake CLAUDIO.

openWakeWord's own reset fills what the models read with the features of 4 s of random noise:
read ~0.5 s later, that noise fired the wake word on silence after every beep (30/10).

Skipped unless the models are downloaded (scripts/download_models.py); JenJo's own models
("claudio", "hey_claudio", not in Git) are checked too when they are there. Run with:
uv run pytest -m slow"""

from __future__ import annotations

import numpy as np
import pytest

from claudio.config import WakeWordModel, WakeWordSettings, load_settings
from claudio.wakeword.detector import OpenWakeWordDetector

pytestmark = pytest.mark.slow

SETTINGS = load_settings()
WAKEWORD_DIR = SETTINGS.resolve(SETTINGS.models_dir) / "wakeword"
BLOCK = 1280
# After the old reset these signals scored up to 0.19 ("hey_jarvis") and 0.59 ("claudio");
# after the fixed one 0.04 and 0.0.
QUIET_PEAK = 0.1
MODELS = [m for m in ("hey_jarvis", "claudio", "hey_claudio") if any(WAKEWORD_DIR.glob(f"{m}*"))]

if not (WAKEWORD_DIR / "melspectrogram.onnx").is_file():
    pytest.skip("modelli non scaricati", allow_module_level=True)


def quiet_signals(seed: int) -> dict[str, np.ndarray]:
    """1 s of the quiet a microphone hears: digital silence (a headset's gate), white noise
    and a darker noise, from far below the voice to just under it."""
    rng = np.random.default_rng(seed)
    n = 13 * BLOCK
    signals = {"zeros": np.zeros(n)}
    for dbfs in (-90, -60, -40):
        rms = 32768 * 10 ** (dbfs / 20)
        signals[f"white {dbfs}"] = rng.standard_normal(n) * rms
        dark = np.cumsum(rng.standard_normal(n))
        dark -= np.convolve(dark, np.ones(400) / 400, "same")
        signals[f"dark {dbfs}"] = dark / np.sqrt(np.mean(dark**2)) * rms
    return {k: np.clip(v, -32768, 32767).astype(np.int16) for k, v in signals.items()}


@pytest.mark.parametrize("model", MODELS)
def test_quiet_audio_after_a_reset_stays_far_below_the_threshold(model: str) -> None:
    detector = OpenWakeWordDetector(
        WakeWordSettings(models=[WakeWordModel(name=model, threshold=0.5)]), WAKEWORD_DIR, BLOCK
    )
    peaks = {}
    for seed in range(3):
        for name, audio in quiet_signals(seed).items():
            detector.reset()
            scores = [
                max(detector._model.predict(audio[i : i + BLOCK]).values())
                for i in range(0, len(audio), BLOCK)
            ]
            peaks[f"{name} #{seed}"] = round(float(max(scores)), 2)
    assert max(peaks.values()) < QUIET_PEAK, peaks
