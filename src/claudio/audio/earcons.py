"""Short confirmation tones, generated at start-up (no audio files)."""

from __future__ import annotations

import numpy as np


def tone(freqs: list[float], duration_s: float, sample_rate: int, volume: float) -> np.ndarray:
    """Sequence of sine tones with short fades, as int16."""
    parts = []
    n = int(sample_rate * duration_s / len(freqs))
    fade = min(int(sample_rate * 0.01), n // 2)
    env = np.ones(n, dtype=np.float32)
    env[:fade] = np.linspace(0, 1, fade)
    env[n - fade :] = np.linspace(1, 0, fade)
    t = np.arange(n, dtype=np.float32) / sample_rate
    for f in freqs:
        parts.append(np.sin(2 * np.pi * f * t) * env)
    audio = np.concatenate(parts) * volume
    return (audio * 32767).astype(np.int16)


def make_earcons(sample_rate: int, volume: float) -> dict[str, np.ndarray]:
    return {
        "wake": tone([880.0, 1320.0], 0.12, sample_rate, volume),  # rising: "I'm listening"
        "cancel": tone([440.0, 330.0], 0.18, sample_rate, volume),  # falling: "never mind"
        "error": tone([220.0], 0.25, sample_rate, volume),
    }
