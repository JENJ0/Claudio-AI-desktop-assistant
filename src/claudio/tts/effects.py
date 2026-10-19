"""Voice effects applied to every synthesised sentence: Claudio's robotic tone.

Presets (tts.effect): "estremo" (strong ring modulation + metallic echo), "jarvis" (clean,
chorus and a short reverb), "ultron" (a lower copy, distortion, echo) and "ultron_estremo"
(half and half, JenJo's choice). Parameters are in ms and Hz, so any sample rate works."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from claudio.i18n import tr

TAIL_S = 0.15  # silence added after each sentence so echoes and reverb can fade out
PEAK = 0.8


def norm(x: np.ndarray, peak: float = PEAK) -> np.ndarray:
    top = float(np.abs(x).max()) if x.size else 0.0
    return x * (peak / top) if top > 0 else x


def ring(x: np.ndarray, sr: int, freq: float, depth: float) -> np.ndarray:
    """Ring modulation: the 'robot' buzz. depth 1 = the full effect."""
    t = np.arange(len(x)) / sr
    return x * ((1 - depth) + depth * np.sin(2 * np.pi * freq * t))


def metallic(
    x: np.ndarray, sr: int, delay_ms: float, decay: float, taps: int, mix: float
) -> np.ndarray:
    """Short repeated echoes (feed-forward comb): a metallic, tube-like colour."""
    d = int(sr * delay_ms / 1000)
    y = x.copy()
    for k in range(1, taps + 1):
        if k * d < len(x):
            y[k * d :] += decay**k * x[: len(x) - k * d]
    return (1 - mix) * x + mix * y / (1 + sum(decay**k for k in range(1, taps + 1)))


def comb(x: np.ndarray, sr: int, delay_ms: float, feedback: float) -> np.ndarray:
    """y[n] = x[n] + feedback * y[n - d], computed one delay-long block at a time."""
    d = max(int(sr * delay_ms / 1000), 1)
    y = x.astype(np.float64, copy=True)
    for start in range(d, len(y), d):
        end = min(start + d, len(y))
        y[start:end] += feedback * y[start - d : end - d]
    return y


def reverb(x: np.ndarray, sr: int, mix: float) -> np.ndarray:
    wet = np.mean([comb(x, sr, ms, 0.55) for ms in (29.7, 37.1, 41.1, 43.7)], axis=0)
    top = float(np.abs(wet).max())
    if top == 0:
        return x
    return (1 - mix) * x + mix * wet / top * float(np.abs(x).max())


def pitch_shift(x: np.ndarray, sr: int, factor: float, grain_ms: float = 40) -> np.ndarray:
    """Granular pitch shift, same duration: factor < 1 lowers the voice."""
    g = int(sr * grain_ms / 1000)
    hop = g // 2
    window = np.hanning(g)
    out = np.zeros(len(x) + g)
    src = np.arange(g) * factor
    grid = np.arange(g)
    for start in range(0, len(x) - g, hop):
        out[start : start + g] += np.interp(src, grid, x[start : start + g]) * window
    return out[: len(x)]


def chorus(
    x: np.ndarray, sr: int, base_ms: float = 12, depth_ms: float = 3, rate: float = 0.4
) -> np.ndarray:
    n = np.arange(len(x))
    delay = (base_ms + depth_ms * np.sin(2 * np.pi * rate * n / sr)) * sr / 1000
    return 0.65 * x + 0.35 * np.interp(n - delay, n, x, left=0.0)


def estremo(x: np.ndarray, sr: int) -> np.ndarray:
    return metallic(ring(ring(x, sr, 120, 1.0), sr, 35, 0.8), sr, 14.0, 0.75, 7, 0.85)


def jarvis(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    y = sosfilt(butter(2, [150, min(6500, sr * 0.45)], btype="bandpass", fs=sr, output="sos"), x)
    y = chorus(y, sr)
    d = int(sr * 0.018)
    y = y + 0.25 * np.concatenate((np.zeros(d), y[:-d]))
    return reverb(y, sr, 0.2)


def ultron(x: np.ndarray, sr: int) -> np.ndarray:
    y = 0.55 * norm(pitch_shift(x, sr, 0.72), 1) + 0.45 * norm(x, 1)
    y = ring(y, sr, 45, 0.35)
    y = np.tanh(3.0 * norm(y, 1))
    return reverb(comb(y, sr, 7, 0.45), sr, 0.25)


def ultron_estremo(x: np.ndarray, sr: int) -> np.ndarray:
    return 0.5 * norm(ultron(x, sr), 1) + 0.5 * norm(estremo(x, sr), 1)


PRESETS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "estremo": estremo,
    "jarvis": jarvis,
    "ultron": ultron,
    "ultron_estremo": ultron_estremo,
}


class VoiceEffect:
    """int16 sentence in, int16 sentence out (TAIL_S longer), peak at 80% of full scale."""

    def __init__(self, name: str) -> None:
        if name not in PRESETS:
            raise ValueError(
                tr(
                    "Effetto voce '{name}' sconosciuto (disponibili: {known})",
                    name=name,
                    known=", ".join(PRESETS),
                )
            )
        self.name = name
        self._fx = PRESETS[name]

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.size == 0:
            return audio
        x = np.concatenate((audio / 32768.0, np.zeros(int(TAIL_S * sample_rate))))
        y = norm(self._fx(x, sample_rate))
        return (np.clip(y, -1, 1) * 32767).astype(np.int16)
