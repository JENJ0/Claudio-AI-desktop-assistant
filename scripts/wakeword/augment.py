"""Audio augmentation for wake word training, with numpy and scipy only.

openWakeWord's own augmentation needs speechbrain, torchaudio and audiomentations, whose
versions break every few Colab updates; this does what the models need with nothing else:
speed, level, room echo, microphone colour and background noise, and places each clip in a
fixed window the way the detector sees it while streaming."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 16000


def to_float(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def read_wav(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz (other rates are resampled)."""
    sr, audio = wavfile.read(path)
    audio = to_float(np.asarray(audio))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample(audio, sr, SR)


def write_wav(path: Path, audio: np.ndarray) -> None:
    wavfile.write(path, SR, to_int16(audio) if audio.dtype != np.int16 else audio)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    ratio = Fraction(sr_out, sr_in).limit_denominator(1000)
    return signal.resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


def trim(audio: np.ndarray, top_db: float = 35.0, pad_s: float = 0.08) -> np.ndarray:
    """The loud part of a clip (10 ms frames within top_db of the loudest), plus a margin."""
    frame = SR // 100
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    if rms.max() <= 0:
        return audio[:0]
    loud = np.flatnonzero(rms >= rms.max() * 10 ** (-top_db / 20))
    pad = int(pad_s * SR)
    start = max(int(loud[0]) * frame - pad, 0)
    end = min((int(loud[-1]) + 1) * frame + pad, len(audio))
    return audio[start:end]


def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """factor > 1 is faster (and higher), like a tape: fast speech is what the models miss."""
    ratio = Fraction(1 / factor).limit_denominator(100)
    return signal.resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


def set_peak(audio: np.ndarray, dbfs: float) -> np.ndarray:
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    return audio if peak == 0 else audio * (10 ** (dbfs / 20) / peak)


def synthetic_rir(rng: np.random.Generator, rt60: float | None = None) -> np.ndarray:
    """A room: the direct sound, then exponentially decaying noise (reverberation time rt60)."""
    rt60 = rt60 if rt60 is not None else float(rng.uniform(0.15, 0.8))
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60)
    delay = int(rng.uniform(0.002, 0.02) * SR)  # first reflection
    rir = np.zeros(n + delay, dtype=np.float32)
    rir[0] = 1.0
    rir[delay:] += tail * float(rng.uniform(0.1, 0.5))
    return rir / np.sqrt(np.sum(rir**2))


def reverb(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    wet = signal.fftconvolve(audio, rir)[: len(audio)].astype(np.float32)
    peak = float(np.abs(audio).max())
    return set_peak(wet, 20 * np.log10(peak)) if peak > 0 else wet


def colour(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A different microphone: cut the lows (cheap mics), the highs (webcams) or both."""
    sos = []
    if rng.random() < 0.6:
        sos.append(signal.butter(2, rng.uniform(80, 400), "highpass", fs=SR, output="sos"))
    if rng.random() < 0.6:
        sos.append(signal.butter(4, rng.uniform(2800, 7000), "lowpass", fs=SR, output="sos"))
    for s in sos:
        audio = signal.sosfilt(s, audio).astype(np.float32)
    return audio


def coloured_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """White, pink or brown noise: fans, hum, air conditioning."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    freqs[0] = freqs[1] if n > 1 else 1.0
    spectrum = spectrum / freqs ** float(rng.choice([0.0, 0.5, 1.0]))
    noise = np.fft.irfft(spectrum, n).astype(np.float32)
    return noise / (np.std(noise) + 1e-9)


class Backgrounds:
    """Random stretches of background audio (16 kHz int16 WAVs, read through mmap)."""

    def __init__(self, files: list[Path]) -> None:
        self.files = [f for f in files if f.stat().st_size > SR * 2]  # at least 0.5 s

    def __len__(self) -> int:
        return len(self.files)

    def segment(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if not self.files:
            return coloured_noise(n, rng)
        path = self.files[int(rng.integers(len(self.files)))]
        sr, data = wavfile.read(path, mmap=True)
        if data.ndim > 1:
            data = data[:, 0]
        if sr != SR:
            data = resample(to_float(np.asarray(data)), sr, SR)
        if len(data) < n:
            data = np.tile(np.asarray(data), n // max(len(data), 1) + 1)
        start = int(rng.integers(0, len(data) - n + 1))
        segment = to_float(np.asarray(data[start : start + n]))
        std = float(np.std(segment))
        return segment / std if std > 1e-6 else coloured_noise(n, rng)


def mix(fg: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    """Foreground with background at snr_db (bg already has unit power)."""
    voiced = fg[np.abs(fg) > 1e-4]
    power = float(np.mean(voiced**2)) if len(voiced) else 0.0
    if power == 0:
        return fg + bg * 0.01
    return fg + bg * np.sqrt(power / 10 ** (snr_db / 10))


def place(
    clip: np.ndarray,
    total: int,
    rng: np.random.Generator,
    at_end: bool,
    end_jitter_s: float = 0.2,
    tail: np.ndarray | None = None,
) -> np.ndarray:
    """A window of `total` samples. A wake word ends near the window's end (the detector
    fires as the phrase ends), optionally with the start of other speech after it, as when
    the command is glued to the wake word; anything else goes anywhere."""
    out = np.zeros(total, dtype=np.float32)
    if len(clip) >= total:
        start = len(clip) - total if at_end else int(rng.integers(0, len(clip) - total + 1))
        return clip[start : start + total].copy()
    if at_end:
        gap = int(rng.uniform(0, end_jitter_s) * SR)
        start = max(total - len(clip) - gap, 0)
        out[start : start + len(clip)] = clip
        end = start + len(clip)
        if tail is not None and end < total:
            room = min(total - end, len(tail))
            out[end : end + room] = tail[:room] * float(
                np.abs(clip).max() / (np.abs(tail).max() + 1e-9)
            )
        return out
    start = int(rng.integers(0, total - len(clip) + 1))
    out[start : start + len(clip)] = clip
    return out


def augment(
    clip: np.ndarray,
    total: int,
    rng: np.random.Generator,
    backgrounds: Backgrounds,
    positive: bool,
    tails: list[np.ndarray] | None = None,
) -> np.ndarray:
    """One training window from one clip: every effect is random, so each call differs."""
    clip = trim(clip)
    if len(clip) == 0:
        clip = np.zeros(SR // 10, dtype=np.float32)
    if rng.random() < 0.5:
        clip = change_speed(clip, float(rng.uniform(0.85, 1.25)))
    clip = set_peak(clip, float(rng.uniform(-12, -1)))
    tail = None
    if positive and tails and rng.random() < 0.3:
        tail = trim(tails[int(rng.integers(len(tails)))])
    out = place(
        clip, total, rng, at_end=positive, end_jitter_s=0.35 if tail is not None else 0.2, tail=tail
    )
    if rng.random() < 0.4:
        out = reverb(out, synthetic_rir(rng))
    if rng.random() < 0.4:
        out = colour(out, rng)
    if rng.random() < 0.8:
        out = mix(out, backgrounds.segment(total, rng), float(rng.uniform(0, 20)))
    out = set_peak(out, float(rng.uniform(-25, -1)))
    return to_int16(out)
