"""Audio output through one persistent stream (no per-sentence open latency)."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling; good enough for speech and beeps."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    n = round(len(audio) * dst_rate / src_rate)
    x = np.linspace(0, len(audio) - 1, n)
    return np.interp(x, np.arange(len(audio)), audio.astype(np.float32)).astype(audio.dtype)


class AudioPlayer:
    def __init__(self, sample_rate: int, device: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._lock = threading.Lock()
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            latency="low",
        )
        stream.start()
        self._stream = stream

    def write(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        """Blocking: returns when the audio has been queued to the device."""
        if self._stream is None:
            self.start()
        audio = resample(audio, sample_rate or self.sample_rate, self.sample_rate)
        with self._lock:
            try:
                self._stream.write(np.ascontiguousarray(audio, dtype=np.int16))
            except Exception as exc:  # device unplugged: reopen once and retry
                log.warning("Uscita audio non disponibile (%s): riapro il dispositivo", exc)
                self.close()
                self.start()
                self._stream.write(np.ascontiguousarray(audio, dtype=np.int16))

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
