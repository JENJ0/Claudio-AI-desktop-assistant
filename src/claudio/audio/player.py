"""Audio output through one persistent stream (no per-sentence open latency)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from claudio.audio.devices import (
    WindowsDefaultOutput,
    describe_device,
    match_endpoint,
    pick_output,
    stream_settings,
)

log = logging.getLogger(__name__)

CHUNK_S = 0.1  # audio is queued in slices this long, so that a sentence can be cut short


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling; good enough for speech and beeps."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    n = round(len(audio) * dst_rate / src_rate)
    x = np.linspace(0, len(audio) - 1, n)
    return np.interp(x, np.arange(len(audio)), audio.astype(np.float32)).astype(audio.dtype)


class AudioPlayer:
    """`device` is the configured spec (name or index), resolved again at every (re)open.

    With no device configured, every sentence and beep goes to the output Windows is using at
    that moment: `windows_default` names it (Core Audio), and the stream moves when it changes."""

    def __init__(
        self,
        sample_rate: int,
        device: str | int = "",
        windows_default: Callable[[], str | None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        follow = device == "" or device is None
        self.windows_default = (windows_default or WindowsDefaultOutput()) if follow else None
        self._lock = threading.Lock()
        self._stream: Any = None
        self._index: int | None = None
        self._default_name: str | None = None

    def start(self) -> None:
        name = self.windows_default() if self.windows_default else None
        self._default_name = name
        self._open(pick_output(self.device, name))

    def _open(self, index: int | None) -> None:
        import sounddevice as sd

        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=index,
                latency="low",
                extra_settings=stream_settings(index),
            )
        except sd.PortAudioError as exc:
            if index is None or self.device not in ("", None):
                raise
            # The default under WASAPI refused to open: PortAudio's own default (MME) still works.
            log.warning("Uscita %s non apribile (%s): uso quella di base", index, exc)
            index = None
            stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype="int16", device=None, latency="low"
            )
        stream.start()
        self._stream, self._index = stream, index
        log.info("Uscita audio: %s", describe_device(index, "output"))

    def _follow_default(self) -> None:
        """Move to Windows' default output if it changed since the last sentence."""
        assert self.windows_default is not None
        name = self.windows_default()
        if not name or name == self._default_name:
            return
        self._default_name = name
        index = match_endpoint(name, "output")
        if index is None:  # plugged in after start-up: PortAudio does not know it yet
            log.warning(
                "Uscita predefinita di Windows cambiata in %s, che non c'era all'avvio: "
                "riavvia CLAUDIO per usarla",
                name,
            )
            return
        if index != self._index or self._stream is None:
            log.info("Uscita predefinita di Windows cambiata: %s", name)
            self.close()
            self._open(index)

    def write(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        cancel: threading.Event | None = None,
    ) -> bool:
        """Blocking: returns when the audio has been queued to the device. With `cancel`,
        it is queued in CHUNK_S slices and stops as soon as the event is set ("basta");
        returns False when it was cut short."""
        audio = resample(audio, sample_rate or self.sample_rate, self.sample_rate)
        data = np.ascontiguousarray(audio, dtype=np.int16)
        step = max(1, int(self.sample_rate * CHUNK_S)) if cancel is not None else len(data)
        with self._lock:
            if self.windows_default is not None:
                self._follow_default()
            if self._stream is None:
                self.start()
            for start in range(0, max(len(data), 1), max(step, 1)):
                if cancel is not None and cancel.is_set():
                    return False
                self._write(data[start : start + step])
        return True

    def _write(self, data: np.ndarray) -> None:
        try:
            self._stream.write(data)
        except Exception as exc:  # device unplugged: reopen once and retry
            log.warning("Uscita audio non disponibile (%s): riapro il dispositivo", exc)
            self.close()
            self.start()
            self._stream.write(data)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
