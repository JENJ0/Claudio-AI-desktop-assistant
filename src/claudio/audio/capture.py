"""Microphone capture: sounddevice callback thread -> asyncio frames."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable

import numpy as np

from claudio.audio.devices import (
    WindowsDefaultInput,
    describe_device,
    pick_device,
    stream_settings,
)

log = logging.getLogger(__name__)

RETRY_S = 1.0
FOLLOW_S = 2.0  # how often the Windows default microphone is checked (no device configured)
STILL_MISSING_EVERY = 30  # retries between two "still missing" warnings (~30 s)


def apply_gain(frame: np.ndarray, gain: float) -> np.ndarray:
    """int16 frame times a linear gain, clipped instead of wrapping around."""
    return np.clip(frame * gain, -32768, 32767).astype(np.int16)


class AudioCapture:
    """Yields 16 kHz mono int16 frames; reopens the device if it disappears.

    `device` is the configured spec (name or index): it is resolved again at every reopen,
    because indexes change when a USB/Bluetooth device reconnects."""

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        device: str | int = "",
        queue_size: int = 64,
        gain_db: float = 0.0,
        windows_default: Callable[[], str | None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.queue_size = queue_size
        self.gain = 1.0
        self.set_gain_db(gain_db)
        self.overflows = 0
        # With no device configured: the microphone Windows uses, followed while running
        # (checked every FOLLOW_S; a change reopens the stream on the new one).
        self.windows_default = (windows_default or WindowsDefaultInput()) if device == "" else None

    def _windows_default(self) -> str | None:
        return self.windows_default() if self.windows_default is not None else None

    def set_gain_db(self, gain_db: float) -> None:
        """Takes effect from the next frame (the callback reads it every time)."""
        self.gain = 10 ** (gain_db / 20)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=self.queue_size)

        def put(frame: np.ndarray) -> None:
            if queue.full():  # consumer is late: drop the oldest frame, never block
                queue.get_nowait()
                self.overflows += 1
            queue.put_nowait(frame)

        def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            if status:
                log.debug("Stato audio in ingresso: %s", status)
            frame = indata[:, 0]
            frame = apply_gain(frame, self.gain) if self.gain != 1.0 else frame.copy()
            loop.call_soon_threadsafe(put, frame)

        portaudio_default = False  # the WASAPI default failed once: use PortAudio's (MME)
        lost_at: float | None = None  # when the microphone went missing
        retries = 0
        while True:
            index = None
            opened_on = self._windows_default()
            try:
                if not portaudio_default:
                    index = pick_device(self.device, "input", opened_on)
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    channels=1,
                    dtype="int16",
                    device=index,
                    callback=callback,
                    extra_settings=stream_settings(index),
                ):
                    log.info(
                        "Microfono: %s%s",
                        describe_device(index, "input"),
                        f", guadagno +{20 * np.log10(self.gain):.0f} dB" if self.gain > 1 else "",
                    )
                    checked = time.monotonic()
                    while True:
                        # No frame for 2 s means the device stopped delivering audio.
                        frame = await asyncio.wait_for(queue.get(), timeout=2.0)
                        if time.monotonic() - checked >= FOLLOW_S:
                            checked = time.monotonic()
                            now = self._windows_default()
                            if now and now != opened_on:
                                log.info("Microfono di Windows cambiato: %s", now)
                                break  # the stream reopens on the new one
                        if lost_at is not None:  # back only once audio flows again
                            log.info(
                                "Microfono di nuovo disponibile dopo %.0f s",
                                time.monotonic() - lost_at,
                            )
                            lost_at, retries = None, 0
                        yield frame
                # Windows switched microphone: re-scan, a just-plugged one is not listed yet.
                portaudio_default = False
                sd._terminate()
                sd._initialize()
            except (sd.PortAudioError, TimeoutError, ValueError) as exc:
                # Unplugged, it fails every second: warn once, then every ~30 s, not 3600 lines/h.
                if lost_at is None:
                    lost_at = time.monotonic()
                    log.warning("Microfono non disponibile (%s): riprovo tra 1 s", exc)
                elif retries % STILL_MISSING_EVERY == 0:
                    log.warning(
                        "Microfono ancora non disponibile da %.0f s (%s): ricollegalo",
                        time.monotonic() - lost_at,
                        exc,
                    )
                retries += 1
                if self.device == "" and index is not None and isinstance(exc, sd.PortAudioError):
                    portaudio_default = True
                await asyncio.sleep(RETRY_S)
                # Re-scan devices so a re-plugged USB microphone is found again.
                sd._terminate()
                sd._initialize()
