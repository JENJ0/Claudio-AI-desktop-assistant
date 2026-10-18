"""Microphone capture: sounddevice callback thread -> asyncio frames."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from claudio.audio.devices import describe_device, resolve_device, stream_settings

log = logging.getLogger(__name__)


class AudioCapture:
    """Yields 16 kHz mono int16 frames; reopens the device if it disappears.

    `device` is the configured spec (name or index): it is resolved again at every reopen,
    because indexes change when a USB/Bluetooth device reconnects."""

    def __init__(
        self, sample_rate: int, block_size: int, device: str | int = "", queue_size: int = 64
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.queue_size = queue_size
        self.overflows = 0

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
            loop.call_soon_threadsafe(put, indata[:, 0].copy())

        while True:
            try:
                index = resolve_device(self.device, "input")
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    channels=1,
                    dtype="int16",
                    device=index,
                    callback=callback,
                    extra_settings=stream_settings(index),
                ):
                    log.info("Microfono: %s", describe_device(index, "input"))
                    while True:
                        # No frame for 2 s means the device stopped delivering audio.
                        yield await asyncio.wait_for(queue.get(), timeout=2.0)
            except (sd.PortAudioError, TimeoutError, ValueError) as exc:
                log.warning("Microfono non disponibile (%s): riprovo tra 1 s", exc)
                await asyncio.sleep(1.0)
                # Re-scan devices so a re-plugged USB microphone is found again.
                sd._terminate()
                sd._initialize()
