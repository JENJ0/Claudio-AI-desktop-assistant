"""Microphone capture: sounddevice callback thread -> asyncio frames."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable

import numpy as np

from claudio.audio.devices import (
    AUTO,
    WindowsDefaultInput,
    choose_input,
    default_device,
    describe_device,
    pick_device,
    probe_inputs,
    stream_settings,
)

log = logging.getLogger(__name__)

RETRY_S = 1.0
FOLLOW_S = 2.0  # how often the Windows default microphone is checked (no device configured)
# input_device = "auto": after QUIET_S with nothing above QUIET_PEAK the other microphones
# are listened to, and Claudio moves only if one of them clearly hears more. Silence alone is
# no reason: JenJo's headset sends digital silence (-96 dBFS) between phrases (24/10 test).
QUIET_PEAK = 100  # int16: about -50 dBFS
QUIET_S = 60.0
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
        chooser: Callable[[int | None], int | None] | None = None,
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
        # "auto": every microphone is listened to and the one hearing the most is used.
        self.chooser = (chooser or self._choose) if device == AUTO else None

    def _choose(self, current: int | None) -> int | None:
        """`current`: the microphone in use (None at start: Windows' own keeps the job)."""
        levels = probe_inputs(self.sample_rate)
        default = current if current is not None else default_device("input")
        index = choose_input(levels, default)
        log.info(
            "Microfoni sentiti: %s",
            ", ".join(f"{i} {db:.0f} dBFS" for i, db in sorted(levels.items())) or "nessuno",
        )
        return index

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
        switch_to: int | None = None  # "auto" found a microphone that hears more
        while True:
            index = None
            opened_on = self._windows_default()
            try:
                if switch_to is not None:
                    index, switch_to = switch_to, None
                elif self.chooser is not None:
                    index = await asyncio.to_thread(self.chooser, None)
                elif not portaudio_default:
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
                    checked = loud = time.monotonic()
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
                        if self.chooser is not None:
                            now_t = time.monotonic()
                            if int(np.abs(frame.astype(np.int32)).max()) > QUIET_PEAK:
                                loud = now_t
                            elif now_t - loud >= QUIET_S:
                                loud = now_t
                                # The stream stays open: the probe opens the others alongside.
                                best = await asyncio.to_thread(self.chooser, index)
                                if best is not None and best != index:
                                    log.info("Un altro microfono sente di più: passo a quello")
                                    switch_to = best
                                    break
                if switch_to is not None:
                    continue  # no re-scan: it would also break the speaker's open stream
                # Windows' microphone changed: re-scan, a just-plugged one is not listed yet.
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
