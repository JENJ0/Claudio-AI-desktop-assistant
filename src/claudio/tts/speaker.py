"""Piper TTS: synthesises the next sentence while the current one plays."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np

from claudio.audio.earcons import make_earcons
from claudio.audio.player import AudioPlayer
from claudio.config import TtsSettings
from claudio.tts.sentence_splitter import group_stream, split_stream

log = logging.getLogger(__name__)


def voice_path(piper_dir: Path, voice: str) -> Path:
    path = piper_dir / f"{voice}.onnx"
    if not path.is_file() or not path.with_suffix(".onnx.json").is_file():
        raise FileNotFoundError(
            f"Voce Piper '{voice}' mancante in {piper_dir}. Esegui: "
            "uv run python scripts/download_models.py"
        )
    return path


# -40 dBFS on int16: below it a sample is silence to the ear.
AUDIBLE_LEVEL = 328


def leading_silence_s(audio: np.ndarray, rate: int) -> float:
    """Seconds before the first audible sample; all silence: the piece's length."""
    # shortcut: only the first chunk is scanned, upgrade if audible_ms outliers need it
    loud = np.flatnonzero(np.abs(audio.astype(np.int32)) > AUDIBLE_LEVEL)
    return float(loud[0] if loud.size else len(audio)) / rate


class PiperSynth:
    def __init__(self, model_path: Path, settings: TtsSettings) -> None:
        from piper import PiperVoice, SynthesisConfig

        self._voice = PiperVoice.load(model_path, use_cuda=settings.use_cuda)
        self._config: Any = SynthesisConfig(
            length_scale=settings.length_scale,
            noise_scale=settings.noise_scale,
            noise_w_scale=settings.noise_w_scale,
        )
        self.sample_rate = int(self._voice.config.sample_rate)

    def synthesize(self, text: str) -> np.ndarray:
        chunks = [
            c.audio_int16_array for c in self._voice.synthesize(text, syn_config=self._config)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


class StreamingSpeaker:
    """Works with any synth exposing synthesize(text) -> int16 array and sample_rate."""

    def __init__(
        self,
        synth: Any,
        player: AudioPlayer,
        beep_volume: float = 0.25,
        first_clause_chars: int = 0,
        group_chars: int = 0,
        group_first_chars: int = 0,
        group_pause_s: float = 0.0,
    ) -> None:
        self.synth = synth
        self.first_clause_chars = first_clause_chars  # the first piece may end at a comma
        # > 0: the sentences of a reply are said together, up to this many characters per
        # piece (Jack keeps one voice); 0: sentence by sentence, the first one at once.
        self.group_chars = group_chars
        self.group_first_chars = group_first_chars  # the first piece leaves at this length
        self.group_pause_s = group_pause_s  # or once Claude's text stops for this long
        self.player = player
        self._earcons = make_earcons(player.sample_rate, beep_volume)
        self._cancel = threading.Event()  # set by interrupt(): the reply stops at once
        self._ducked = False  # lowered by duck(): logged when it goes back to full volume
        # Wakes speak() when it waits for the next piece: Claude may be running a tool for
        # seconds while the first one has already been said (IT-1 5.1).
        self._wake: Callable[[], None] | None = None
        # The sentences of the current reply that started playing; the last one may have been
        # cut short. Where an interrupted reply resumes.
        self.played: list[str] = []
        # Seconds from "first audio" to the first audible sample (queue, device, the piece's
        # silence): set just before on_first_audio fires.
        self.first_sound_delay_s = 0.0

    def interrupt(self) -> None:
        """Stop the reply being spoken ("basta" heard while speaking)."""
        self._cancel.set()
        if cancel := getattr(self.synth, "cancel", None):  # a worker generating audio stops too
            cancel()
        if (wake := self._wake) is not None:
            wake()

    def duck(self, volume: float | None) -> None:
        """Lower the reply being spoken to `volume` (a voice may be interrupting); None
        brings it back to full volume. The player fades between the two (DUCK_RAMP_S)."""
        full = volume is None or volume >= 1.0
        if full and self._ducked:  # only the change: every reply starts with duck(None)
            log.info("Risposta di nuovo a volume pieno")
        self._ducked = not full
        self.player.gain = 1.0 if volume is None else volume

    def render(self, sentence: str) -> tuple[np.ndarray, int]:
        synth = self.synth
        audio = synth.synthesize(sentence)
        rate = int(synth.sample_rate)
        return audio, rate

    async def render_stream(self, sentence: str) -> AsyncIterator[tuple[np.ndarray, int]]:
        """The audio of a sentence. A synth with stream() (Jack) is played as it is generated,
        so the first words start before the sentence is finished."""
        synth = self.synth
        stream = getattr(synth, "stream", None)
        if stream is None:
            yield await asyncio.to_thread(self.render, sentence)
            return
        chunks = iter(stream(sentence))
        try:
            while (chunk := await asyncio.to_thread(next, chunks, None)) is not None:
                yield chunk, int(synth.sample_rate)
        finally:
            with contextlib.suppress(ValueError):  # still running in its thread: it ends by itself
                chunks.close()  # an abandoned stream stops the worker

    def warm_up(self) -> None:
        """The first sentence of a voice is slow: say one now. Jack's worker warms up by
        itself, so its fallback is warmed instead (it speaks if Jack is not there in time)."""
        base = getattr(self.synth, "fallback", self.synth)
        base.synthesize("Ciao.")

    async def beep(self, kind: str = "wake") -> None:
        await asyncio.to_thread(self.player.write, self._earcons[kind])

    def queued_s(self) -> float:
        """Seconds until what was played has come out of the speakers."""
        return self.player.queued_s()

    async def speak(
        self,
        text_stream: AsyncIterator[str],
        on_first_audio: Callable[[], None] | None = None,
    ) -> None:
        self._reset_for_reply()
        # Rendering runs ahead of playback: the next sentences are ready before they are due
        # (Jack's whole-sentence RVC needs it); "basta" drops whatever is queued. The first
        # piece of each sentence carries its text.
        queue: asyncio.Queue[tuple[np.ndarray, int, str | None] | None] = asyncio.Queue(maxsize=16)
        producer = asyncio.create_task(self._produce(text_stream, queue))
        loop = asyncio.get_running_loop()

        def wake() -> None:  # interrupt() may come from another thread
            loop.call_soon_threadsafe(_end_now, queue)

        self._wake = wake
        try:
            await self._play(queue, on_first_audio)
        finally:
            self._wake = None
            if not producer.done():
                producer.cancel()
            while not producer.done():  # it ends by queueing None: make room for it
                while not queue.empty():
                    queue.get_nowait()
                await asyncio.sleep(0)
            (result,) = await asyncio.gather(producer, return_exceptions=True)
        if isinstance(result, Exception):
            raise result

    def _reset_for_reply(self) -> None:
        self._cancel.clear()
        # Every reply starts at full volume, at once: after a silence the player has nothing to
        # fade (a fade-in would only swallow the first syllable).
        self.duck(None)
        self.played = []
        self.first_sound_delay_s = 0.0
        if begin := getattr(self.synth, "begin_reply", None):  # Jack: one voice per reply
            begin()

    async def _produce(
        self,
        text_stream: AsyncIterator[str],
        queue: asyncio.Queue[tuple[np.ndarray, int, str | None] | None],
    ) -> None:
        try:
            if self.group_chars:
                pieces = group_stream(
                    text_stream,
                    self.group_chars,
                    self.group_first_chars,
                    self.group_pause_s,
                )
            else:
                pieces = split_stream(text_stream, first_clause_chars=self.first_clause_chars)
            async for sentence in pieces:
                log.debug("TTS: %s", sentence)
                text: str | None = sentence
                async for audio, rate in self.render_stream(sentence):
                    await queue.put((audio, rate, text))
                    text = None
        finally:
            await queue.put(None)

    async def _play(
        self,
        queue: asyncio.Queue[tuple[np.ndarray, int, str | None] | None],
        on_first_audio: Callable[[], None] | None,
    ) -> None:
        first = True
        # "basta" while waiting for a piece, or while it was made: nothing more plays,
        # and it is not counted as played (the resume and the echo filter read that).
        while (item := await queue.get()) is not None and not self._cancel.is_set():
            audio, rate, sentence = item
            if first and on_first_audio is not None:
                self.first_sound_delay_s = self.player.starts_in_s() + leading_silence_s(
                    audio, rate
                )
                on_first_audio()
            first = False
            if sentence is not None:
                self.played.append(sentence)
            done = await asyncio.to_thread(self.player.write, audio, rate, self._cancel)
            if done is False or self._cancel.is_set():
                break
        if self._cancel.is_set():
            log.info("Risposta interrotta")


def _end_now(queue: asyncio.Queue[Any]) -> None:
    """speak() waiting for the next piece stops at once ("basta" while Claude runs a tool)."""
    with contextlib.suppress(asyncio.QueueFull):  # full: it is playing and sees the cancel
        queue.put_nowait(None)
