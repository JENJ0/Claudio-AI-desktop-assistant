"""Piper TTS: synthesises the next sentence while the current one plays, each language
with its own voice."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from claudio.audio.earcons import make_earcons
from claudio.audio.player import AudioPlayer
from claudio.config import DEFAULT_LANGUAGE, TtsSettings
from claudio.i18n import tr
from claudio.tts.sentence_splitter import split_stream

log = logging.getLogger(__name__)


def voice_path(piper_dir: Path, voice: str) -> Path:
    path = piper_dir / f"{voice}.onnx"
    if not path.is_file() or not path.with_suffix(".onnx.json").is_file():
        raise FileNotFoundError(
            tr(
                "Voce Piper '{voice}' mancante in {folder}. Esegui: "
                "uv run python scripts/download_models.py",
                voice=voice,
                folder=piper_dir,
            )
        )
    return path


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
    """Works with any synth exposing synthesize(text) -> int16 array and sample_rate.
    `voices` are the other languages' synths (a language without one uses `synth`);
    `effect` (int16 audio, sample rate) -> int16 audio colours every sentence."""

    def __init__(
        self,
        synth: Any,
        player: AudioPlayer,
        beep_volume: float = 0.25,
        effect: Callable[[np.ndarray, int], np.ndarray] | None = None,
        voices: Mapping[str, Any] | None = None,
        first_clause_chars: int = 0,
    ) -> None:
        self.synth = synth
        self.first_clause_chars = first_clause_chars  # the first piece may end at a comma
        self.voices = dict(voices or {})
        self.player = player
        self.effect = effect
        self._earcons = make_earcons(player.sample_rate, beep_volume)
        self._cancel = threading.Event()  # set by interrupt(): the reply stops at once

    def interrupt(self) -> None:
        """Stop the reply being spoken ("basta" heard while speaking)."""
        self._cancel.set()
        for synth in (self.synth, *self.voices.values()):
            if cancel := getattr(synth, "cancel", None):  # a worker generating audio stops too
                cancel()

    def duck(self, volume: float | None) -> None:
        """Lower the reply being spoken to `volume` (a voice may be interrupting); None
        brings it back to full volume."""
        self.player.gain = 1.0 if volume is None else volume

    def voice(self, language: str) -> Any:
        return self.voices.get(language, self.synth)

    def render(self, sentence: str, language: str = DEFAULT_LANGUAGE) -> tuple[np.ndarray, int]:
        synth = self.voice(language)
        audio = synth.synthesize(sentence)
        rate = int(synth.sample_rate)
        return (self.effect(audio, rate) if self.effect else audio), rate

    async def render_stream(
        self, sentence: str, language: str
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """The audio of a sentence. A synth with stream() (Jack) is played as it is generated,
        so the first words start before the sentence is finished; with a voice effect, the
        sentence is rendered whole, as the effect needs it."""
        synth = self.voice(language)
        stream = getattr(synth, "stream", None)
        if stream is None or self.effect:
            yield await asyncio.to_thread(self.render, sentence, language)
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
        audio = base.synthesize("Ciao.")
        if self.effect:
            self.effect(audio, int(base.sample_rate))
        for language in self.voices:
            self.render("Hi.", language)

    async def beep(self, kind: str = "wake") -> None:
        await asyncio.to_thread(self.player.write, self._earcons[kind])

    async def speak(
        self,
        text_stream: AsyncIterator[str],
        on_first_audio: Callable[[], None] | None = None,
        language: str | Callable[[], str] = DEFAULT_LANGUAGE,
    ) -> None:
        """`language` may be a callable, read at every sentence: the responder only knows the
        language of its reply once it has matched the request."""
        self._cancel.clear()
        self.duck(None)  # every reply starts at full volume
        queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue(maxsize=2)

        async def produce() -> None:
            try:
                async for sentence in split_stream(
                    text_stream, first_clause_chars=self.first_clause_chars
                ):
                    lang = language() if callable(language) else language
                    log.debug("TTS (%s): %s", lang, sentence)
                    async for item in self.render_stream(sentence, lang):
                        await queue.put(item)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        first = True
        try:
            while (item := await queue.get()) is not None:
                if first and on_first_audio is not None:
                    on_first_audio()
                first = False
                done = await asyncio.to_thread(self.player.write, *item, self._cancel)
                if done is False or self._cancel.is_set():
                    log.info("Risposta interrotta")
                    break
        finally:
            if not producer.done():
                producer.cancel()
            while not producer.done():  # it ends by queueing None: make room for it
                while not queue.empty():
                    queue.get_nowait()
                await asyncio.sleep(0)
            (result,) = await asyncio.gather(producer, return_exceptions=True)
        if isinstance(result, Exception):
            raise result
