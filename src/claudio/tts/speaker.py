"""Piper TTS: synthesises the next sentence while the current one plays."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np

from claudio.audio.earcons import make_earcons
from claudio.audio.player import AudioPlayer
from claudio.config import TtsSettings
from claudio.tts.sentence_splitter import split_stream

log = logging.getLogger(__name__)


def voice_path(piper_dir: Path, voice: str) -> Path:
    path = piper_dir / f"{voice}.onnx"
    if not path.is_file() or not path.with_suffix(".onnx.json").is_file():
        raise FileNotFoundError(
            f"Voce Piper '{voice}' mancante in {piper_dir}. "
            "Esegui: uv run python scripts/download_models.py"
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
    """Works with any synth exposing synthesize(text) -> int16 array and sample_rate."""

    def __init__(self, synth: Any, player: AudioPlayer, beep_volume: float = 0.25) -> None:
        self.synth = synth
        self.player = player
        self._earcons = make_earcons(player.sample_rate, beep_volume)

    def warm_up(self) -> None:
        self.synth.synthesize("Ciao.")

    async def beep(self, kind: str = "wake") -> None:
        await asyncio.to_thread(self.player.write, self._earcons[kind])

    async def speak(
        self, text_stream: AsyncIterator[str], on_first_audio: Callable[[], None] | None = None
    ) -> None:
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=2)

        async def produce() -> None:
            try:
                async for sentence in split_stream(text_stream):
                    log.debug("TTS: %s", sentence)
                    audio = await asyncio.to_thread(self.synth.synthesize, sentence)
                    await queue.put(audio)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        first = True
        try:
            while (audio := await queue.get()) is not None:
                if first and on_first_audio is not None:
                    on_first_audio()
                first = False
                await asyncio.to_thread(self.player.write, audio, self.synth.sample_rate)
        finally:
            if not producer.done():
                producer.cancel()
            (result,) = await asyncio.gather(producer, return_exceptions=True)
        if isinstance(result, Exception):
            raise result
