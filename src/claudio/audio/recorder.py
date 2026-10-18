"""Keeps a WAV of each request (1 s before the wake word, then everything heard after it),
to tune VAD and STT on real audio."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


class UtteranceRecorder:
    def __init__(self, directory: Path, keep: int = 50, sample_rate: int = 16000) -> None:
        self.directory = directory
        self.keep = keep
        self.sample_rate = sample_rate
        directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, audio: np.ndarray, outcome: str) -> str | None:
        import soundfile as sf

        if audio.size == 0:
            return None
        name = f"{datetime.now():%Y%m%d-%H%M%S-%f}"[:-3] + f"_{outcome}.wav"
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        sf.write(self.directory / name, pcm, self.sample_rate)
        for old in sorted(self.directory.glob("*.wav"))[: -self.keep]:
            old.unlink(missing_ok=True)
        return name
