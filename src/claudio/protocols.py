"""Component interfaces. Tests use fakes; later phases swap implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Protocol

import numpy as np


class WakeWordDetector(Protocol):
    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        """Score one int16 frame; returns (wake word, score) when one fires."""
        ...

    def reset(self) -> None: ...


class SpeechProbability(Protocol):
    window_size: int

    def __call__(self, window: np.ndarray) -> float:
        """Speech probability of one float32 window of `window_size` samples."""
        ...

    def reset(self) -> None: ...


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz float32 mono audio; returns "" when nothing was said."""
        ...


class Responder(Protocol):
    def respond(self, text: str) -> AsyncIterator[str]:
        """Stream the reply text. Phase 1: one chunk. Phase 2: Haiku tokens."""
        ...


class Speaker(Protocol):
    async def speak(
        self, text_stream: AsyncIterator[str], on_first_audio: Callable[[], None] | None = None
    ) -> None: ...

    async def beep(self, kind: str = "wake") -> None: ...
