"""Component interfaces. Tests use fakes; later phases swap implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from claudio.config import DEFAULT_LANGUAGE


class WakeWordDetector(Protocol):
    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        """Score one int16 frame; returns (wake word, score) when one fires."""
        ...

    def pop_near_miss(self) -> tuple[str, float] | None:
        """(wake word, peak score) of an attempt that stayed under the threshold, once."""
        ...

    def reset(self) -> None: ...


@runtime_checkable
class StrictWake(Protocol):
    """A detector that can demand a higher score: while CLAUDIO is off ("spegniti") a wake
    has to be clearer before the phrase after it is transcribed."""

    strict: bool


class SpeechProbability(Protocol):
    window_size: int

    def __call__(self, window: np.ndarray) -> float:
        """Speech probability of one float32 window of `window_size` samples."""
        ...

    def reset(self) -> None: ...


class Transcriber(Protocol):
    language: str  # of the last transcription

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> str:
        """Transcribe 16 kHz float32 mono audio; returns "" when nothing was said.
        `language`: skip the detection and use that one."""
        ...


class Responder(Protocol):
    language: str  # of the reply being streamed, known by its first chunk
    last_route: dict[str, Any]  # how the last request was handled ("route", "command"...)

    @property
    def awaiting(self) -> bool:
        """A confirmation question is open: the answer comes without the wake word."""
        ...

    @property
    def needs_voice(self) -> bool:
        """An action waits for a spoken yes: a message typed in the chat never gives it."""
        ...

    @property
    def asked_question(self) -> bool:
        """Claude's last reply ended with a question: the next phrase is its answer."""
        ...

    @property
    def answer_expected(self) -> bool:
        """asked_question, and the question is recent: a short phrase is still its answer."""
        ...

    def prepare(self) -> None:
        """A request is coming (wake word, or speech in the conversation window): open
        connections now, while JenJo is still talking."""
        ...

    def understands(self, text: str, language: str = DEFAULT_LANGUAGE, memory: bool = True) -> bool:
        """A local command matches `text` (memory=True: or a memory request)."""
        ...

    def is_stop(self, text: str) -> bool:
        """`text` is only "basta" / "stop": said while CLAUDIO speaks, it interrupts."""
        ...

    def cancel_pending(self, silence: bool = False) -> None:
        """Drop an armed level 2-3 action (nothing was answered)."""
        ...

    def propose(self, text: str, language: str) -> str:
        """A request worked out from a misheard phrase: the question to ask first ("Hai
        detto: ...?"); the next phrase is its answer (awaiting)."""
        ...

    def respond(self, text: str, language: str = DEFAULT_LANGUAGE) -> AsyncIterator[str]:
        """Stream the reply to `text`, heard in `language`. Phase 1: one chunk. Phase 2:
        Haiku tokens."""
        ...


class LevelControl(Protocol):
    gain_db: float

    def observe(self, stats: Mapping[str, float]) -> None:
        """Level stats of a request in which speech was heard; may adjust the input gain."""
        ...


class Speaker(Protocol):
    # The sentences of the last reply that started playing (the last one may be cut short).
    played: list[str]

    async def speak(
        self,
        text_stream: AsyncIterator[str],
        on_first_audio: Callable[[], None] | None = None,
        language: str | Callable[[], str] = DEFAULT_LANGUAGE,
    ) -> None: ...

    async def beep(self, kind: str = "wake") -> None: ...

    def queued_s(self) -> float:
        """Seconds until what speak() and beep() handed over has come out of the speakers
        (they return once the audio is queued)."""
        ...

    def interrupt(self) -> None:
        """Stop the reply being spoken."""
        ...

    def duck(self, volume: float | None) -> None:
        """Lower the reply being spoken to `volume` (0-1); None: full volume again."""
        ...
