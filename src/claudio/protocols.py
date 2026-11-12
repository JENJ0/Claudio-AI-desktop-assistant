"""Component interfaces. Tests use fakes; later phases swap implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


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
    # True while no transcribed word may be logged (CLAUDIO is off); the Assistant sets it.
    hides_text: Callable[[], bool] | None

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz float32 mono audio; returns "" when nothing was said."""
        ...


class Responder(Protocol):
    last_route: dict[str, Any]  # how the last request was handled ("route", "command"...)
    claude_replied: bool  # the reply being streamed (or the last one) came from Claude
    can_resume: Callable[[], bool] | None  # set by the Assistant: "continua" has a rest to say

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

    def understands(self, text: str, memory: bool = True) -> bool:
        """A local command matches `text` (memory=True: or a memory request)."""
        ...

    def is_stop(self, text: str) -> bool:
        """`text` is only "basta" / "stop", or ends a phrase to Claudio with it: said while
        CLAUDIO speaks, it interrupts."""
        ...

    def cancel_pending(
        self, silence: bool = False, cut: bool = False, takeover: bool = False
    ) -> None:
        """Drop an armed level 2-3 action (nothing was answered); `cut`: the reply was stopped.
        `takeover`: a phrase was said over the reply and is the answer to Claude's question."""
        ...

    def stop_acting(self) -> None:
        """A new request cut the reply: the rest of it may still be read, it starts nothing."""
        ...

    def reply_cut(self, played: Sequence[str], handle: object | None = None) -> object | None:
        """The reply was cut after `played`, the pieces that started playing: the conversation
        with Claude keeps that much of it. Returns a handle of the cut reply (None: not
        Claude's), which a later call passes back to name that reply and no other."""
        ...

    def action_dropped(self, handle: object | None) -> bool:
        """The cut reply `handle` names had an action that was stopped and never ran."""
        ...

    def propose(self, text: str) -> str:
        """A request worked out from a misheard phrase: the question to ask first ("Hai
        detto: ...?"); the next phrase is its answer (awaiting)."""
        ...

    def respond(self, text: str, named: bool = False) -> AsyncIterator[str]:
        """Stream the reply to `text`. Phase 1: one chunk. Phase 2:
        Haiku tokens. `named`: the wake audio held the name, though `text` does not."""
        ...


class LevelControl(Protocol):
    gain_db: float

    def observe(self, stats: Mapping[str, float]) -> None:
        """Level stats of a request in which speech was heard; may adjust the input gain."""
        ...


class _Marker(str):
    """A chunk of a reply stream that is not text: equal to "", so whatever joins the chunks
    (chat, history, metrics) ignores it; the speaker finds it by identity."""

    def __repr__(self) -> str:
        return "CUT_SHORT"


# In a reply stream: the reply was cut at the token limit (30/10: a story stopped inside
# "Attraverso" and the voice said "Attra"). The speaker drops the unfinished sentence before
# it; what follows (a closing sentence) is said. Test it with `chunk is CUT_SHORT`.
CUT_SHORT = _Marker()


class Speaker(Protocol):
    # The sentences of the last reply that started playing (the last one may be cut short).
    played: list[str]
    # Seconds from the first audio being handed over to the first audible sample.
    first_sound_delay_s: float

    async def speak(
        self,
        text_stream: AsyncIterator[str],
        on_first_audio: Callable[[], None] | None = None,
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
