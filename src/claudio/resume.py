"""Resuming a reply cut short by a new request: "raccontami una storia" ... "che ore sono?"
... "Sono le 19 e 44." ... "Dicevo: ..." (JenJo, 25/10)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence

from claudio.router.normalizer import normalize

# The new request ends the old reply instead of pausing it ("ok, va bene, grazie").
CLOSERS = frozenset(
    {"ok", "okay", "va", "bene", "grazie", "basta", "stop", "lascia", "stare", "perfetto"}
    | {"capito", "non", "importa", "niente", "fa", "nulla", "thanks", "thank", "you", "fine"}
    | {"claudio", "ehi", "hey", "e", "si", "no", "allora", "dai"}
)
RESUME_PREFIX = {"it": "Dicevo:", "en": "As I was saying:"}


class DrainedStream:
    """Reads `stream` to its end in a task of its own, also after the speaker stops listening
    to it: a reply cut by a new request is kept whole, so it can be resumed. `read()` is the
    stream as the speaker sees it; errors reach it as they would reach the speaker."""

    def __init__(self, stream: AsyncIterator[str]) -> None:
        self.parts: list[str] = []
        self._new = asyncio.Event()
        self._task = asyncio.create_task(self._pull(stream))

    async def _pull(self, stream: AsyncIterator[str]) -> None:
        try:
            async for chunk in stream:
                self.parts.append(chunk)
                self._new.set()
        finally:
            self._new.set()

    async def read(self) -> AsyncIterator[str]:
        sent = 0
        while True:
            self._new.clear()
            while sent < len(self.parts):
                yield self.parts[sent]
                sent += 1
            if self._task.done():
                if not self._task.cancelled() and (error := self._task.exception()):
                    raise error
                if sent == len(self.parts):
                    return
                continue
            await self._new.wait()

    @property
    def text(self) -> str:
        return "".join(self.parts)

    async def finish(self, timeout_s: float) -> None:
        """Wait for the rest of the reply; after `timeout_s` keep what arrived."""
        if self._task.done():
            return
        with contextlib.suppress(TimeoutError, asyncio.CancelledError, Exception):
            await asyncio.wait_for(asyncio.shield(self._task), timeout_s)
        self.cancel()

    def cancel(self) -> None:
        """Stop reading (the reply was stopped for good, nothing to resume)."""
        if not self._task.done():
            self._task.cancel()


def remainder(text: str, played: Sequence[str]) -> str:
    """What is left of `text` from the last sentence that started playing (said again whole,
    as it was cut), or all of it when nothing played."""
    start = position = 0
    for sentence in played:
        found = text.find(sentence.strip(), position)
        if found < 0:
            continue
        start, position = found, found + len(sentence.strip())
    return text[start:].strip()


def ends_the_reply(request: str) -> bool:
    """True when every word of the request that cut the reply is a closer ("ok va bene
    grazie"): the old reply is not resumed."""
    words = normalize(request, drop_fillers=False).split()
    return all(word in CLOSERS for word in words)
