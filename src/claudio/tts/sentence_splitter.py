"""Split streamed text into sentences so speech can start after the first one."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator

ABBREVIATIONS = {
    "ecc",
    "dott",
    "dr",
    "sig",
    "sigg",
    "sigra",
    "prof",
    "avv",
    "ing",
    "arch",
    "geom",
    "es",
    "pag",
    "pagg",
    "n",
    "nr",
    "art",
    "cap",
    "tel",
    "ca",
    "vs",
    "etc",
    "rag",
    "mons",
    "s",
    "sez",
    "fig",
    "vol",
    "cfr",
    "p",
    "pp",
}
_END_CHARS = ".!?…;\n"
_ABBREV_TAIL = re.compile(r"([A-Za-zÀ-ÿ]+)\.$")


class SentenceSplitter:
    """Incremental: feed() chunks, get complete sentences; flush() returns the rest.

    `first_clause_chars` > 0: the FIRST piece may end at a comma once it is at least that
    long ("Se in Italia sono le 22 e 51," is spoken while Claude writes the rest), so the
    voice starts sooner. 0: whole sentences only."""

    def __init__(self, max_chars: int = 220, first_clause_chars: int = 0) -> None:
        self.max_chars = max_chars
        self.first_clause_chars = first_clause_chars
        self._buf = ""
        self._first = True

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out = []
        while True:
            cut = self._find_cut()
            if self._first and self.first_clause_chars:
                clause = self._clause_cut()
                if clause is not None and (cut is None or clause < cut):
                    cut = clause
            if cut is None:
                break
            sentence, self._buf = self._buf[:cut].strip(), self._buf[cut:]
            if sentence:
                out.append(sentence)
                self._first = False
        return out

    def _clause_cut(self) -> int | None:
        """After the first comma followed by a space (not "3,5") past first_clause_chars."""
        start = self.first_clause_chars
        while (i := self._buf.find(",", start)) != -1 and i + 1 < len(self._buf):
            if self._buf[i + 1].isspace():
                return i + 1
            start = i + 1
        return None

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []

    def _find_cut(self) -> int | None:
        buf = self._buf
        for i, ch in enumerate(buf):
            if ch not in _END_CHARS:
                continue
            if ch == "\n":
                return i + 1
            # Absorb runs like "?!" or "..." and closing quotes/brackets.
            j = i + 1
            while j < len(buf) and buf[j] in _END_CHARS.replace("\n", "") + "\"'»)":
                j += 1
            if j >= len(buf):
                return None  # wait: the next chunk may continue the number/abbreviation
            if not buf[j].isspace():
                if ch != "\n" and buf[j].isupper() and i > 0 and buf[i - 1].islower():
                    return j  # "Cerco il nome.Nessuna cartella": two turns glued together
                continue  # "3.5", "www.sito.it", "10:30", "U.S.A."
            if ch == "." and j == i + 1 and self._is_abbreviation(buf[: i + 1]):
                continue
            return j
        if len(buf) > self.max_chars:  # very long sentence: cut at the last comma
            comma = buf.rfind(",", 0, self.max_chars)
            if comma > 0:
                return comma + 1
        return None

    @staticmethod
    def _is_abbreviation(text: str) -> bool:
        m = _ABBREV_TAIL.search(text)
        if not m:
            return False
        word = m.group(1)
        if len(word) == 1 and word.isupper():
            return True  # initial: "G. Verdi"
        return word.lower() in ABBREVIATIONS


async def split_stream(
    chunks: AsyncIterator[str], max_chars: int = 220, first_clause_chars: int = 0
) -> AsyncIterator[str]:
    splitter = SentenceSplitter(max_chars, first_clause_chars)
    async for chunk in chunks:
        for sentence in splitter.feed(chunk):
            yield sentence
    for sentence in splitter.flush():
        yield sentence


_PAUSED = ""  # the text stopped: split_stream never yields an empty sentence


async def group_stream(
    text: AsyncIterator[str], max_chars: int, first_chars: int = 0, pause_s: float = 0.0
) -> AsyncIterator[str]:
    """The sentences of streamed `text` joined into pieces of up to max_chars, each one
    said in a single breath.

    Jack's voice drifts between sentences generated one by one (each draws its own take
    of Jack from the training lines); a reply in few generations keeps one voice (JenJo
    28/10: tone changing between sentences). But waiting for max_chars before the first
    piece delayed every reply by up to 2 s (JenJo 29/10: M1 up to 3.4 s), so a piece takes
    only what has ARRIVED by the time it is asked for: the first one leaves as soon as it
    has `first_chars` (or the reply is over), and the rest of the reply, which Claude
    writes while the first piece is generated and played, goes out together in the next
    one. A sentence longer than max_chars stays whole.

    `pause_s` > 0: the first piece also leaves, shorter, once no new text has come for
    that long: Claude is running a tool (30/10 14:07, "Ci guardo, socio." waited 2.1 s for
    three calls). It is timed on the text, not on whole sentences, so a long sentence still
    being written is not a pause; whitespace alone does not count as new text.
    """
    loop = asyncio.get_running_loop()
    moved = loop.time()  # when the last text arrived
    ready: asyncio.Queue[str | None] = asyncio.Queue()

    async def noted() -> AsyncIterator[str]:
        nonlocal moved
        async for chunk in text:
            if chunk.strip():
                moved = loop.time()
            yield chunk

    async def read() -> None:
        try:
            async for sentence in split_stream(noted()):
                await ready.put(sentence)
        finally:
            await ready.put(None)

    async def more() -> str | None:
        """The next sentence for the first piece; _PAUSED once the text stopped coming."""
        while pause_s > 0:
            left = pause_s - (loop.time() - moved)
            if left <= 0:
                return _PAUSED
            with contextlib.suppress(TimeoutError):
                return await asyncio.wait_for(ready.get(), left)
        return await ready.get()

    reader = asyncio.create_task(read())
    held: str | None = None  # a sentence that did not fit the previous piece
    first = True
    try:
        while True:
            piece = held or await ready.get()
            held = None
            if piece is None:
                break
            ended = False
            # Wait for more text only while the first piece is too short to start with;
            # afterwards take what is there and never wait.
            while True:
                if first and len(piece) < first_chars:
                    nxt = await more()
                    if nxt == _PAUSED:
                        break  # say what is there while Claude is busy
                elif ready.empty():
                    break
                else:
                    nxt = ready.get_nowait()
                if nxt is None:
                    ended = True
                    break
                if len(piece) + 1 + len(nxt) > max_chars:
                    held = nxt
                    break
                piece = f"{piece} {nxt}"
            first = False
            yield piece
            if ended:
                break
        await reader  # a failure of the text stream reaches the caller
    finally:
        if not reader.done():
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
