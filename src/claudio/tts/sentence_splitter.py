"""Split streamed text into sentences so speech can start after the first one."""

from __future__ import annotations

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
    """Incremental: feed() chunks, get complete sentences; flush() returns the rest."""

    def __init__(self, max_chars: int = 220) -> None:
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence, self._buf = self._buf[:cut].strip(), self._buf[cut:]
            if sentence:
                out.append(sentence)
        return out

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
                continue  # "3.5", "www.sito.it", "10:30"
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


async def split_stream(chunks: AsyncIterator[str], max_chars: int = 220) -> AsyncIterator[str]:
    splitter = SentenceSplitter(max_chars)
    async for chunk in chunks:
        for sentence in splitter.feed(chunk):
            yield sentence
    for sentence in splitter.flush():
        yield sentence
