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


async def group_stream(sentences: AsyncIterator[str], max_chars: int) -> AsyncIterator[str]:
    """Sentences joined into pieces of up to max_chars, each one said in a single breath.

    Jack's voice drifts between sentences generated one by one (each draws its own take
    of Jack from the training lines); a whole reply in one generation keeps one voice
    (JenJo 28/10: tone changing between sentences). A piece waits for its text, so a
    reply starts once it is complete or long enough. A sentence longer than max_chars
    stays whole.
    """
    piece = ""
    async for sentence in sentences:
        if piece and len(piece) + 1 + len(sentence) > max_chars:
            yield piece
            piece = ""
        piece = f"{piece} {sentence}" if piece else sentence
    if piece:
        yield piece
