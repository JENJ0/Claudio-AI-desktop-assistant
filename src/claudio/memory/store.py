"""Long-term memory (Phase 4): what JenJo asked Claudio to remember, kept in SQLite.

One row per memory, in JenJo's own words ("mia sorella si chiama Anna"): a fact, or a standing
rule ("chiamami capo", said as "da ora in poi chiamami capo"). The set is small (things said
aloud, a few hundred at most): it is read once at start and kept in a list, so building the
prompt and searching never touch the disk. A connection is opened per write and a lock keeps
the list and the file in step, so the store works from any thread (the PC tools run on a
worker thread).

Never stored: what looks like a password, a PIN, a card or an IBAN. Every memory goes to
Anthropic with the requests, and a voice assistant has no business keeping those."""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from claudio.router.normalizer import normalize

log = logging.getLogger(__name__)

Kind = Literal["fact", "rule"]
Source = Literal["jenjo", "claude"]  # who decided: JenJo (by voice or in the chat) or Claude
Refusal = Literal["empty", "too_long", "sensitive", "full"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'rule')),
    source TEXT NOT NULL CHECK (source IN ('jenjo', 'claude')),
    created REAL NOT NULL                  -- epoch seconds
);
"""

# On normalised text (lowercase, no accents or punctuation). A word is enough: "il pin del
# bancomat è..." is refused even without the digits.
_SENSITIVE = re.compile(
    r"\b(password|passwd|pass word|passphrase|pin|puk|cvv|cvc|iban|otp|seed|token"
    r"|parola d ordine|chiave api|api key|carta di credito|numero della carta"
    r"|codice (segreto|di sicurezza|fiscale|della carta|di accesso|dell allarme|pin))\b"
)
_LONG_NUMBER = re.compile(r"\d{13,}")  # a card (13-19 digits) or an IBAN; a phone is shorter
_DIGIT_GAP = re.compile(r"(?<=\d) (?=\d)")  # "4111 1111 1111 1111": Whisper spaces the groups
_TRIM = " \t\r\n.,;:!?\"'«»“”"


@dataclass(frozen=True)
class Memory:
    id: int
    text: str
    kind: Kind
    source: Source
    created: float

    @property
    def day(self) -> str:
        """The local date it was said, e.g. 2026-10-30: a newer memory beats an older one."""
        return datetime.fromtimestamp(self.created).date().isoformat()


class MemoryRefused(ValueError):
    """Not stored; `reason` says why (the caller says it aloud)."""

    def __init__(self, reason: Refusal) -> None:
        super().__init__(reason)
        self.reason: Refusal = reason


def sensitive(text: str) -> bool:
    """A password, a PIN, a card, an IBAN...: never kept, never sent."""
    norm = _DIGIT_GAP.sub("", normalize(text, drop_fillers=False))
    return bool(_SENSITIVE.search(norm) or _LONG_NUMBER.search(norm))


def clean(text: str) -> str:
    """One line, without the quotes and the full stop Whisper puts around a phrase."""
    return " ".join(text.split()).strip(_TRIM)


class MemoryStore:
    def __init__(
        self,
        path: Path,
        max_items: int = 500,
        max_chars: int = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.max_items = max_items
        self.max_chars = max_chars
        self.clock = clock
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            rows = db.execute(
                "SELECT id, text, kind, source, created FROM memories ORDER BY id"
            ).fetchall()
        self._items = [Memory(int(r[0]), str(r[1]), r[2], r[3], float(r[4])) for r in rows]
        self._added: list[Memory] = []  # this run's memories, newest last ("dimenticalo")
        self.version = 0  # bumped by every change: the prompt is rebuilt only then

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:  # commit or roll back
                yield db
        finally:
            db.close()

    def items(self) -> tuple[Memory, ...]:
        """Oldest first."""
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, text: str, kind: Kind = "fact", source: Source = "jenjo") -> tuple[Memory, bool]:
        """Keep `text`; (the memory, True) when new, (the one already there, False) when it was
        known. Raises MemoryRefused when it is empty, too long, sensitive or the store is full."""
        text = clean(text)
        if not text:
            raise MemoryRefused("empty")
        if len(text) > self.max_chars:
            raise MemoryRefused("too_long")
        if sensitive(text):
            log.warning("Ricordo rifiutato: sembra una password o un codice")
            raise MemoryRefused("sensitive")
        key = normalize(text, drop_fillers=False)
        with self._lock:
            for known in self._items:
                if known.kind == kind and normalize(known.text, drop_fillers=False) == key:
                    return known, False
            if len(self._items) >= self.max_items:
                raise MemoryRefused("full")
            created = self.clock()
            with self._connect() as db:
                cur = db.execute(
                    "INSERT INTO memories (text, kind, source, created) VALUES (?, ?, ?, ?)",
                    (text, kind, source, created),
                )
                memory = Memory(int(cur.lastrowid or 0), text, kind, source, created)
            self._items.append(memory)
            self._added.append(memory)
            self.version += 1
        log.info("Ricordo salvato (%s, %s): %s", kind, source, text)
        return memory, True

    def remove(self, ids: Sequence[int]) -> list[Memory]:
        """Forget these memories; the ones that were there, in their order."""
        wanted = set(ids)
        with self._lock:
            gone = [m for m in self._items if m.id in wanted]
            if not gone:
                return []
            with self._connect() as db:
                db.executemany("DELETE FROM memories WHERE id = ?", [(m.id,) for m in gone])
            self._items = [m for m in self._items if m.id not in wanted]
            self._added = [m for m in self._added if m.id not in wanted]
            self.version += 1
        for memory in gone:
            log.info("Ricordo dimenticato: %s", memory.text)
        return gone

    def clear(self) -> int:
        """Forget everything; how many there were."""
        with self._lock:
            count = len(self._items)
            with self._connect() as db:
                db.execute("DELETE FROM memories")
            self._items, self._added = [], []
            self.version += 1
        log.info("Memoria svuotata: %d ricordi dimenticati", count)
        return count

    def last_added(self) -> Memory | None:
        """The newest memory of this run that is still there ("dimenticalo" forgets it)."""
        return self._added[-1] if self._added else None
