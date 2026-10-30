"""Long-term memory (Phase 4): what JenJo asked Claudio to remember, kept in SQLite.

One row per memory, in JenJo's own words ("mia sorella si chiama Anna"): a fact, or a standing
rule ("chiamami capo", said as "da ora in poi chiamami capo"). The set is small (things said
aloud, a few hundred at most): it is read once at start and kept in a list, so building the
prompt and searching never touch the disk. A connection is opened per write and a lock keeps
the list and the file in step, so the store works from any thread. A write waits at most
BUSY_S for a database another program holds (it runs on the event loop: the voice must not
freeze), and a forgotten memory is overwritten in the file, not only unlinked.

Never stored: what looks like a password, a PIN, a code, a card or an IBAN. Every memory goes
to Anthropic with the requests, and a voice assistant has no business keeping those."""

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

# A locked database (another program, a second Claudio): give up after this. SQLite's busy
# handler sleeps longer than asked on Windows (about twice), and the voice waits meanwhile.
BUSY_S = 0.25

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'rule')),
    source TEXT NOT NULL CHECK (source IN ('jenjo', 'claude')),
    created REAL NOT NULL                  -- epoch seconds
);
"""

# On normalised text (lowercase, no accents, punctuation as spaces, digit groups joined).
# Always a secret, value or not: "il pin del bancomat", "la chiave del wifi"...
_ALWAYS = re.compile(
    r"\b(pin|pins|puk|cvv|cvc|iban|otp|passwd|passcodes?|passphrases?|seed phrase|tokens?"
    r"|parola d ordine|chiav[ei] api|api keys?|codice fiscale|codic[ei] (segret[oi]|di sblocco"
    r"|di sicurezza|di accesso|pin|dell allarme|dell antifurto|del bancomat)"
    r"|security codes?|access codes?|pin codes?"
    r"|chiav[ei] (del |della |di )?(wifi|wi fi|rete)|(wifi|wi fi|network) (keys?|passwords?)"
    r"|passwords? (del |della |di )?(wifi|wi fi|rete)"
    r"|chiav[ei] (privat[ae]|segret[ae]|di accesso|d accesso|di sicurezza)|private keys?"
    r"|secret keys?|credenziali|credentials|passaporto|passport|carta d identita"
    r"|frase di recupero|parole di recupero|recovery phrase|seed words)\b"
)
# A password, then within six words its value: digits, a word mixing letters and digits, or
# "è" / "is" (the value follows). "Uso Bitwarden per le password" says no value: kept.
_PASSWORD = re.compile(r"\b(passwords?|passcodes?)\b(?P<after>(?: \w+){1,6})")
_PASSWORD_VALUE = re.compile(
    r"\b(\d{3,}|(?=[a-z]*\d)(?=\d*[a-z])[a-z\d]{4,}|e|sono|is|are|era|was)\b"
)
# A code, a combination, a key or a card, then within six words a value in digits (or
# letters mixed with digits). "Il codice di Claudio è su GitHub", "la mia carta di credito è
# una Visa" have none: kept.
_SECRET = re.compile(
    r"\b(codic[ei]|codes?|combinazion[ei]|chiav[ei]|carta di credito|carte di credito"
    r"|numero della carta|card numbers?|credit cards?)\b(?P<after>(?: \w+){1,6})"
)
_VALUE = re.compile(r"\b(\d{3,}|(?=[a-z]*\d)(?=\d*[a-z])[a-z\d]{4,})\b")
# On the text as said: "la password di Netflix: gattonero" (the colon is the "è").
_COLON = re.compile(
    r"\b(password|passcode|pin|codice|codici|code|chiave|key|combinazione)\w*\b[^.:\n]{0,30}:\s*\S",
    re.IGNORECASE,
)
_NOT_SECRET = re.compile(
    r"\b(codice postale|codice di avviamento postale|postal code|zip code|area code"
    r"|dress code|codice della strada|codice civile|codice penale)\b"
)
# 13+ digits: a card (13-19) or an IBAN; not a phone with its 00 prefix (at most 17).
_LONG_NUMBER = re.compile(r"(?<!\d)(?!00\d{11,15}(?!\d))\d{13,}")
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
    """A password, a PIN, a code, a card, an IBAN...: never kept, never sent."""
    norm = _DIGIT_GAP.sub("", normalize(text, drop_fillers=False))
    if _ALWAYS.search(norm) or _LONG_NUMBER.search(norm) or _COLON.search(text):
        return True
    if any(_PASSWORD_VALUE.search(m.group("after")) for m in _PASSWORD.finditer(norm)):
        return True
    plain = _NOT_SECRET.sub(" ", norm)
    return any(_VALUE.search(m.group("after")) for m in _SECRET.finditer(plain))


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
            rows = db.execute(
                "SELECT id, text, kind, source, created FROM memories ORDER BY id"
            ).fetchall()
        self._items = [Memory(int(r[0]), str(r[1]), r[2], r[3], float(r[4])) for r in rows]
        self._added: list[Memory] = []  # this run's memories, newest last ("dimenticalo")
        self.version = 0  # bumped by every change: the prompt is rebuilt only then
        self.removals = 0  # bumped by every forget: Claude's conversation starts over then

    def refresh(self) -> None:
        """The file was deleted while Claudio runs (JenJo emptying the memory by hand): the
        memories are gone here too, so they are no longer sent to Claude and a new one never
        takes the number of an old one. A stat per call: cheap enough for every request."""
        if self._items and not self.path.exists():
            log.warning("Il file della memoria non c'è più: riparto senza ricordi")
            self._items, self._added = [], []
            self.version += 1
            self.removals += 1

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """The schema each time: a file deleted while Claudio runs comes back empty and
        working, instead of failing every save until a restart."""
        if hasattr(self, "_items"):
            self.refresh()
        db = sqlite3.connect(self.path, timeout=BUSY_S)
        try:
            db.execute("PRAGMA secure_delete = ON")  # a forgotten text is zeroed in the file
            db.executescript(SCHEMA)
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
        known. Raises MemoryRefused when it is empty, too long, sensitive or the store is full,
        sqlite3.Error when the database cannot be written."""
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
            self.refresh()
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
            self.removals += 1
        log.info("Ricordi dimenticati: %s", ", ".join(str(m.id) for m in gone))
        return gone

    def clear(self) -> int:
        """Forget everything; how many there were."""
        with self._lock:
            count = len(self._items)
            with self._connect() as db:
                db.execute("DELETE FROM memories")
            self._items, self._added = [], []
            self.version += 1
            self.removals += 1
        log.info("Memoria svuotata: %d ricordi dimenticati", count)
        return count

    def last_added(self) -> Memory | None:
        """The newest memory of this run that is still there."""
        return self._added[-1] if self._added else None
