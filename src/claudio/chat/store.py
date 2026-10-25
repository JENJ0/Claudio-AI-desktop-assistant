"""SQLite store of the conversations (data/claudio.db). It is written by the assistant, one
request at a time, and read by the chat window, a different process: WAL mode lets both work
at once. A connection is opened per operation: the calls are rare and it keeps threads simple.

The `inbox` table goes the other way: the window writes what JenJo types, the assistant takes
it (no port, no socket between the two processes: the database is the only link)."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'claudio')),
    text TEXT NOT NULL,
    route TEXT,
    model TEXT,
    cost REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    m2_ms REAL
);
CREATE INDEX IF NOT EXISTS messages_ts ON messages (ts);
CREATE INDEX IF NOT EXISTS messages_session ON messages (session, id);
CREATE TABLE IF NOT EXISTS inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    text TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'taken', 'expired'))
);
CREATE INDEX IF NOT EXISTS inbox_state ON inbox (state, id);
"""

MESSAGE_COLUMNS = (
    "id, ts, session, role, text, route, model, cost, input_tokens, output_tokens, m2_ms"
)
MAX_LIMIT = 500
MAX_TYPED_CHARS = 1000  # a written request: more is a paste, not a question for a voice


def _like(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class ChatStore:
    def __init__(self, path: Path, session: str = "") -> None:
        self.path = path
        self.session = session
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:  # commit or roll back
                yield db
        finally:
            db.close()

    def add(self, row: dict[str, Any]) -> None:
        """A finished request (a metrics row): what was heard and what Claudio answered.
        Near misses and timeouts have neither and are skipped. Never raises: the chat must not
        break a conversation."""
        text, reply = (row.get("text") or "").strip(), (row.get("reply") or "").strip()
        if reply and row.get("shown"):  # the exact command waiting for "confermo" (v0.3.2)
            reply = f"{reply}\n{row['shown']}"
        if not text and not reply:
            return
        ts = float(row.get("ts") or time.time())
        try:
            with self._connect() as db:
                if text:
                    db.execute(
                        "INSERT INTO messages (ts, session, role, text) VALUES (?, ?, 'user', ?)",
                        (ts, self.session, text),
                    )
                if reply:
                    db.execute(
                        "INSERT INTO messages (ts, session, role, text, route, model, cost,"
                        " input_tokens, output_tokens, m2_ms)"
                        " VALUES (?, ?, 'claudio', ?, ?, ?, ?, ?, ?, ?)",
                        (
                            ts + 0.001,  # after the question, also when sorted by time
                            self.session,
                            reply,
                            row.get("route"),
                            row.get("model"),
                            row.get("cost"),
                            row.get("input_tokens"),
                            row.get("output_tokens"),
                            row.get("m2_ms"),
                        ),
                    )
        except sqlite3.Error:
            log.exception("Chat non salvata")

    def messages(
        self,
        session: str | None = None,
        query: str = "",
        after_id: int = 0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Oldest first. With a `query` the text is searched in every session; `after_id`
        returns only what is newer (the window polls with it)."""
        where: list[str] = ["id > ?"]
        args: list[Any] = [after_id]
        if session:
            where.append("session = ?")
            args.append(session)
        if query.strip():
            where.append("text LIKE ? ESCAPE '\\'")
            args.append(_like(query.strip()))
        limit = max(1, min(limit, MAX_LIMIT))
        # The newest `limit`, shown oldest first.
        sql = (
            f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE {' AND '.join(where)}"
            " ORDER BY id DESC LIMIT ?"
        )
        with self._connect() as db:
            rows = db.execute(sql, (*args, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Newest first: when it started, how many messages, what it cost, how it begins."""
        sql = """
            SELECT m.session AS session, MIN(m.ts) AS started, MAX(m.ts) AS last,
                   COUNT(*) AS messages, COALESCE(SUM(m.cost), 0) AS cost,
                   (SELECT text FROM messages u WHERE u.session = m.session AND u.role = 'user'
                    ORDER BY u.id LIMIT 1) AS preview
            FROM messages m GROUP BY m.session ORDER BY MAX(m.id) DESC LIMIT ?
        """
        with self._connect() as db:
            return [dict(r) for r in db.execute(sql, (max(1, min(limit, MAX_LIMIT)),))]

    def cost_since(self, since_ts: float) -> float:
        with self._connect() as db:
            (total,) = db.execute(
                "SELECT COALESCE(SUM(cost), 0) FROM messages WHERE ts >= ?", (since_ts,)
            ).fetchone()
        return float(total)

    def purge_older_than(self, days: int, now: float | None = None) -> int:
        """Privacy: everything the microphone picked up goes after `days` days."""
        cutoff = (now or time.time()) - days * 86400
        with self._connect() as db:
            db.execute("DELETE FROM inbox WHERE ts < ?", (cutoff,))
            return db.execute("DELETE FROM messages WHERE ts < ?", (cutoff,)).rowcount

    # --- the inbox: written in the window, answered by the assistant ---------------------

    def enqueue(self, text: str, now: float | None = None) -> int:
        """A message typed in the window; its id lets the window follow it."""
        text = " ".join(text.split())[:MAX_TYPED_CHARS]
        if not text:
            raise ValueError("empty message")
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO inbox (ts, text) VALUES (?, ?)", (now or time.time(), text)
            )
            return int(cur.lastrowid or 0)

    def take_pending(self, max_age_s: float, now: float | None = None) -> list[tuple[int, str]]:
        """The messages waiting, oldest first, marked as taken. Older than `max_age_s` (typed
        while the assistant was off) they expire: an answer minutes later would surprise."""
        cutoff = (now or time.time()) - max_age_s
        with self._connect() as db:
            db.execute(
                "UPDATE inbox SET state = 'expired' WHERE state = 'pending' AND ts < ?", (cutoff,)
            )
            rows = db.execute(
                "SELECT id, text FROM inbox WHERE state = 'pending' ORDER BY id"
            ).fetchall()
            if rows:
                db.executemany(
                    "UPDATE inbox SET state = 'taken' WHERE id = ?", [(r["id"],) for r in rows]
                )
        return [(int(r["id"]), str(r["text"])) for r in rows]

    def inbox_state(self, message_id: int) -> str | None:
        """pending (the assistant has not seen it: is it running?), taken, expired."""
        with self._connect() as db:
            row = db.execute("SELECT state FROM inbox WHERE id = ?", (message_id,)).fetchone()
        return str(row["state"]) if row else None
