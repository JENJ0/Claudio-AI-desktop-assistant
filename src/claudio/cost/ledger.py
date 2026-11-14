"""The spending ledger: one SQLite row per Claude call, with the cache tokens of each."""

from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path

from claudio.cost.pricing import CacheTtl, Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                      -- local time, ISO 8601
    day TEXT NOT NULL,                     -- local date: what the daily budget sums
    model TEXT NOT NULL,
    ttl TEXT NOT NULL,                     -- cache TTL the request asked for: 5m | 1h
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_creation_input_tokens INTEGER NOT NULL,
    cache_read_input_tokens INTEGER NOT NULL,
    cache_write_5m_tokens INTEGER NOT NULL,
    cache_write_1h_tokens INTEGER NOT NULL,
    cost REAL NOT NULL                     -- in the currency of config/pricing.toml
)
"""
INDEX = "CREATE INDEX IF NOT EXISTS calls_day ON calls (day)"

VOICE_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                      -- local time, ISO 8601
    day TEXT NOT NULL,                     -- local date
    model TEXT NOT NULL,
    characters INTEGER NOT NULL,           -- characters sent to the voice
    credits INTEGER                        -- the character-cost header; NULL when missing
);
CREATE TABLE IF NOT EXISTS voice_notices (
    period INTEGER NOT NULL,               -- reset unix time of the billing period
    level TEXT NOT NULL,                   -- warn | out
    ts TEXT NOT NULL,
    PRIMARY KEY (period, level)
);
"""


class SpendLedger:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(SCHEMA)
        self._db.execute(INDEX)
        self._db.commit()

    def record(self, model: str, usage: Usage, cost: float, ttl: CacheTtl, when: datetime) -> None:
        self._db.execute(
            "INSERT INTO calls (ts, day, model, ttl, input_tokens, output_tokens,"
            " cache_creation_input_tokens, cache_read_input_tokens,"
            " cache_write_5m_tokens, cache_write_1h_tokens, cost)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                when.isoformat(timespec="seconds"),
                when.date().isoformat(),
                model,
                ttl,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
                usage.cache_write_5m_tokens,
                usage.cache_write_1h_tokens,
                cost,
            ),
        )
        self._db.commit()

    def spent(self, day: date) -> float:
        row = self._db.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM calls WHERE day = ?", (day.isoformat(),)
        ).fetchone()
        return float(row[0])

    def calls(self, day: date) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM calls WHERE day = ?", (day.isoformat(),)
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._db.close()


class VoiceLedger:
    """ElevenLabs credits per request, and the notices already given per billing period.

    The synth writes from different threads, so it has its own connection and a lock.
    """

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self._db.executescript(VOICE_SCHEMA)
            self._db.commit()

    def record(self, model: str, characters: int, credits: int | None, when: datetime) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO voice (ts, day, model, characters, credits) VALUES (?, ?, ?, ?, ?)",
                (
                    when.isoformat(timespec="seconds"),
                    when.date().isoformat(),
                    model,
                    characters,
                    credits,
                ),
            )
            self._db.commit()

    def claim_notice(self, period: int, level: str, when: datetime) -> bool:
        """True only the first time a notice is claimed for this period."""
        with self._lock:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO voice_notices (period, level, ts) VALUES (?, ?, ?)",
                (period, level, when.isoformat(timespec="seconds")),
            )
            self._db.commit()
            return cursor.rowcount == 1

    def close(self) -> None:
        with self._lock:
            self._db.close()
