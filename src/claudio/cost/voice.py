"""ElevenLabs credits: usage, monthly limit, reset, and the two notices (80% and 100%)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from claudio.cost.ledger import VoiceLedger

WARN_TEXT = "Attenzione, la voce online ha usato l'80 per cento dei crediti del mese."
OUT_TEXT = "Crediti della voce online finiti: {until} parlo con la voce locale."

MONTHS = (
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
)


def _int_or_none(value: object) -> int | None:
    # bool is an int subclass: reject it so `True` never counts as 1 credit.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class VoiceCredits:
    def __init__(
        self,
        ledger: VoiceLedger,
        warn_at: float = 0.8,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._ledger = ledger
        self._warn_at = warn_at
        self._clock = clock
        self._used: int | None = None
        self._limit = 0
        self._reset = 0  # 0 = unknown
        self._marked_out: int | None = None  # the period (reset) a 402 said was out
        # A cut reply leaves its old stream running while the next one starts: spends can overlap.
        self._lock = threading.Lock()

    @property
    def known(self) -> bool:
        return self._used is not None

    def update(self, subscription: object) -> None:
        """Read the subscription JSON; anything unexpected means unknown usage."""
        data: dict[str, Any] = subscription if isinstance(subscription, dict) else {}
        used = _int_or_none(data.get("character_count"))
        limit = _int_or_none(data.get("character_limit"))
        with self._lock:
            if used is None or limit is None or limit <= 0:  # a zero limit is not "out"
                self._used = None
                return
            self._used, self._limit = used, limit
            self._reset = _int_or_none(data.get("next_character_count_reset_unix")) or 0
            if self._marked_out != self._reset:  # a new period clears a 402's "out"
                self._marked_out = None

    def spend(self, model: str, characters: int, credits: int | None) -> str | None:
        with self._lock:
            self._ledger.record(model, characters, credits, self._now())
            if self._used is None or credits is None:
                return None
            self._used += credits
            return self._claim_notice()

    def mark_out(self) -> str | None:
        """The server refused for lack of credits (HTTP 402): out until the reset."""
        with self._lock:
            self._marked_out = self._reset
            return self._claim_notice()

    def startup_notice(self) -> str | None:
        with self._lock:
            return self._claim_notice() if self.known else None

    def exhausted(self) -> bool:
        return self._level() == "out" and not self.stale()

    def stale(self) -> bool:
        return self._reset > 0 and self._clock() >= self._reset

    def summary(self) -> str:
        if self._used is None:
            return "Crediti ElevenLabs: uso sconosciuto"
        percent = round(100 * self._used / self._limit) if self._limit else 0
        text = f"Crediti ElevenLabs: {self._used} su {self._limit} ({percent}%)"
        if self._reset:
            when = datetime.fromtimestamp(self._reset)
            text += f", reset il {when:%d/%m %H:%M}"
        return text

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock())

    def _level(self) -> str | None:
        if self._marked_out is not None:
            return "out"
        if self._used is None:
            return None
        if self._used >= self._limit:
            return "out"
        return "warn" if self._used >= self._limit * self._warn_at else None

    def _claim_notice(self) -> str | None:
        level = self._level()
        if level is None or self.stale():  # no notice of an expired period
            return None
        now = self._now()
        if level == "out":
            # The 100% text covers the 80% one: claim it too so it never comes later.
            self._ledger.claim_notice(self._reset, "warn", now)
        if not self._ledger.claim_notice(self._reset, level, now):
            return None
        return WARN_TEXT if level == "warn" else OUT_TEXT.format(until=self._until())

    def _until(self) -> str:
        if not self._reset:
            return "fino al rinnovo"
        when = datetime.fromtimestamp(self._reset)
        return f"fino al {when.day} {MONTHS[when.month - 1]}"
