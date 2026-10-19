"""Spending counter with a daily budget: warns at 80 %, blocks at 100 %.

The Claude responder (Phase 2) calls `check()` before every request and `record()` after it.
Local commands never touch it: they cost nothing and keep working when the budget is spent."""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from claudio.cost.ledger import SpendLedger
from claudio.cost.pricing import CacheTtl, Pricing, Usage

log = logging.getLogger(__name__)


class BudgetState(enum.StrEnum):
    OK = "ok"
    WARN = "warn"  # 80 % or more spent: said aloud once a day
    BLOCKED = "blocked"  # 100 % spent: no more Claude calls today


@dataclass(frozen=True)
class BudgetStatus:
    state: BudgetState
    spent: float
    limit: float
    announce: bool = False  # WARN, and not yet said today

    @property
    def percent(self) -> int:
        return round(self.spent / self.limit * 100) if self.limit > 0 else 0


class CostTracker:
    def __init__(
        self,
        pricing: Pricing,
        ledger: SpendLedger,
        daily_budget: float,
        warn_at: float = 0.8,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        if daily_budget <= 0 or not 0 < warn_at < 1:
            raise ValueError("daily_budget must be > 0 and warn_at between 0 and 1")
        self.pricing = pricing
        self.ledger = ledger
        self.daily_budget = daily_budget
        self.warn_at = warn_at
        self._now = now
        self._warned_on: date | None = None  # the day the 80 % warning was said

    def check(self) -> BudgetStatus:
        """Where today's spending stands. `announce` is true once per day, on the first check
        at or over the warning level (and on the first blocked check of a day)."""
        today = self._now().date()
        spent = round(self.ledger.spent(today), 6)  # 0.8 must not read as 0.7999999999999999
        if spent >= self.daily_budget:
            state = BudgetState.BLOCKED
        elif spent >= self.daily_budget * self.warn_at:
            state = BudgetState.WARN
        else:
            state = BudgetState.OK
        announce = state is not BudgetState.OK and self._warned_on != today
        if announce:
            self._warned_on = today
            log.warning("Budget giornaliero: %d%% speso", round(spent / self.daily_budget * 100))
        return BudgetStatus(state, spent, self.daily_budget, announce)

    def record(self, model: str, usage: Usage, ttl: CacheTtl = "5m") -> float:
        """Add one call to today's spending; returns its cost."""
        cost = self.pricing.cost(model, usage)
        self.ledger.record(model, usage, cost, ttl, self._now())
        log.info(
            "Costo chiamata %s: %.5f %s",
            model,
            cost,
            self.pricing.currency,
            extra={"model": model, "cost": cost},
        )
        return cost

    def spent_today(self) -> float:
        return self.ledger.spent(self._now().date())

    def calls_today(self) -> int:
        return self.ledger.calls(self._now().date())


def open_tracker(
    pricing: Pricing, db_path: Path, daily_budget: float, warn_at: float
) -> CostTracker:
    return CostTracker(pricing, SpendLedger(db_path), daily_budget, warn_at)
