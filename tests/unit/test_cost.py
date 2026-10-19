from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from claudio.config import PROJECT_ROOT
from claudio.cost.ledger import SpendLedger
from claudio.cost.pricing import Pricing, PricingError, Usage, load_pricing
from claudio.cost.tracker import BudgetState, CostTracker

PRICES = PROJECT_ROOT / "config" / "pricing.toml"
NOON = datetime(2026, 10, 19, 12, 0)


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


class Clock:
    def __init__(self, now: datetime = NOON) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def tracker(pricing: Pricing, budget: float = 1.0, clock: Clock | None = None) -> CostTracker:
    return CostTracker(pricing, SpendLedger(":memory:"), budget, now=clock or Clock())


def test_shipped_prices_cover_the_three_models(pricing: Pricing) -> None:
    assert set(pricing.models) >= {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"}


def test_plain_cost(pricing: Pricing) -> None:
    # 1M input + 1M output on Haiku 5.5 = 0.10 + 0.50
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert pricing.cost("claude-haiku-5-5", usage) == pytest.approx(0.60)


def test_cache_write_multipliers_and_read(pricing: Pricing) -> None:
    usage = Usage(
        cache_write_5m_tokens=1_000_000,
        cache_write_1h_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    # Haiku: input 0.10 -> write 5m x1.25, write 1h x2, read x0.1 (no explicit cache_read)
    assert pricing.cost("claude-haiku-5-5", usage) == pytest.approx(0.125 + 0.20 + 0.01)
    # Sonnet has an explicit cache_read price of 0.20
    assert pricing.cost("claude-sonnet-5-5", Usage(cache_read_input_tokens=1_000_000)) == (
        pytest.approx(0.20)
    )


def test_unknown_model_is_an_error(pricing: Pricing) -> None:
    with pytest.raises(PricingError, match="claude-nope"):
        pricing.cost("claude-nope", Usage())


def test_usage_from_api_object_with_ttl_breakdown() -> None:
    api = SimpleNamespace(
        input_tokens=10,
        output_tokens=20,
        cache_read_input_tokens=300,
        cache_creation_input_tokens=150,
        cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=50),
    )
    usage = Usage.from_api(api)
    assert (usage.cache_write_5m_tokens, usage.cache_write_1h_tokens) == (100, 50)
    assert usage.cache_creation_input_tokens == 150
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens) == (10, 20, 300)


def test_usage_from_api_dict_without_breakdown_uses_the_requested_ttl() -> None:
    api = {"input_tokens": 1, "output_tokens": 2, "cache_creation_input_tokens": 80}
    assert Usage.from_api(api, "1h").cache_write_1h_tokens == 80
    assert Usage.from_api(api).cache_write_5m_tokens == 80


def test_usage_from_api_handles_nulls() -> None:
    api = {"input_tokens": 5, "output_tokens": 6, "cache_read_input_tokens": None}
    usage = Usage.from_api(api)
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def test_negative_tokens_are_refused() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        Usage(input_tokens=-1)


def test_ledger_rows_are_kept_per_day(pricing: Pricing, tmp_path: Path) -> None:
    clock = Clock()
    db = tmp_path / "sub" / "spend.db"
    t = CostTracker(pricing, SpendLedger(db), 1.0, now=clock)
    t.record("claude-haiku-5-5", Usage(input_tokens=1_000_000))
    clock.now += timedelta(days=1)
    t.record("claude-haiku-5-5", Usage(input_tokens=2_000_000), ttl="1h")
    assert t.spent_today() == pytest.approx(0.20)
    assert t.calls_today() == 1
    # the file survives a restart
    again = CostTracker(pricing, SpendLedger(db), 1.0, now=clock)
    assert again.spent_today() == pytest.approx(0.20)
    clock.now -= timedelta(days=1)
    assert again.spent_today() == pytest.approx(0.10)


def test_ledger_stores_cache_tokens_model_and_ttl(pricing: Pricing) -> None:
    t = tracker(pricing)
    t.record(
        "claude-haiku-5-5",
        Usage(cache_write_1h_tokens=7, cache_read_input_tokens=9),
        ttl="1h",
    )
    row = t.ledger._db.execute(
        "SELECT model, ttl, cache_creation_input_tokens, cache_read_input_tokens FROM calls"
    ).fetchone()
    assert row == ("claude-haiku-5-5", "1h", 7, 9)


def test_budget_ok_warn_blocked_and_announce_once(pricing: Pricing) -> None:
    t = tracker(pricing, budget=1.0)
    one_dollar_in_haiku = Usage(output_tokens=2_000_000)  # 2M x $0.50 = $1.00
    assert t.check().state is BudgetState.OK
    t.record("claude-haiku-5-5", Usage(output_tokens=1_000_000))  # $0.50
    assert t.check().state is BudgetState.OK
    t.record("claude-haiku-5-5", Usage(output_tokens=700_000))  # $0.85
    first = t.check()
    assert first.state is BudgetState.WARN and first.announce and first.percent == 85
    assert not t.check().announce  # said once
    t.record("claude-haiku-5-5", Usage(output_tokens=300_000))  # $1.00
    blocked = t.check()
    assert blocked.state is BudgetState.BLOCKED
    assert one_dollar_in_haiku.output_tokens == 2_000_000


def test_warning_at_exactly_80_percent(pricing: Pricing) -> None:
    t = tracker(pricing)
    t.record("claude-haiku-5-5", Usage(output_tokens=1_600_000))  # $0.80
    assert t.check().state is BudgetState.WARN


def test_a_new_day_resets_budget_and_warning(pricing: Pricing) -> None:
    clock = Clock()
    t = tracker(pricing, clock=clock)
    t.record("claude-haiku-5-5", Usage(output_tokens=2_000_000))
    assert t.check().state is BudgetState.BLOCKED
    clock.now += timedelta(days=1)
    assert t.check().state is BudgetState.OK
    t.record("claude-haiku-5-5", Usage(output_tokens=1_700_000))
    assert t.check().announce  # announced again on the new day


@pytest.mark.parametrize(("budget", "warn"), [(0, 0.8), (-1, 0.8), (1, 0), (1, 1)])
def test_bad_budget_settings(pricing: Pricing, budget: float, warn: float) -> None:
    with pytest.raises(ValueError, match="daily_budget"):
        CostTracker(pricing, SpendLedger(":memory:"), budget, warn)


def test_pricing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match=r"non trovato|not found"):
        load_pricing(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[models.x]\ninput = -1\noutput = 1\n", encoding="utf-8")
    with pytest.raises(PricingError, match=r"models\.x\.input"):
        load_pricing(bad)
    broken = tmp_path / "broken.toml"
    broken.write_text("models = [", encoding="utf-8")
    with pytest.raises(PricingError, match="TOML"):
        load_pricing(broken)
