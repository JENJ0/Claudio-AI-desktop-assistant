"""VoiceLedger and VoiceCredits: ElevenLabs credits counted, warned once a month, out at 100%."""

from __future__ import annotations

import threading
from datetime import datetime

import pytest

from claudio.cost.ledger import VoiceLedger
from claudio.cost.voice import WARN_TEXT, VoiceCredits

# Local noon, so the day is the same in any timezone (CI runs in UTC).
RESET = int(datetime(2026, 12, 14, 12, 0).timestamp())
NOW = datetime(2026, 11, 14, 10, 0)
OUT_14_DEC = "Crediti della voce online finiti: fino al 14 dicembre parlo con la voce locale."


def sub(used: int, limit: int = 40000, reset: int | None = RESET) -> dict[str, object]:
    data: dict[str, object] = {"character_count": used, "character_limit": limit}
    if reset is not None:
        data["next_character_count_reset_unix"] = reset
    return data


def credits(
    used: int | None = None, ledger: VoiceLedger | None = None, now: float | None = None
) -> VoiceCredits:
    clock_now = now if now is not None else NOW.timestamp()
    voice = VoiceCredits(ledger or VoiceLedger(":memory:"), clock=lambda: clock_now)
    if used is not None:
        voice.update(sub(used))
    return voice


def test_ledger_records_each_request() -> None:
    ledger = VoiceLedger(":memory:")
    ledger.record("eleven_v4_turbo", 121, 24, NOW)
    ledger.record("eleven_v4_turbo", 30, None, NOW)
    rows = ledger._db.execute("SELECT characters, credits, day FROM voice ORDER BY id").fetchall()
    assert rows == [(121, 24, "2026-11-14"), (30, None, "2026-11-14")]


def test_a_notice_is_claimed_once_per_period() -> None:
    ledger = VoiceLedger(":memory:")
    assert ledger.claim_notice(RESET, "warn", NOW) is True
    assert ledger.claim_notice(RESET, "warn", NOW) is False
    assert ledger.claim_notice(RESET + 1, "warn", NOW) is True


def test_update_reads_the_subscription() -> None:
    voice = credits()
    voice.update(sub(13661))
    assert voice.known
    assert voice.summary() == "Crediti ElevenLabs: 13661 su 40000 (34%), reset il 14/12 12:00"
    voice.update({"character_count": "x"})
    assert not voice.known


def test_the_warning_once_at_80_percent() -> None:
    voice = credits(31990)
    assert voice.spend("m", 9, 9) is None
    assert voice.spend("m", 1, 1) == WARN_TEXT
    assert voice.spend("m", 5, 5) is None


def test_out_of_credits_says_until_the_reset() -> None:
    voice = credits(39990)
    assert voice.spend("m", 20, 20) == OUT_14_DEC
    assert voice.exhausted()
    assert voice.spend("m", 1, 1) is None


def test_the_warning_does_not_repeat_after_a_restart() -> None:
    ledger = VoiceLedger(":memory:")
    assert credits(31990, ledger).spend("m", 10, 10) == WARN_TEXT
    assert credits(32000, ledger).startup_notice() is None


def test_startup_notice_when_already_over() -> None:
    assert credits(36000).startup_notice() == WARN_TEXT
    assert credits(40000).startup_notice() == OUT_14_DEC
    assert credits(1000).startup_notice() is None


def test_the_reset_ends_the_period() -> None:
    voice = credits(40000, now=RESET + 60)
    assert voice.stale()
    assert not voice.exhausted()


def test_unknown_usage_records_but_never_warns() -> None:
    ledger = VoiceLedger(":memory:")
    voice = credits(None, ledger)
    assert voice.spend("m", 24, 24) is None
    assert ledger._db.execute("SELECT COUNT(*) FROM voice").fetchone()[0] == 1
    assert voice.startup_notice() is None
    assert not voice.exhausted()


def test_a_request_without_cost_header_writes_null_and_keeps_the_counter() -> None:
    ledger = VoiceLedger(":memory:")
    voice = credits(31999, ledger)
    assert voice.spend("m", 30, None) is None
    assert ledger._db.execute("SELECT credits FROM voice").fetchone()[0] is None
    assert voice.spend("m", 1, 1) == WARN_TEXT


def test_a_missing_reset_says_until_renewal() -> None:
    voice = credits()
    voice.update(sub(40000, reset=None))
    assert voice.startup_notice() == (
        "Crediti della voce online finiti: fino al rinnovo parlo con la voce locale."
    )
    assert not voice.stale()


def test_one_spend_crossing_both_thresholds_says_only_the_out_text() -> None:
    ledger = VoiceLedger(":memory:")
    voice = credits(31000, ledger)
    assert voice.spend("m", 9000, 9000) == OUT_14_DEC
    assert credits(32000, ledger).startup_notice() is None


@pytest.mark.parametrize(
    "bad", [{}, {"character_count": 1}, {"character_count": True, "character_limit": 5}]
)
def test_incomplete_subscription_is_unknown(bad: dict[str, object]) -> None:
    voice = credits()
    voice.update(bad)
    assert not voice.known


def test_an_expired_period_claims_no_notice() -> None:
    ledger = VoiceLedger(":memory:")
    voice = credits(40000, ledger, now=RESET + 60)
    assert voice.startup_notice() is None
    assert voice.spend("m", 1, 1) is None
    assert ledger.claim_notice(RESET, "out", NOW) is True  # nothing was claimed


@pytest.mark.parametrize("body", [None, [], "x", 5])
def test_update_with_a_non_dict_means_unknown(body: object) -> None:
    voice = credits(1000)
    voice.update(body)
    assert not voice.known


def test_concurrent_spends_lose_no_credits() -> None:
    voice = credits(0)
    voice._limit = 10**9

    def work() -> None:
        for _ in range(2000):
            voice.spend("m", 1, 1)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert voice._used == 8000


def test_a_limit_of_zero_means_unknown_usage() -> None:
    voice = credits()
    voice.update(sub(0, limit=0))
    assert not voice.known and not voice.exhausted()
    assert voice.startup_notice() is None
    assert voice.spend("m", 5, 5) is None
    voice.update(sub(5, limit=-1))
    assert not voice.known and not voice.exhausted()


def test_mark_out_is_exhausted_until_the_reset_and_says_it_once() -> None:
    voice = credits(1000)
    assert not voice.exhausted()
    assert voice.mark_out() == OUT_14_DEC
    assert voice.exhausted()
    assert voice.mark_out() is None  # the notice was claimed
    voice.update(sub(1010))  # the same period: still out
    assert voice.exhausted()


def test_mark_out_claims_the_warning_too() -> None:
    ledger = VoiceLedger(":memory:")
    assert credits(1000, ledger).mark_out() == OUT_14_DEC
    assert credits(36000, ledger).startup_notice() is None


def test_mark_out_works_with_unknown_usage() -> None:
    voice = credits()
    assert voice.mark_out() is not None
    assert voice.exhausted()


def test_a_new_period_clears_mark_out() -> None:
    voice = credits(1000)
    voice.mark_out()
    voice.update(sub(5, reset=RESET + 30 * 86400))
    assert not voice.exhausted()
    assert voice.mark_out() == OUT_14_DEC.replace("14 dicembre", "13 gennaio")
