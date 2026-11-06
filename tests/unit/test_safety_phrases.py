"""The phrases JenJo records for the safety audio test (scripts/safety_phrases.tsv)."""

from __future__ import annotations

import sys

import pytest

from claudio.config import PROJECT_ROOT
from claudio.safety.confirmation import LEVEL_EXPLICIT, Answer, parse_answer

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import record_samples

PHRASES = record_samples.load_safety_phrases(PROJECT_ROOT / "scripts" / "safety_phrases.tsv")


@pytest.mark.parametrize(("expected", "phrase"), PHRASES)
def test_every_safety_phrase_parses_as_expected(expected: str, phrase: str) -> None:
    assert parse_answer(phrase) is Answer(expected)


@pytest.mark.parametrize(("expected", "phrase"), PHRASES)
def test_at_level_3_only_an_explicit_confirmation_is_a_yes(expected: str, phrase: str) -> None:
    explicit = expected == "yes" and "confermo" in phrase
    assert (parse_answer(phrase, LEVEL_EXPLICIT) is Answer.YES) is explicit


def test_the_list_has_both_sides_and_no_repeated_take_name() -> None:
    assert {e for e, _ in PHRASES} == {"yes", "no", "unclear"}
    stems = [f"{e}/{record_samples.slug(p)}" for e, p in PHRASES]
    assert len(stems) == len(set(stems))
