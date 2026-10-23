"""CLAUDIO.md: the personality is read once, copied from the default, cut, and sent frozen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from claudio.brain.claude import ClaudeBrain
from claudio.brain.personality import DEFAULT_FILE, load_personality
from claudio.brain.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TOOLS, system_prompt
from claudio.config import ClaudeSettings
from claudio.cost.pricing import Pricing, load_pricing
from tests.unit.test_claude_brain import FakeClient, Monotonic
from tests.unit.test_cost import PRICES, tracker


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


def test_the_first_start_copies_the_default(tmp_path: Path) -> None:
    default = tmp_path / "default.md"
    default.write_text("<!-- note for JenJo -->\n# Chi sei\nSei Jack.\n", encoding="utf-8")
    mine = tmp_path / "data" / "CLAUDIO.md"
    assert load_personality(mine, 3000, default) == "# Chi sei\nSei Jack."  # notes never sent
    assert mine.read_text(encoding="utf-8").startswith("<!-- note")


def test_jenjos_copy_wins_and_is_never_overwritten(tmp_path: Path) -> None:
    default = tmp_path / "default.md"
    default.write_text("Sei Jack.", encoding="utf-8")
    mine = tmp_path / "CLAUDIO.md"
    mine.write_text("Sei un pirata.", encoding="utf-8")
    assert load_personality(mine, 3000, default) == "Sei un pirata."
    assert mine.read_text(encoding="utf-8") == "Sei un pirata."


def test_too_long_is_cut_at_a_line_end(tmp_path: Path) -> None:
    mine = tmp_path / "CLAUDIO.md"
    mine.write_text("riga uno\nriga due\nriga tre", encoding="utf-8")
    assert load_personality(mine, 20, tmp_path / "none.md") == "riga uno\nriga due"


def test_off_or_missing_means_no_personality(tmp_path: Path) -> None:
    assert load_personality(tmp_path / "CLAUDIO.md", 3000, tmp_path / "none.md") == ""
    assert load_personality(Path(DEFAULT_FILE), 0) == ""


def test_the_default_is_short_and_in_character() -> None:
    text = load_personality(Path(DEFAULT_FILE), 3000)
    assert "Jack il Bello" in text and "<!--" not in text
    assert len(text) < 1500  # it goes with every request


def test_the_personality_replaces_the_neutral_tone() -> None:
    assert system_prompt(False) == SYSTEM_PROMPT and system_prompt(True) == SYSTEM_PROMPT_TOOLS
    prompt = system_prompt(True, "Sei Jack.")
    assert prompt.startswith(SYSTEM_PROMPT_TOOLS.split("- Your tone")[0])
    assert "a little ironic" not in prompt and prompt.rstrip().endswith("Sei Jack.")
    assert "Never pretend to know" in prompt  # the honesty rule stays


def test_the_rules_ask_for_short_italian_replies() -> None:
    assert "Answer in Italian" in SYSTEM_PROMPT and "two at most" in SYSTEM_PROMPT


async def test_every_call_sends_the_same_frozen_prompt(pricing: Pricing) -> None:
    client: Any = FakeClient()
    b = ClaudeBrain(
        client,
        tracker(pricing, 1.0),
        ClaudeSettings(tell_time=False),
        clock=Monotonic(),
        personality="Sei Jack.",
    )
    for _ in range(2):
        "".join([c async for c in b.ask("ciao come va", "it")])
    assert (
        client.calls[0]["system"] == client.calls[1]["system"] == system_prompt(False, "Sei Jack.")
    )
