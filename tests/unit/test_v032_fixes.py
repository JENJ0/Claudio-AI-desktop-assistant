"""v0.3.2 fixes from JenJo's v0.3.1 test (24/10 logs): short follow-ups in the conversation
window are answered, and a question followed by a short sentence still waits for its answer."""

from __future__ import annotations

import pytest

from claudio.brain.claude import ClaudeBrain
from claudio.config import ClaudeSettings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.intake import short_request
from claudio.router.matcher import CommandMatcher
from tests.unit.test_claude_brain import FakeClient, Monotonic, hybrid, say
from tests.unit.test_cost import PRICES, tracker


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


@pytest.mark.parametrize(
    ("text", "wanted"),
    [
        ("Allora scrivi.", True),  # dropped on 24/10 00:25
        ("Scrivila pure.", True),
        ("Apri Chrome.", True),
        ("Felice.", False),  # one word: only as the answer to a question
        ("Vai via.", False),
        ("Ma dai.", False),
        ("Scrivi.", False),
    ],
)
def test_short_requests(text: str, wanted: bool) -> None:
    assert short_request(text) is wanted


async def test_a_question_before_a_last_sentence_still_waits_for_the_answer(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    """24/10 00:14: "Che tema vuoi? Dimmelo e la scrivo subito." then "Felice." was dropped."""
    client = FakeClient(["Che tema vuoi? Dimmelo e la scrivo subito."])
    b = ClaudeBrain(
        client, tracker(pricing, 1.0), ClaudeSettings(tell_time=False), clock=Monotonic()
    )
    r = hybrid(matcher, b)
    await say(r, "scrivimi una poesia")
    assert r.asked_question
    client.chunks = ["Ecco una lunga risposta senza domande, che finisce con un punto."]
    await say(r, "grazie mille davvero")
    assert not r.asked_question
