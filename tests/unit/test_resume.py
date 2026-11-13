"""resume.py: the part of a cut reply still to say, and reading a reply to its end."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from claudio.resume import DrainedStream, ends_the_reply, is_story, remainder, worth_resuming

STORY = "C'era una volta un robot. Viveva in una fabbrica. Un giorno scappò. Fine."


def test_remainder_starts_again_at_the_sentence_that_was_cut() -> None:
    played = ["C'era una volta un robot.", "Viveva in una fabbrica."]
    assert remainder(STORY, played) == "Viveva in una fabbrica. Un giorno scappò. Fine."


def test_remainder_is_everything_when_nothing_played() -> None:
    assert remainder(STORY, []) == STORY


def test_remainder_skips_a_sentence_it_cannot_find() -> None:
    assert remainder(STORY, ["Un giorno scappò.", "ripetuto male"]) == "Un giorno scappò. Fine."


@pytest.mark.parametrize(
    ("request_text", "ends"),
    [
        ("Ok, va bene, grazie.", True),
        ("Basta così, grazie", False),  # "così" is not a closer: the stop words handle "basta"
        ("Grazie", True),
        ("Che ore sono?", False),
        ("Aspetta, che giorno è oggi?", False),
    ],
)
def test_ends_the_reply(request_text: str, ends: bool) -> None:
    assert ends_the_reply(request_text) is ends


LONG = STORY * 3  # 222 characters: a story worth going on with


@pytest.mark.parametrize(
    ("reply", "worth"),
    [
        (LONG, True),
        (STORY, False),  # 74 characters: an answer, banter, a joke
        (f"{LONG} Vuoi il seguito?", True),  # a story asking for more is still a story (17:08)
        (f"{LONG} Ne vuoi un'altra? Dimmelo e la racconto.", True),
        (f"Una storia? Eccola. {LONG}", True),  # a "?" at the start is no question to JenJo
    ],
)
def test_worth_resuming(reply: str, worth: bool) -> None:
    assert worth_resuming(reply, 200) is worth


async def slow(chunks: list[str], fail: bool = False) -> AsyncIterator[str]:
    for chunk in chunks:
        await asyncio.sleep(0.01)
        yield chunk
    if fail:
        raise RuntimeError("Claude down")


async def test_a_stream_is_read_to_its_end_after_the_reader_stops() -> None:
    drained = DrainedStream(slow(["Uno. ", "Due. ", "Tre."]))
    async for chunk in drained.read():
        assert chunk == "Uno. "
        break  # the speaker was cut after the first sentence
    await drained.finish(1.0)
    assert drained.text == "Uno. Due. Tre."


async def test_read_gives_every_chunk_and_errors() -> None:
    drained = DrainedStream(slow(["a", "b"], fail=True))
    got: list[str] = []
    with pytest.raises(RuntimeError):
        async for chunk in drained.read():
            got.append(chunk)
    assert got == ["a", "b"]


async def test_finish_gives_up_after_the_timeout() -> None:
    drained = DrainedStream(slow(["a"] * 100))
    await drained.finish(0.05)
    assert 0 < len(drained.parts) < 100


# H2 ter 13/11: the two answers about the Moon were long enough to be resumed (224 and 183
# characters) but they were no stories: "riprendi" said their last sentence again.
MOON_1 = (
    "Mi sta benissimo, socio, come tutto il resto: la luna mi dona, ovviamente. Se invece volevi "
    "sapere quanto dista, sono circa trecentoottantaquattromila chilometri, più o meno la "
    "distanza tra il tuo cervello e un'idea decente."
)
LONG_STORY = (
    "C'era una volta un robot che viveva in una fabbrica abbandonata alla periferia della città. "
    "Ogni notte accendeva le luci e contava i pezzi di ricambio rimasti sugli scaffali. Un giorno "
    "trovò una porta che nessuno aveva mai aperto, e decise di attraversarla."
)


@pytest.mark.parametrize(
    ("reply", "story"),
    [
        (MOON_1, False),  # 224 characters, two sentences: an answer
        (LONG_STORY, True),  # three sentences, over 200 characters
        ("Uno. Due. Tre. Quattro.", False),  # sentences, but too short to be worth resuming
        ("", False),
    ],
)
def test_is_story(reply: str, story: bool) -> None:
    assert is_story(reply, 200) is story
