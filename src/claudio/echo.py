"""CLAUDIO's own voice heard back through the speakers.

With speakers instead of headphones the microphone hears the reply, and Whisper turns that
echo into new words: "Sono le 19 e 42" -> "Se non le diciamo", "dell'universo è in linea"
-> "dell'università e lì" (JenJo's log, 26/10 19:41). Few of those words are the reply's own,
so the letters are compared instead: the phrase heard against the stretch of the reply that
sounds most like it."""

from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz import fuzz

from claudio.router.normalizer import normalize
from claudio.tts.number_words import digits_to_words_it

# Words heard while speaking that are mostly in the reply: CLAUDIO's own voice.
WORD_SHARE = 0.6
# The echo heard now comes from the sentence playing or the one before it (the output buffer
# and the room add well under a second): a long story is not compared whole, where any
# phrase would find letters that look like it.
RECENT_SENTENCES = 2
RECENT_CHARS = 240  # the same bound when the sentences are not known


def _letters(text: str) -> str:
    """Lowercase letters and digits only, numbers as the voice says them ("19" ->
    "diciannove"): Whisper splits and joins the echo's words at random."""
    return normalize(digits_to_words_it(text), drop_fillers=False).replace(" ", "")


def recent_reply(played: Sequence[str], reply: str) -> str:
    """What the microphone can be hearing now: the last sentences that started playing, or,
    without them, the end of the reply."""
    if played:
        return " ".join(played[-RECENT_SENTENCES:])
    return reply[-RECENT_CHARS:]


def similarity(heard: str, reply: str) -> float:
    """0-100: how much `heard` sounds like some stretch of `reply`."""
    a, b = _letters(heard), _letters(reply)
    if not a or not b:
        return 0.0
    return float(fuzz.partial_ratio(a, b))


def word_share(heard: str, reply: str) -> float:
    """The share of the words heard that are words of the reply."""
    words = normalize(heard, drop_fillers=False).split()
    if not words:
        return 0.0
    said = set(normalize(digits_to_words_it(reply), drop_fillers=False).split())
    said |= set(normalize(reply, drop_fillers=False).split())
    return sum(w in said for w in words) / len(words)


def is_echo(heard: str, recent: str, reply: str, min_similarity: float) -> bool:
    """`heard` is the reply's echo: mostly its words (anywhere in it), or letters close to
    what was just said. min_similarity 0 turns the letter check off."""
    if not heard or not (recent or reply):
        return False
    if word_share(heard, reply) >= WORD_SHARE:
        return True
    return min_similarity > 0 and similarity(heard, recent) >= min_similarity
