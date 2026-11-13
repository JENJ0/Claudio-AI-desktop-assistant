"""CLAUDIO's own voice heard back through the speakers.

With speakers instead of headphones the microphone hears the reply, and Whisper turns that
echo into new words: "Sono le 19 e 42" -> "Se non le diciamo", "dell'universo è in linea"
-> "dell'università e lì" (JenJo's log, 26/10 19:41). Few of those words are the reply's own,
so the letters are compared instead: the phrase heard against the stretch of the reply that
sounds most like it."""

from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz import fuzz

from claudio.router.normalizer import GREETINGS, heard_wake_word, is_wake_word, normalize
from claudio.tts.number_words import digits_to_words_it

# Words heard while speaking that are mostly in the reply: CLAUDIO's own voice.
WORD_SHARE = 0.6
# The echo heard now comes from the sentence playing or the one before it (the output buffer
# and the room add well under a second): a long story is not compared whole, where any
# phrase would find letters that look like it.
RECENT_SENTENCES = 2
RECENT_CHARS = 240  # the same bound when the sentences are not known
# A phrase with the name over a reply that says it too: its words around the name this close
# (rapidfuzz ratio) to the words around one of CLAUDIO's own names are the echo. Bench of 13/11
# (H2's stories and three more, 1092 requests, 74 echoes): 21 requests lost, all one word or the
# story's own words around its name ("Claudio, sì" over "Claudio si alzò"), and one echo that
# Whisper garbled beyond recognition.
NAME_ECHO = 70


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


def _words(text: str) -> list[str]:
    return normalize(digits_to_words_it(text), drop_fillers=False).split()


def _is_name(word: str) -> bool:
    return word not in GREETINGS and is_wake_word(word)


def echo_around_name(heard: str, said: str, now: str) -> bool:
    """`heard` has the name and so does the reply (a story about CLAUDIO, H2 12/11): the name
    tells nothing, the words around it do. The echo repeats the words around one of the names
    in `said`, what has played; JenJo says others ("Claudio, no" against "Claudio lo salvò").
    A lone name is the echo when `now`, what is playing, says it."""
    words = _words(heard)
    at = next((i for i, w in enumerate(words) if _is_name(w)), None)
    if at is None:
        return False
    before = [w for w in words[:at] if w not in GREETINGS]
    after = [w for w in words[at + 1 :] if not _is_name(w)]
    if not before and not after:
        return heard_wake_word(now)
    mine = " ".join(before + after)
    # One word ("Claudio, sì") finds its like ("Claudio si alzò") somewhere in a long story:
    # it is compared with what plays now only.
    story = _words(now if len(before) + len(after) == 1 else said)

    def sizes(n: int) -> range:
        """As many story words as JenJo's, give or take one Whisper dropped or split."""
        return range(max(n - 1, 0), n + 3) if n else range(1)

    def alike(j: int) -> bool:
        """The story's words around its name at `j` sound like `mine`."""
        return any(
            fuzz.ratio(mine, " ".join(story[max(j - b, 0) : j] + story[j + 1 : j + 1 + a]))
            >= NAME_ECHO
            for b in sizes(len(before))
            for a in sizes(len(after))
        )

    return any(alike(j) for j, w in enumerate(story) if _is_name(w))
