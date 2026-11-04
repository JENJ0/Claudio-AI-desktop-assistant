"""Safety levels and the spoken confirmation.

Level 1: runs at once (open, search, read). Level 2: CLAUDIO asks and a yes runs it (type,
close an app, move a file). Level 3: CLAUDIO reads a summary of what will happen and only an
explicit "confermo" / "confirm" runs it (delete, terminal, install, network).

Anything that is not a clear yes cancels: a TV in the background, a cough or a wrong word
never confirms an action. A level 2 yes also holds nothing else (_only_yes), and the
Assistant does not count an answer said far from the microphone (intake.py,
noise.answer_min_dbfs)."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from claudio.config import DEFAULT_LANGUAGE
from claudio.router.matcher import Match
from claudio.router.normalizer import is_wake_word, normalize, strip_wake_words

LEVEL_SAFE, LEVEL_CONFIRM, LEVEL_EXPLICIT = 1, 2, 3
LEVELS = (LEVEL_SAFE, LEVEL_CONFIRM, LEVEL_EXPLICIT)

# Said when a command does not carry its own question (`confirm` in commands.toml).
ASK_LEVEL_2 = {"it": "Confermi?", "en": "Do you confirm?"}
ASK_LEVEL_3 = {
    "it": "Per procedere dì: confermo.",  # short (JenJo, 23/10): the summary comes before
    "en": "To go ahead, say: I confirm.",
}
CANCELLED = {"it": "Va bene, non lo faccio.", "en": "Okay, I won't do it."}
NOT_UNDERSTOOD = {"it": "Non ho capito: non lo faccio.", "en": "I didn't catch that: not doing it."}

# Normalised (lowercase, no accents or punctuation). A refusal anywhere wins over a yes.
NO_WORDS = frozenset(
    {"no", "non", "annulla", "annullare", "ferma", "fermo", "basta", "stop", "niente", "mai"}
    | {"nope", "dont", "don", "cancel", "abort", "nevermind", "never", "negative", "wait"}
    | {"aspetta", "lascia", "lascialo", "ferme"}
)
YES_WORDS = frozenset(
    {"si", "sii", "certo", "certamente", "ok", "okay", "esatto", "procedi", "avanti", "fallo"}
    | {"confermo", "conferma", "confermato", "yes", "yeah", "yep", "sure", "confirm"}
    | {"confirmed", "proceed", "affirmative"}
)
# Level 3 accepts only the explicit words (a plain "sì" is too easy to say by accident).
EXPLICIT_WORDS = frozenset({"confermo", "conferma", "confirm", "confirmed"})
YES_PHRASES = ("va bene", "fai pure", "go ahead", "do it", "i confirm")
# All a level 2 yes may hold besides yes words, whole yes phrases and the name: "sì, grazie
# mille" is a yes, "si chiama Anna" has a yes word and says something else.
YES_COURTESY_PHRASES = (
    "grazie mille",
    "per favore",
    "per piacere",
    "per cortesia",
    "thank you very much",
    "thank you",
)
YES_COURTESY_WORDS = frozenset(
    {"grazie", "davvero", "pure", "gentilmente", "allora", "dai", "ma", "please", "thanks"}
    | {"ehm", "eh", "uhm", "um", "uh", "ah"}
)
EXPLICIT_PHRASES = ("i confirm", "yes i confirm", "si confermo")


class Answer(enum.Enum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


def parse_answer(text: str, level: int = LEVEL_CONFIRM) -> Answer:
    """What the person said to "do you confirm?". At level 3 only an explicit
    confirmation is a yes; an ordinary yes is unclear (the action is not run). Below it, a yes
    word is not enough: every word must be one a yes may hold."""
    norm = normalize(text, drop_fillers=False)
    words = strip_wake_words(norm.split())
    if not words:
        return Answer.UNCLEAR
    if NO_WORDS.intersection(words):
        return Answer.NO
    joined = " ".join(words)
    if level >= LEVEL_EXPLICIT:
        explicit = EXPLICIT_WORDS.intersection(words) or any(p in joined for p in EXPLICIT_PHRASES)
        return Answer.YES if explicit else Answer.UNCLEAR
    said_yes = YES_WORDS.intersection(words) or any(p in joined for p in YES_PHRASES)
    return Answer.YES if said_yes and _only_yes(words) else Answer.UNCLEAR


def _only_yes(words: list[str]) -> bool:
    """Nothing but a yes: take the whole phrases out, then every word left must be one a yes
    may hold."""
    rest = f" {' '.join(words)} "
    for phrase in (*YES_PHRASES, *YES_COURTESY_PHRASES):
        rest = rest.replace(f" {phrase} ", " ")
    return all(w in YES_WORDS or w in YES_COURTESY_WORDS or is_wake_word(w) for w in rest.split())


@dataclass(frozen=True)
class Pending:
    """An action waiting for its yes."""

    match: Match
    language: str  # of the question: the answer is given in it

    @property
    def level(self) -> int:
        return self.match.command.level


def question(match: Match, language: str) -> str:
    """What CLAUDIO asks before running `match`: the command's own `confirm` text (for level 3,
    the summary of what will happen), else the generic one of its level."""
    cmd = match.command
    own = cmd.confirm_in(language)
    if own:
        return own
    generic = ASK_LEVEL_3 if cmd.level >= LEVEL_EXPLICIT else ASK_LEVEL_2
    return generic.get(language) or generic[DEFAULT_LANGUAGE]
