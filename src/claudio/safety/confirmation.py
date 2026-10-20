"""Safety levels and the spoken confirmation.

Level 1: runs at once (open, search, read). Level 2: CLAUDIO asks and a yes runs it (type,
close an app, move a file). Level 3: CLAUDIO reads a summary of what will happen and only an
explicit "confermo" / "confirm" runs it (delete, terminal, install, network).

Anything that is not a clear yes cancels: a TV in the background, a cough or a wrong word
never confirms an action."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from claudio.config import DEFAULT_LANGUAGE
from claudio.router.matcher import Match
from claudio.router.normalizer import normalize, strip_wake_words

LEVEL_SAFE, LEVEL_CONFIRM, LEVEL_EXPLICIT = 1, 2, 3
LEVELS = (LEVEL_SAFE, LEVEL_CONFIRM, LEVEL_EXPLICIT)

# Said when a command does not carry its own question (`confirm` in commands.toml).
ASK_LEVEL_2 = {"it": "Confermi?", "en": "Do you confirm?"}
ASK_LEVEL_3 = {
    "it": "Questa azione è delicata. Per procedere dì: confermo.",
    "en": "This action is sensitive. To go ahead, say: I confirm.",
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
EXPLICIT_PHRASES = ("i confirm", "yes i confirm", "si confermo")


class Answer(enum.Enum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


def parse_answer(text: str, level: int = LEVEL_CONFIRM) -> Answer:
    """What the person said to "do you confirm?". At level 3 only an explicit
    confirmation is a yes; an ordinary yes is unclear (the action is not run)."""
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
    if YES_WORDS.intersection(words) or any(p in joined for p in YES_PHRASES):
        return Answer.YES
    return Answer.UNCLEAR


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
