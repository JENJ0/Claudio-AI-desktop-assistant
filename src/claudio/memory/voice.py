"""What JenJo says to Claudio's memory, understood locally: free, instant, and working also
offline or with the day's budget spent.

Phrases with free text are recognised here: "ricordati che...", "da ora in poi...",
"dimentica...", "cosa ricordi di...". The fixed ones ("cosa ricordi?", "dimenticalo",
"dimentica tutto") are commands in config/commands.toml (action "memory"), so they get the
router's exact matching and, for "dimentica tutto", its "confermo"; they run here too (run()).

"ricordami di chiamare la mamma" is a reminder, not a memory: it is left to Claude."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from claudio.config import DEFAULT_LANGUAGE
from claudio.memory.relevance import rank, stems
from claudio.memory.store import Kind, Memory, MemoryRefused, MemoryStore
from claudio.router.normalizer import normalize, strip_accents, strip_wake_words

Action = Literal["remember", "rule", "forget", "recall"]

FORGET_MIN = 0.5  # share of the words of "dimentica X" a memory must have to be X
CLEAR_LEAD = 0.25  # the closest memory is X when it beats the next one by this much
RECALL_MIN = 0.5
RECALL_MAX = 3  # memories read out for "cosa ricordi di X", or offered by "Quale dimentico?"
LIST_MAX = 3  # newest memories read out for "cosa ricordi?"
CHOICE_TTL_S = 45.0  # "Quale dimentico?" waits this long for its answer
SPOKEN_MAX = 80  # a memory longer than this is not read back when saved

_SEP = r"[\W_]+"
_ACCENTED = {"a": "[aàá]", "e": "[eèé]", "i": "[iìí]", "o": "[oòó]", "u": "[uùú]"}

IT_POLITE = ("per favore", "per piacere", "ti prego", "puoi", "potresti", "devi", "ora", "allora")
IT_REMEMBER = (
    *("ricordati", "ricorda", "ricordatelo", "ricordarti", "ricordare", "ricordatevi"),
    *("voglio che ti ricordi", "vorrei che ti ricordassi", "tieni a mente", "tieni presente"),
    *("memorizza", "memorizzati", "segnati", "segna", "segnatelo", "annotati", "annota"),
    *("prendi nota", "impara", "sappi", "non dimenticare", "non dimenticarti", "non scordare"),
    "non scordarti",
)
IT_ADVERBS = ("bene", "sempre", "pure", "anche", "per favore")
IT_RULE = (
    *("da ora in poi", "d ora in poi", "da adesso in poi", "da oggi in poi", "da ora in avanti"),
    *("d ora in avanti", "da qui in avanti", "d ora innanzi", "da ora in futuro"),
)
IT_FORGET = (
    *("dimentica", "dimenticati", "scordati", "scorda", "cancella dalla memoria"),
    *("togli dalla memoria", "cancella il ricordo", "elimina il ricordo", "rimuovi il ricordo"),
    *("non ricordare piu", "non ricordarti piu"),
)
IT_FORGET_JOIN = ("che", "di", "il ricordo di", "il ricordo che", "quello di", "quello che")
IT_RECALL = (
    *("cosa ricordi", "che cosa ricordi", "cosa ti ricordi", "che cosa ti ricordi"),
    *("che ricordi hai", "cosa ti ho detto", "che cosa ti ho detto", "che ti ho detto"),
)
IT_ABOUT = (
    *("di", "su", "riguardo a", "riguardo", "a proposito di", "circa", "del", "della", "dello"),
    *("dei", "degli", "delle", "sul", "sulla", "sullo", "sui", "sugli", "sulle"),
)
EN_POLITE = ("please", "now", "and")
EN_REMEMBER = ("remember", "don t forget", "do not forget", "keep in mind", "make a note", "note")
EN_ADVERBS = ("always", "well", "please")
EN_RULE = ("from now on", "from today on", "going forward", "from here on")
EN_FORGET = ("forget", "forget about", "stop remembering")
EN_FORGET_JOIN = ("that", "about")
EN_RECALL = ("what do you remember", "what did i tell you", "what do you recall")
EN_ABOUT = ("about", "of", "on", "regarding")
# Courtesy at the end of the phrase is not part of the memory ("..., grazie").
_COURTESY = re.compile(
    r"(?:[\W_]+(?:grazie(?:[\W_]+mille)?|per[\W_]+favore|per[\W_]+piacere|thanks|thank[\W_]+you"
    r"|please))+[\W_]*$",
    re.IGNORECASE,
)

SAY = {
    "it": {
        "saved": "Me lo ricordo: {text}.",
        "saved_long": "Fatto, me lo ricordo.",
        "rule": "D'accordo, da ora in poi: {text}.",
        "known": "Lo sapevo già.",
        "empty": "Cosa devo ricordare?",
        "too_long": "È troppo lungo: dimmelo in una frase.",
        "sensitive": "Questo non lo salvo: password e codici andrebbero su internet a ogni "
        "richiesta.",
        "full": "Ho la memoria piena: fammi dimenticare qualcosa prima.",
        "forgot": "Dimenticato: {text}.",
        "forgot_many": "Fatto, ne ho dimenticati {count}.",
        "not_found": "Non ricordo niente del genere.",
        "which": "Ne ricordo {count}: {items}. Quale dimentico?",
        "kept": "Va bene, non dimentico niente.",
        "recall": "Mi hai detto: {items}.",
        "nothing": "Non ricordo ancora niente. Dimmi: ricordati che, e poi la cosa.",
        "list": "Ricordo {count} cose. Le ultime: {items}.",
        "list_one": "Ricordo una cosa: {items}.",
        "nothing_new": "Non ho niente di nuovo da dimenticare.",
        "wiped": "Fatto: ho dimenticato tutto.",
        "ruled": "da ora in poi {text}",
    },
    "en": {
        "saved": "I'll remember: {text}.",
        "saved_long": "Done, I'll remember it.",
        "rule": "All right, from now on: {text}.",
        "known": "I knew that already.",
        "empty": "What should I remember?",
        "too_long": "That's too long: tell me in one sentence.",
        "sensitive": "I won't keep that: passwords and codes would go over the internet with "
        "every request.",
        "full": "My memory is full: have me forget something first.",
        "forgot": "Forgotten: {text}.",
        "forgot_many": "Done, I forgot {count} of them.",
        "not_found": "I don't remember anything like that.",
        "which": "I remember {count}: {items}. Which one should I forget?",
        "kept": "Okay, I won't forget anything.",
        "recall": "You told me: {items}.",
        "nothing": "I don't remember anything yet. Say: remember that, and then the thing.",
        "list": "I remember {count} things. The latest: {items}.",
        "list_one": "I remember one thing: {items}.",
        "nothing_new": "There's nothing new to forget.",
        "wiped": "Done: I forgot everything.",
        "ruled": "from now on {text}",
    },
}

# The answer to "Quale dimentico?" (normalised words).
_ORDINALS = {
    **dict.fromkeys(("primo", "prima", "uno", "first", "one"), 0),
    **dict.fromkeys(("secondo", "seconda", "due", "second", "two"), 1),
    **dict.fromkeys(("terzo", "terza", "tre", "third", "three"), 2),
}
_ALL = frozenset({"tutti", "tutte", "entrambi", "entrambe", "both", "all"})
_NONE = frozenset({"no", "nessuno", "nessuna", "niente", "none", "neither", "nothing", "annulla"})


@dataclass(frozen=True)
class Intent:
    action: Action
    text: str  # what to remember, to forget or to look for, as said
    language: str


@dataclass(frozen=True)
class Reply:
    text: str
    language: str
    outcome: str  # for the metrics: saved, rule, known, refused, forgot, which, recall...


def _pattern(phrase: str) -> str:
    """A phrase as a regex: any punctuation or spacing between its words, any accent."""
    return _SEP.join(
        "".join(_ACCENTED.get(c, re.escape(c)) for c in word) for word in phrase.split()
    )


def _alternatives(phrases: Sequence[str]) -> str:
    """Longest first: "ricordatelo" is tried before "ricorda"."""
    return "(?:" + "|".join(_pattern(p) for p in sorted(phrases, key=len, reverse=True)) + ")"


@dataclass(frozen=True)
class _Words:
    """A language's trigger words. `that` joins "ricordati" and the fact ("che"); `always`
    makes "ricordati sempre di..." a rule."""

    language: str
    polite: Sequence[str]
    remember: Sequence[str]
    adverbs: Sequence[str]
    that: str
    always: str
    rule: Sequence[str]
    forget: Sequence[str]
    forget_join: Sequence[str]
    recall: Sequence[str]
    about: Sequence[str]


def _grammar(w: _Words) -> tuple[tuple[Action, re.Pattern[str]], ...]:
    """The patterns of one language, in the order they are tried: forgetting and asking come
    first, and a rule before a fact ("ricordati che da ora in poi..." is a rule)."""
    start = rf"^(?:{_alternatives(w.polite)}{_SEP})*"
    remember = _alternatives(w.remember)
    told = rf"{remember}(?:{_SEP}{_alternatives(w.adverbs)})?{_SEP}{_pattern(w.that)}"
    body = rf"{_SEP}(?P<body>.+)$"
    forget_join = rf"(?:{_SEP}{_alternatives(w.forget_join)})?"
    patterns: tuple[tuple[Action, str], ...] = (
        ("forget", rf"{start}{_alternatives(w.forget)}{forget_join}{body}"),
        ("recall", rf"{start}{_alternatives(w.recall)}{_SEP}{_alternatives(w.about)}{body}"),
        ("rule", rf"{start}(?:{told}{_SEP})?{_alternatives(w.rule)}{body}"),
        ("rule", rf"{start}{remember}{_SEP}{_pattern(w.always)}{body}"),
        ("remember", rf"{start}{told}{body}"),
    )
    return tuple((a, re.compile(p, re.IGNORECASE | re.DOTALL)) for a, p in patterns)


GRAMMARS = {
    "it": _grammar(
        _Words("it", IT_POLITE, IT_REMEMBER, IT_ADVERBS, "che", "sempre di", IT_RULE, IT_FORGET,
               IT_FORGET_JOIN, IT_RECALL, IT_ABOUT)
    ),
    "en": _grammar(
        _Words("en", EN_POLITE, EN_REMEMBER, EN_ADVERBS, "that", "always to", EN_RULE,
               EN_FORGET, EN_FORGET_JOIN, EN_RECALL, EN_ABOUT)
    ),
}  # fmt: skip


def without_wake_words(text: str) -> str:
    """`text` from where the request starts: "Ehi Claudio, ricordati che..." -> "ricordati
    che...", with its own spelling, accents and capitals (they are what gets stored)."""
    words = list(re.finditer(r"[^\W_]+", text))
    norm = [strip_accents(w.group().lower()) for w in words]
    skipped = len(norm) - len(strip_wake_words(norm))
    if skipped == 0:
        return text.strip()
    return text[words[skipped].start() :].strip() if skipped < len(words) else ""


def parse(text: str) -> Intent | None:
    """The memory request in `text`, or None when it is not one. Forgetting or asking about
    nothing but function words ("dimentica tutto", "forget it") is not one either."""
    said = without_wake_words(text)
    for language, patterns in GRAMMARS.items():
        for action, pattern in patterns:
            m = pattern.match(said)
            if m is None:
                continue
            body = _COURTESY.sub("", m.group("body")).strip()
            if action in ("forget", "recall") and not stems(body):
                return None
            return Intent(action, body, language)
    return None


class MemoryVoice:
    """Answers the memory requests the router hands over (handle) and the memory commands of
    commands.toml (run)."""

    def __init__(self, store: MemoryStore, clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.clock = clock
        # "Quale dimentico?": the memories it asked about, its language and when.
        self._choice: tuple[list[Memory], str, float] | None = None

    @property
    def awaiting(self) -> bool:
        """The last reply asked which memory to forget: the next phrase is the answer."""
        return self._choice is not None and self.clock() - self._choice[2] < CHOICE_TTL_S

    def cancel(self) -> None:
        self._choice = None

    def handle(self, text: str) -> Reply | None:
        """The reply to a memory request; None when `text` is not one (the router goes on)."""
        waiting = self.awaiting
        choice, self._choice = self._choice, None
        intent = parse(text)
        if intent is None and waiting and choice is not None:
            return self._choose(text, choice[0], choice[1])
        if intent is None:
            return None
        if intent.action in ("remember", "rule"):
            return self._remember(intent)
        if intent.action == "forget":
            return self._forget(intent)
        return self._recall(intent)

    def run(self, do: str, language: str = DEFAULT_LANGUAGE) -> str:
        """A memory command of commands.toml: list, forget_last or wipe."""
        say = _say(language)
        if do == "list":
            items = self.store.items()
            if not items:
                return say["nothing"]
            latest = _spoken(list(reversed(items[-LIST_MAX:])), language)
            key = "list_one" if len(items) == 1 else "list"
            return say[key].format(count=len(items), items=latest)
        if do == "forget_last":
            last = self.store.last_added()
            if last is None:
                return say["nothing_new"]
            self.store.remove([last.id])
            return say["forgot"].format(text=_as_said(last, language))
        if do == "wipe":
            return say["wiped"] if self.store.clear() else say["nothing"]
        raise ValueError(f"unknown memory command: {do!r}")

    def _remember(self, intent: Intent) -> Reply:
        say = _say(intent.language)
        kind: Kind = "rule" if intent.action == "rule" else "fact"
        try:
            memory, new = self.store.add(intent.text, kind, "jenjo")
        except MemoryRefused as exc:
            return Reply(say[exc.reason], intent.language, "refused")
        if not new:
            return Reply(say["known"], intent.language, "known")
        if kind == "rule":
            return Reply(say["rule"].format(text=memory.text), intent.language, "rule")
        if len(memory.text) > SPOKEN_MAX:
            return Reply(say["saved_long"], intent.language, "saved")
        return Reply(say["saved"].format(text=memory.text), intent.language, "saved")

    def _forget(self, intent: Intent) -> Reply:
        say = _say(intent.language)
        found = [s for s in rank(intent.text, self.store.items()) if s.coverage >= FORGET_MIN]
        if not found:
            return Reply(say["not_found"], intent.language, "not_found")
        close = [s for s in found if s.coverage > found[0].coverage - CLEAR_LEAD]
        if len(close) == 1:
            return self._forgot([found[0].memory], intent.language)
        options = [s.memory for s in close[:RECALL_MAX]]
        self._choice = (options, intent.language, self.clock())
        items = _spoken(options, intent.language)
        return Reply(say["which"].format(count=len(options), items=items), intent.language, "which")

    def _choose(self, text: str, options: list[Memory], language: str) -> Reply | None:
        """The answer to "Quale dimentico?": "tutti e due", "il primo", words of the one to
        forget, or "nessuno". Anything else is a new request (None)."""
        words = normalize(text, drop_fillers=False).split()
        if _ALL.intersection(words):
            return self._forgot(options, language)
        picked = {_ORDINALS[w] for w in words if w in _ORDINALS and _ORDINALS[w] < len(options)}
        if len(picked) == 1:
            return self._forgot([options[picked.pop()]], language)
        scored = rank(text, options)
        if scored and (len(scored) == 1 or scored[0].coverage - scored[1].coverage >= CLEAR_LEAD):
            return self._forgot([scored[0].memory], language)
        if _NONE.intersection(words):
            return Reply(_say(language)["kept"], language, "kept")
        return None

    def _forgot(self, memories: list[Memory], language: str) -> Reply:
        say = _say(language)
        gone = self.store.remove([m.id for m in memories])
        if len(gone) == 1:
            return Reply(say["forgot"].format(text=_as_said(gone[0], language)), language, "forgot")
        return Reply(say["forgot_many"].format(count=len(gone)), language, "forgot")

    def _recall(self, intent: Intent) -> Reply | None:
        """What JenJo said about the subject; None when nothing matches: Claude answers then,
        as it sees the memories and knows what words mean, not only how they are spelt."""
        found = [s for s in rank(intent.text, self.store.items()) if s.coverage >= RECALL_MIN]
        if not found:
            return None
        items = _spoken([s.memory for s in found[:RECALL_MAX]], intent.language)
        return Reply(_say(intent.language)["recall"].format(items=items), intent.language, "recall")


def _say(language: str) -> dict[str, str]:
    return SAY.get(language) or SAY[DEFAULT_LANGUAGE]


def _as_said(memory: Memory, language: str) -> str:
    """A rule is read with its "da ora in poi": "chiamami capo" alone sounds like an order."""
    return (
        _say(language)["ruled"].format(text=memory.text) if memory.kind == "rule" else memory.text
    )


def _spoken(memories: Sequence[Memory], language: str) -> str:
    return "; ".join(_as_said(m, language) for m in memories)
