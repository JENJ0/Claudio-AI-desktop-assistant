"""What JenJo says to Claudio's memory, understood locally: free, instant, and working also
offline or with the day's budget spent.

Phrases with free text are recognised here: "ricordati che...", "da ora in poi...",
"dimentica...", "dimenticalo", "cosa ricordi di...". The fixed ones ("cosa ricordi?",
"dimentica tutto") are commands in config/commands.toml (action "memory"): they get the
router's exact matching and, for "dimentica tutto", its "confermo"; they run here (run()).

Forgetting never guesses. "dimentica X" forgets at once only one memory that has every word
of X as a whole word (not "anni" for "Anna"), said with the wake word; otherwise Claudio asks
("Dimentico questo: ...?", or "Quale dimentico?" when two or three are as close). A rule heard
without the wake word (the TV in the conversation window: "da ora in poi, risparmia...") is
asked about too. The answer is taken only when it is one: a short yes or no, "la seconda",
"tutti e due", words of one of them; never the echo of the question, never a longer request
("alza il volume di due"), which is answered as a request instead.

"ricordami di chiamare la mamma" is a reminder, not a memory: it is left to Claude."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from claudio.config import DEFAULT_LANGUAGE
from claudio.echo import WORD_SHARE, word_share
from claudio.memory.relevance import STOPWORDS, rank, same_word, stems
from claudio.memory.store import Kind, Memory, MemoryRefused, MemoryStore
from claudio.router.normalizer import normalize, strip_accents, strip_wake_words
from claudio.safety.confirmation import (
    NO_WORDS,
    Answer,
    has_question_mark,
    parse_answer,
    words_without_name,
)

Action = Literal["remember", "rule", "forget", "forget_it", "forget_last", "recall"]

FORGET_MIN = 0.5  # share of the words of "dimentica X" a memory must have to be a candidate
CLEAR_LEAD = 0.25  # the closest memory is X when it beats the next one by this much
RECALL_MIN = 0.5
RECALL_MAX = 3  # memories read out for "cosa ricordi di X", or offered by "Quale dimentico?"
LIST_MAX = 3  # newest memories read out for "cosa ricordi?"
QUESTION_TTL_S = 45.0  # "Quale dimentico?" waits this long for its answer
TALKED_TTL_S = 60.0  # "dimenticalo" means the memory just talked about, this recently
SPOKEN_MAX = 80  # a memory longer than this is not read back when saved
ANSWER_WORDS = 4  # "tutti e due", "la seconda": an answer is short
NAMED_ANSWER_WORDS = 6  # "quella che abita a Torino"
# All a named answer may hold besides the memory's own words: "chi abita a Torino" is a question.
NAMED_ANSWER_CONNECTORS = frozenset(
    {"quella", "quello", "la", "il", "l", "di", "del", "della", "dei", "su", "sul", "sulla"}
)
# Only after a lead ("quella che abita a Torino"): at the start they make a question
# ("che riunione è", "dove abita a Torino"), a delete.
NAMED_ANSWER_LEADS = frozenset({"quella", "quello", "la", "il", "l"})
NAMED_ANSWER_AFTER_LEAD = frozenset({"che", "dove", "con", "cosa"})

# The rule question's own phrases, whole: "sì, tienila come regola" is a yes, "tienila come
# regola" is not, nor is "come sì" or "sì a". Longest first, so "tienila come regola" is one.
_RULE_QUESTION = (
    *("tienila come regola", "tienilo come regola"),
    *("tienila", "tienilo"),
)
RULE_QUESTION_PHRASES = re.compile(rf"\b(?:{'|'.join(_RULE_QUESTION)})\b")

_SEP = r"[\W_]+"
_ACCENTED = {"a": "[aàá]", "e": "[eèé]", "i": "[iìí]", "o": "[oòó]", "u": "[uùú]"}

# What may come before the request: courtesy and the spoken openers ("ok", "senti", "e"...).
IT_OPENERS = (
    *("per favore", "per piacere", "ti prego", "mi raccomando", "puoi", "potresti", "devi"),
    *("ora", "allora", "ok", "okay", "ehm", "eh", "senti", "ascolta", "scusa", "si", "e"),
    *("dai", "beh", "comunque", "poi", "anche", "ah", "oh"),
)
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
# "Dimenticalo": the memory just talked about. "Scordatelo!" is not here: it means "no way".
IT_FORGET_IT = (
    *("dimenticalo", "dimenticala", "dimenticatelo", "dimenticatela", "non ricordarlo"),
    *("non ricordarla", "cancellalo", "cancellala"),
)
IT_FORGET_LAST = ("dimentica", "dimenticati", "scordati")
# "cancella l'ultimo" may be about typed text: only "cancella l'ultimo ricordo" is the memory.
IT_DELETE = ("cancella", "togli", "elimina", "rimuovi")
IT_LAST_MEMORY = ("l ultimo ricordo", "l ultima cosa che ti ho detto di ricordare")
IT_LAST = (
    *("l ultimo ricordo", "l ultima cosa che ti ho detto", "l ultima cosa", "l ultimo", "l ultima"),
    *("quello che ti ho appena detto", "quello che ti ho detto", "cosa ti ho appena detto"),
    *("cosa ti ho detto", "quello che ho appena detto", "quello che ho detto"),
)
IT_RECALL = (
    *("cosa ricordi", "che cosa ricordi", "cosa ti ricordi", "che cosa ti ricordi"),
    *("che ricordi hai", "cosa ti ho detto", "che cosa ti ho detto", "che ti ho detto"),
)
IT_ABOUT = (
    *("di", "su", "riguardo a", "riguardo", "a proposito di", "circa", "del", "della", "dello"),
    *("dei", "degli", "delle", "sul", "sulla", "sullo", "sui", "sugli", "sulle"),
)
# Courtesy at the start or the end of what is remembered is not part of it ("..., grazie").
_COURTESY = (
    r"(?:grazie(?:[\W_]+mille)?|per[\W_]+favore|per[\W_]+piacere|per[\W_]+cortesia"
    r"|ti[\W_]+prego)"
)
_COURTESY_END = re.compile(rf"(?:[\W_]+{_COURTESY})+[\W_]*$", re.IGNORECASE)
_COURTESY_START = re.compile(rf"^(?:{_COURTESY}[\W_]+)+", re.IGNORECASE)

SAY = {
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
    "which": "Ne ricordo {count}: {items}. Quale dimentico?",
    "forget_one": "Dimentico questo: {items}?",
    "keep_rule": "Da ora in poi: {text}. La tengo come regola?",
    "kept": "Va bene, non dimentico niente.",
    "rule_dropped": "Va bene, non la salvo.",
    "unclear": "Non ho capito: non dimentico niente.",
    "rule_unclear": "Non ho capito: non la salvo.",
    "recall": "Mi hai detto: {items}.",
    "nothing": "Non ricordo ancora niente. Dimmi: ricordati che, e poi la cosa.",
    "list": "Ricordo {count} cose. Le ultime: {items}.",
    "list_one": "Ricordo una cosa: {items}.",
    "nothing_new": "Non ho niente di nuovo da dimenticare.",
    "not_saved": "Non l'avevo salvato.",
    "wiped": "Fatto: ho dimenticato tutto.",
    "ruled": "da ora in poi {text}",
}

# The answer to "Quale dimentico?" (normalised words).
_ORDINALS = {
    **dict.fromkeys(("primo", "prima", "uno"), 0),
    **dict.fromkeys(("secondo", "seconda", "due"), 1),
    **dict.fromkeys(("terzo", "terza", "tre"), 2),
    **dict.fromkeys(("ultimo", "ultima"), -1),
}
_ORDINAL_FILLER = frozenset(
    {"il", "la", "l", "lo", "quello", "quella", "numero", "ricordo", "cosa", "e", "ok", "si"}
)
_ALL = frozenset({"tutti", "tutte", "entrambi", "entrambe"})
_ALL_FILLER = frozenset({"e", "due", "tre", "quanti", "i", "le", "gli"})
_OPENER_WORDS = frozenset(
    {"ok", "okay", "allora", "senti", "ascolta", "scusa", "dai", "beh", "ehm", "eh", "ah", "oh"}
    | {"comunque"}
)
_COURTESY_WORDS = frozenset(
    {"grazie", "mille", "favore", "piacere", "cortesia", "prego", "per", "ti"}
)
_NONE = frozenset(NO_WORDS | {"nessuno", "nessuna", "stare"})
_FORGET_VERBS = frozenset(
    {"dimentica", "dimenticati", "scordati", "scorda", "cancella", "togli", "elimina", "rimuovi"}
)


@dataclass(frozen=True)
class Intent:
    action: Action
    text: str  # what to remember, to forget or to look for, as said
    language: str


@dataclass(frozen=True)
class Reply:
    text: str  # "" when nothing is to be said (the echo of Claudio's own question)
    language: str
    outcome: str  # for the metrics: saved, rule, known, refused, forgot, which, recall...


@dataclass(frozen=True)
class _Question:
    """Claudio asked before forgetting or before keeping a rule: the next phrase may be
    the answer."""

    kind: Literal["which", "forget", "rule"]
    options: tuple[Memory, ...]  # which / forget: the memories asked about
    rule: str  # rule: the rule to keep
    language: str
    said: str  # the question as spoken: its echo is not an answer
    asked: float


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
    openers: Sequence[str]
    remember: Sequence[str]
    adverbs: Sequence[str]
    that: str
    always: str
    rule: Sequence[str]
    forget: Sequence[str]
    forget_join: Sequence[str]
    forget_it: Sequence[str]
    forget_last: Sequence[str]
    last: Sequence[str]
    delete: Sequence[str]
    last_memory: Sequence[str]
    recall: Sequence[str]
    about: Sequence[str]


def _grammar(w: _Words) -> tuple[tuple[Action, re.Pattern[str]], ...]:
    """The patterns of one language, in the order they are tried: forgetting and asking come
    first, and a rule before a fact ("ricordati che da ora in poi..." is a rule)."""
    start = rf"^(?:{_alternatives(w.openers)}{_SEP})*"
    remember = _alternatives(w.remember)
    told = rf"{remember}(?:{_SEP}{_alternatives(w.adverbs)})?(?:{_SEP}{_pattern(w.that)}|\s*:)"
    body = rf"{_SEP}(?P<body>.+)$"
    end = r"[\W_]*$"
    forget_join = rf"(?:{_SEP}{_alternatives(w.forget_join)})?"
    patterns: tuple[tuple[Action, str], ...] = (
        ("forget_it", rf"{start}(?P<body>{_alternatives(w.forget_it)}){end}"),
        (
            "forget_last",
            rf"{start}{_alternatives(w.forget_last)}{_SEP}(?P<body>{_alternatives(w.last)}){end}",
        ),
        (
            "forget_last",
            rf"{start}{_alternatives(w.delete)}{_SEP}(?P<body>{_alternatives(w.last_memory)}){end}",
        ),
        ("forget", rf"{start}{_alternatives(w.forget)}{forget_join}{body}"),
        ("recall", rf"{start}{_alternatives(w.recall)}{_SEP}{_alternatives(w.about)}{body}"),
        ("rule", rf"{start}(?:{told}{_SEP})?{_alternatives(w.rule)}{body}"),
        ("rule", rf"{start}{remember}{_SEP}{_pattern(w.always)}{body}"),
        ("remember", rf"{start}{told}{body}"),
    )
    return tuple((a, re.compile(p, re.IGNORECASE | re.DOTALL)) for a, p in patterns)


GRAMMAR = _grammar(
    _Words("it", IT_OPENERS, IT_REMEMBER, IT_ADVERBS, "che", "sempre di", IT_RULE,
           IT_FORGET, IT_FORGET_JOIN, IT_FORGET_IT, IT_FORGET_LAST, IT_LAST, IT_DELETE,
           IT_LAST_MEMORY, IT_RECALL, IT_ABOUT)
)  # fmt: skip


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
    nothing but function words ("dimentica tutto") is not one either."""
    said = without_wake_words(text)
    for action, pattern in GRAMMAR:
        m = pattern.match(said)
        if m is None:
            continue
        body = _COURTESY_START.sub("", _COURTESY_END.sub("", m.group("body"))).strip()
        if action in ("forget", "recall") and not stems(body):
            return None
        return Intent(action, body, DEFAULT_LANGUAGE)
    return None


class MemoryVoice:
    """Answers the memory requests the router hands over (handle) and the memory commands of
    commands.toml (run)."""

    def __init__(self, store: MemoryStore, clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.clock = clock
        self._question: _Question | None = None
        # The memory last saved, found or read out, and when (the store's wall clock); None
        # as the memory: the last thing talked about was refused ("dimenticalo" then means it).
        self._talked: tuple[Memory | None, float] | None = None
        self._moved = -float("inf")  # when a request went elsewhere: older talk is over

    @property
    def awaiting(self) -> bool:
        """Claudio asked which memory to forget, or whether to keep a rule: the next phrase
        may be the answer."""
        q = self._question
        return q is not None and self.clock() - q.asked < QUESTION_TTL_S

    def understands(self, text: str) -> bool:
        """A memory request: the router's business (no closed list, no guess, a request
        also in the conversation window). The answer to Claudio's question is not counted
        here: the assistant already listens for it as an answer."""
        return parse(text) is not None

    def cancel(self) -> None:
        self._question = None

    def moved_on(self) -> None:
        """The request went to a command or to Claude: "dimenticalo" after it is not about
        a memory talked about before."""
        self._question, self._talked = None, None
        self._moved = self.store.clock()

    def handle(self, text: str, addressed: bool = True) -> Reply | None:
        """The reply to a memory request; None when `text` is not one (the router goes on).
        `addressed`: said with the wake word (or typed); without it a rule or a forget is
        asked about first."""
        addressed = addressed and not has_question_mark(text)  # "Dimenticalo?" asks, never forgets
        question = self._question if self.awaiting else None
        self._question = None
        if question is not None:
            answer = self._answer(text, question)
            if answer is not None:
                return answer
        intent = parse(text)
        if intent is None:
            return None
        if intent.action in ("remember", "rule"):
            return self._remember(intent, addressed)
        if intent.action == "forget":
            return self._forget(intent, addressed)
        if intent.action in ("forget_it", "forget_last"):
            return self._forget_last(intent, addressed)
        return self._recall(intent)

    def run(self, do: str, language: str = DEFAULT_LANGUAGE) -> str:
        """A memory command of commands.toml: list or wipe (level 3: "confermo" was heard)."""
        self._question = None
        if do == "list":
            items = self.store.items()
            if not items:
                return SAY["nothing"]
            latest = _spoken(list(reversed(items[-LIST_MAX:])), language)
            if len(items) == 1:
                self._talk(items[0])  # "dimenticalo" now means it
            key = "list_one" if len(items) == 1 else "list"
            return SAY[key].format(count=len(items), items=latest)
        if do == "wipe":
            self._talked = None
            return SAY["wiped"] if self.store.clear() else SAY["nothing"]
        raise ValueError(f"unknown memory command: {do!r}")

    # -- saving --

    def _remember(self, intent: Intent, addressed: bool) -> Reply:
        kind: Kind = "rule" if intent.action == "rule" else "fact"
        if kind == "rule" and not addressed:
            text = " ".join(intent.text.split()).strip(" .!?")
            return self._ask(_Question("rule", (), text, intent.language, "", 0.0))
        try:
            memory, new = self.store.add(intent.text, kind, "jenjo")
        except MemoryRefused as exc:
            self._talked = (None, self.store.clock())  # "ok, dimenticalo": nothing to forget
            return Reply(SAY[exc.reason], intent.language, "refused")
        self._talk(memory)
        if not new:
            return Reply(SAY["known"], intent.language, "known")
        if kind == "rule":
            return Reply(SAY["rule"].format(text=memory.text), intent.language, "rule")
        if len(memory.text) > SPOKEN_MAX:
            return Reply(SAY["saved_long"], intent.language, "saved")
        return Reply(SAY["saved"].format(text=memory.text), intent.language, "saved")

    # -- forgetting --

    def _forget(self, intent: Intent, addressed: bool) -> Reply | None:
        """None when nothing comes close: "dimentica la storia di prima e raccontami una
        barzelletta" is not about a memory, and Claude (who sees them all) answers it."""
        found = [s for s in rank(intent.text, self.store.items()) if s.coverage >= FORGET_MIN]
        if not found:
            return None
        close = [s for s in found if s.coverage > found[0].coverage - CLEAR_LEAD]
        if len(close) == 1:
            best = close[0]
            if addressed and best.coverage == 1.0 and best.exact == 1.0:
                return self._forgot([best.memory], intent.language)
            options: tuple[Memory, ...] = (best.memory,)
            return self._ask(_Question("forget", options, "", intent.language, "", 0.0))
        options = tuple(s.memory for s in close[:RECALL_MAX])
        return self._ask(_Question("which", options, "", intent.language, "", 0.0))

    def _forget_last(self, intent: Intent, addressed: bool) -> Reply | None:
        """ "Dimenticalo" forgets the memory just talked about (None without one: then it is
        not about a memory); after a refusal there is nothing to forget. "Dimentica l'ultimo
        ricordo" with nothing just talked about asks about the last one of this run."""
        target, refused = self._recent()
        if refused:
            return Reply(SAY["not_saved"], intent.language, "not_saved")
        if target is not None:
            if not addressed:
                return self._ask(_Question("forget", (target,), "", intent.language, "", 0.0))
            return self._forgot([target], intent.language)
        if intent.action == "forget_it":
            return None
        last = self.store.last_added()
        if last is None:
            return Reply(SAY["nothing_new"], intent.language, "nothing_new")
        return self._ask(_Question("forget", (last,), "", intent.language, "", 0.0))

    def _forgot(self, memories: Sequence[Memory], language: str) -> Reply:
        gone = self.store.remove([m.id for m in memories])
        if self._talked is not None and self._talked[0] in gone:
            self._talked = None
        if len(gone) == 1:
            return Reply(SAY["forgot"].format(text=_as_said(gone[0], language)), language, "forgot")
        return Reply(SAY["forgot_many"].format(count=len(gone)), language, "forgot")

    # -- the question and its answer --

    def _ask(self, q: _Question) -> Reply:
        if q.kind == "rule":
            said = SAY["keep_rule"].format(text=q.rule)
        elif q.kind == "forget":
            said = SAY["forget_one"].format(items=_spoken(q.options, q.language))
            self._talk(q.options[0])  # "dimenticalo" now means this one
        else:
            items = _spoken(q.options, q.language)
            said = SAY["which"].format(count=len(q.options), items=items)
        self._question = _Question(q.kind, q.options, q.rule, q.language, said, self.clock())
        outcome = {"which": "which", "forget": "forget_one", "rule": "rule_asked"}[q.kind]
        return Reply(said, q.language, outcome)

    def _answer(self, text: str, q: _Question) -> Reply | None:
        """The answer to the question, or None when `text` is something else (the question
        is dropped and `text` answered as usual): a new memory request, any longer phrase,
        a question, a command. Only a no, a yes ("sì", "dimenticalo"), "la seconda", "tutti
        e due" or the words of one of them is an answer."""
        unclear = SAY["rule_unclear" if q.kind == "rule" else "unclear"]
        words = _answer_words(text)
        if not words:  # said far from the microphone (or nothing): never a yes
            return Reply(unclear, q.language, "unclear")
        if _echo(text, q):
            self._question = q  # Claudio's own question in the microphone: still waiting
            return Reply("", q.language, "echo")
        asked = has_question_mark(text)  # parse_answer gets the words, without the "?"
        intent = parse(text)
        if intent is not None and intent.action == "forget_it":
            if asked:
                return None  # "Dimenticalo?" asks: nothing is forgotten
            if q.kind == "forget":
                return self._forgot(q.options, q.language)  # "Dimenticalo." is the yes
            if q.kind == "rule":
                return Reply(SAY["rule_dropped"], q.language, "kept")
            self._question = q  # which one? "dimenticalo" does not say
            return Reply(q.said, q.language, "which")
        if words[0] in _FORGET_VERBS:  # "dimentica la prima": the answer, if the rest is one
            words = words[1:]
        elif intent is not None:
            return None  # "ricordati che non mangio la carne": a new request
        if not words:
            return None
        if words[0] in _NONE or (len(words) <= ANSWER_WORDS and _NONE.intersection(words)):
            kept = SAY["rule_dropped" if q.kind == "rule" else "kept"]
            return Reply(kept, q.language, "kept")
        if q.kind != "which":
            if not asked and parse_answer(" ".join(_without_question(q, words))) is Answer.YES:
                return (
                    self._keep_rule(q) if q.kind == "rule" else self._forgot(q.options, q.language)
                )
            return None
        return self._which_answer(words, text, q)

    def _which_answer(self, words: list[str], text: str, q: _Question) -> Reply | None:
        """The answer to "which one?": all of them, an ordinal or one named. `words` is the
        stripped answer, `text` the raw one (a "?" makes any answer a question)."""
        if has_question_mark(text):
            return None
        if _all(words):
            return self._forgot(q.options, q.language)
        index = _ordinal(words, len(q.options))
        if index is not None:
            return self._forgot([q.options[index]], q.language)
        if len(words) <= NAMED_ANSWER_WORDS:
            # "quella di Torino": every word in one of them, as a whole word
            allowed = NAMED_ANSWER_CONNECTORS
            if words[0] in NAMED_ANSWER_LEADS:
                allowed = allowed | NAMED_ANSWER_AFTER_LEAD
            said = [w for w in words if w not in allowed]
            whole = [s for s in rank(" ".join(said), q.options) if s.exact == 1.0]
            if len(whole) == 1 and _holds(whole[0].memory, said):
                return self._forgot([whole[0].memory], q.language)
        return None

    def _keep_rule(self, q: _Question) -> Reply:
        return self._remember(Intent("rule", q.rule, q.language), addressed=True)

    # -- reading back --

    def _recall(self, intent: Intent) -> Reply | None:
        """What JenJo said about the subject; None when nothing matches: Claude answers then,
        as it sees the memories and knows what words mean, not only how they are spelt."""
        found = [s for s in rank(intent.text, self.store.items()) if s.coverage >= RECALL_MIN]
        if not found:
            return None
        shown = [s.memory for s in found[:RECALL_MAX]]
        if len(shown) == 1:
            self._talk(shown[0])
        items = _spoken(shown, intent.language)
        return Reply(SAY["recall"].format(items=items), intent.language, "recall")

    def _talk(self, memory: Memory) -> None:
        self._talked = (memory, self.store.clock())

    def _recent(self) -> tuple[Memory | None, bool]:
        """The memory talked about in the last TALKED_TTL_S and since the last request that
        went elsewhere: read out, saved, found or asked about by JenJo, or saved by Claude
        with its tool. (None, True) when the last thing talked about was refused."""
        now = self.store.clock()
        candidates: list[tuple[Memory | None, float]] = []
        if self._talked is not None and now - self._talked[1] <= TALKED_TTL_S:
            candidates.append(self._talked)
        last = self.store.last_added()
        if last is not None and now - last.created <= TALKED_TTL_S:
            candidates.append((last, last.created))
        candidates = [c for c in candidates if c[1] > self._moved]
        if not candidates:
            return None, False
        memory = max(candidates, key=lambda c: c[1])[0]
        if memory is None:
            return None, True
        alive = {m.id for m in self.store.items()}
        return (memory, False) if memory.id in alive else (None, False)


def _answer_words(text: str) -> list[str]:
    """The words of an answer without the name, the openers ("ok", "allora") and the
    courtesy ("grazie", "per favore")."""
    words = words_without_name(normalize(text, drop_fillers=False).split())
    while words and words[0] in _OPENER_WORDS:
        words = words[1:]
    while words and words[-1] in _COURTESY_WORDS:
        words = words[:-1]
    return words


def _all(words: Sequence[str]) -> bool:
    """ "tutti e due", "entrambe", "both": nothing else ("chiudi tutte le finestre" is a
    request, not an answer)."""
    rest = [w for w in words if w not in _ORDINAL_FILLER and w not in _ALL_FILLER]
    return bool(rest) and all(w in _ALL for w in rest)


def _ordinal(words: Sequence[str], count: int) -> int | None:
    """ "la seconda", "il primo", "the second one", "l'ultima": the answer is the ordinal
    and nothing else ("alza il volume di due" is not an answer)."""
    rest = [w for w in words if w not in _ORDINAL_FILLER]
    if len(rest) != 1 or rest[0] not in _ORDINALS:
        return None
    index = _ORDINALS[rest[0]]
    index = count - 1 if index < 0 else index
    return index if index < count else None


def _without_question(q: _Question, words: Sequence[str]) -> list[str]:
    """At the rule question its own phrases are no part of the answer ("sì, tienila come
    regola")."""
    if q.kind != "rule":
        return list(words)
    return RULE_QUESTION_PHRASES.sub(" ", " ".join(words)).split()


def _holds(memory: Memory, said: Sequence[str]) -> bool:
    """Every word of `said` is in `memory`, stopwords too: `rank` does not look at those."""
    own = normalize(memory.text, drop_fillers=False).split()
    return all(any(same_word(word, w) for w in own) for word in said)


def _echo(text: str, q: _Question) -> bool:
    """Claudio's own question heard back: mostly its words, among them the question's own
    ("quale", "dimentico", "regola"), which an answer does not repeat."""
    key = {"which": "which", "forget": "forget_one", "rule": "keep_rule"}[q.kind]
    own = set(normalize(re.sub(r"\{\w+\}", " ", SAY[key]), drop_fillers=False).split())
    own -= STOPWORDS
    heard = set(normalize(text, drop_fillers=False).split())
    return bool(heard & own) and word_share(text, q.said) >= WORD_SHARE


def _as_said(memory: Memory, language: str) -> str:
    """A rule is read with its "da ora in poi": "chiamami capo" alone sounds like an order."""
    return (
        SAY["ruled"].format(text=memory.text) if memory.kind == "rule" else memory.text
    )


def _spoken(memories: Sequence[Memory], language: str) -> str:
    return "; ".join(_as_said(m, language) for m in memories)
