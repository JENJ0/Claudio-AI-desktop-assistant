"""What to do with a transcribed phrase: answer it, listen again, or drop it (and why).

The Assistant is the audio state machine; this module holds the rules that decide whether a
phrase is a request. They are pure (no audio, no asyncio): the Assistant runs Whisper when a
rule asks for it (the retry, the closed list, the name before the request) and calls decide().

Drops cost nothing: a false wake ends in silence, the TV in the conversation window or noise
heard as words ends with the low beep, and none of them reaches Claude."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from claudio.config import NoiseSettings
from claudio.protocols import Responder
from claudio.router.normalizer import heard_wake_word, normalize, only_wake_words
from claudio.wakeword.verifier import wake_confirmed

CONVERSATION = "conversation"  # the wake_word of a phrase heard in the conversation window
# The wake_word of a request said over CLAUDIO's reply: close to the microphone and already
# heard as words by Whisper, so it is a request without the name.
BARGE_IN = "barge_in"
# An answer to Claudio's own question ("Confermi?", "quale delle due?") is a few words.
ANSWER_MAX_WORDS = 4


class Segment(NamedTuple):
    """Whisper's confidence in one segment of a transcription."""

    no_speech_prob: float
    avg_logprob: float


class Hearing(NamedTuple):
    """How sure Whisper was of a request that goes to Claude: Claude is told when it was
    unsure, with the other transcriptions it considered, to work out what JenJo said."""

    unsure: bool = False
    alternatives: tuple[str, ...] = ()


def unsure(segments: Sequence[Segment], below_logprob: float) -> bool:
    """Whisper doubted at least one segment. On JenJo's recordings (25/10) the misheard
    requests scored -0.25..-1.1, the right ones -0.01..-0.17."""
    return any(s.avg_logprob < below_logprob for s in segments)


class Kind(enum.StrEnum):
    """The values are the outcomes written in metrics.jsonl."""

    ANSWER = "ok"
    RELISTEN = "relisten"  # only the wake word: the request is still to come
    EMPTY = "empty"  # nothing (or only the name) was heard
    FALSE_WAKE = "false_wake"  # woke on something that only starts like the name
    NOT_REQUEST = "conversation_end"  # the TV or someone else, in the conversation window
    NOISE = "noise"  # noise transcribed as words: not worth a call to Claude
    CONTINUE = "continue"  # left halfway ("apri Chrome e cerca..."): listen for the rest


@dataclass(frozen=True)
class Decision:
    kind: Kind
    text: str  # what the responder gets
    reason: str = ""  # why a phrase is dropped, or why its text was changed

    @property
    def dropped(self) -> bool:
        return self.kind not in (Kind.ANSWER, Kind.RELISTEN, Kind.CONTINUE)


@dataclass(frozen=True)
class Heard:
    """What is known about a phrase besides its text."""

    wake_word: str  # the wake word model, CONVERSATION, BARGE_IN, "follow_up" or "typed"
    relistened: bool = False  # the request after a phrase that was only the wake word
    second_chance: bool = False  # Whisper already heard the name in a near miss
    name_heard: bool = False  # Whisper heard the name in the wake word audio itself
    speech_dbfs: float | None = None  # loudest 32 ms of speech; None: not measured
    segments: Sequence[Segment] = ()  # empty: Whisper's confidence is not known
    continued: bool = False  # already waited once for the rest of a phrase left halfway


@dataclass
class RequestFilter:
    """The rules, with their settings. The Assistant exposes the settings as its own
    attributes (tests and __main__ set them there)."""

    verify_wake: bool = False
    # None: Claude is off, the conversation window keeps only the local commands.
    claude_min_words: int | None = None
    only_addressed: bool = True
    closed_list_max_words: int = 0  # 0: no limit
    retry_language: str | None = None
    noise: NoiseSettings | None = None  # None: noise is never judged
    continue_incomplete: bool = False  # a phrase left halfway waits for its end

    # -- before decide(): is another Whisper pass worth it? --

    def wants_retry(self, text: str, heard: str, responder: Responder) -> bool:
        """Transcribe again in retry_language: heard in another one and not understood."""
        return (
            self.retry_language is not None
            and heard != self.retry_language
            and bool(text)
            and not only_wake_words(text)
            and not responder.awaiting  # a yes/no: no command to match
            and not responder.understands(text, heard)
        )

    def wants_closed_list(self, text: str, heard: str, responder: Responder) -> bool:
        """Ask Whisper for its n-best: not understood, short and not a question. Whisper
        writes the "?" only on questions, and those are for Claude (~100 ms saved)."""
        return (
            bool(text)
            and not only_wake_words(text)
            and not text.rstrip().endswith("?")
            and not responder.awaiting
            and not (self.closed_list_max_words and words(text) > self.closed_list_max_words)
            and not responder.understands(text, heard)
        )

    # -- the decision --

    def decide(self, text: str, language: str, heard: Heard, responder: Responder) -> Decision:
        """A FALSE_WAKE may still be a request said after a pause: the Assistant lets Whisper
        listen to the wake word itself and, if the name is there, decides again with
        name_heard."""
        if only_wake_words(text) and not heard.relistened:
            return Decision(Kind.RELISTEN, text)
        if not text or only_wake_words(text):
            return Decision(Kind.EMPTY, text)
        if self._false_wake(text, language, heard, responder):
            return Decision(Kind.FALSE_WAKE, text, "name_not_heard")
        if responder.awaiting:
            return self._answer(text, heard)
        if heard.wake_word == CONVERSATION and not self.is_request(text, language, responder):
            return Decision(Kind.NOT_REQUEST, text, "not_a_request")
        if (cause := self.noise_cause(text, heard)) and not responder.understands(text, language):
            return Decision(Kind.NOISE, text, cause)
        if (
            self.continue_incomplete
            and not heard.continued
            and incomplete(text)
            and not responder.understands(text, language)
        ):
            return Decision(Kind.CONTINUE, text, "incomplete")
        return Decision(Kind.ANSWER, text)

    def is_request(self, text: str, language: str, responder: Responder) -> bool:
        """Heard in the conversation window: a command, or (with Claude) a phrase that sounds
        addressed to Claudio, short ones too when they say the name. The TV's "e poi gli ho
        detto di no" is not: it costs nothing."""
        if responder.understands(text, language):
            return True
        if self.claude_min_words is not None and responder.asked_question:
            return True  # Claude asked "da dove parti?": "Milano." is the answer
        if self.claude_min_words is None:
            return False
        if words(text) < self.claude_min_words:
            # "Allora scrivi." (23/10) and "Claudio QDT." (26/10): dropped, now answered.
            return short_request(text) or heard_wake_word(text)
        return not self.only_addressed or addressed_to_claudio(text)

    def noise_cause(self, text: str, heard: Heard) -> str:
        """ "quiet" (nobody near the microphone), "unsure" (a short phrase Whisper was unsure
        of from start to end) or "" (speech). Without figures nothing is judged."""
        noise = self.noise
        if noise is None or not noise.enabled:
            return ""
        if heard.speech_dbfs is not None and heard.speech_dbfs < noise.min_speech_dbfs:
            return "quiet"
        unsure = bool(heard.segments) and all(
            s.no_speech_prob > noise.no_speech_prob or s.avg_logprob < noise.avg_logprob
            for s in heard.segments
        )
        return "unsure" if unsure and words(text) <= noise.max_words else ""

    def _answer(self, text: str, heard: Heard) -> Decision:
        """The answer to "Confermi?": one said far from the microphone (the TV's "sì") is
        passed on as nothing, so the action is not run ("Non ho capito: non lo faccio.")."""
        noise = self.noise
        if (
            noise is not None
            and noise.enabled
            and heard.speech_dbfs is not None
            and heard.speech_dbfs < noise.answer_min_dbfs
        ):
            return Decision(Kind.ANSWER, "", "quiet_answer")
        return Decision(Kind.ANSWER, text)

    def _false_wake(self, text: str, language: str, heard: Heard, responder: Responder) -> bool:
        """Only a real wake word detection is judged: not the conversation window, not a
        yes/no answer, not a second chance (Whisper already heard the name), not the request
        after a lone wake word (the name was in the phrase before) and not a short answer to
        a question Claude just asked (26/10: "Confermo." said after the window had closed
        woke the detector and was thrown away)."""
        if (
            not self.verify_wake
            or heard.name_heard
            or heard.wake_word in (CONVERSATION, BARGE_IN)
            or responder.awaiting
            or heard.second_chance
            or heard.relistened
            or (responder.answer_expected and words(text) <= ANSWER_MAX_WORDS)
        ):
            return False
        return not wake_confirmed(text, responder.understands(text, language))


# How a follow-up request to Claudio usually starts, when it is not a question.
REQUEST_STARTS = frozenset(
    [
        "dimmi",
        "spiegami",
        "raccontami",
        "descrivimi",
        "dammi",
        "fammi",
        "consigliami",
        "suggeriscimi",
        "traduci",
        "calcola",
        "elenca",
        "riassumi",
        "continua",
        "aiutami",
        "scrivi",
        "scrivila",
        "scrivilo",
        "scrivimi",
        "apri",
        "aprila",
        "aprilo",
        "chiudi",
        "cerca",
        "trova",
        "leggi",
        "leggimi",
        "mostrami",
        "fallo",
        "procedi",
        "perche",
        "come",
        "cosa",
        "chi",
        "dove",
        "quando",
        "quale",
        "quanto",
        "tell",
        "explain",
        "describe",
        "give",
        "show",
        "recommend",
        "suggest",
        "translate",
        "calculate",
        "list",
        "summarize",
        "continue",
        "help",
        "write",
        "open",
        "close",
        "find",
        "read",
        "look",
        "why",
        "how",
        "what",
        "who",
        "where",
        "when",
        "which",
    ]
)


def addressed_to_claudio(text: str) -> bool:
    """A question (Whisper writes the "?"), a request verb at the start ("dimmi di più") or
    the name: worth asking Claude without the wake word."""
    tokens = normalize(text, drop_fillers=False).split()
    return (
        text.rstrip().endswith("?")
        or (bool(tokens) and tokens[0] in REQUEST_STARTS)
        or heard_wake_word(text)
    )


def short_request(text: str) -> bool:
    """Two words with a request verb after the fillers: "allora scrivi", "aprila pure"."""
    tokens = normalize(text, drop_fillers=True).split()
    return len(tokens) >= 1 and words(text) >= 2 and tokens[0] in REQUEST_STARTS


def words(text: str) -> int:
    return len(normalize(text, drop_fillers=False).split())


# A phrase ending on one of these is not finished: "apri Chrome e cerca", "metti la".
DANGLING_WORDS = frozenset(
    {"e", "o", "oppure", "ma", "poi", "di", "del", "della", "dei", "delle", "a", "al", "alla"}
    | {"da", "dal", "dalla", "in", "nel", "nella", "con", "su", "sul", "sulla", "per", "tra"}
    | {"fra", "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "che", "cerca", "apri"}
    | {"scrivi", "digita", "metti", "chiama", "and", "or", "the", "to", "of", "for", "with"}
    | {"search", "open", "type"}
)
TRAILING_DOTS = ("...", "…")


def incomplete(text: str) -> bool:
    """Whisper's "..." where the voice trailed off, or a last word that needs more after it:
    on 26/10 "Apri Chrome e cerca..." (twice) and "Dimmi..." were cut by a pause while JenJo
    thought of the words. A question ("?") is complete."""
    stripped = text.rstrip()
    if stripped.endswith(TRAILING_DOTS):
        return True
    if stripped.endswith("?"):
        return False
    tokens = normalize(text, drop_fillers=False).split()
    return len(tokens) >= 2 and tokens[-1] in DANGLING_WORDS


# Sounds Whisper writes for a cough, a breath or the TV ("Mm-hmm.", "Ehm"): not words.
INTERJECTIONS = frozenset(
    {"mm", "mmm", "mmh", "mh", "mhm", "hm", "hmm", "hmmm", "uh", "huh", "um", "uhm", "ehm"}
    | {"eh", "ah", "oh", "uhh", "ahh", "ohh"}
)


def content_words(text: str) -> int:
    """The words in `text` that are not interjections: "Mm-hmm." has none."""
    return sum(w not in INTERJECTIONS for w in normalize(text, drop_fillers=False).split())


# CLAUDIO's own voice heard again through the speakers, as Whisper writes it: rarely the same
# words ("Ciao, campione" -> "ciao cappiore", "dell'universo è in linea" -> "dell'università e
# lì", "il grande mistero" -> "che grande vista": logs of 26/10).
ECHO_SHARE = 0.6  # this share of the heard words sounds like words of the reply
ECHO_WORD = 75  # rapidfuzz ratio of a heard word to a reply word: the same word, misheard
ECHO_PHRASE = 77  # partial_ratio of three or more heard words to the reply: a piece of it
ECHO_MIN_CHARS = 4  # shorter words ("e", "il", "lì") are in every reply: they say nothing


def echo_of(text: str, reply: str) -> bool:
    """`text`, heard while the reply was spoken, is the reply itself coming back into the
    microphone: most of its words sound like the reply's, or it is a misheard piece of it.
    The name is JenJo talking to CLAUDIO, unless the reply says it too."""
    from rapidfuzz import fuzz

    heard = normalize(text, drop_fillers=False).split()
    said = normalize(reply, drop_fillers=False).split()
    if not heard or not said:
        return False
    if heard_wake_word(text) and not heard_wake_word(reply):
        return False
    vocabulary = set(said)
    long_words = [w for w in vocabulary if len(w) >= ECHO_MIN_CHARS]
    checked = [w for w in heard if len(w) >= ECHO_MIN_CHARS] or heard
    alike = sum(
        w in vocabulary or any(fuzz.ratio(w, r) >= ECHO_WORD for r in long_words) for w in checked
    )
    if alike / len(checked) >= ECHO_SHARE:
        return True
    return len(heard) >= 3 and fuzz.partial_ratio(" ".join(heard), " ".join(said)) >= ECHO_PHRASE
