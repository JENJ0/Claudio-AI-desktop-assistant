"""The responder: local commands first (free), then `fallback` (Claude, Phase 2) for what no
command understands."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from claudio.config import (
    DEFAULT_LANGUAGE,
    ERROR_REPLY,
    UNKNOWN_REPLY,
    GuessSettings,
    in_language,
)
from claudio.echo import WORD_SHARE, word_share
from claudio.intake import Doubt, Guess, Hearing, Recovery, Verdict, asks_something, verdict
from claudio.router.actions import ActionRunner
from claudio.router.matcher import Command, CommandMatcher, Match, Near
from claudio.router.normalizer import extract_number, normalize, strip_wake_words
from claudio.safety.confirmation import (
    CANCELLED,
    LEVEL_CONFIRM,
    LEVEL_SAFE,
    NO_WORDS,
    NOT_UNDERSTOOD,
    YES_WORDS,
    Answer,
    Pending,
    parse_answer,
    question,
)
from claudio.system import ON
from claudio.wakeword.verifier import after_clipped_name

log = logging.getLogger(__name__)

ANSWER_WINDOW_S = 45.0  # then the question is old: a phrase without the name is not its answer

# Asked with (request, reply language): Claude's reply, streamed.
Fallback = Callable[[str, str], AsyncGenerator[str, None]]
# Claude's guess of what JenJo said (brain/guess.py), given the closest commands.
GuessCall = Callable[[Doubt, Sequence[Near]], Awaitable[Guess | None]]

# Claude guessed the request without being sure: Claudio asks before answering it.
DID_YOU_SAY = {"it": "Hai detto: {text}?", "en": "Did you say: {text}?"}
NOT_THAT = {"it": "Va bene, lascio stare.", "en": "Okay, never mind."}


@dataclass(frozen=True)
class Proposal:
    """A guessed request waiting for JenJo's yes ("Hai detto: alza il volume?")."""

    text: str
    language: str
    question: str  # as said: its echo is not an answer


class LocalResponder:
    def __init__(
        self,
        matcher: CommandMatcher,
        actions: ActionRunner,
        unknown_reply: dict[str, str] | None = None,
        error_reply: dict[str, str] | None = None,
        voiced: Collection[str] | None = None,
        fallback: Fallback | None = None,
        fallback_info: Callable[[], dict[str, Any]] | None = None,
        on_prepare: Callable[[], None] | None = None,
        fallback_awaiting: Callable[[], bool] | None = None,
        fallback_cancel: Callable[[], None] | None = None,
        fallback_heard: Callable[[Hearing], None] | None = None,
        fallback_guess: GuessCall | None = None,
        fallback_guess_info: Callable[[], dict[str, Any]] | None = None,
        guess: GuessSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.matcher = matcher
        self.actions = actions
        self.unknown_reply = unknown_reply or UNKNOWN_REPLY
        self.error_reply = error_reply or ERROR_REPLY
        self.voiced = voiced  # languages with their own voice (None: all of them)
        self.fallback = fallback  # None: "Questo non lo so ancora fare"
        self.fallback_info = fallback_info  # details of the last fallback call, for the metrics
        self.on_prepare = on_prepare  # a request is coming: get ready (Claude's connection)
        # Claude asked "Confermi?" before a PC action: the next phrase is its answer.
        self.fallback_awaiting = fallback_awaiting
        self.fallback_cancel = fallback_cancel
        self.fallback_heard = fallback_heard  # how sure Whisper was, for Claude's next request
        # A phrase about to be dropped that sounds like a command: Claude guesses what JenJo
        # said (None: dropped as always). `guess` holds the thresholds.
        self.fallback_guess = fallback_guess
        self.fallback_guess_info = fallback_guess_info
        self.guess_settings = guess or GuessSettings()
        self._proposal: Proposal | None = None  # "Hai detto: ...?" waiting for its yes
        self.last_route: dict[str, Any] = {}
        self.language = DEFAULT_LANGUAGE  # of the reply being spoken
        self._pending: Pending | None = None  # a level 2-3 action waiting for its yes
        self._last_reply: tuple[str, str] | None = None  # (text, language) for "ripeti"
        self.clock = clock
        self._last_reply_at = -float("inf")

    def prepare(self) -> None:
        """JenJo started speaking to Claudio: whatever the request needs can get ready now.
        Never raises: a slow or failed preparation only loses the head start."""
        system = self.actions.system
        if self.on_prepare is None or system.paused or system.off:
            return
        try:
            self.on_prepare()
        except Exception:
            log.exception("Preparazione della risposta non riuscita")

    def understands(self, text: str, language: str = DEFAULT_LANGUAGE) -> bool:
        """A local command matches `text` (the assistant retries in Italian when not)."""
        return self.matcher.match(text, language) is not None

    def is_stop(self, text: str) -> bool:
        """`text` is only "basta" / "stop" (in any language): said while CLAUDIO speaks, it
        interrupts. Repeated words count once ("basta, basta")."""
        words = normalize(text, drop_fillers=False).split()
        deduped = " ".join(w for i, w in enumerate(words) if i == 0 or w != words[i - 1])
        for candidate in {text, deduped}:
            for language in self.matcher.languages:
                match = self.matcher.match(candidate, language)
                if match is not None and match.command.action == "stop":
                    return True
        return False

    @property
    def awaiting(self) -> bool:
        """True after a confirmation question: the assistant listens again, no wake word."""
        return self._pending is not None or self._proposal is not None or self._fallback_waits()

    def _fallback_waits(self) -> bool:
        return (
            self.fallback is not None
            and self.fallback_awaiting is not None
            and self.fallback_awaiting()
        )

    @property
    def asked_question(self) -> bool:
        """Claude's last reply ended with a question, maybe followed by a short sentence
        ("Che tema vuoi? Dimmelo e la scrivo."): the next phrase is its answer."""
        return (
            self.last_route.get("route") == "claude"
            and self._last_reply is not None
            and asks_something(self._last_reply[0])
        )

    @property
    def answer_expected(self) -> bool:
        """The question was asked less than ANSWER_WINDOW_S ago (counted from the end of the
        reply's text): "Confermo." after the conversation window closed is still its answer."""
        return self.asked_question and self.clock() - self._last_reply_at <= ANSWER_WINDOW_S

    def heard(self, hearing: Hearing) -> None:
        """How sure Whisper was of the request about to be answered (Claude is told)."""
        if self.fallback_heard is not None:
            self.fallback_heard(hearing)

    async def guess(self, doubt: Doubt) -> Recovery | None:
        """A phrase about to be dropped: when it, or one of Whisper's other guesses, sounds
        like a command ([guess] near_floor), Claude says which command JenJo most likely
        asked and how likely. ASK also arms the question ("Hai detto: ...?"): the next
        phrase is its answer. None when nothing sounds close (no call, no cost), when the
        voice was far from the microphone (never acted on: no call either), when Claude is
        not there or when Claudio is paused or off (only "svegliati" / "accenditi" are heard
        then, for free)."""
        settings = self.guess_settings
        system = self.actions.system
        if (
            self.fallback_guess is None
            or not settings.enabled
            or doubt.close is False
            or system.paused
            or system.off
            or self.awaiting
        ):
            return None
        texts = [doubt.text, *doubt.alternatives]
        texts += [rest for t in texts if (rest := after_clipped_name(t))]  # "Audio, alsa..."
        near = self.matcher.nearest(texts, doubt.language, settings.candidates)
        if not near or near[0].score < settings.near_floor:
            return None
        guess = await self.fallback_guess(doubt, near)
        info: dict[str, Any] = dict(self.fallback_guess_info()) if self.fallback_guess_info else {}
        info["guess_near"] = f"{near[0].command.id} {near[0].score:.0f}"
        if guess is None:
            return Recovery(Verdict.DROP, "", info)
        command = self._guessed_command(guess)
        said = self._request_for(guess, command)
        # Only a command is recovered: any other phrase stays dropped, as before.
        result = (
            Verdict.DROP
            if command is None
            else verdict(guess, doubt, settings.act_at, settings.ask_at, command.action)
        )
        info.update(guess=said, guess_p=round(guess.probability, 2), guess_verdict=result.value)
        log.info(
            "Frase incerta %r: forse %r (%d%%, %s) -> %s",
            doubt.text,
            said,
            round(guess.probability * 100),
            guess.why or "-",
            result.value,
            extra={"route": "guess"},
        )
        if result is Verdict.ASK:
            return Recovery(result, said, info, self.propose(said, doubt.language))
        return Recovery(result, said, info)

    def _guessed_command(self, guess: Guess) -> Command | None:
        """The local command Claude named, or the one its words match ("" said: none)."""
        if guess.command:
            return self.matcher.command(guess.command)
        match = self.matcher.match(guess.said) if guess.said.strip() else None
        return match.command if match else None

    def _request_for(self, guess: Guess, command: Command | None) -> str:
        """What to answer: Claude's words, or (when they do not match the command it named,
        e.g. an exact-only one said differently) that command's own phrase, number kept."""
        said = guess.said.strip()
        if command is None:
            return said
        match = self.matcher.match(said)
        if match is not None and match.command.id == command.id:
            return said
        _, number = extract_number(normalize(said))
        phrase = command.phrases[0]
        return f"{phrase} {number}" if number is not None else phrase

    def propose(self, text: str, language: str) -> str:
        """Claude guessed `text` without being sure: Claudio asks first. The next phrase is
        the answer (awaiting): a yes answers `text` with no second "Confermi?" for a level 2
        command; "no, abbassa il volume" is a new request."""
        question = in_language(DID_YOU_SAY, language).format(text=text.strip().rstrip(".?!"))
        self._proposal = Proposal(text, language, question)
        self.language = language
        self.last_route = {"route": "guess", "guess": text, "language": language}
        log.info("Chiedo conferma della richiesta dedotta: %r", text, extra={"route": "guess"})
        return question

    def cancel_pending(self, silence: bool = False) -> None:
        """Nothing was said after the question (or it was not understood as speech). After
        mere `silence` an action Claude proposed stays armed until it expires: JenJo may
        answer later with the wake word ("Hey Claudio, confermo")."""
        if self._pending is not None:
            log.info(
                "Conferma scaduta: %s non eseguito",
                self._pending.match.command.id,
                extra={"route": "confirm_timeout"},
            )
            self._pending = None
        if self._proposal is not None:
            log.info("Nessuna risposta: %r non eseguito", self._proposal.text)
            self._proposal = None
        if self.fallback_cancel is not None and not silence:
            self.fallback_cancel()

    def _reply_language(self, language: str, has_text: bool) -> str:
        """`language` when there is a reply and a voice for it, else Italian: the text and
        the voice reading it always agree."""
        voiced = self.voiced is None or language in self.voiced
        return language if has_text and voiced else DEFAULT_LANGUAGE

    async def respond(self, text: str, language: str = DEFAULT_LANGUAGE) -> AsyncIterator[str]:
        """Reply in the language of the command's phrase (`language` when none matched):
        "open chrome" gets an English reply even if Whisper thought it was Italian.
        Right after a confirmation question, `text` is its answer. The reply is kept for
        "ripeti"."""
        parts: list[str] = []
        async for chunk in self._respond(text, language):
            parts.append(chunk)
            yield chunk
        reply = "".join(parts).strip()
        if reply and self.last_route.get("command") != "repeat":
            self._last_reply = (reply, self.language)
            self._last_reply_at = self.clock()

    async def _respond(
        self, text: str, language: str, confirmed: bool = False
    ) -> AsyncIterator[str]:
        """`confirmed`: JenJo already said yes to this very request ("Hai detto: chiudi la
        finestra?" "Sì"): a level 2 command runs without asking again."""
        if self._pending is not None:
            async for chunk in self._answer(text):
                yield chunk
            return
        if self._proposal is not None:
            async for chunk in self._answer_proposal(text, language):
                yield chunk
            return
        if self._fallback_waits():
            # The answer to Claude's "Confermi?" goes to Claude, never to a local command, and
            # is answered in the language (and voice) of the question.
            log.info("Risposta alla conferma di Claude: %r", text, extra={"route": "claude"})
            async for chunk in self._ask_fallback(text):
                yield chunk
            return
        match = self.matcher.match(text, language)
        if match:
            self.language = self._reply_language(
                match.language, match.command.has_reply_in(match.language)
            )
        else:
            self.language = self._reply_language(language, bool(self.unknown_reply.get(language)))
        if self.actions.system.off and not _turns_on(match):
            self.last_route = {"route": "off"}
            log.info("Spento, ignoro: %r", text, extra={"route": "off"})
            return  # off: not even a beep, only "accenditi" is answered
        if self.actions.system.paused and not _is_system(match):
            self.last_route = {"route": "paused"}
            log.info("In pausa, ignoro: %r", text, extra={"route": "paused"})
            return  # nothing to say: the assistant answers with a low beep
        if match is None and self.fallback is not None:
            self.language = self._reply_language(language, True)
            log.info("Non è un comando, chiedo a Claude: %r", text, extra={"route": "claude"})
            async for chunk in self._ask_fallback(text):
                yield chunk
            return
        if match is None:
            self.last_route = {"route": "unknown", "language": self.language}
            log.info("Richiesta non riconosciuta: %r", text, extra={"route": "unknown"})
            yield in_language(self.unknown_reply, self.language)
            return
        self.last_route = {
            "route": "local",
            "command": match.command.id,
            "score": round(match.score, 1),
            "language": self.language,
        }
        if match.command.action == "repeat":
            async for chunk in self._repeat(match):
                yield chunk
            return
        if match.command.action == "stop":
            log.info("Basta: fine della conversazione", extra={"route": "local"})
            return  # nothing to say: the assistant beeps and stops listening
        if match.command.level > LEVEL_SAFE and not (
            confirmed and match.command.level <= LEVEL_CONFIRM
        ):
            self._pending = Pending(match, self.language)
            self.last_route.update(route="confirm", level=match.command.level)
            log.info(
                "Conferma richiesta (livello %d): %s",
                match.command.level,
                match.command.id,
                extra={"route": "confirm", "level": match.command.level},
            )
            yield question(match, self.language)
            return
        async for chunk in self._execute(match):
            yield chunk

    async def _answer_proposal(self, text: str, language: str) -> AsyncIterator[str]:
        """The answer to "Hai detto: ...?". A yes (also "sì, grazie") answers the guessed
        request; "no" alone drops it; anything else ("no, abbassa il volume", "apri Chrome")
        is what JenJo wants instead, answered as a new request."""
        proposal, self._proposal = self._proposal, None
        assert proposal is not None
        answer = parse_answer(text)
        words = strip_wake_words(normalize(text, drop_fillers=False).split())
        if answer is Answer.UNCLEAR and words and words[0] in YES_WORDS:
            answer = Answer.YES  # "sì, grazie mille": a yes with something after it
        if answer is Answer.UNCLEAR and word_share(text, proposal.question) >= WORD_SHARE:
            # The question's own echo ("...detto spegniti"), never a yes: JenJo says "sì".
            self.last_route = {"route": "guess_echo", "guess": proposal.text}
            log.info("La mia domanda nel microfono, non è una risposta: %r", text)
            return
        if answer is Answer.YES:
            log.info("Sì: rispondo a %r", proposal.text, extra={"route": "guess"})
            async for chunk in self._respond(proposal.text, proposal.language, confirmed=True):
                yield chunk
            return
        rest = [w for w in words if w not in NO_WORDS] if answer is Answer.NO else words
        if len(rest) >= 2 or (rest and answer is Answer.UNCLEAR):
            again = " ".join(rest) if answer is Answer.NO else text
            log.info("Non era %r: nuova richiesta %r", proposal.text, again)
            async for chunk in self._respond(again, language):
                yield chunk
            return
        self.language = proposal.language
        self.last_route = {"route": "guess_refused", "guess": proposal.text}
        log.info("Richiesta dedotta rifiutata: %r", proposal.text, extra={"route": "guess"})
        yield in_language(NOT_THAT if answer is Answer.NO else NOT_UNDERSTOOD, self.language)

    async def _ask_fallback(self, text: str) -> AsyncIterator[str]:
        assert self.fallback is not None
        self.last_route = {"route": "claude", "language": self.language}
        try:
            async with contextlib.aclosing(self.fallback(text, self.language)) as reply:
                async for chunk in reply:
                    yield chunk
        finally:
            if self.fallback_info is not None:
                self.last_route.update(self.fallback_info())

    async def _repeat(self, match: Match) -> AsyncIterator[str]:
        if self._last_reply is None:
            yield str(match.command.param_in("empty", self.language) or "")
            return
        text, self.language = self._last_reply  # same words, same voice
        self.last_route["language"] = self.language
        log.info("Ripeto: %r", text, extra={"route": "local"})
        yield text

    async def _answer(self, text: str) -> AsyncIterator[str]:
        pending, self._pending = self._pending, None
        assert pending is not None
        match, level = pending.match, pending.level
        self.language = pending.language  # the answer is in the language of the question
        answer = parse_answer(text, level)
        self.last_route = {
            "route": "confirmed" if answer is Answer.YES else "refused",
            "command": match.command.id,
            "level": level,
            "answer": answer.value,
            "language": self.language,
        }
        if answer is Answer.YES:
            log.info("Conferma data (livello %d): eseguo %s", level, match.command.id)
            async for chunk in self._execute(match):
                yield chunk
            return
        if answer is Answer.NO:
            log.info("Conferma rifiutata (livello %d): %s non eseguito", level, match.command.id)
            yield in_language(CANCELLED, self.language)
        else:
            log.info("Conferma non chiara (livello %d): %s non eseguito", level, match.command.id)
            yield in_language(NOT_UNDERSTOOD, self.language)

    async def _execute(self, match: Match) -> AsyncIterator[str]:
        try:
            reply = self.actions.execute(match, self.language)
        except Exception:
            log.exception("Errore eseguendo %s", match.command.id)
            self.last_route["error"] = True
            self.language = self._reply_language(
                self.language, bool(self.error_reply.get(self.language))
            )
            yield in_language(self.error_reply, self.language)
            return
        if reply:
            yield reply


def _is_system(match: Match | None) -> bool:
    return match is not None and match.command.action == "system"


def _turns_on(match: Match | None) -> bool:
    """ "Accenditi": the one command heard while CLAUDIO is off."""
    return _is_system(match) and match is not None and match.command.params.get("do") == ON
