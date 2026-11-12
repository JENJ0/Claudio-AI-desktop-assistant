"""The responder: local commands first (free), then `fallback` (Claude, Phase 2) for what no
command understands."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from claudio.config import (
    DEFAULT_LANGUAGE,
    ERROR_REPLY,
    UNKNOWN_REPLY,
    GuessSettings,
)
from claudio.echo import WORD_SHARE, word_share
from claudio.intake import (
    NEVER_ASKED,
    PRECALL,
    Doubt,
    Guess,
    Hearing,
    Recovery,
    Verdict,
    asks_something,
    verdict,
)
from claudio.router.actions import ActionRunner
from claudio.router.matcher import Command, CommandMatcher, Match, Near
from claudio.router.normalizer import (
    after_wake_word,
    extract_number,
    heard_wake_word,
    normalize,
    said_in,
    strip_wake_words,
)
from claudio.safety.confirmation import (
    ACTION_TTL_S,
    ASK_AGAIN,
    ASK_AGAIN_EXPLICIT,
    CANCELLED,
    LEVEL_CONFIRM,
    LEVEL_EXPLICIT,
    LEVEL_SAFE,
    NO_WORDS,
    NOT_UNDERSTOOD,
    YES_WORDS,
    Answer,
    Pending,
    has_question_mark,
    parse_answer,
    question,
)
from claudio.system import HIDDEN, ON
from claudio.tts.sentence_splitter import joined_reply
from claudio.wakeword.verifier import after_clipped_name

if TYPE_CHECKING:
    from claudio.memory.voice import MemoryVoice

log = logging.getLogger(__name__)

# The routes whose reply Claude is told about (not the memory, a guess, repeat or stop).
NOTED_ROUTES = frozenset({"local", "confirm", "confirmed", "refused"})
ANSWER_WINDOW_S = 45.0  # then the question is old: a phrase without the name is not its answer

# Asked with (request, reply language): Claude's reply, streamed.
Fallback = Callable[[str, str], AsyncGenerator[str, None]]
# Claude's guess of what JenJo said (brain/guess.py), given the closest commands.
GuessCall = Callable[[Doubt, Sequence[Near]], Awaitable[Guess | None]]

# Claude guessed the request without being sure: Claudio asks before answering it.
DID_YOU_SAY = "Hai detto: {text}?"
NOT_THAT = "Va bene, lascio stare."
MEMORY_ERROR = "Non riesco a usare la memoria in questo momento."


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
        unknown_reply: str | None = None,
        error_reply: str | None = None,
        fallback: Fallback | None = None,
        fallback_info: Callable[[], dict[str, Any]] | None = None,
        on_prepare: Callable[[], None] | None = None,
        fallback_awaiting: Callable[[], bool] | None = None,
        fallback_cancel: Callable[[bool], None] | None = None,
        fallback_repeats: Callable[[str], bool] | None = None,
        fallback_stop: Callable[[], None] | None = None,
        fallback_note: Callable[[str, str], None] | None = None,
        fallback_cut: Callable[[Sequence[str], object | None], object | None] | None = None,
        fallback_dropped: Callable[[object | None], bool] | None = None,
        fallback_heard: Callable[[Hearing], None] | None = None,
        fallback_guess: GuessCall | None = None,
        fallback_guess_info: Callable[[], dict[str, Any]] | None = None,
        guess: GuessSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
        memory: MemoryVoice | None = None,
    ) -> None:
        self.matcher = matcher
        self.actions = actions
        self.unknown_reply = unknown_reply or UNKNOWN_REPLY
        self.error_reply = error_reply or ERROR_REPLY
        self.fallback = fallback  # None: "Questo non lo so ancora fare"
        self.fallback_info = fallback_info  # details of the last fallback call, for the metrics
        self.on_prepare = on_prepare  # a request is coming: get ready (Claude's connection)
        # Claude asked "Confermi?" before a PC action: the next phrase is its answer.
        self.fallback_awaiting = fallback_awaiting
        self.fallback_cancel = fallback_cancel
        self.fallback_repeats = fallback_repeats  # the phrase says Claude's waiting command again
        self.fallback_stop = fallback_stop  # a new request cut Claude's reply: it no longer acts
        # A local command answered: Claude's history gets (request, reply), see NOTED_ROUTES.
        self.fallback_note = fallback_note
        self.fallback_cut = fallback_cut  # Claude's reply was cut: what was heard of it
        self.fallback_dropped = fallback_dropped  # that cut reply's action was stopped
        # "continua" has something to say again (set by the Assistant): else it goes to Claude.
        self.can_resume: Callable[[], bool] | None = None
        self.claude_replied = False  # the reply being spoken came from Claude
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
        # "ricordati che...", "dimentica...", "cosa ricordi di...": free text, understood
        # before any command (None: no long-term memory).
        self.memory = memory
        self._addressed = True  # the next request was said with the wake word (or typed)

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

    def understands(self, text: str, language: str = DEFAULT_LANGUAGE, memory: bool = True) -> bool:
        """A local command matches `text` (the assistant retries in Italian when not), or it
        is a memory request: no closed list or guess may turn "dimentica Anna" into another
        command, and it is a request also in the conversation window. memory=False: commands
        only (words said over a reply need the name to be a memory request). While CLAUDIO
        is off only "accenditi" counts: no other command nor memory request is answered."""
        match = self.matcher.match(text, language)
        if self.actions.system.off:
            return _turns_on(match)
        if memory and self.memory is not None and self.memory.understands(text):
            return True
        return match is not None

    def is_stop(self, text: str) -> bool:
        """`text` is only "basta" / "stop" (in any language), or a phrase to Claudio ending with
        it (the matcher's rule): said while CLAUDIO speaks, it interrupts. Repeated words count
        once ("basta, basta")."""
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
        return (
            self._pending_open()
            or self._proposal is not None
            or self._fallback_waits()
            or (self.memory is not None and self.memory.awaiting)
        )

    @property
    def needs_voice(self) -> bool:
        """An action waits for a spoken yes (never given in the chat window). The memory's
        question ("Quale dimentico?") is not one: it can be answered by typing too."""
        return self._pending_open() or self._proposal is not None or self._fallback_waits()

    def _pending_open(self) -> bool:
        """A local question was asked less than ACTION_TTL_S ago: it still waits for its yes."""
        return self._pending is not None and self.clock() - self._pending.asked < ACTION_TTL_S

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
        """How sure Whisper was of the request about to be answered (Claude is told), and
        whether it was said to Claudio (the memory asks first when it was not)."""
        self._addressed = hearing.addressed
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
        precall = doubt.reason == PRECALL
        if (
            self.fallback_guess is None
            or not settings.enabled
            or (precall and not settings.precall)
            or doubt.close is False
            or system.paused
            or system.off
            or self.awaiting
        ):
            return None
        texts = [doubt.text, *doubt.alternatives]
        texts += [rest for t in texts if (rest := after_clipped_name(t))]  # "Audio, alsa..."
        near = self.matcher.nearest(texts, doubt.language, settings.candidates)
        if not precall and (not near or near[0].score < settings.near_floor):
            return None  # a pre-call is asked whatever the closeness: the request is for Claude
        guess = await self.fallback_guess(doubt, near)
        info: dict[str, Any] = dict(self.fallback_guess_info()) if self.fallback_guess_info else {}
        info["guess_near"] = f"{near[0].command.id} {near[0].score:.0f}" if near else "-"
        if guess is None:
            return Recovery(Verdict.DROP, "", info)
        command = self._guessed_command(guess)
        said = self._request_for(guess, command)
        # Only a command is recovered: any other phrase stays dropped, as before.
        result = (
            Verdict.DROP
            if command is None
            else verdict(guess, doubt, settings.act_at, settings.ask_at, self._action(command))
        )
        if precall:
            said = self._precall_reading(result, command, said, guess.probability)
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

    def _precall_reading(
        self, result: Verdict, command: Command | None, said: str, probability: float
    ) -> str:
        """What a dropped pre-call leaves for the main model: Haiku's own words when no command
        matches them and it is sure enough to ask; nothing otherwise (a command it read but the
        verdict dropped is no reading to hand over)."""
        if result is Verdict.DROP and (
            command is not None or probability < self.guess_settings.ask_at
        ):
            return ""
        return said

    def _action(self, command: Command) -> str:
        """The command's action as the verdict sees it. Right after Claude asked something a
        guessed "basta" is asked like any other command: "Pasta." may be the answer to "Che
        piatto vuoi?", and stopping would throw it away."""
        if self.answer_expected and command.action in NEVER_ASKED:
            return ""
        return command.action

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
        question = DID_YOU_SAY.format(text=quoted(text))
        self.claude_replied = False  # a reply of its own, which never goes through respond()
        self._proposal = Proposal(text, language, question)
        self.language = language
        self.last_route = {"route": "guess", "guess": text, "language": language}
        log.info("Chiedo conferma della richiesta dedotta: %r", text, extra={"route": "guess"})
        return question

    def cancel_pending(
        self, silence: bool = False, cut: bool = False, takeover: bool = False
    ) -> None:
        """Nothing was said after the question (or it was not understood as speech). After
        mere `silence` a question (Claude's or the router's) stays armed until it expires:
        JenJo may answer later with the wake word ("Hey Claudio, confermo"). `takeover`: a
        phrase was said over the reply and answers a question next: it stays (either kind)."""
        if self._pending is not None and not silence and not takeover:
            log.info(
                "Conferma scaduta: %s non eseguito",
                self._pending.match.command.id,
                extra={"route": "confirm_timeout"},
            )
            self._pending = None
        if self._proposal is not None:
            log.info("Nessuna risposta: %r non eseguito", self._proposal.text)
            self._proposal = None
        if self.fallback_cancel is not None and not silence and not takeover:
            self.fallback_cancel(cut)
        if self.memory is not None:
            # "Quale dimentico?" without an answer, even after mere silence: nothing is
            # forgotten, and a phrase much later is never taken for the answer.
            self.memory.cancel()

    def reply_cut(self, played: Sequence[str], handle: object | None = None) -> object | None:
        """The reply was cut after `played`: Claude's history keeps that much of it. Only a reply
        of Claude's, or the one `handle` (from the first cut) names: None for any other. A
        question of the router's whose last sentence was not among `played` is marked unheard."""
        if handle is None and not self.claude_replied:
            self._disarm_unheard(played)
            return None
        if self.fallback_cut is None:
            return None
        return self.fallback_cut(played, handle)

    def _disarm_unheard(self, played: Sequence[str]) -> None:
        """Like Claude's question (rule 6): it stays armed, and a yes or no has it said again."""
        pending = self._pending
        if pending is None or not pending.heard:
            return
        asked = (
            self._ask_again(pending.match, pending.level)
            if pending.asked_again
            else question(pending.match, pending.language)
        )
        if not said_in(asked, " ".join(played)):
            self._pending = replace(pending, heard=False)

    def action_dropped(self, handle: object | None) -> bool:
        """The reply `handle` names was cut with an action that never ran."""
        return self.fallback_dropped is not None and self.fallback_dropped(handle)

    def stop_acting(self) -> None:
        """A new request was said over the reply: Claude's stream goes on, for "Dicevo", but
        it runs nothing more and asks nothing."""
        if self.fallback_stop is not None:
            self.fallback_stop()

    async def respond(
        self, text: str, language: str = DEFAULT_LANGUAGE, named: bool = False
    ) -> AsyncIterator[str]:
        """Right after a confirmation question, `text` is its answer. The reply is kept for
        "ripeti". `named`: the wake audio held the name, though `text` does not."""
        parts: list[str] = []
        self.claude_replied = False
        quiet = self._quiet()
        async for chunk in self._respond(text, language, named=named):
            parts.append(chunk)
            yield chunk
        reply = joined_reply(parts)  # as said: a reply cut at the token limit has no fragment
        if reply and self.last_route.get("command") != "repeat":
            self._last_reply = (reply, self.language)
            self._last_reply_at = self.clock()
            if not quiet and not self._quiet():
                self._note_for_claude(text, reply)

    def _quiet(self) -> bool:
        system = self.actions.system
        return system.off or system.paused

    def _note_for_claude(self, text: str, reply: str) -> None:
        """Claude's next request sees what a local command just answered, or it would not know
        it happened (and could answer "che ore sono?" differently, or repeat the action)."""
        if self.fallback_note is not None and self.last_route.get("route") in NOTED_ROUTES:
            self.fallback_note(text, reply)

    async def _respond(
        self, text: str, language: str, confirmed: bool = False, named: bool = False
    ) -> AsyncIterator[str]:
        """`confirmed`: JenJo already said yes to this very request ("Hai detto: chiudi la
        finestra?" "Sì"): a level 2 command runs without asking again."""
        answer = self._open_question(text, language, named)
        if answer is not None:
            async for chunk in answer:
                yield chunk
            return
        match = self.matcher.match(text, language)
        if match and match.command.action == "resume" and not self._has_rest():
            match = None  # nothing to say again: "continua" is a request like any other
        if self.actions.system.off and not _turns_on(match):
            self.last_route = {"route": "off"}
            log.info("Spento, ignoro: %r", HIDDEN, extra={"route": "off"})
            return  # off: not even a beep, only "accenditi" is answered
        if self.actions.system.paused and not _is_system(match):
            self.last_route = {"route": "paused"}
            log.info("In pausa, ignoro: %r", text, extra={"route": "paused"})
            return  # nothing to say: the assistant answers with a low beep
        addressed, self._addressed = self._addressed, True
        said = self._memory_reply(text, match, addressed)
        if said is not None:
            if said:
                yield said
            return
        if match is None and self.fallback is not None:
            log.info("Non è un comando, chiedo a Claude: %r", text, extra={"route": "claude"})
            async for chunk in self._ask_fallback(text):
                yield chunk
            return
        if match is None:
            self.last_route = {"route": "unknown", "language": self.language}
            log.info("Richiesta non riconosciuta: %r", text, extra={"route": "unknown"})
            yield self.unknown_reply
            return
        async for chunk in self._run_command(match, confirmed):
            yield chunk

    def _has_rest(self) -> bool:
        return self.can_resume is not None and self.can_resume()

    def _open_question(
        self, text: str, language: str, named: bool = False
    ) -> AsyncIterator[str] | None:
        """The answer to a question waiting for one, not started yet (None: no question). Off
        or paused (the tray, with "Confermi?" still waiting), a question is over: "sì" runs
        nothing."""
        if self._quiet():
            self.cancel_pending()
            return None
        if self._pending is not None and not self._pending_open():
            log.info(
                "Domanda scaduta: %s non eseguito",
                self._pending.match.command.id,
                extra={"route": "confirm_timeout"},
            )
            self._pending = None  # the phrase is a normal request
        if self._pending is not None:
            return self._answer(text, language, named)
        if self._proposal is not None:
            return self._answer_proposal(text, language)
        if self._fallback_waits():
            if self._name_over_question(text, named):
                return self._cancel_by_name(text, language, "claude")
            # The answer to Claude's "Confermi?" goes to Claude, never to a local command, and
            # is answered in the language (and voice) of the question.
            log.info("Risposta alla conferma di Claude: %r", text, extra={"route": "claude"})
            return self._ask_fallback(text)
        return None

    def _memory_reply(self, text: str, match: Match | None, addressed: bool) -> str | None:
        """What the memory says to `text`: None when it is not the memory's, "" when it was
        handled in silence (the echo of Claudio's own question). Closes the memory's open
        question when the phrase moved on, and records the route and language of its reply."""
        if self.memory is None:
            return None
        if _is_memory(match):
            self.memory.moved_on()  # "cosa ricordi?" instead of an answer: the question is over
            return None
        # Before a fuzzy command: "ricordati che la pausa pranzo è alle 13" is no "pausa".
        try:
            reply = self.memory.handle(text, addressed)
        except (sqlite3.Error, OSError):  # locked by another program, disk full...
            log.exception("Memoria non disponibile")
            self.last_route = {"route": "memory", "memory": "error", "error": True}
            return MEMORY_ERROR
        if reply is None:
            self.memory.moved_on()  # a command or Claude: "dimenticalo" is not about before
            return None
        self.last_route = {
            "route": "memory",
            "memory": reply.outcome,
            "language": self.language,
        }
        log.info("Memoria: %s", reply.outcome, extra={"route": "memory"})
        return reply.text

    async def _run_command(self, match: Match, confirmed: bool) -> AsyncIterator[str]:
        """The command `match`: run, or asked about first when its level needs a yes."""
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
        if match.command.action == "resume":
            log.info("Continua: ridico il testo mancante", extra={"route": "local"})
            return  # nothing to say here: the assistant says the rest, with "Dicevo"
        if match.command.level > LEVEL_SAFE and not (
            confirmed and match.command.level <= LEVEL_CONFIRM
        ):
            self._pending = Pending(match, self.language, self.clock())
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
        request; "no" alone drops it; a phrase that is neither ("no, abbassa il volume", "apri
        Chrome", "sì, chiudi la porta") is what JenJo wants instead, answered as a new request."""
        proposal, self._proposal = self._proposal, None
        assert proposal is not None
        words = strip_wake_words(normalize(text, drop_fillers=False).split())
        repeated = _yes_then_guess(text, words, proposal.text)
        answer = Answer.YES if repeated else parse_answer(text)
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
        rest = _after_no(words) if answer is Answer.NO else words
        if len(rest) >= 2 or (rest and answer is Answer.UNCLEAR):
            again = " ".join(rest) if answer is Answer.NO else text
            log.info("Non era %r: nuova richiesta %r", proposal.text, again)
            async for chunk in self._respond(again, language):
                yield chunk
            return
        self.language = proposal.language
        self.last_route = {"route": "guess_refused", "guess": proposal.text}
        log.info("Richiesta dedotta rifiutata: %r", proposal.text, extra={"route": "guess"})
        yield NOT_THAT if answer is Answer.NO else NOT_UNDERSTOOD

    async def _ask_fallback(self, text: str) -> AsyncIterator[str]:
        assert self.fallback is not None
        self.claude_replied = True
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
            yield str(match.command.params.get("empty") or "")
            return
        text, self.language = self._last_reply  # same words, same voice
        self.last_route["language"] = self.language
        log.info("Ripeto: %r", text, extra={"route": "local"})
        yield text

    async def _answer(self, text: str, language: str, named: bool = False) -> AsyncIterator[str]:
        pending, self._pending = self._pending, None
        assert pending is not None
        match, level = pending.match, pending.level
        self.language = pending.language  # the answer is in the language of the question
        answer = parse_answer(text, level)
        if self._name_over_question(
            text, named
        ):  # level 2 on purpose: "Claudio, sì" at level 3 asks again
            async for chunk in self._cancel_by_name(text, language, match.command.id):
                yield chunk
            return
        unheard = not pending.heard
        if unheard and not pending.said_unheard and answer is not Answer.UNCLEAR:
            # The voice never got to the question: it is said again, once (rule 6).
            self._pending = replace(pending, heard=True, said_unheard=True, asked=self.clock())
            self.last_route = {
                "route": "confirm",
                "command": match.command.id,
                "level": level,
                "language": self.language,
                "asked_unheard": True,
            }
            log.info("Domanda non sentita: la ripeto (%s)", match.command.id)
            yield question(match, self.language)
            return
        if unheard and answer is Answer.YES:
            answer = Answer.UNCLEAR  # nobody heard the question, even said again: never a yes
        if answer is Answer.UNCLEAR and not pending.asked_again:
            self._pending = replace(pending, asked_again=True, asked=self.clock())
            self.last_route = {
                "route": "confirm",
                "command": match.command.id,
                "level": level,
                "language": self.language,
                "asked_again": True,
            }
            log.info("Conferma non sentita: chiedo ancora (%s)", match.command.id)
            yield self._ask_again(match, level)
            return
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
            yield CANCELLED
        else:
            log.info("Conferma non chiara (livello %d): %s non eseguito", level, match.command.id)
            yield NOT_UNDERSTOOD

    def _name_over_question(self, text: str, named: bool = False) -> bool:
        """Not a yes or no at level 2, with the name (in the words, or in the wake audio:
        `named`), and not Claude's waiting command said again: a new call, for either kind
        of question."""
        return (
            parse_answer(text) is Answer.UNCLEAR
            and (heard_wake_word(text) or named)
            and not (self.fallback_repeats is not None and self.fallback_repeats(text))
        )

    async def _cancel_by_name(self, text: str, language: str, what: str) -> AsyncIterator[str]:
        """A new call, not an answer ("Claudio, che idea hai?", 05/11 17:20): the open question
        (Claude's or the router's) is cancelled and the words after the name are answered as a
        request of their own, router first. Nothing after the name: only the cancel."""
        log.info("Il nome sopra la domanda: %s annullato", what)
        if what == "claude" and self.fallback_cancel is not None:
            self.fallback_cancel(False)  # the brain notes the calls as not run
        self.language = language if what == "claude" else self.language
        self.last_route = {"route": "refused", "answer": "name"}
        if what != "claude":
            self.last_route["command"] = what
        cancelled = CANCELLED
        # The name only in the wake audio, not in the words: the whole phrase is the request.
        rest = after_wake_word(text) if heard_wake_word(text) else text
        claude_next = self.fallback is not None and self.matcher.match(rest, language) is None
        if what != "claude" and rest and claude_next and self.fallback_note is not None:
            self.fallback_note(text, cancelled)  # Claude answers next: its history must say so
        yield cancelled
        if rest:
            yield " "
            async for chunk in self._respond(rest, language):
                yield chunk
            self.last_route["cancelled_by_name"] = True  # the new request's route replaced it

    def _ask_again(self, match: Match, level: int) -> str:
        """The second question: the command's own phrase, then yes/no (or "confermo")."""
        phrases = match.command.phrases_in(self.language) or match.command.phrases
        what = phrases[0][:1].upper() + phrases[0][1:]
        ask = ASK_AGAIN_EXPLICIT if level >= LEVEL_EXPLICIT else ASK_AGAIN
        return ask.format(what=what)

    async def _execute(self, match: Match) -> AsyncIterator[str]:
        try:
            reply = self.actions.execute(match, self.language)
        except Exception:
            log.exception("Errore eseguendo %s", match.command.id)
            self.last_route["error"] = True
            yield self.error_reply
            return
        if reply:
            yield reply


def _yes_then_guess(text: str, words: list[str], guess: str) -> bool:
    """A yes word and exactly the guessed request ("sì, chiudi la finestra"): the answer to
    "Hai detto: chiudi la finestra?", not the question heard back."""
    same = words[1:] == normalize(guess, drop_fillers=False).split()
    return bool(words) and words[0] in YES_WORDS and same and not has_question_mark(text)


def _after_no(words: list[str]) -> list[str]:
    """What follows a refusal: "no, abbassa il volume" is a new request. "non spegnerti adesso"
    refuses what it names: taking the "non" out would run it."""
    rest = words[1:] if words and words[0] in ("no", "nope") else []
    return [] if NO_WORDS.intersection(rest) else rest


def quoted(text: str) -> str:
    """A request repeated inside Claudio's question: no final stop, first letter in lower
    case ("Spegniti." -> "spegniti") unless the word is an acronym ("PC")."""
    text = text.strip().rstrip(".?!")
    return text[:1].lower() + text[1:] if text[1:2].islower() else text


def _is_system(match: Match | None) -> bool:
    return match is not None and match.command.action == "system"


def _is_memory(match: Match | None) -> bool:
    """A memory command of commands.toml ("dimentica tutto"): the router runs it."""
    return match is not None and match.command.action == "memory"


def _turns_on(match: Match | None) -> bool:
    """ "Accenditi": the one command heard while CLAUDIO is off."""
    return _is_system(match) and match is not None and match.command.params.get("do") == ON
