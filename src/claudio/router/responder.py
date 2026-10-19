"""Phase 1 responder: local router only. Phase 2 adds a Claude responder behind
the same streaming interface."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Collection
from typing import Any

from claudio.config import DEFAULT_LANGUAGE, ERROR_REPLY, UNKNOWN_REPLY, in_language
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher, Match
from claudio.safety.confirmation import (
    CANCELLED,
    LEVEL_SAFE,
    NOT_UNDERSTOOD,
    Answer,
    Pending,
    parse_answer,
    question,
)

log = logging.getLogger(__name__)


class LocalResponder:
    def __init__(
        self,
        matcher: CommandMatcher,
        actions: ActionRunner,
        unknown_reply: dict[str, str] | None = None,
        error_reply: dict[str, str] | None = None,
        voiced: Collection[str] | None = None,
    ) -> None:
        self.matcher = matcher
        self.actions = actions
        self.unknown_reply = unknown_reply or UNKNOWN_REPLY
        self.error_reply = error_reply or ERROR_REPLY
        self.voiced = voiced  # languages with their own voice (None: all of them)
        self.last_route: dict[str, Any] = {}
        self.language = DEFAULT_LANGUAGE  # of the reply being spoken
        self._pending: Pending | None = None  # a level 2-3 action waiting for its yes

    @property
    def awaiting(self) -> bool:
        """True after a confirmation question: the assistant listens again, no wake word."""
        return self._pending is not None

    def cancel_pending(self) -> None:
        """Nothing was said after the question (or it was not understood as speech)."""
        if self._pending is not None:
            log.info(
                "Conferma scaduta: %s non eseguito",
                self._pending.match.command.id,
                extra={"route": "confirm_timeout"},
            )
            self._pending = None

    def _reply_language(self, language: str, has_text: bool) -> str:
        """`language` when there is a reply and a voice for it, else Italian: the text and
        the voice reading it always agree."""
        voiced = self.voiced is None or language in self.voiced
        return language if has_text and voiced else DEFAULT_LANGUAGE

    async def respond(self, text: str, language: str = DEFAULT_LANGUAGE) -> AsyncIterator[str]:
        """Reply in the language of the command's phrase (`language` when none matched):
        "open chrome" gets an English reply even if Whisper thought it was Italian.
        Right after a confirmation question, `text` is its answer."""
        if self._pending is not None:
            async for chunk in self._answer(text):
                yield chunk
            return
        match = self.matcher.match(text, language)
        if match:
            self.language = self._reply_language(
                match.language, match.command.has_reply_in(match.language)
            )
        else:
            self.language = self._reply_language(language, bool(self.unknown_reply.get(language)))
        if self.actions.system.paused and not _is_system(match):
            self.last_route = {"route": "paused"}
            log.info("In pausa, ignoro: %r", text, extra={"route": "paused"})
            return  # nothing to say: the assistant answers with a low beep
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
        if match.command.level > LEVEL_SAFE:
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
