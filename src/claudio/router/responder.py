"""Phase 1 responder: local router only. Phase 2 adds a Claude responder behind
the same streaming interface."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from claudio.config import DEFAULT_LANGUAGE, ERROR_REPLY, UNKNOWN_REPLY, in_language
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher, Match

log = logging.getLogger(__name__)


class LocalResponder:
    def __init__(
        self,
        matcher: CommandMatcher,
        actions: ActionRunner,
        unknown_reply: dict[str, str] | None = None,
        error_reply: dict[str, str] | None = None,
    ) -> None:
        self.matcher = matcher
        self.actions = actions
        self.unknown_reply = unknown_reply or UNKNOWN_REPLY
        self.error_reply = error_reply or ERROR_REPLY
        self.last_route: dict[str, Any] = {}
        self.language = DEFAULT_LANGUAGE  # of the reply being spoken

    async def respond(self, text: str, language: str = DEFAULT_LANGUAGE) -> AsyncIterator[str]:
        """Reply in the language of the command's phrase (`language` when none matched):
        "open chrome" gets an English reply even if Whisper thought it was Italian."""
        match = self.matcher.match(text, language)
        self.language = match.language if match else language
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
        try:
            reply = self.actions.execute(match, self.language)
        except Exception:
            log.exception("Errore eseguendo %s", match.command.id)
            self.last_route["error"] = True
            yield in_language(self.error_reply, self.language)
            return
        if reply:
            yield reply


def _is_system(match: Match | None) -> bool:
    return match is not None and match.command.action == "system"
