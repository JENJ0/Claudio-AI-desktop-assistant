"""Phase 1 responder: local router only. Phase 2 adds a Claude responder behind
the same streaming interface."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher, Match

log = logging.getLogger(__name__)


class LocalResponder:
    def __init__(self, matcher: CommandMatcher, actions: ActionRunner, unknown_reply: str) -> None:
        self.matcher = matcher
        self.actions = actions
        self.unknown_reply = unknown_reply
        self.last_route: dict[str, Any] = {}

    async def respond(self, text: str) -> AsyncIterator[str]:
        match = self.matcher.match(text)
        if self.actions.system.paused and not _is_system(match):
            self.last_route = {"route": "paused"}
            log.info("In pausa, ignoro: %r", text, extra={"route": "paused"})
            return  # nothing to say: the assistant answers with a low beep
        if match is None:
            self.last_route = {"route": "unknown"}
            log.info("Richiesta non riconosciuta: %r", text, extra={"route": "unknown"})
            yield self.unknown_reply
            return
        self.last_route = {
            "route": "local",
            "command": match.command.id,
            "score": round(match.score, 1),
        }
        try:
            reply = self.actions.execute(match)
        except Exception:
            log.exception("Errore eseguendo %s", match.command.id)
            self.last_route["error"] = True
            yield "Non ci sono riuscito."
            return
        if reply:
            yield reply


def _is_system(match: Match | None) -> bool:
    return match is not None and match.command.action == "system"
