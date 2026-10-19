"""Pause, wake up and shut down by voice: the state the router and the assistant share.

While paused CLAUDIO still hears the wake word, but answers only "svegliati" and "spegniti":
every other request is dropped locally, at no cost."""

from __future__ import annotations

import logging

from claudio.i18n import tr

log = logging.getLogger(__name__)

PAUSE, RESUME, SHUTDOWN = "pause", "resume", "shutdown"


class SystemControl:
    def __init__(self) -> None:
        self.paused = False
        self.stopping = False  # set by "spegniti": the assistant exits after its reply

    def apply(self, what: str) -> bool:
        """Do `what` (pause, resume, shutdown); False when there was nothing to change."""
        if what == PAUSE:
            changed, self.paused = not self.paused, True
        elif what == RESUME:
            changed, self.paused = self.paused, False
        elif what == SHUTDOWN:
            changed, self.stopping = not self.stopping, True
        else:
            raise ValueError(tr("Comando di sistema sconosciuto '{command}'", command=what))
        if changed:
            log.info("Sistema: %s", what, extra={"system": what})
        return changed
