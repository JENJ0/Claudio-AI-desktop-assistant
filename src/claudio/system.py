"""Pause, wake up, off, on and shut down by voice: the state the router and the assistant share.

While paused CLAUDIO still hears the wake word, but answers only the system commands: every
other request is dropped locally, at no cost, with a low beep.

Off ("Claudio, spegniti") goes further: no beeps, no replies, no Claude, no second Whisper
pass. Only the wake word listens, with a slightly higher threshold, and the one phrase heard
after it is matched against "accenditi" alone. "Chiuditi" still closes the program."""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

PAUSE, RESUME, SHUTDOWN, OFF, ON = "pause", "resume", "shutdown", "off", "on"
# In the logs and the rows, in place of a phrase heard while off: one was heard, and it is gone.
HIDDEN = "<hidden: off>"


class SystemControl:
    def __init__(self) -> None:
        self.paused = False
        self.off = False  # set by "spegniti": silent until "accenditi"
        self.stopping = False  # set by "chiuditi": the assistant exits after its reply
        self._listeners: list[Callable[[str], None]] = []

    def subscribe(self, listener: Callable[[str], None]) -> None:
        """Call `listener(what)` after every change, by voice or from the tray menu (the tray
        icon turns gray on "spegniti")."""
        self._listeners.append(listener)

    def apply(self, what: str) -> bool:
        """Do `what` (pause, resume, off, on, shutdown); False when there was nothing to
        change."""
        if what == PAUSE:
            changed, self.paused = not self.paused, True
        elif what == RESUME:
            changed, self.paused = self.paused, False
        elif what == OFF:
            changed, self.off = not self.off, True
        elif what == ON:
            # "accenditi" also ends a pause: after it CLAUDIO answers everything.
            changed = self.off or self.paused
            self.off = self.paused = False
        elif what == SHUTDOWN:
            changed, self.stopping = not self.stopping, True
        else:
            raise ValueError(f"Comando di sistema sconosciuto '{what}'")
        if changed:
            log.info("Sistema: %s", what, extra={"system": what})
            for listener in self._listeners:
                try:
                    listener(what)
                except Exception:  # a listener never undoes or blocks the command
                    log.exception("Avviso del cambio di stato non riuscito")
        return changed
