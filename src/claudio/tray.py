"""CLAUDIO's icon in the system tray, next to the clock: its state at a glance, and a menu to
turn it off or on, pause it, open the chat and close it without saying a word.

The color follows the state: blue while the models load, green when on, amber when paused,
gray when off. A left click opens the chat. pystray runs the Windows message loop in a
thread of its own ("tray"): every click is handed to the assistant's event loop
(call_soon_threadsafe), the thread that owns CLAUDIO's state, exactly as a voice command
would change it. Without pystray (not installed) CLAUDIO runs as before, with a warning."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import importlib
import logging
import threading
from collections.abc import Callable
from typing import Any

from claudio import __version__
from claudio.system import OFF, ON, PAUSE, RESUME, SHUTDOWN, SystemControl

log = logging.getLogger(__name__)

STARTING, ON_STATE, PAUSED, OFF_STATE, CLOSING = "starting", "on", "paused", "off", "closing"
COLORS = {
    STARTING: (37, 99, 235),  # blue: the models are loading, nothing listens yet
    ON_STATE: (22, 163, 74),  # green: listening for the wake word
    PAUSED: (217, 119, 6),  # amber: only "svegliati" is answered
    OFF_STATE: (107, 114, 128),  # gray: only "accenditi" is heard
    CLOSING: (107, 114, 128),
}
ICON_SIZE = 64  # pixels; Windows scales it to the tray (16-32 px)
STOP_WAIT_S = 2.0


def tray_state(system: SystemControl, ready: bool) -> str:
    if system.stopping:
        return CLOSING
    if not ready:
        return STARTING
    if system.off:
        return OFF_STATE
    return PAUSED if system.paused else ON_STATE


def tooltip(state: str) -> str:
    """The text shown when the mouse rests on the icon."""
    texts = {
        STARTING: "si avvia...",
        ON_STATE: "acceso",
        PAUSED: 'in pausa: di\' "Claudio, svegliati"',
        OFF_STATE: 'spento: di\' "Claudio, accenditi"',
        CLOSING: "si chiude...",
    }
    return f"CLAUDIO {__version__}: {texts[state]}"


def icon_image(state: str, size: int = ICON_SIZE) -> Any:
    """A colored disc with a white "C", drawn now: no image file to ship or lose."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((1, 1, size - 2, size - 2), fill=(*COLORS[state], 255))
    margin = size * 0.26
    draw.arc(
        (margin, margin, size - 1 - margin, size - 1 - margin),
        start=45,
        end=315,
        fill=(255, 255, 255, 255),
        width=max(2, size // 9),
    )
    return image


class Tray:
    """The icon and its menu. `chat`: the chat window launcher (None: no "open the chat")."""

    def __init__(
        self,
        system: SystemControl,
        loop: asyncio.AbstractEventLoop,
        chat: Any = None,
        backend: Any = None,
    ) -> None:
        self.system = system
        self.loop = loop
        self.chat = chat
        self.ready = False  # set once the models have loaded
        self._backend = backend  # the pystray module; None: imported by start()
        self._icon: Any = None
        self._thread: threading.Thread | None = None
        self._shown: str | None = None
        system.subscribe(lambda _what: self.refresh())  # "spegniti" by voice turns it gray

    @property
    def state(self) -> str:
        return tray_state(self.system, self.ready)

    def start(self) -> bool:
        """Show the icon; False when pystray is missing or the tray cannot be used."""
        try:
            pystray = self._backend or importlib.import_module("pystray")
            state = self.state
            self._icon = pystray.Icon(
                "CLAUDIO", icon_image(state), tooltip(state), self._menu(pystray)
            )
        except Exception as exc:  # ImportError when it is not installed
            log.warning(
                "Icona nella system tray non disponibile (%s): CLAUDIO va avanti senza", exc
            )
            return False
        self._shown = state
        self._thread = threading.Thread(target=self._run, name="tray", daemon=True)
        self._thread.start()
        # However CLAUDIO ends ("chiuditi", a model that fails to load, an error), the icon
        # goes with it: a daemon thread killed at exit would leave it next to the clock.
        atexit.register(self.stop)
        log.info("Icona nella system tray: %s", state)
        return True

    def _run(self) -> None:
        try:
            self._icon.run()
        except Exception:
            log.exception("Icona nella system tray chiusa per errore")

    def set_ready(self) -> None:
        self.ready = True
        self.refresh()

    def refresh(self) -> None:
        """Show the current state: color, tooltip, menu (called on the event loop thread)."""
        icon, state = self._icon, self.state
        if icon is None or state == self._shown:
            return
        try:
            icon.icon = icon_image(state)
            icon.title = tooltip(state)
            icon.update_menu()  # the menu is built in advance on Windows
            self._shown = state
        except Exception:  # the tray must never break a voice command
            log.exception("Icona nella system tray non aggiornata")

    def stop(self) -> None:
        """Remove the icon (none is left behind next to the clock) and end its thread."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                log.exception("Icona nella system tray non chiusa")
        if self._thread is not None:
            self._thread.join(timeout=STOP_WAIT_S)
            if self._thread.is_alive():  # stopped before pystray's loop had started
                log.warning("Icona nella system tray: il suo thread non si è fermato")
        self._icon = self._thread = None

    # The menu. pystray reads the labels and the ticks when the menu is built, in its thread.

    def _menu(self, pystray: Any) -> Any:
        item, separator = pystray.MenuItem, pystray.Menu.SEPARATOR
        items = [
            item(lambda _: "Accendi" if self.system.off else "Spegni", self._on_off),
            item(
                "Pausa",
                self._pause,
                checked=lambda _: self.system.paused,
                enabled=lambda _: not self.system.off,
            ),
        ]
        if self.chat is not None:
            items.append(item("Apri la chat", self._open_chat, default=True))
        items += [separator, item("Chiudi CLAUDIO", self._close)]
        return pystray.Menu(*items)

    def _on_off(self) -> None:
        self._later(self.system.apply, ON if self.system.off else OFF)

    def _pause(self) -> None:
        self._later(self.system.apply, RESUME if self.system.paused else PAUSE)

    def _open_chat(self) -> None:
        if self.chat is not None:
            self._later(self.chat.open)

    def _close(self) -> None:
        log.info("Chiusura chiesta dall'icona nella system tray")
        self._later(self.system.apply, SHUTDOWN)

    def _later(self, action: Callable[..., Any], *args: Any) -> None:
        """Run `action` on the event loop: CLAUDIO's state changes on one thread only."""
        with contextlib.suppress(RuntimeError):  # the loop has closed: CLAUDIO is going
            self.loop.call_soon_threadsafe(action, *args)


def start_tray(enabled: bool, system: SystemControl, chat: Any = None) -> Tray | None:
    """The tray icon, shown at once (blue while the models load); None when it is switched
    off ([startup] tray) or cannot be shown. Called from the running event loop."""
    if not enabled:
        return None
    tray = Tray(system, asyncio.get_running_loop(), chat)
    return tray if tray.start() else None
