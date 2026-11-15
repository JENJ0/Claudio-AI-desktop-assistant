"""Level-1 actions (no confirmation needed) and the system commands (pause, wake up, shut
down). Side effects go through ActionContext, so tests check what would run without
launching anything."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from claudio.chat.launcher import ChatLauncher
from claudio.cost.tracker import CostTracker
from claudio.router.matcher import Match
from claudio.system import SystemControl

if TYPE_CHECKING:
    from claudio.memory.voice import MemoryVoice

log = logging.getLogger(__name__)

# Windows virtual-key codes for the media keys: they reach whatever is playing (Spotify,
# YouTube in the browser...), even in the background.
MEDIA_KEYS = {
    "volume_up": 0xAF,
    "volume_down": 0xAE,
    "mute": 0xAD,
    "play_pause": 0xB3,
    "next_track": 0xB0,
    "previous_track": 0xB1,
}
WM_CLOSE = 0x0010
NO_WINDOW = "Non vedo una finestra da chiudere."
GONE = "La finestra non c'è più."
# Desktop and taskbar are not windows to close (Progman answers WM_CLOSE with the shutdown dialog).
SHELL_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"})
APPS: dict[str, list[str]] = {
    # "start" resolves Chrome through its App Paths registration.
    "chrome": ["cmd", "/c", "start", "", "chrome"],
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
}
KNOWN_FOLDERS = {
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
}
FALLBACK_FOLDERS = {"downloads": "Downloads", "documents": "Documents"}

WEEKDAYS = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")
MONTHS = (
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
)


def known_folder(folder_id: str) -> Path | None:
    """Real location of Downloads/Documents (they are often moved, e.g. to OneDrive)."""
    if sys.platform != "win32":
        return None
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    guid = GUID.from_buffer_copy(uuid.UUID(folder_id).bytes_le)
    path_ptr = ctypes.c_wchar_p()
    hr = ctypes.windll.shell32.SHGetKnownFolderPath(
        ctypes.byref(guid), 0, None, ctypes.byref(path_ptr)
    )
    try:
        return Path(path_ptr.value) if hr == 0 and path_ptr.value else None
    finally:
        ctypes.windll.ole32.CoTaskMemFree(path_ptr)


def _run(args: list[str]) -> None:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(args, creationflags=flags, close_fds=True)


def _open_path(path: str) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])


def _press_key(vk: int) -> None:
    import ctypes

    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP


def _front_window() -> Target | None:
    import win32gui

    handle = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(handle) if handle else ""
    if not title or win32gui.GetClassName(handle) in SHELL_CLASSES:
        return None
    target = Target(handle, title)
    # a title of only `*`, spaces or `-` has no name to say
    return target if target.name.strip(" -") else None


def _close_handle(handle: int) -> bool:
    """WM_CLOSE as the X button: the app may still ask to save."""
    import win32gui

    if not win32gui.IsWindow(handle):
        return False
    win32gui.PostMessage(handle, WM_CLOSE, 0, 0)
    return True


def _refuse(*args: object) -> NoReturn:
    raise RuntimeError("a real action needs ActionContext.real()")


@dataclass(frozen=True)
class Target:
    """A window the close command is about: its handle and full title."""

    handle: int
    title: str

    @property
    def name(self) -> str:
        """The app part of the title: no leading `*`, nothing before the last " - ".
        A title ending in " - " has no app part: the whole title is the name."""
        title = self.title.lstrip("* ")
        return title.rsplit(" - ", 1)[-1].strip() or title.strip(" -")


@dataclass
class ActionContext:
    """Refuses every real action by default (05/11: a probe's default context sent a real
    Alt+F4); only `real()` acts. `now`, `folder_lookup` and `home` just read."""

    run: Callable[[list[str]], None] = _refuse
    open_path: Callable[[str], None] = _refuse
    press_key: Callable[[int], None] = _refuse
    front_window: Callable[[], Target | None] = lambda: None
    close_window: Callable[[int], bool] = _refuse
    now: Callable[[], datetime] = datetime.now
    folder_lookup: Callable[[str], Path | None] = known_folder
    home: Path = field(default_factory=Path.home)

    @classmethod
    def real(cls) -> ActionContext:
        return cls(
            run=_run,
            open_path=_open_path,
            press_key=_press_key,
            front_window=_front_window,
            close_window=_close_handle,
        )


def say_time(now: datetime) -> str:
    h, m = now.hour, now.minute
    if h == 1:
        head = "È l'una"
    elif h == 0:
        head = "È mezzanotte"
    else:
        head = f"Sono le {h}"
    if m == 0:
        return f"{head} in punto."
    return f"{head} e {m}."


def say_date(now: datetime) -> str:
    day = "primo" if now.day == 1 else str(now.day)
    return f"Oggi è {WEEKDAYS[now.weekday()]} {day} {MONTHS[now.month - 1]} {now.year}."


def say_spend(spent: float, budget: float, calls: int) -> str:
    percent = round(spent / budget * 100) if budget > 0 else 0
    if calls == 0:
        return "Oggi non hai speso nulla."
    amount = f"{spent:.2f}".replace(".", ",")
    return f"Oggi hai speso {amount} dollari, il {percent} per cento del budget."


Handler = Callable[[Match], "str | None"]


class ActionRunner:
    def __init__(
        self,
        ctx: ActionContext | None = None,
        system: SystemControl | None = None,
        tracker: CostTracker | None = None,
        chat: ChatLauncher | None = None,
        memory: MemoryVoice | None = None,
    ) -> None:
        self.ctx = ctx or ActionContext()
        self.system = system or SystemControl()
        self.tracker = tracker  # the spending counter ("quanto ho speso oggi")
        self.chat = chat  # the chat window ("apri la chat")
        self.memory = memory  # long-term memory ("cosa ricordi?", "dimentica tutto")
        self._handlers: dict[str, Handler] = {
            "open_app": self._open_app,
            "open_folder": self._open_folder,
            "media_key": self._media_key,
            "tell_time": lambda m: say_time(self.ctx.now()),
            "tell_date": lambda m: say_date(self.ctx.now()),
            "system": self._system,
            "tell_spend": self._tell_spend,
            "chat": self._chat,
            "memory": self._memory,
        }

    def target(self, match: Match) -> Target | None:
        """The window the command is about (only close_window has one), read now."""
        return self.ctx.front_window() if match.command.action == "close_window" else None

    def execute(self, match: Match, target: Target | None = None) -> str:
        """Run the command's action; return the text to speak. `target` is the window
        captured when the question was asked; without it close_window reads it now."""
        cmd = match.command
        if cmd.action == "close_window":
            text, closed = self._close_window(match, target or self.target(match))
            log.info(
                "Azione eseguita: %s",
                cmd.id,
                extra={
                    "action": cmd.action,
                    "command": cmd.id,
                    "score": match.score,
                    "closed": closed,
                },
            )
            return text
        handler = self._handlers.get(cmd.action)
        if handler is None:
            raise ValueError(f"Azione sconosciuta '{cmd.action}' nel comando '{cmd.id}'")
        spoken = handler(match)
        log.info(
            "Azione eseguita: %s",
            cmd.id,
            extra={
                "action": cmd.action,
                "command": cmd.id,
                "params": cmd.params,
                "score": match.score,
            },
        )
        return spoken if spoken is not None else cmd.reply

    def _open_app(self, match: Match) -> None:
        self.ctx.run(APPS[match.command.params["app"]])

    def _open_folder(self, match: Match) -> None:
        folder = match.command.params["folder"]
        if folder == "explorer":
            self.ctx.run(["explorer.exe"])
            return
        path = None
        if folder in KNOWN_FOLDERS:
            path = self.ctx.folder_lookup(KNOWN_FOLDERS[folder])
        path = path or self.ctx.home / FALLBACK_FOLDERS.get(folder, folder)
        self.ctx.open_path(str(path))

    def _system(self, match: Match) -> str | None:
        """The command's reply when something changed, params["already"] when not (pausing
        while already paused)."""
        changed = self.system.apply(match.command.params["do"])
        return None if changed else str(match.command.params.get("already") or "")

    def _close_window(self, match: Match, target: Target | None) -> tuple[str, bool]:
        """Level 2: the responder has already asked and heard a yes. Returns the text to
        speak and whether WM_CLOSE was sent."""
        if target is None:
            log.info("Nessuna finestra da chiudere")
            return NO_WINDOW, False
        if not self.ctx.close_window(target.handle):
            log.info("Finestra «%s» non più aperta: niente da chiudere", target.title)
            return GONE, False
        log.info("Chiusa «%s» (WM_CLOSE)", target.title)
        return match.command.reply, True

    def _chat(self, match: Match) -> str | None:
        """params["do"]: open or close; params["already"] when there is nothing to do."""
        if self.chat is None:
            return str(match.command.params.get("unavailable") or "")
        what = match.command.params["do"]
        if what not in ("open", "close"):
            raise ValueError(f"Comando chat sconosciuto '{what}'")
        changed = self.chat.open() if what == "open" else self.chat.close()
        return None if changed else str(match.command.params.get("already") or "")

    def _memory(self, match: Match) -> str:
        """params["do"]: list, forget_last or wipe (level 3: "confermo" was heard)."""
        if self.memory is None:
            return str(match.command.params.get("unavailable") or "")
        return self.memory.run(str(match.command.params["do"]))

    def _tell_spend(self, match: Match) -> str:
        if self.tracker is None:
            return say_spend(0.0, 1.0, 0)
        return say_spend(
            self.tracker.spent_today(),
            self.tracker.daily_budget,
            self.tracker.calls_today(),
        )

    def _media_key(self, match: Match) -> None:
        params = match.command.params
        steps = match.number if match.number is not None else int(params.get("steps", 1))
        steps = max(1, min(steps, 50))  # one press = 2% on Windows
        vk = MEDIA_KEYS[params["key"]]
        for _ in range(steps):
            self.ctx.press_key(vk)
