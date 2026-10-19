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

from claudio.config import DEFAULT_LANGUAGE
from claudio.cost.tracker import CostTracker
from claudio.i18n import tr
from claudio.router.matcher import Match
from claudio.system import SystemControl

log = logging.getLogger(__name__)

# Windows virtual-key codes for the media keys.
MEDIA_KEYS = {"volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD}
VK_MENU, VK_F4 = 0x12, 0x73  # Alt, F4: closes the window in front
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


def _press_combo(vks: tuple[int, ...]) -> None:
    """Hold the keys down in order, release them in reverse (Alt+F4)."""
    import ctypes

    user32 = ctypes.windll.user32
    for vk in vks:
        user32.keybd_event(vk, 0, 0, 0)
    for vk in reversed(vks):
        user32.keybd_event(vk, 0, 2, 0)


@dataclass
class ActionContext:
    run: Callable[[list[str]], None] = _run
    open_path: Callable[[str], None] = _open_path
    press_key: Callable[[int], None] = _press_key
    press_combo: Callable[[tuple[int, ...]], None] = _press_combo
    now: Callable[[], datetime] = datetime.now
    folder_lookup: Callable[[str], Path | None] = known_folder
    home: Path = field(default_factory=Path.home)


EN_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
EN_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def say_time(now: datetime, language: str = DEFAULT_LANGUAGE) -> str:
    h, m = now.hour, now.minute
    if language == "en":  # "It's 2 oh 5 p.m.": digits with a colon are read "two zero five"
        h12 = h % 12 or 12
        suffix = "a.m." if h < 12 else "p.m."
        if m == 0:
            return f"It's {h12} o'clock." if h not in (0, 12) else f"It's 12 {suffix}"
        return f"It's {h12} {'oh ' if m < 10 else ''}{m} {suffix}"
    if h == 1:
        head = "È l'una"
    elif h == 0:
        head = "È mezzanotte"
    else:
        head = f"Sono le {h}"
    if m == 0:
        return f"{head} in punto."
    return f"{head} e {m}."


def say_date(now: datetime, language: str = DEFAULT_LANGUAGE) -> str:
    if language == "en":
        d = now.day
        suffix = "th" if 10 < d % 100 < 14 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        weekday, month = EN_WEEKDAYS[now.weekday()], EN_MONTHS[now.month - 1]
        return f"Today is {weekday}, {month} {d}{suffix}, {now.year}."
    day = "primo" if now.day == 1 else str(now.day)
    return f"Oggi è {WEEKDAYS[now.weekday()]} {day} {MONTHS[now.month - 1]} {now.year}."


def say_spend(spent: float, budget: float, calls: int, language: str = DEFAULT_LANGUAGE) -> str:
    percent = round(spent / budget * 100) if budget > 0 else 0
    if language == "en":
        if calls == 0:
            return "You haven't spent anything today."
        return f"Today you spent {spent:.2f} dollars, {percent} percent of the budget."
    if calls == 0:
        return "Oggi non hai speso nulla."
    amount = f"{spent:.2f}".replace(".", ",")
    return f"Oggi hai speso {amount} dollari, il {percent} per cento del budget."


Handler = Callable[[Match, str], "str | None"]


class ActionRunner:
    def __init__(
        self,
        ctx: ActionContext | None = None,
        system: SystemControl | None = None,
        tracker: CostTracker | None = None,
    ) -> None:
        self.ctx = ctx or ActionContext()
        self.system = system or SystemControl()
        self.tracker = tracker  # the spending counter ("quanto ho speso oggi")
        self._handlers: dict[str, Handler] = {
            "open_app": self._open_app,
            "open_folder": self._open_folder,
            "media_key": self._media_key,
            "tell_time": lambda m, lang: say_time(self.ctx.now(), lang),
            "tell_date": lambda m, lang: say_date(self.ctx.now(), lang),
            "system": self._system,
            "tell_spend": self._tell_spend,
            "close_window": self._close_window,
        }

    def execute(self, match: Match, language: str = DEFAULT_LANGUAGE) -> str:
        """Run the command's action; return the text to speak, in `language`."""
        cmd = match.command
        handler = self._handlers.get(cmd.action)
        if handler is None:
            raise ValueError(
                tr(
                    "Azione sconosciuta '{action}' nel comando '{command}'",
                    action=cmd.action,
                    command=cmd.id,
                )
            )
        spoken = handler(match, language)
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
        return spoken if spoken is not None else cmd.reply_in(language)

    def _open_app(self, match: Match, language: str) -> None:
        self.ctx.run(APPS[match.command.params["app"]])

    def _open_folder(self, match: Match, language: str) -> None:
        folder = match.command.params["folder"]
        if folder == "explorer":
            self.ctx.run(["explorer.exe"])
            return
        path = None
        if folder in KNOWN_FOLDERS:
            path = self.ctx.folder_lookup(KNOWN_FOLDERS[folder])
        path = path or self.ctx.home / FALLBACK_FOLDERS.get(folder, folder)
        self.ctx.open_path(str(path))

    def _system(self, match: Match, language: str) -> str | None:
        """The command's reply when something changed, params["already"] when not (pausing
        while already paused)."""
        changed = self.system.apply(match.command.params["do"])
        return None if changed else str(match.command.param_in("already", language) or "")

    def _close_window(self, match: Match, language: str) -> None:
        """Level 2: the responder has already asked and heard a yes."""
        self.ctx.press_combo((VK_MENU, VK_F4))

    def _tell_spend(self, match: Match, language: str) -> str:
        if self.tracker is None:
            return say_spend(0.0, 1.0, 0, language)
        return say_spend(
            self.tracker.spent_today(),
            self.tracker.daily_budget,
            self.tracker.calls_today(),
            language,
        )

    def _media_key(self, match: Match, language: str) -> None:
        params = match.command.params
        steps = match.number if match.number is not None else int(params.get("steps", 1))
        steps = max(1, min(steps, 50))  # one press = 2% on Windows
        vk = MEDIA_KEYS[params["key"]]
        for _ in range(steps):
            self.ctx.press_key(vk)
