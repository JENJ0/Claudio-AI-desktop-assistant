"""Level-1 actions (no confirmation needed). Side effects go through ActionContext,
so tests check what would run without launching anything."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from claudio.router.matcher import Match

log = logging.getLogger(__name__)

# Windows virtual-key codes for the media keys.
MEDIA_KEYS = {"volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD}
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


@dataclass
class ActionContext:
    run: Callable[[list[str]], None] = _run
    open_path: Callable[[str], None] = _open_path
    press_key: Callable[[int], None] = _press_key
    now: Callable[[], datetime] = datetime.now
    folder_lookup: Callable[[str], Path | None] = known_folder
    home: Path = field(default_factory=Path.home)


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


class ActionRunner:
    def __init__(self, ctx: ActionContext | None = None) -> None:
        self.ctx = ctx or ActionContext()
        self._handlers: dict[str, Callable[[dict[str, Any], Match], str | None]] = {
            "open_app": self._open_app,
            "open_folder": self._open_folder,
            "media_key": self._media_key,
            "tell_time": lambda p, m: say_time(self.ctx.now()),
            "tell_date": lambda p, m: say_date(self.ctx.now()),
        }

    def execute(self, match: Match) -> str:
        """Run the command's action; return the text to speak."""
        cmd = match.command
        handler = self._handlers.get(cmd.action)
        if handler is None:
            raise ValueError(f"Azione sconosciuta '{cmd.action}' nel comando '{cmd.id}'")
        spoken = handler(cmd.params, match)
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

    def _open_app(self, params: dict[str, Any], match: Match) -> None:
        self.ctx.run(APPS[params["app"]])

    def _open_folder(self, params: dict[str, Any], match: Match) -> None:
        folder = params["folder"]
        if folder == "explorer":
            self.ctx.run(["explorer.exe"])
            return
        path = None
        if folder in KNOWN_FOLDERS:
            path = self.ctx.folder_lookup(KNOWN_FOLDERS[folder])
        path = path or self.ctx.home / FALLBACK_FOLDERS.get(folder, folder)
        self.ctx.open_path(str(path))

    def _media_key(self, params: dict[str, Any], match: Match) -> None:
        steps = match.number if match.number is not None else int(params.get("steps", 1))
        steps = max(1, min(steps, 50))  # one press = 2% on Windows
        vk = MEDIA_KEYS[params["key"]]
        for _ in range(steps):
            self.ctx.press_key(vk)
