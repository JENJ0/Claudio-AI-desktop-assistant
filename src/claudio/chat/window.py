"""`claudio --ui`: the chat in its own window (pywebview, Windows WebView2) over the local
server. It reads data/claudio.db, so it works with the assistant running or not; the running
assistant opens it the same way when asked ("apri la chat", see chat/launcher.py)."""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any

from claudio.chat.server import free_port, serve_in_thread
from claudio.chat.store import ChatStore

log = logging.getLogger(__name__)

HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOSIZE, SWP_NOMOVE, SWP_ASYNCWINDOWPOS = 0x0001, 0x0002, 0x4000


def bring_to_front(window: Any, user32: Any = None) -> None:
    """Opened by voice, the window belongs to a background process and Windows leaves it
    behind the others: raise it once (top-most, then not), without keeping it on top.

    pywebview runs the "shown" handler in its own thread. Setting WinForms' TopMost there
    is a cross-thread call, and the window froze ("Non risponde", 23/10). SetWindowPos with
    SWP_ASYNCWINDOWPOS only posts the request to the window's own thread."""
    native = getattr(window, "native", None)
    if native is None:
        return
    if user32 is None:
        if sys.platform != "win32":
            return
        user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
    try:
        hwnd = ctypes.c_void_p(int(native.Handle.ToInt64()))
        flags = SWP_NOMOVE | SWP_NOSIZE | SWP_ASYNCWINDOWPOS
        for where in (HWND_TOPMOST, HWND_NOTOPMOST):
            user32.SetWindowPos(hwnd, ctypes.c_void_p(where), 0, 0, 0, 0, flags)
    except Exception:  # a window left behind is better than no window
        log.warning("Finestra della chat non portata in primo piano", exc_info=True)


def open_window(db_path: Path) -> None:
    import webview  # Windows only: imported here so the tests and the assistant do not need it

    store = ChatStore(db_path)
    port = free_port()
    server = serve_in_thread(store, port)
    try:
        window = webview.create_window(
            "Claudio", f"http://127.0.0.1:{port}/", width=980, height=720
        )
        if window is not None:
            window.events.shown += lambda: bring_to_front(window)
        webview.start()  # returns when the window is closed
    finally:
        server.should_exit = True
