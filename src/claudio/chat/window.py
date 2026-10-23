"""`claudio --ui`: the chat in its own window (pywebview, Windows WebView2) over the local
server. It reads data/claudio.db, so it works with the assistant running or not; the running
assistant opens it the same way when asked ("apri la chat", see chat/launcher.py)."""

from __future__ import annotations

from pathlib import Path

from claudio.chat.server import free_port, serve_in_thread
from claudio.chat.store import ChatStore


def open_window(db_path: Path) -> None:
    import webview  # Windows only: imported here so the tests and the assistant do not need it

    store = ChatStore(db_path)
    port = free_port()
    server = serve_in_thread(store, port)
    try:
        # Opened by voice, the window belongs to a background process: Windows would leave it
        # behind the others. On top until shown, then a normal window.
        window = webview.create_window(
            "Claudio", f"http://127.0.0.1:{port}/", width=980, height=720, on_top=True
        )
        if window is not None:
            window.events.shown += lambda: setattr(window, "on_top", False)
        webview.start()  # returns when the window is closed
    finally:
        server.should_exit = True
