"""`claudio --ui`: the chat in its own window (pywebview, Windows WebView2) over the local
server. It reads data/claudio.db, so it works with the assistant running or not."""

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
        webview.create_window("Claudio", f"http://127.0.0.1:{port}/", width=980, height=720)
        webview.start()  # returns when the window is closed
    finally:
        server.should_exit = True
