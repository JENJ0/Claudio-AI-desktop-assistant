"""Local web server behind the chat window. It listens on 127.0.0.1 only and refuses any other
Host header (a web page cannot reach it through DNS rebinding). Read-only."""

from __future__ import annotations

import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from claudio.chat.store import ChatStore

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def create_app(store: ChatStore) -> FastAPI:
    app = FastAPI(title="Claudio", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def local_only(request: Request, call_next: Any) -> Response:
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in LOCAL_HOSTS:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        response: Response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline'"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/sessions")
    def sessions() -> list[dict[str, Any]]:
        return store.sessions()

    @app.get("/api/messages")
    def messages(
        session: str | None = None,
        q: str = Query("", max_length=200),
        after_id: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return store.messages(session=session, query=q, after_id=after_id, limit=limit)

    @app.get("/api/today")
    def today() -> dict[str, float]:
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        return {"cost": store.cost_since(midnight), "now": time.time()}

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def serve_in_thread(store: ChatStore, port: int) -> uvicorn.Server:
    """Start the server in a daemon thread and wait until it accepts connections."""
    config = uvicorn.Config(
        create_app(store), host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="chat-server", daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("chat server did not start")
    return server
