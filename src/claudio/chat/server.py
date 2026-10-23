"""Local web server behind the chat window. It listens on 127.0.0.1 only and refuses any other
Host header (a web page cannot reach it through DNS rebinding).

The only write is POST /api/send (a message typed in the window, for the assistant). A web page
open in a browser could still post to 127.0.0.1, so it needs: a JSON body (a plain HTML form
cannot send one), no foreign Origin, and the random token this server put in its own page."""

from __future__ import annotations

import logging
import secrets
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from claudio.chat.store import MAX_TYPED_CHARS, ChatStore

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
TOKEN_PLACEHOLDER = "__CLAUDIO_TOKEN__"


class Typed(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TYPED_CHARS)
    token: str = Field(max_length=200)


def _local_origin(origin: str | None) -> bool:
    if not origin:
        return True  # same-origin GETs and the window itself may send none
    host = origin.removeprefix("http://").split("/", 1)[0].rsplit(":", 1)[0]
    return origin.startswith("http://") and host in LOCAL_HOSTS


def create_app(store: ChatStore, token: str | None = None) -> FastAPI:
    app = FastAPI(title="Claudio", docs_url=None, redoc_url=None, openapi_url=None)
    token = token or secrets.token_urlsafe(32)  # new at every start of the window
    page = (STATIC / "index.html").read_text(encoding="utf-8").replace(TOKEN_PLACEHOLDER, token)

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
    def index() -> HTMLResponse:
        return HTMLResponse(page)

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

    @app.post("/api/send")
    def send(message: Typed, request: Request) -> dict[str, int]:
        # FastAPI reads the body as JSON only with a JSON content type: a form post is a 422.
        if not _local_origin(request.headers.get("origin")) or not secrets.compare_digest(
            message.token, token
        ):
            log.warning(
                "Messaggio dalla finestra rifiutato (origine %s)", request.headers.get("origin")
            )
            raise HTTPException(status_code=403, detail="forbidden")
        try:
            message_id = store.enqueue(message.text)
            log.info("Messaggio dalla finestra in coda: %d", message_id)
            return {"id": message_id}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="empty") from exc

    @app.get("/api/inbox/{message_id}")
    def inbox(message_id: int) -> dict[str, str | None]:
        return {"state": store.inbox_state(message_id)}

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
