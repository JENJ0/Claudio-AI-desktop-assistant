import time

import pytest
from fastapi.testclient import TestClient

from claudio.chat.server import create_app
from claudio.chat.store import ChatStore
from claudio.metrics import MetricsWriter


@pytest.fixture
def store(tmp_path) -> ChatStore:
    return ChatStore(tmp_path / "claudio.db", session="2026-10-23_10-00")


def row(text: str, reply: str, ts: float | None = None, **extra) -> dict:
    return {"ts": ts or time.time(), "text": text, "reply": reply, **extra}


def test_a_request_becomes_a_question_and_an_answer(store) -> None:
    store.add(row("che ore sono", "Sono le dieci.", route="local", m2_ms=640))
    msgs = store.messages()
    assert [(m["role"], m["text"]) for m in msgs] == [
        ("user", "che ore sono"),
        ("claudio", "Sono le dieci."),
    ]
    assert msgs[1]["route"] == "local" and msgs[1]["m2_ms"] == 640


def test_rows_without_text_are_skipped(store) -> None:
    store.add({"ts": time.time(), "outcome": "near_miss"})
    store.add(row("  ", ""))
    assert store.messages() == []


def test_cost_is_kept_on_the_answer(store) -> None:
    store.add(row("domanda", "risposta", route="claude", model="haiku", cost=0.0002))
    assert store.cost_since(0) == pytest.approx(0.0002)
    assert store.cost_since(time.time() + 10) == 0


def test_search_finds_text_in_every_session_and_escapes_wildcards(tmp_path) -> None:
    ChatStore(tmp_path / "c.db", "a").add(row("apri chrome", "Fatto."))
    other = ChatStore(tmp_path / "c.db", "b")
    other.add(row("100% sicuro?", "Sì."))
    assert [m["text"] for m in other.messages(query="chrome")] == ["apri chrome"]
    assert [m["text"] for m in other.messages(query="100%")] == ["100% sicuro?"]
    assert other.messages(query="%") == [m for m in other.messages() if "%" in m["text"]]


def test_after_id_returns_only_what_is_new(store) -> None:
    store.add(row("uno", "1"))
    last = store.messages()[-1]["id"]
    store.add(row("due", "2"))
    assert [m["text"] for m in store.messages(after_id=last)] == ["due", "2"]


def test_sessions_are_listed_newest_first_with_a_preview(tmp_path) -> None:
    ChatStore(tmp_path / "c.db", "old").add(row("prima", "ok", cost=0.001))
    new = ChatStore(tmp_path / "c.db", "new")
    new.add(row("dopo", "ok"))
    sessions = new.sessions()
    assert [s["session"] for s in sessions] == ["new", "old"]
    assert sessions[1]["preview"] == "prima" and sessions[1]["cost"] == pytest.approx(0.001)


def test_purge_forgets_old_messages(store) -> None:
    now = time.time()
    store.add(row("vecchio", "x", ts=now - 40 * 86400))
    store.add(row("nuovo", "y", ts=now))
    assert store.purge_older_than(30, now=now) == 2
    assert [m["text"] for m in store.messages()] == ["nuovo", "y"]


def test_metrics_writer_feeds_the_chat(tmp_path, store) -> None:
    class Inter:
        def summary(self) -> dict:
            return {
                "text": "ciao",
                "reply": "Ciao!",
                "m2_ms": 500,
                "m1_ms": 400,
                "stt_ms": 90,
                "tts_first_ms": 80,
            }

    MetricsWriter(tmp_path / "m.jsonl", chat=store).write(Inter())  # type: ignore[arg-type]
    assert [m["text"] for m in store.messages()] == ["ciao", "Ciao!"]


def test_a_broken_database_does_not_raise(tmp_path) -> None:
    store = ChatStore(tmp_path / "c.db", "s")
    store.path.unlink()
    store.path.mkdir()  # not a file anymore: sqlite fails
    store.add(row("ciao", "ciao"))  # logged, not raised


def test_server_serves_the_page_and_the_data(store) -> None:
    store.add(row("ciao", "Ciao!", cost=0.5))
    client = TestClient(create_app(store), base_url="http://127.0.0.1:8000")
    assert "Claudio" in client.get("/").text
    assert [m["text"] for m in client.get("/api/messages").json()] == ["ciao", "Ciao!"]
    assert client.get("/api/messages", params={"q": "ciao"}).json()[0]["text"] == "ciao"
    assert client.get("/api/sessions").json()[0]["messages"] == 2
    assert client.get("/api/today").json()["cost"] == pytest.approx(0.5)


def test_server_refuses_foreign_host_headers(store) -> None:
    client = TestClient(create_app(store), base_url="http://evil.example")
    assert client.get("/api/messages").status_code == 403


def test_server_rejects_bad_parameters(store) -> None:
    client = TestClient(create_app(store), base_url="http://127.0.0.1:8000")
    assert client.get("/api/messages", params={"limit": 100000}).status_code == 422
    assert client.get("/api/messages", params={"after_id": -1}).status_code == 422


# --- v0.2.2: writing to Claudio from the window ----------------------------------------


def test_inbox_hands_each_message_over_once(store) -> None:
    first = store.enqueue("  apri   chrome ")
    store.enqueue("che ore sono")
    assert store.inbox_state(first) == "pending"
    assert store.take_pending(max_age_s=60) == [(first, "apri chrome"), (first + 1, "che ore sono")]
    assert store.take_pending(max_age_s=60) == []
    assert store.inbox_state(first) == "taken"
    assert store.inbox_state(999) is None


def test_inbox_drops_what_was_written_while_claudio_was_off(store) -> None:
    old = store.enqueue("ciao", now=time.time() - 600)
    assert store.take_pending(max_age_s=120) == []
    assert store.inbox_state(old) == "expired"


def test_inbox_refuses_empty_text(store) -> None:
    with pytest.raises(ValueError):
        store.enqueue("   ")


def page_token(client: TestClient) -> str:
    html = client.get("/").text
    return html.split('const TOKEN = "', 1)[1].split('"', 1)[0]


def test_the_window_can_send_with_its_token(store) -> None:
    client = TestClient(create_app(store), base_url="http://127.0.0.1:8000")
    token = page_token(client)
    assert token and token != "__CLAUDIO_TOKEN__"
    sent = client.post("/api/send", json={"text": "ciao Claudio", "token": token})
    assert sent.status_code == 200
    message_id = sent.json()["id"]
    assert client.get(f"/api/inbox/{message_id}").json() == {"state": "pending"}
    assert store.take_pending(60) == [(message_id, "ciao Claudio")]


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"json": {"text": "ciao", "token": "wrong"}}, 403),
        (
            {"json": {"text": "ciao", "token": "TOKEN"}, "headers": {"origin": "https://evil.it"}},
            403,
        ),
        ({"data": {"text": "ciao", "token": "TOKEN"}}, 422),  # an HTML form: not JSON
        ({"json": {"text": "", "token": "TOKEN"}}, 422),
        ({"json": {"text": "x" * 1001, "token": "TOKEN"}}, 422),
    ],
)
def test_send_refuses_anything_but_the_window(store, kwargs, status) -> None:
    client = TestClient(create_app(store, token="TOKEN"), base_url="http://127.0.0.1:8000")
    assert client.post("/api/send", **kwargs).status_code == status
    assert store.take_pending(60) == []


def test_send_accepts_the_local_origin(store) -> None:
    client = TestClient(create_app(store, token="T"), base_url="http://127.0.0.1:8000")
    headers = {"origin": "http://127.0.0.1:8000"}
    assert client.post("/api/send", json={"text": "ok", "token": "T"}, headers=headers).is_success


def test_the_window_is_raised_without_staying_on_top() -> None:
    import ctypes
    from types import SimpleNamespace

    from claudio.chat.window import HWND_NOTOPMOST, HWND_TOPMOST, bring_to_front

    calls: list[tuple] = []
    user32 = SimpleNamespace(SetWindowPos=lambda *a: calls.append(a))
    handle = SimpleNamespace(ToInt64=lambda: 1234)
    bring_to_front(SimpleNamespace(native=SimpleNamespace(Handle=handle)), user32)
    assert [c[0].value for c in calls] == [1234, 1234]
    # c_void_p keeps the -1/-2 sentinels as their unsigned bit pattern, as Win32 reads them
    expected = [ctypes.c_void_p(h).value for h in (HWND_TOPMOST, HWND_NOTOPMOST)]
    assert [c[1].value for c in calls] == expected
    assert all(c[6] & 0x4000 for c in calls)  # asynchronous: posted to the window's thread
    bring_to_front(SimpleNamespace(native=None), user32)  # not created yet: nothing
    broken = SimpleNamespace(native=SimpleNamespace(Handle=None))
    bring_to_front(broken, user32)  # never raises
    assert len(calls) == 2
