"""The PC tools without Windows (v0.3.1): levels fixed in code, the confirmation question,
the dispatcher, the Claude tool loop and the spoken yes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from claudio.brain.claude import ClaudeBrain
from claudio.brain.prompts import SYSTEM_PROMPT_TOOLS
from claudio.config import ClaudeSettings, PcSettings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.pc.files import Found
from claudio.pc.tools import DEFINITIONS, LEVELS, App, PcTools, ToolCall, Window
from claudio.router.matcher import CommandMatcher
from tests.unit.test_claude_brain import Monotonic, ScriptedClient, hybrid, say
from tests.unit.test_cost import PRICES, tracker


class FakeDesktop:
    """Records what would happen on Windows."""

    def __init__(self) -> None:
        self.done: list[tuple[str, Any]] = []
        self.app_list = [
            App("Google Chrome", "Chrome"),
            App("Spotify", "SpotifyAB.Spotify"),
            App("Blocco note", "notepad"),
        ]
        self.open_windows = [
            Window(1, "Senza nome - Blocco note", "notepad.exe", active=True),
            Window(2, "YouTube - Google Chrome", "chrome.exe"),
            Window(3, "KeePassXC", "keepassxc.exe"),
        ]
        self.index: list[Found] | None = None
        self.text = "Riga uno\nRiga due"

    def thread_init(self) -> None:
        self.done.append(("thread_init", None))

    def apps(self) -> list[App]:
        self.done.append(("apps", None))
        return self.app_list

    def launch(self, app: App) -> None:
        self.done.append(("launch", app.name))

    def open_path(self, path: Path) -> None:
        self.done.append(("open", path))

    def search_index(
        self, terms: list[str], kind: str | None, roots: list[Path], limit: int
    ) -> list[Found] | None:
        self.done.append(("index", terms))
        return self.index

    def windows(self) -> list[Window]:
        return self.open_windows

    def read_window(self, window: Window, max_chars: int, timeout_s: float) -> str:
        self.done.append(("read", window.title))
        return self.text

    def focus(self, window: Window) -> None:
        self.done.append(("focus", window.title))

    def close(self, window: Window) -> None:
        self.done.append(("close", window.title))

    def send_keys(self, sequence: str) -> None:
        self.done.append(("keys", sequence))


@pytest.fixture
def home(tmp_path: Path) -> Path:
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Documents").mkdir()
    (tmp_path / "Downloads" / "fattura.pdf").write_text("x")
    return tmp_path


@pytest.fixture
def desk() -> FakeDesktop:
    return FakeDesktop()


@pytest.fixture
def tools(desk: FakeDesktop, home: Path) -> PcTools:
    t = PcTools(desk, PcSettings(), home=home)
    yield t
    t.close()


@pytest.fixture
def verbose(desk: FakeDesktop, home: Path) -> PcTools:
    """The question that reads the action out ([pc] confirm_brief = false)."""
    t = PcTools(desk, PcSettings(confirm_brief=False), home=home)
    yield t
    t.close()


def call(tool: str, **args: Any) -> ToolCall:
    return ToolCall(f"id_{tool}", tool, args)


# --- levels and definitions -----------------------------------------------------------------


def test_every_tool_has_a_definition_and_a_level() -> None:
    assert {d["name"] for d in DEFINITIONS} == set(LEVELS)
    for d in DEFINITIONS:
        assert d["input_schema"]["additionalProperties"] is False


def test_levels_are_fixed_in_code(tools: PcTools) -> None:
    assert tools.level(call("open_app", name="x")) == 1
    assert tools.level(call("read_window")) == 1
    assert tools.level(call("type_text", text="ciao")) == 2
    assert tools.level(call("close_window", window="x")) == 2
    assert tools.level(call("move_file", source="a", destination="b")) == 2


def test_harmless_keys_run_at_once_the_others_ask(tools: PcTools) -> None:
    assert tools.level(call("press_keys", keys=["page_down", "ctrl+t"])) == 1
    assert tools.level(call("press_keys", keys=["ctrl+t", "enter"])) == 2
    assert tools.level(call("press_keys", keys=["alt+f4"])) == 2
    assert tools.level(call("press_keys", keys=["shift+delete"])) == 2


def test_prudent_setting_asks_for_every_key(desk: FakeDesktop, home: Path) -> None:
    t = PcTools(desk, PcSettings(safe_keys=[]), home=home)
    assert t.level(call("press_keys", keys=["down"])) == 2
    t.close()


async def test_level_2_asks_only_confermi_by_default(tools: PcTools) -> None:
    """JenJo (23/10): "Confermi?" alone, without repeating the action."""
    q = await tools.question(
        [call("type_text", text="ciao", window="chrome"), call("press_keys", keys=["enter"])],
        "it",
    )
    assert q == "Confermi?"
    assert await tools.question([call("close_window", window="")], "en") == "Do you confirm?"


async def test_level_3_still_says_what_will_run(
    tools: PcTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit "confermo" must know to what: the brief question is for level 2 only."""
    monkeypatch.setattr(tools, "level", lambda c: 3)
    q = await tools.question([call("close_window", window="")], "it")
    assert q.startswith('Chiudo la finestra "Senza nome - Blocco note"') and "confermo" in q


async def test_the_question_says_what_will_run(verbose: PcTools) -> None:
    q = await verbose.question(
        [call("type_text", text="ciao", window="chrome"), call("press_keys", keys=["enter"])],
        "it",
    )
    assert q == 'Scrivo "ciao" in "YouTube - Google Chrome", poi premo enter. Confermi?'
    q = await verbose.question([call("close_window", window="")], "en")
    assert q == 'I\'ll close the window "Senza nome - Blocco note". Do you confirm?'
    q = await verbose.question(
        [call("move_file", source="~/Downloads/a.pdf", destination="~")], "it"
    )
    assert q.startswith('Sposto "a.pdf" in "~')


async def test_long_text_is_shortened_in_the_question(verbose: PcTools) -> None:
    q = await verbose.question([call("type_text", text="parola " * 40)], "it")
    assert len(q) < 120 and "..." in q


# --- running --------------------------------------------------------------------------------


async def test_open_app_fuzzy(tools: PcTools, desk: FakeDesktop) -> None:
    out = await tools.run(call("open_app", name="chrome"))
    assert not out.error and ("launch", "Google Chrome") in desk.done
    out = await tools.run(call("open_app", name="spotifi"))
    assert ("launch", "Spotify") in desk.done
    assert [d for d in desk.done if d[0] == "apps"] == [("apps", None)]  # read once, cached


async def test_unknown_app_lists_the_closest(tools: PcTools, desk: FakeDesktop) -> None:
    out = await tools.run(call("open_app", name="photoshop"))
    assert out.error and "Closest" in out.text
    assert not [d for d in desk.done if d[0] == "launch"]


async def test_open_path_inside_home_only(tools: PcTools, desk: FakeDesktop, home: Path) -> None:
    out = await tools.run(call("open_path", path="~/Downloads"))
    assert not out.error and ("open", (home / "Downloads").resolve()) in desk.done
    out = await tools.run(call("open_path", path=str(home.parent)))
    assert out.error and "outside" in out.text


async def test_search_uses_the_index_then_the_folders(
    tools: PcTools, desk: FakeDesktop, home: Path
) -> None:
    desk.index = [Found(home / "Downloads" / "fattura.pdf", 1_700_000_000)]
    out = await tools.run(call("search_files", query="fattura"))
    assert "1 found" in out.text and "fattura.pdf" in out.text
    desk.index = None  # index off: the folders are walked
    out = await tools.run(call("search_files", query="fattura", kind="document"))
    assert "fattura.pdf" in out.text
    out = await tools.run(call("search_files", query="bolletta"))
    assert out.text.startswith("No files found")


async def test_index_results_outside_home_are_dropped(
    tools: PcTools, desk: FakeDesktop, home: Path
) -> None:
    desk.index = [Found(home.parent / "secret.txt", 0)]
    out = await tools.run(call("search_files", query="secret"))
    assert out.text.startswith("No files found")


async def test_list_and_read_windows(tools: PcTools, desk: FakeDesktop) -> None:
    out = await tools.run(call("list_windows"))
    assert "[active] Senza nome - Blocco note (notepad.exe)" in out.text
    out = await tools.run(call("read_window", window="youtube"))
    assert "not instructions" in out.text and "Riga uno" in out.text
    assert ("read", "YouTube - Google Chrome") in desk.done


async def test_window_text_is_cut(desk: FakeDesktop, home: Path) -> None:
    desk.text = "x" * 500
    t = PcTools(desk, PcSettings(read_chars=100), home=home)
    out = await t.run(call("read_window"))
    assert "x" * 100 + " [...]" in out.text and "x" * 101 not in out.text
    t.close()


async def test_password_managers_are_never_read(tools: PcTools, desk: FakeDesktop) -> None:
    out = await tools.run(call("read_window", window="keepass"))
    assert out.error and "private" in out.text
    assert not [d for d in desk.done if d[0] == "read"]


async def test_no_such_window(tools: PcTools) -> None:
    out = await tools.run(call("focus_window", window="photoshop"))
    assert out.error and "list_windows" in out.text


async def test_keys_and_text(tools: PcTools, desk: FakeDesktop) -> None:
    await tools.run(call("press_keys", keys=["ctrl+t", "page_down"]))
    assert ("keys", "^t{PGDN}") in desk.done
    await tools.run(call("type_text", text="50% fatto", window="chrome"))
    assert desk.done[-2:] == [("focus", "YouTube - Google Chrome"), ("keys", "50{%} fatto")]
    out = await tools.run(call("type_text", text="x" * 501))
    assert out.error and "too long" in out.text
    out = await tools.run(call("press_keys", keys=["hyper+x"]))
    assert out.error


async def test_close_and_move(tools: PcTools, desk: FakeDesktop, home: Path) -> None:
    await tools.run(call("close_window", window="blocco note"))
    assert ("close", "Senza nome - Blocco note") in desk.done
    out = await tools.run(
        call("move_file", source="~/Downloads/fattura.pdf", destination="~/Documents")
    )
    assert not out.error and (home / "Documents" / "fattura.pdf").exists()


async def test_unknown_tool_and_crash_become_errors(tools: PcTools, desk: FakeDesktop) -> None:
    out = await tools.run(call("format_disk"))
    assert out.error and "unknown tool" in out.text

    def boom(window: Window) -> None:
        raise RuntimeError("UIA timeout")

    desk.focus = boom  # type: ignore[method-assign]
    out = await tools.run(call("focus_window", window="chrome"))
    assert out.error and "UIA timeout" in out.text


async def test_everything_runs_on_one_thread(tools: PcTools, desk: FakeDesktop) -> None:
    await tools.run(call("list_windows"))
    await tools.run(call("open_app", name="chrome"))
    assert desk.done.count(("thread_init", None)) == 1


# --- the Claude loop ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


def use(tool: str, **args: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=f"tu_{tool}", name=tool, input=args)


def text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def acting_brain(pricing: Pricing, client: Any, tools: PcTools, **pc: Any) -> ClaudeBrain:
    return ClaudeBrain(
        client,
        tracker(pricing, 1.0),
        ClaudeSettings(tell_time=False),
        clock=Monotonic(),
        tools=tools,
        pc=PcSettings(**pc),
    )


async def ask(b: ClaudeBrain, said: str) -> str:
    return "".join([c async for c in b.ask(said, "it")])


async def test_level_1_runs_and_haiku_answers(
    pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("open_app", name="spotify")]),
        (["Fatto, Spotify è aperto."], "end_turn", [text("Fatto, Spotify è aperto.")]),
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "apri spotify") == "Fatto, Spotify è aperto."
    assert ("launch", "Spotify") in desk.done
    first, second = client.calls
    assert first["system"] == SYSTEM_PROMPT_TOOLS
    assert [t["name"] for t in first["tools"]][:2] == ["ask_sonnet", "open_app"]
    assert first["tools"] == second["tools"]  # same prefix: cached
    assert second["messages"][-2]["content"][0]["type"] == "tool_use"
    result = second["messages"][-1]["content"][0]
    assert result == {
        "type": "tool_result",
        "tool_use_id": "tu_open_app",
        "content": "Started Spotify.",
    }
    assert b.last_call["tools"] == ["open_app(name=spotify) ok"]
    assert not b.awaiting


async def test_several_tools_in_one_reply(
    pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("list_windows"), use("read_window", window="youtube")]),
        (["Stai guardando YouTube."], "end_turn", []),
    )
    await ask(acting_brain(pricing, client, tools), "cosa c'è sullo schermo")
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_list_windows", "tu_read_window"]


async def test_errors_go_back_to_haiku(pricing: Pricing, tools: PcTools) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("open_path", path="C:/Windows/System32")]),
        (["Non posso aprire quella cartella."], "end_turn", []),
    )
    await ask(acting_brain(pricing, client, tools), "apri system32")
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


async def test_rounds_are_limited(pricing: Pricing, tools: PcTools) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("list_windows")]),
        ([], "tool_use", [use("list_windows")]),
        (["Basta così."], "end_turn", []),
    )
    await ask(acting_brain(pricing, client, tools, max_rounds=3), "fai tante cose")
    assert len(client.calls) == 3
    assert "tool_choice" not in client.calls[1]
    assert client.calls[2]["tool_choice"] == {"type": "none"}  # the last round: words only
    assert client.calls[2]["tools"] == client.calls[0]["tools"]


async def test_level_2_asks_and_runs_after_a_yes(
    pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    client = ScriptedClient(
        (["Ok. "], "tool_use", [text("Ok. "), use("type_text", text="ciao", window="blocco")]),
        (["Scritto."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    reply = await ask(b, "scrivi ciao nel blocco note")
    assert reply == "Ok. Confermi?"
    assert b.awaiting and not [d for d in desk.done if d[0] == "keys"]  # nothing typed yet
    assert b.last_call["tools"] == ["type_text(text=ciao, window=blocco) ask"]
    assert await ask(b, "sì") == "Scritto."
    assert ("keys", "ciao") in desk.done and not b.awaiting
    assert client.calls[1]["messages"][-1]["content"][0]["content"] == "Typed 4 characters."
    assert client.calls[1]["messages"][0]["content"].endswith("scrivi ciao nel blocco note")


@pytest.mark.parametrize(("answer", "reply"), [("no", "Va bene"), ("boh forse", "Non ho capito")])
async def test_anything_but_yes_cancels(
    pricing: Pricing, tools: PcTools, desk: FakeDesktop, answer: str, reply: str
) -> None:
    client = ScriptedClient(([], "tool_use", [use("close_window", window="chrome")]))
    b = acting_brain(pricing, client, tools)
    await ask(b, "chiudi chrome")
    assert (await ask(b, answer)).startswith(reply)
    assert not [d for d in desk.done if d[0] == "close"] and len(client.calls) == 1


async def test_level_1_calls_wait_with_the_level_2_one(
    pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    """One reply, one question: focus and type run together, in order, after the yes."""
    client = ScriptedClient(
        ([], "tool_use", [use("focus_window", window="chrome"), use("press_keys", keys=["enter"])]),
        (["Inviato."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    assert (await ask(b, "premi invio su chrome")).endswith("Confermi?")
    assert not [d for d in desk.done if d[0] in ("focus", "keys")]
    await ask(b, "sì")
    assert [d for d in desk.done if d[0] in ("focus", "keys")] == [
        ("focus", "YouTube - Google Chrome"),
        ("keys", "{ENTER}"),
    ]


async def test_the_question_expires(pricing: Pricing, tools: PcTools, desk: FakeDesktop) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("close_window", window="chrome")]),
        (["Ciao."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    await ask(b, "chiudi chrome")
    b.clock.t += 61  # type: ignore[attr-defined]
    assert not b.awaiting
    await ask(b, "sì")  # a new request now, not the answer
    assert not [d for d in desk.done if d[0] == "close"]


async def test_cancel_pending(pricing: Pricing, tools: PcTools, desk: FakeDesktop) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("close_window", window="chrome")]),
        (["Ciao."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    await ask(b, "chiudi chrome")
    b.cancel_pending()
    assert not b.awaiting
    await ask(b, "sì")
    assert not [d for d in desk.done if d[0] == "close"]


async def test_history_keeps_only_words(pricing: Pricing, tools: PcTools) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("list_windows")]),
        (["Hai tre finestre."], "end_turn", []),
        (["Prego."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    await ask(b, "quante finestre ho")
    await ask(b, "grazie")
    history = client.calls[2]["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[1]["content"] == "Hai tre finestre."


async def test_without_tools_the_old_prompt(pricing: Pricing) -> None:
    client = ScriptedClient((["Ciao."], "end_turn", []))
    b = ClaudeBrain(
        client, tracker(pricing, 1.0), ClaudeSettings(tell_time=False), clock=Monotonic()
    )
    await ask(b, "ciao")
    assert [t["name"] for t in client.calls[0]["tools"]] == ["ask_sonnet"]
    assert client.calls[0]["system"] != SYSTEM_PROMPT_TOOLS


# --- through the router ---------------------------------------------------------------------


async def test_the_router_waits_for_the_answer(
    matcher: CommandMatcher, pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    """After "Confermi?" the responder is awaiting (no wake word needed) and the answer goes
    to Claude even when it sounds like a local command."""
    client = ScriptedClient(
        ([], "tool_use", [use("close_window", window="chrome")]),
        (["Chiusa."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    r = hybrid(matcher, b)
    r.fallback_awaiting = lambda: b.awaiting
    r.fallback_cancel = b.cancel_pending
    await say(r, "chiudi la finestra di chrome")
    assert r.awaiting
    assert await say(r, "sì") == "Chiusa."
    assert ("close", "YouTube - Google Chrome") in desk.done and not r.awaiting


async def test_the_router_cancels_the_action(
    matcher: CommandMatcher, pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    client = ScriptedClient(([], "tool_use", [use("close_window", window="chrome")]))
    b = acting_brain(pricing, client, tools)
    r = hybrid(matcher, b)
    r.fallback_awaiting = lambda: b.awaiting
    r.fallback_cancel = b.cancel_pending
    await say(r, "chiudi la finestra di chrome")
    r.cancel_pending()  # silence after the question
    assert not r.awaiting and not b.awaiting


# --- the real backend (Windows CI only) -----------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows APIs")
def test_windows_backend_loads_and_lists() -> None:
    from claudio.pc.windows import WindowsDesktop

    desk = WindowsDesktop()
    desk.thread_init()
    assert isinstance(desk.windows(), list)  # the CI runner may have no visible window
    import pywinauto.keyboard  # noqa: F401  # installed and importable


async def test_a_call_cut_by_max_tokens_never_runs(
    pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    client = ScriptedClient((["Apro"], "max_tokens", [use("open_app", name="spot")]))
    await ask(acting_brain(pricing, client, tools), "apri spotify")
    assert not [d for d in desk.done if d[0] == "launch"] and len(client.calls) == 1


async def test_sonnet_is_not_offered_in_the_middle_of_an_action(
    pricing: Pricing, tools: PcTools
) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("list_windows"), use("ask_sonnet")]),
        (["Ecco."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "guarda le finestre") == "Ecco."
    sent = client.calls[1]["messages"][-2]["content"]
    assert [blk["name"] for blk in sent] == ["list_windows"]  # every tool_use has its result


async def test_the_answer_keeps_the_question_language(
    matcher: CommandMatcher, pricing: Pricing, tools: PcTools
) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("close_window", window="chrome")]),
        (["Chiusa."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    r = hybrid(matcher, b)
    r.fallback_awaiting = lambda: b.awaiting
    await say(r, "chiudi la finestra di chrome")
    await say(r, "ok", language="en")  # Whisper heard "ok" as English
    assert r.language == "it"
