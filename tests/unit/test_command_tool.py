"""IT-0: Claudio's own commands as a tool for Claude. "Claudio, ok, la chata" (26/10) reached
Claude, which could guess the chat window but had no tool to open it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from claudio.brain.claude import CUT_NOTE
from claudio.config import PcSettings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.pc.tools import PcTools
from claudio.router.actions import MEDIA_KEYS, ActionRunner
from claudio.router.command_tool import NAME, CommandTool
from claudio.router.matcher import load_commands
from claudio.system import SystemControl
from tests.conftest import COMMANDS_FILE
from tests.unit.test_actions import Recorder
from tests.unit.test_claude_brain import ScriptedClient
from tests.unit.test_cost import PRICES, tracker
from tests.unit.test_pc_tools import FakeDesktop, acting_brain, ask, call, text, use


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


@pytest.fixture
def system() -> SystemControl:
    return SystemControl()


@pytest.fixture
def tool(rec: Recorder, system: SystemControl) -> CommandTool:
    return CommandTool(load_commands(COMMANDS_FILE), ActionRunner(rec.ctx(), system=system))


@pytest.fixture
def tools(tool: CommandTool, tmp_path: Path) -> PcTools:
    t = PcTools(FakeDesktop(), PcSettings(), home=tmp_path, commands=tool)
    yield t
    t.close()


def test_only_what_claude_cannot_do_otherwise(tool: CommandTool) -> None:
    offered = set(tool.definition["input_schema"]["properties"]["command"]["enum"])
    assert {"volume_up", "mute", "media_next", "open_chat", "pause", "shutdown"} <= offered
    assert {"tell_spend"} <= offered
    # Claude opens apps and folders itself, knows the time, and is never asked while paused;
    # "basta" and "ripeti" never reach it.
    assert not offered & {"open_chrome", "open_downloads", "tell_time", "resume", "stop"}
    assert not offered & {"repeat", "close_window"}
    assert "volume_up (alza il volume)" in tool.definition["description"]


def test_switching_off_and_pausing_are_confirmed(tool: CommandTool) -> None:
    assert tool.level({"command": "shutdown"}) == 2
    assert tool.level({"command": "pause"}) == 2
    assert tool.level({"command": "volume_up"}) == 1
    assert tool.level({"command": "nonsense"}) == 1  # nothing runs: the result says why


def test_runs_through_the_routers_actions(
    tool: CommandTool, rec: Recorder, system: SystemControl
) -> None:
    assert tool.run({"command": "volume_up", "number": 3}) == "Volume alzato."
    assert rec.keys == [MEDIA_KEYS["volume_up"]] * 3
    assert tool.run({"command": "shutdown"}, "en") == "See you soon."  # the request's language
    assert system.stopping


def test_number_is_kept_in_range(tool: CommandTool, rec: Recorder) -> None:
    tool.run({"command": "media_next", "number": 500})
    assert len(rec.keys) == 50


async def test_offered_with_the_pc_tools(tools: PcTools, rec: Recorder) -> None:
    assert tools.definitions()[-1]["name"] == NAME
    out = await tools.run(call(NAME, command="mute"))
    assert not out.error and rec.keys == [MEDIA_KEYS["mute"]]
    out = await tools.run(call(NAME, command="format_disk"))
    assert out.error and "unknown command" in out.text


async def test_runs_on_the_thread_of_the_spending_ledger(
    pricing: Pricing, rec: Recorder, tmp_path: Path
) -> None:
    # PC test 29/10: on the PC tools' worker thread SQLite refused the ledger opened here.
    tool = CommandTool(
        load_commands(COMMANDS_FILE), ActionRunner(rec.ctx(), tracker=tracker(pricing))
    )
    t = PcTools(FakeDesktop(), PcSettings(), home=tmp_path, commands=tool)
    try:
        out = await t.run(call(NAME, command="tell_spend"), "en")
    finally:
        t.close()
    assert not out.error and "spent" in out.text


async def test_the_question_names_the_command(tool: CommandTool, tmp_path: Path) -> None:
    verbose = PcTools(FakeDesktop(), PcSettings(confirm_brief=False), home=tmp_path, commands=tool)
    try:
        q = await verbose.question([call(NAME, command="shutdown")], "it")
    finally:
        verbose.close()
    assert q == "Chiuditi. Confermi?"  # "spegniti" is off now, "chiuditi" closes


async def test_garbled_request_runs_the_command_and_says_its_reply(
    pricing: Pricing, tools: PcTools, rec: Recorder
) -> None:
    """IT-1 5.3: the command's own reply, at once: no second call just to say "Fatto"."""
    client = ScriptedClient(([], "tool_use", [use(NAME, command="volume_up", last=True)]))
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "Audio alsa il volu") == "Volume alzato."
    assert rec.keys and set(rec.keys) == {MEDIA_KEYS["volume_up"]}
    assert len(client.calls) == 1
    assert b.last_call["tools"] == ["claudio_command(command=volume_up, last=True) ok"]
    # The history keeps the words and what ran, as after a reply of Claude's.
    later = ScriptedClient((["Ok."], "end_turn", []))
    b.client = later
    await ask(b, "grazie")
    assert later.calls[0]["messages"][1] == {
        "role": "assistant",
        "content": "Volume alzato. [Claudio ran: claudio_command ok]",
    }


async def test_what_haiku_said_comes_first_and_a_media_key_stays_silent(
    pricing: Pricing, tools: PcTools, rec: Recorder
) -> None:
    said = "Metto in pausa, socio."
    client = ScriptedClient(
        ([said], "tool_use", [text(said), use(NAME, command="media_play_pause", last=True)]),
        ([], "tool_use", [use(NAME, command="mute", last=True)]),
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "ferma la musica") == f"{said} "  # play/pause has no reply
    assert await ask(b, "togli l'audio") == ""  # nothing said: silent, as with the router
    assert len(client.calls) == 2 and len(rec.keys) == 2


async def test_claude_still_reads_mixed_or_failed_rounds(pricing: Pricing, tools: PcTools) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use(NAME, command="volume_up"), use("open_app", name="spotify")]),
        (["Spotify aperto, volume su."], "end_turn", []),
        ([], "tool_use", [use(NAME, command="format_disk", last=True)]),
        (["Non posso."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "apri spotify e alza") == "Spotify aperto, volume su."
    assert client.calls[1]["messages"][-1]["content"][0]["content"] == "Volume alzato."
    assert await ask(b, "formatta il disco") == "Non posso."
    assert len(client.calls) == 4


async def test_a_command_after_a_tool_without_words_still_goes_back_to_claude(
    pricing: Pricing, tools: PcTools
) -> None:
    client = ScriptedClient(
        (
            [],
            "tool_use",
            [use("open_app", name="spotify"), use(NAME, command="volume_up", last=True)],
        ),
        (["Spotify aperto, volume su."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "apri spotify e alza") == "Spotify aperto, volume su."
    assert len(client.calls) == 2


async def test_a_cut_at_the_commands_own_reply_is_marked(pricing: Pricing, tools: PcTools) -> None:
    client = ScriptedClient(([], "tool_use", [use(NAME, command="volume_up", last=True)]))
    b = acting_brain(pricing, client, tools)
    stream = b.ask("Audio alsa il volu", "it")
    assert await anext(stream) == "Volume alzato."
    await stream.aclose()
    # The finished entry before this one is the doubled cut bookkeeping: a known bug, not pinned.
    # No action note: a known bug to fix (a cut loses the read-outside taint), not intended.
    assert b._history[-1] == ("[it] Audio alsa il volu", f"Volume alzato. {CUT_NOTE}")


async def test_nothing_claude_meant_to_add_is_dropped(pricing: Pricing, tools: PcTools) -> None:
    """Review of 5.3: the shortcut ends a request only when Claude marked the command
    `last` and nothing else ran or was asked for in it."""
    sonnet = SimpleNamespace(type="tool_use", id="tu_s", name="ask_sonnet", input={})
    client = ScriptedClient(
        # "abbassa il volume e dimmi che ore sono": not marked, the time comes next.
        ([], "tool_use", [use(NAME, command="volume_down")]),
        (["Fatto, e sono le 21:30."], "end_turn", []),
        # "apri spotify e poi alza il volume": the command comes in the second round.
        ([], "tool_use", [use("open_app", name="spotify")]),
        ([], "tool_use", [use(NAME, command="volume_up", last=True)]),
        (["Spotify aperto e volume alzato."], "end_turn", []),
        # "abbassa il volume e parlami della relatività": Haiku also asks for Sonnet.
        ([], "tool_use", [use(NAME, command="volume_down", last=True), sonnet]),
        (["La relatività dice che..."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "abbassa il volume e dimmi che ore sono") == "Fatto, e sono le 21:30."
    assert await ask(b, "apri spotify e poi alza il volume") == "Spotify aperto e volume alzato."
    assert await ask(b, "abbassa e parlami della relatività") == "La relatività dice che..."
    assert len(client.calls) == 7


async def test_switching_off_waits_for_the_yes(
    pricing: Pricing, tools: PcTools, system: SystemControl
) -> None:
    client = ScriptedClient(([], "tool_use", [use(NAME, command="shutdown", last=True)]))
    b = acting_brain(pricing, client, tools)
    assert (await ask(b, "Claudio, spegni ti")).endswith("Confermi?")
    assert not system.stopping and b.awaiting
    assert await ask(b, "sì") == "A presto."  # the command's own reply
    assert system.stopping
    assert len(client.calls) == 1  # no call after the yes


async def test_a_silent_command_stays_silent_after_the_yes(
    pricing: Pricing, tools: PcTools, rec: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "level", lambda c: 2)  # no silent command asks in commands.toml
    client = ScriptedClient(([], "tool_use", [use(NAME, command="mute", last=True)]))
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "togli l'audio") == "Confermi?"
    assert [c async for c in b.ask("sì", "it")] == []
    assert rec.keys == [MEDIA_KEYS["mute"]] and len(client.calls) == 1
    assert b._history[-1] == (
        "[it] togli l'audio",
        "Confermi? [Claudio ran: claudio_command ok]",
    )
