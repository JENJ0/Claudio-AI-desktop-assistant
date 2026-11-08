"""IT-0: Claudio's own commands as a tool for Claude. "Claudio, ok, la chata" (26/10) reached
Claude, which could guess the chat window but had no tool to open it."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from claudio.__main__ import build_responder
from claudio.brain.claude import CUT_NOTE
from claudio.config import PcSettings, load_settings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.pc.tools import PcTools
from claudio.router.actions import MEDIA_KEYS, ActionRunner
from claudio.router.command_tool import NAME, CommandTool
from claudio.router.matcher import CommandMatcher, load_commands
from claudio.router.responder import LocalResponder
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


async def test_a_note_before_the_command_leaves_only_its_reply(
    pricing: Pricing, tools: PcTools
) -> None:
    note = "[Claudio ran: open_app ok]"
    client = ScriptedClient(
        ([note], "tool_use", [text(note), use(NAME, command="volume_up", last=True)])
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "Audio alsa il volu") == "Volume alzato."
    assert list(b._history) == [
        ("[it] Audio alsa il volu", "Volume alzato. [Claudio ran: claudio_command ok]")
    ]


async def test_a_media_key_stays_silent_whatever_haiku_said_before_it(
    pricing: Pricing, tools: PcTools, rec: Recorder
) -> None:
    said = "Metto in pausa, socio."
    client = ScriptedClient(
        ([said], "tool_use", [text(said), use(NAME, command="media_play_pause", last=True)]),
        ([], "tool_use", [use(NAME, command="mute", last=True)]),
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "ferma la musica") == ""  # Task 10: the words before it are held, dropped
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
    assert b._history[-1] == (
        "[it] Audio alsa il volu",
        f"Volume alzato. {CUT_NOTE} [Claudio ran: claudio_command ok]",
    )


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
    # The model said nothing before the question: it reads the action out (Task 10 fix round 1).
    assert await ask(b, "togli l'audio") == "Togli l audio. Confermi?"
    assert [c async for c in b.ask("sì", "it")] == []
    assert rec.keys == [MEDIA_KEYS["mute"]] and len(client.calls) == 1
    assert b._history[-1] == (
        "[it] togli l'audio",
        "Togli l audio. Confermi? [Claudio ran: claudio_command ok]",
    )


# --- the waiting command said again counts as yes ---------------------------------------------


OFF_REPLY = "Mi spengo. Per riaccendermi di' Claudio, accenditi."
OFF_REPLY_EN = "Turning off. To turn me back on, say Claudio, turn on."
ASKED_02_11 = (
    "Spegnermi? Va bene, socio, ma il mondo perderà il suo unico vero eroe. Confermi che mi spengo?"
)
NOT_UNDERSTOOD_IT = "Non ho capito: non lo faccio."
NOT_UNDERSTOOD_EN = "I didn't catch that: not doing it."


def waiting(*calls: SimpleNamespace) -> ScriptedClient:
    """A reply that calls these tools, without a word."""
    return ScriptedClient(([], "tool_use", list(calls)))


async def test_the_sequence_of_02_11_switches_off_at_the_repeat(
    pricing: Pricing, tools: PcTools, system: SystemControl
) -> None:
    """16:23:52: "Claudio, spegni." got "Non ho capito": 4 phrases and ~25 s to switch off."""
    client = ScriptedClient(
        ([ASKED_02_11], "tool_use", [text(ASKED_02_11), use(NAME, command="off", last=True)])
    )
    b = acting_brain(pricing, client, tools)
    assert await ask(b, "Ehi Claudio, spenitri.") == f"{ASKED_02_11} Confermi?"
    assert not system.off and b.awaiting
    assert await ask(b, "Claudio, spegni.") == OFF_REPLY
    assert system.off and not b.awaiting
    assert len(client.calls) == 1  # what a yes costs: no call after it
    assert b.last_call["confirm"] == "yes"
    assert b.last_call["tools"] == ["claudio_command(command=off, last=True) ok"]


async def test_the_repeat_reaches_the_brain_through_the_router(
    matcher: CommandMatcher, pricing: Pricing, tools: PcTools, system: SystemControl
) -> None:
    client = waiting(use(NAME, command="off", last=True))
    b = acting_brain(pricing, client, tools)
    r = LocalResponder(
        matcher,
        ActionRunner(Recorder().ctx(), system=system),
        fallback=b.ask,
        fallback_info=lambda: b.last_call,
        fallback_awaiting=lambda: b.awaiting,
    )
    asked = "".join([c async for c in r.respond("Ehi Claudio, spenitri.", "it")])
    assert asked == "Spegniti. Confermi?"  # nothing said before it: the action is read out
    assert "".join([c async for c in r.respond("Claudio, spegni.", "it")]) == OFF_REPLY
    assert system.off and len(client.calls) == 1


@pytest.mark.parametrize(
    ("command", "said", "state"),
    [
        ("off", "spegniti", "off"),
        ("off", "Claudio, spegni ti", "off"),
        ("off", "Ehi Claudio, spegniti per favore", "off"),
        ("off", "puoi spegnerti", "off"),
        ("off", "sì, spegniti", "off"),  # 84.2 with its "sì": the yes word is not compared
        ("off", "ok, spegniti", "off"),
        ("off", "Claudio, certo, spegni", "off"),
        # Longer than any answer: a new request, were it not the very command that waits.
        ("off", "Claudio, smetti di ascoltare per favore adesso", "off"),
        ("shutdown", "Claudio, chiuditi.", "stopping"),
        ("pause", "vai in pausa", "paused"),
    ],
)
async def test_the_waiting_command_said_again_is_a_yes(
    pricing: Pricing,
    tools: PcTools,
    system: SystemControl,
    command: str,
    said: str,
    state: str,
) -> None:
    client = waiting(use(NAME, command=command, last=True))
    b = acting_brain(pricing, client, tools)
    await ask(b, "parole confuse")
    assert not getattr(system, state)
    await ask(b, said)
    assert getattr(system, state)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("asked_in", "said", "reply", "runs"),
    [
        ("en", "Switch yourself off.", OFF_REPLY_EN, True),
        ("en", "Sure, go silent.", OFF_REPLY_EN, True),
        ("it", "Claudio, spegni.", OFF_REPLY, True),
        ("it", "Switch yourself off.", NOT_UNDERSTOOD_IT, False),
        ("en", "Claudio, spegni.", NOT_UNDERSTOOD_EN, False),
    ],
)
async def test_a_repeat_counts_only_in_the_language_of_the_question(
    pricing: Pricing,
    tools: PcTools,
    system: SystemControl,
    asked_in: str,
    said: str,
    reply: str,
    runs: bool,
) -> None:
    b = acting_brain(pricing, waiting(use(NAME, command="off", last=True)), tools)
    [_ async for _ in b.ask("garbled words", asked_in)]
    assert "".join([c async for c in b.ask(said, asked_in)]) == reply
    assert system.off is runs


async def test_a_word_close_to_a_phrase_of_another_language_closes_nothing(
    pricing: Pricing, tools: PcTools, system: SystemControl
) -> None:
    """ "qui" is 85.7 from the English "quit": over an Italian question it is only "qui"."""
    b = acting_brain(pricing, waiting(use(NAME, command="shutdown", last=True)), tools)
    await ask(b, "parole confuse")
    assert await ask(b, "qui") == NOT_UNDERSTOOD_IT
    assert not system.stopping


@pytest.mark.parametrize(
    "said",
    [
        "non spegnerti",
        "no spegniti",
        "no, spegniti",
        "spegniti no",
        "Claudio, non spegnerti",
        "sì, non spegnerti",
        "stop listening",
    ],
)
async def test_a_refusal_wins_over_a_repeat(
    pricing: Pricing, tools: PcTools, system: SystemControl, said: str
) -> None:
    b = acting_brain(pricing, waiting(use(NAME, command="off", last=True)), tools)
    await ask(b, "parole confuse")
    assert await ask(b, said) == "Va bene, non lo faccio."
    assert not system.off


@pytest.mark.parametrize(
    "said",
    [
        "spegni la luce",
        "sì, spegni la luce",
        "accenditi",
        "apri chrome",
        "vai in pausa",  # a command too, but not the one that waits
        "Ti prendo i niti.",  # 02/11 16:23:57
        "Spegniti?",  # a question is no yes
        "Spegniti?!",
        "Spegniti?.",
        "Spegniti?...",
        "Spegniti? Grazie.",
        "Sì, spegniti?!",
        "Spegni?!",
        "Claudio.",
    ],
)
async def test_what_is_not_the_waiting_command_is_no_yes(
    pricing: Pricing, tools: PcTools, system: SystemControl, said: str
) -> None:
    client = waiting(use(NAME, command="off", last=True))
    b = acting_brain(pricing, client, tools)
    await ask(b, "parole confuse")
    assert await ask(b, said) == NOT_UNDERSTOOD_IT
    assert not system.off and not system.paused
    assert len(client.calls) == 1


async def test_a_longer_request_still_replaces_the_waiting_command(
    pricing: Pricing, tools: PcTools, system: SystemControl
) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use(NAME, command="off", last=True)]),
        (["Qui non ho luci."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    await ask(b, "parole confuse")
    assert await ask(b, "spegni la luce della cucina per favore") == "Qui non ho luci."
    assert not system.off and len(client.calls) == 2


@pytest.mark.parametrize(
    "calls",
    [
        [use(NAME, command="off", last=True), use("open_app", name="spotify")],
        [use(NAME, command="off"), use(NAME, command="shutdown")],
        [use(NAME, command="off", last=True), use(NAME, command="off", last=True)],
    ],
)
async def test_an_action_of_several_calls_is_never_a_repeat(
    pricing: Pricing, tools: PcTools, system: SystemControl, calls: list[SimpleNamespace]
) -> None:
    b = acting_brain(pricing, waiting(*calls), tools)
    await ask(b, "parole confuse")
    assert await ask(b, "spegniti") == NOT_UNDERSTOOD_IT
    assert not system.off and not system.stopping


async def test_saying_another_tool_again_is_no_yes(pricing: Pricing, tools: PcTools) -> None:
    b = acting_brain(pricing, waiting(use("close_window", window="chrome")), tools)
    await ask(b, "chiudi chrome")
    assert await ask(b, "chiudi chrome") == NOT_UNDERSTOOD_IT
    assert not [d for d in tools.desktop.done if d[0] == "close"]
    # Not even a tool whose argument is called "command" too and names one of ours.
    assert not tools.repeats([call("run_command", command="off")], "spegniti", "it")


async def test_a_repeat_never_stands_in_for_the_explicit_confirmation(
    pricing: Pricing, rec: Recorder, system: SystemControl, tmp_path: Path
) -> None:
    """No command Claude may call sits at level 3 in commands.toml today: this one does."""
    commands = [replace(c, level=3) if c.id == "off" else c for c in load_commands(COMMANDS_FILE)]
    tool = CommandTool(commands, ActionRunner(rec.ctx(), system=system))
    tools = PcTools(FakeDesktop(), PcSettings(), home=tmp_path, commands=tool)
    try:
        call_off = ([], "tool_use", [use(NAME, command="off", last=True)])
        b = acting_brain(pricing, ScriptedClient(call_off, call_off), tools)
        await ask(b, "parole confuse")
        assert await ask(b, "spegniti") == NOT_UNDERSTOOD_IT
        assert not system.off
        await ask(b, "parole confuse")
        assert await ask(b, "confermo") == OFF_REPLY  # it did wait for the explicit word
        assert system.off
    finally:
        tools.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the PC tools are built only on Windows")
@pytest.mark.parametrize(("threshold", "repeats"), [(85, True), (90, False)])
def test_a_repeat_is_as_close_as_the_matcher_asks(
    tmp_path: Path, threshold: int, repeats: bool
) -> None:
    """ "spegni" is 85.7 from "spegniti": `fuzzy_threshold` in the settings decides."""
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "cost": settings.cost.model_copy(update={"db_file": tmp_path / "spend.db"}),
            "memory": settings.memory.model_copy(update={"enabled": False}),
            "router": settings.router.model_copy(update={"fuzzy_threshold": threshold}),
        }
    )
    brain = build_responder(settings, Recorder().ctx()).fallback.__self__  # type: ignore[union-attr]
    try:
        off = [call(NAME, command="off")]
        assert brain.tools.repeats(off, "Claudio, spegni.", "it") is repeats
    finally:
        brain.tools.close()
