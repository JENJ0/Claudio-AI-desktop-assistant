"""The memory in the running assistant: the router answers memory requests before any
command and without Claude, Claude sees the memories (cached) and may use its tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from claudio.__main__ import build_memory
from claudio.brain.claude import ClaudeBrain
from claudio.config import ClaudeSettings, MemorySettings, PcSettings, Settings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.memory.prompt import HEADER, MemoryPrompt
from claudio.memory.store import MemoryStore
from claudio.memory.tools import MemoryTools
from claudio.memory.voice import MemoryVoice
from claudio.pc.tools import PcTools
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import LocalResponder
from claudio.system import SystemControl
from tests.unit.test_actions import Recorder
from tests.unit.test_claude_brain import FakeClient, Monotonic, ScriptedClient, ask, cached
from tests.unit.test_cost import PRICES, tracker
from tests.unit.test_pc_tools import FakeDesktop, text, use


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


class Ticking:
    """A wall clock one second further at every call: each memory is newer than the last."""

    def __init__(self) -> None:
        self.t = 1_793_000_000.0

    def __call__(self) -> float:
        self.t += 1
        return self.t


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db", clock=Ticking())


def responder(
    matcher: CommandMatcher,
    store: MemoryStore,
    fallback: Any = None,
    system: SystemControl | None = None,
    memory: bool = True,
) -> LocalResponder:
    voice = MemoryVoice(store) if memory else None
    return LocalResponder(
        matcher,
        ActionRunner(Recorder().ctx(), system=system, memory=voice),
        fallback=fallback,
        memory=voice,
    )


async def say(r: LocalResponder, said: str, language: str = "it") -> str:
    return "".join([c async for c in r.respond(said, language)])


# --- the router ---------------------------------------------------------------------------


async def test_remember_is_local_and_free(
    matcher: CommandMatcher, store: MemoryStore, pricing: Pricing
) -> None:
    client = FakeClient()
    b = ClaudeBrain(client, tracker(pricing, 1.0), ClaudeSettings(tell_time=False))
    r = responder(matcher, store, fallback=b.ask)
    reply = await say(r, "Claudio, ricordati che mia sorella si chiama Anna.")
    assert reply == "Me lo ricordo: mia sorella si chiama Anna."
    assert client.calls == [] and r.last_route == {
        "route": "memory",
        "memory": "saved",
        "language": "it",
    }
    assert await say(r, "ripeti") == reply


async def test_a_memory_wins_over_a_fuzzy_command(
    matcher: CommandMatcher, store: MemoryStore
) -> None:
    """ "fai una pausa" is a command: "ricordati che..." with "pausa" in it is a memory."""
    system = SystemControl()
    r = responder(matcher, store, system=system)
    await say(r, "ricordati che la pausa pranzo è alle 13")
    assert not system.paused and store.items()[0].text == "la pausa pranzo è alle 13"


async def test_the_fixed_memory_commands(matcher: CommandMatcher, store: MemoryStore) -> None:
    r = responder(matcher, store)
    await say(r, "ricordati che il gatto si chiama Leo")
    assert await say(r, "Claudio, cosa ricordi?") == "Ricordo una cosa: il gatto si chiama Leo."
    assert r.last_route["route"] == "local" and r.last_route["command"] == "memory_list"
    assert await say(r, "dimenticalo") == "Dimenticato: il gatto si chiama Leo."
    assert len(store) == 0


async def test_forgetting_everything_needs_confermo(
    matcher: CommandMatcher, store: MemoryStore
) -> None:
    r = responder(matcher, store)
    await say(r, "ricordati che il gatto si chiama Leo")
    assert await say(r, "dimentica tutto") == (
        "Cancello tutti i ricordi, per sempre. Per procedere dì: confermo."
    )
    assert r.awaiting
    assert await say(r, "sì") == "Non ho capito: non lo faccio."  # level 3: "sì" is not enough
    assert len(store) == 1
    await say(r, "dimentica tutto")
    assert await say(r, "confermo") == "Fatto: ho dimenticato tutto."
    assert len(store) == 0


async def test_which_one_keeps_listening(matcher: CommandMatcher, store: MemoryStore) -> None:
    r = responder(matcher, store)
    store.add("mia sorella si chiama Anna")
    store.add("mia sorella abita a Torino")
    assert await say(r, "dimentica mia sorella") == (  # the newest first
        "Ne ricordo 2: mia sorella abita a Torino; mia sorella si chiama Anna. Quale dimentico?"
    )
    assert r.awaiting  # the assistant listens for the answer without the wake word
    assert await say(r, "la prima") == "Dimenticato: mia sorella abita a Torino."
    assert not r.awaiting


async def test_basta_drops_the_question(matcher: CommandMatcher, store: MemoryStore) -> None:
    r = responder(matcher, store)
    store.add("mia sorella si chiama Anna")
    store.add("mia sorella abita a Torino")
    await say(r, "dimentica mia sorella")
    r.cancel_pending(silence=True)
    assert r.awaiting  # silence alone: still waiting, until it expires
    r.cancel_pending()
    assert not r.awaiting and len(store) == 2


async def test_paused_ignores_memory_requests(matcher: CommandMatcher, store: MemoryStore) -> None:
    system = SystemControl()
    system.apply("pause")
    r = responder(matcher, store, system=system)
    assert await say(r, "ricordati che il gatto si chiama Leo") == ""
    assert len(store) == 0


async def test_recall_without_a_match_goes_to_claude(
    matcher: CommandMatcher, store: MemoryStore, pricing: Pricing
) -> None:
    client = FakeClient(["Non me ne hai mai parlato."])
    b = ClaudeBrain(client, tracker(pricing, 1.0), ClaudeSettings(tell_time=False))
    r = responder(matcher, store, fallback=b.ask)
    store.add("mia madre ama le orchidee")
    assert await say(r, "cosa ricordi di mamma") == "Non me ne hai mai parlato."
    assert r.last_route["route"] == "claude"


async def test_without_memory_nothing_changes(matcher: CommandMatcher, store: MemoryStore) -> None:
    r = responder(matcher, store, memory=False)
    assert await say(r, "cosa ricordi") == "La memoria è spenta."
    assert await say(r, "ricordati che il gatto si chiama Leo") == "Questo non lo so ancora fare."
    assert len(store) == 0


# --- Claude ---------------------------------------------------------------------------------


def brain(pricing: Pricing, client: Any, store: MemoryStore, **settings: Any) -> ClaudeBrain:
    settings.setdefault("tell_time", False)
    return ClaudeBrain(
        client,
        tracker(pricing, 1.0),
        ClaudeSettings(**settings),
        clock=Monotonic(),
        memory=MemoryPrompt(store, block_chars=2000, relevant=3),
    )


async def test_claude_sees_the_memories_in_their_own_cached_block(
    pricing: Pricing, store: MemoryStore
) -> None:
    client = ScriptedClient(*[(["Ok."], "end_turn", [])] * 3)
    b = brain(pricing, client, store)
    await ask(b, "ciao")
    assert len(client.calls[0]["system"]) == 1  # nothing to remember: as before
    store.add("mia sorella si chiama Anna")
    await ask(b, "come si chiama mia sorella")
    await ask(b, "e il gatto")
    first, second = client.calls[1]["system"], client.calls[2]["system"]
    assert first == second  # byte for byte: the cache reads the block back
    assert first[0] == cached(b.system)[0]  # the system prompt keeps its own marker
    assert first[1]["text"].startswith(HEADER) and "mia sorella si chiama Anna" in first[1]["text"]
    assert first[1]["cache_control"] == {"type": "ephemeral"}
    markers = sum("cache_control" in block for block in first) + 1  # + the automatic one
    assert markers <= 4  # the API allows four


async def test_without_cache_the_memories_follow_the_system_prompt(
    pricing: Pricing, store: MemoryStore
) -> None:
    store.add("mia sorella si chiama Anna")
    client = ScriptedClient((["Ok."], "end_turn", []))
    b = brain(pricing, client, store, cache=False)
    await ask(b, "ciao")
    system = client.calls[0]["system"]
    assert isinstance(system, str) and system.startswith(b.system.rstrip())
    assert system.endswith("mia sorella si chiama Anna")


async def test_facts_outside_the_block_come_with_the_request(
    pricing: Pricing, store: MemoryStore
) -> None:
    store.add("il wifi del garage si chiama Tana")
    client = ScriptedClient((["Tana."], "end_turn", []))
    b = ClaudeBrain(
        client,
        tracker(pricing, 1.0),
        ClaudeSettings(tell_time=False),
        clock=Monotonic(),
        memory=MemoryPrompt(store, block_chars=0, relevant=3),
    )
    await ask(b, "come si chiama il wifi del garage")
    assert client.calls[0]["messages"][-1]["content"] == (
        f"[it] [memory: [1 | {store.items()[0].day}] il wifi del garage si chiama Tana] "
        "come si chiama il wifi del garage"
    )


async def test_claude_saves_with_its_tool(pricing: Pricing, store: MemoryStore) -> None:
    tools = PcTools(FakeDesktop(), PcSettings(), memory=MemoryTools(store))
    try:
        client = ScriptedClient(
            ([], "tool_use", [use("remember", text="JenJo tifa Napoli")]),
            (["Me lo segno."], "end_turn", [text("Me lo segno.")]),
        )
        b = ClaudeBrain(
            client,
            tracker(pricing, 1.0),
            ClaudeSettings(tell_time=False),
            clock=Monotonic(),
            tools=tools,
            pc=PcSettings(),
        )
        assert await ask(b, "segnatelo che tifo Napoli") == "Me lo segno."
        names = [t["name"] for t in client.calls[0]["tools"]]
        assert names[-2:] == ["remember", "forget"]
        assert [(m.text, m.source) for m in store.items()] == [("JenJo tifa Napoli", "claude")]
        assert b.last_call["tools"] == ["remember(text=JenJo tifa Napoli) ok"]
    finally:
        tools.close()


async def test_claude_forgets_only_after_a_yes(pricing: Pricing, store: MemoryStore) -> None:
    store.add("mia sorella si chiama Anna")
    tools = PcTools(FakeDesktop(), PcSettings(confirm_brief=False), memory=MemoryTools(store))
    try:
        client = ScriptedClient(
            ([], "tool_use", [use("forget", ids=[1])]),
            (["Fatto."], "end_turn", [text("Fatto.")]),
        )
        b = ClaudeBrain(
            client,
            tracker(pricing, 1.0),
            ClaudeSettings(tell_time=False),
            clock=Monotonic(),
            tools=tools,
            pc=PcSettings(confirm_brief=False),
        )
        asked = await ask(b, "scordati di mia sorella")
        assert asked.startswith('Dimentico "mia sorella si chiama Anna"') and b.awaiting
        assert len(store) == 1  # nothing forgotten before the yes
        await ask(b, "sì")
        assert len(store) == 0
    finally:
        tools.close()


# --- start-up -------------------------------------------------------------------------------


def test_build_memory(tmp_path: Path) -> None:
    on = Settings(memory=MemorySettings(db_file=tmp_path / "memory.db"))
    store = build_memory(on)
    assert store is not None and store.path == tmp_path / "memory.db"
    assert build_memory(Settings(memory=MemorySettings(enabled=False))) is None


def test_a_broken_database_leaves_claudio_without_memory(tmp_path: Path) -> None:
    broken = tmp_path / "memory.db"
    broken.write_bytes(b"not a database at all" * 100)
    assert build_memory(Settings(memory=MemorySettings(db_file=broken))) is None


def test_the_shipped_settings() -> None:
    memory = Settings().memory
    assert memory.enabled and memory.db_file == Path("data/memory.db")
    assert memory.block_chars == 2000 and memory.relevant == 3 and not memory.learn
