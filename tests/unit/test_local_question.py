"""A local "Confermi?" survives silence like Claude's, local commands are in Claude's history,
and a typed message never answers a question asked by voice (Task 9a)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from claudio.app import TYPED_CANCELS_QUESTION, Assistant
from claudio.audio.vad import Endpointer
from claudio.brain.claude import ACTION_TTL_S as CLAUDE_TTL_S
from claudio.brain.claude import ACTIONS_NOTE, LOCAL_NOTE, ClaudeBrain
from claudio.config import ClaudeSettings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.memory.store import MemoryStore
from claudio.memory.voice import MemoryVoice
from claudio.router.actions import VK_F4, VK_MENU, ActionRunner
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import LocalResponder
from claudio.safety.confirmation import ACTION_TTL_S
from tests.unit.test_actions import Recorder
from tests.unit.test_app import AudioClock, FakeSpeaker, FakeStt, FakeWake
from tests.unit.test_claude_brain import FakeClient, Monotonic
from tests.unit.test_cost import PRICES, tracker


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


def local(matcher: CommandMatcher, rec: Recorder, clock: Monotonic, **hooks: Any) -> LocalResponder:
    return LocalResponder(matcher, ActionRunner(rec.ctx()), clock=clock, **hooks)


async def say(r: LocalResponder, text: str, language: str = "it") -> str:
    return "".join([c async for c in r.respond(text, language)])


# --- rule 1: the question waits, like Claude's ------------------------------------------


def test_one_window_for_both_kinds_of_question() -> None:
    assert ACTION_TTL_S == CLAUDE_TTL_S == 60.0


async def test_the_confirmation_holds_at_59_seconds(matcher: CommandMatcher) -> None:
    rec, clock = Recorder(), Monotonic()
    r = local(matcher, rec, clock)
    await say(r, "chiudi la finestra")
    clock.t += 59
    assert r.awaiting
    assert await say(r, "Claudio, confermo") == "Fatto."
    assert rec.combos == [(VK_MENU, VK_F4)]


async def test_the_confirmation_is_over_at_61_seconds(matcher: CommandMatcher) -> None:
    rec, clock = Recorder(), Monotonic()
    r = local(matcher, rec, clock)
    await say(r, "chiudi la finestra")
    clock.t += 61
    assert not r.awaiting and not r.needs_voice
    assert await say(r, "Claudio, confermo") == "Questo non lo so ancora fare."  # a normal request
    assert rec.combos == [] and r.last_route["route"] == "unknown"


async def test_silence_leaves_the_question_waiting_and_a_no_refuses(
    matcher: CommandMatcher,
) -> None:
    rec, clock = Recorder(), Monotonic()
    r = local(matcher, rec, clock)
    await say(r, "chiudi la finestra")
    r.cancel_pending(silence=True)
    assert r.awaiting
    assert await say(r, "Claudio, no") == "Va bene, non lo faccio."
    assert rec.combos == [] and r.last_route["route"] == "refused"


async def test_cancel_without_silence_still_clears_the_question(matcher: CommandMatcher) -> None:
    r = local(matcher, Recorder(), Monotonic())
    await say(r, "chiudi la finestra")
    r.cancel_pending()
    assert not r.awaiting


# --- rule 2: local commands in Claude's history -----------------------------------------


def brain(pricing: Pricing, client: FakeClient, clock: Monotonic) -> ClaudeBrain:
    return ClaudeBrain(client, tracker(pricing, 1.0), ClaudeSettings(tell_time=False), clock=clock)


def wired(
    matcher: CommandMatcher, b: ClaudeBrain, clock: Monotonic, **extra: Any
) -> LocalResponder:
    return local(
        matcher,
        Recorder(),
        clock,
        fallback=b.ask,
        fallback_info=lambda: b.last_call,
        fallback_note=b.note_local,
        **extra,
    )


async def test_claude_is_told_what_a_local_command_answered(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    client, clock = FakeClient(["Dante."]), Monotonic()
    r = wired(matcher, brain(pricing, client, clock), clock)
    time_said = await say(r, "che ore sono")
    await say(r, "chi ha scritto la Divina Commedia")
    assert client.calls[0]["messages"] == [
        {"role": "user", "content": "che ore sono"},
        {"role": "assistant", "content": f"{time_said} {LOCAL_NOTE}"},
        {"role": "user", "content": "[it] chi ha scritto la Divina Commedia"},
    ]


async def test_the_confirmation_turns_are_noted_too(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    b = brain(pricing, FakeClient(), clock := Monotonic())
    r = wired(matcher, b, clock)
    asked = await say(r, "chiudi la finestra")
    done = await say(r, "sì")
    assert list(b._history) == [
        ("chiudi la finestra", f"{asked} {LOCAL_NOTE}"),
        ("sì", f"{done} {LOCAL_NOTE}"),
    ]
    await say(r, "chiudi la finestra")
    refused = await say(r, "no")
    assert list(b._history)[-1] == ("no", f"{refused} {LOCAL_NOTE}")


async def test_nothing_is_written_while_off_or_paused(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    b = brain(pricing, FakeClient(), clock := Monotonic())
    r = wired(matcher, b, clock)
    r.actions.system.apply("pause")
    await say(r, "che ore sono")
    r.actions.system.apply("resume")
    r.actions.system.apply("off")
    await say(r, "che ore sono")
    assert list(b._history) == []


async def test_nothing_is_written_for_repeat_stop_or_a_memory_reply(
    matcher: CommandMatcher, pricing: Pricing, tmp_path: Path
) -> None:
    b = brain(pricing, FakeClient(), clock := Monotonic())
    voice = MemoryVoice(MemoryStore(tmp_path / "memory.db"))
    r = wired(matcher, b, clock, memory=voice)
    await say(r, "ricordati che il gatto si chiama Leo")
    assert r.last_route["route"] == "memory"
    await say(r, "ripeti")
    await say(r, "basta")
    assert list(b._history) == []


async def test_a_local_turn_is_never_read_as_outside_text(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    b = brain(pricing, FakeClient(), clock := Monotonic())
    r = wired(matcher, b, clock)
    before = b._read_outside()
    await say(r, "che ore sono")
    assert ACTIONS_NOTE not in LOCAL_NOTE
    assert b._read_outside() is before is False


def test_note_local_prunes_like_the_other_turns(pricing: Pricing) -> None:
    clock = Monotonic()
    b = ClaudeBrain(
        FakeClient(),
        tracker(pricing, 1.0),
        ClaudeSettings(tell_time=False, history_turns=2),
        clock=clock,
    )
    for n in range(3):
        b.note_local(f"q{n}", f"a{n}")
    assert list(b._history) == [("q1", f"a1 {LOCAL_NOTE}"), ("q2", f"a2 {LOCAL_NOTE}")]
    assert b._last_turn == clock.t


# --- rule 7: a typed message never answers a spoken question -----------------------------


def typed_assistant(fake_vad: Any, vad_settings: Any, r: LocalResponder) -> Assistant:
    return Assistant(
        FakeWake({3}),
        Endpointer(fake_vad, vad_settings),
        FakeStt([]),
        r,
        FakeSpeaker(),
        clock=AudioClock(),
        system=r.actions.system,
    )


async def test_a_typed_yes_never_confirms_a_local_question(
    fake_vad: Any, vad_settings: Any, matcher: CommandMatcher
) -> None:
    rec = Recorder()
    r = local(matcher, rec, Monotonic())
    assistant = typed_assistant(fake_vad, vad_settings, r)
    await say(r, "chiudi la finestra")  # armed by voice
    reply = await assistant.handle_typed("sì", "it")
    assert reply.startswith(TYPED_CANCELS_QUESTION["it"])
    assert rec.combos == [] and not r.awaiting


async def test_a_typed_yes_never_confirms_claudes_question(
    fake_vad: Any, vad_settings: Any, matcher: CommandMatcher
) -> None:
    armed = [True]
    events: list[str] = []

    async def claude(text: str, language: str) -> AsyncIterator[str]:
        events.append(f"ask {text}")
        yield "Ok."

    def cancel(cut: bool) -> None:
        events.append("cancel")
        armed.clear()

    r = local(
        matcher,
        Recorder(),
        Monotonic(),
        fallback=claude,
        fallback_awaiting=lambda: bool(armed),
        fallback_cancel=cancel,
    )
    assistant = typed_assistant(fake_vad, vad_settings, r)
    reply = await assistant.handle_typed("sì", "it")
    assert reply.startswith(TYPED_CANCELS_QUESTION["it"])
    assert events == ["cancel", "ask sì"]  # the question is gone before the text is a request


# --- fix round 1 -----------------------------------------------------------------------------


@pytest.mark.parametrize("what", ["off", "pause"])
async def test_switching_off_or_pausing_ends_every_question(
    fake_vad: Any, vad_settings: Any, matcher: CommandMatcher, what: str
) -> None:
    rec = Recorder()
    r = local(matcher, rec, Monotonic())
    typed_assistant(fake_vad, vad_settings, r)  # subscribes to the system changes
    await say(r, "chiudi la finestra")
    r.cancel_pending(silence=True)
    r.actions.system.apply(what)
    r.actions.system.apply("on")
    assert not r.awaiting
    assert await say(r, "Claudio, sì") == "Questo non lo so ancora fare."
    assert rec.combos == []


async def test_a_pause_ends_claudes_question_and_the_proposal_too(
    fake_vad: Any, vad_settings: Any, matcher: CommandMatcher
) -> None:
    armed = [True]
    r = local(
        matcher,
        Recorder(),
        Monotonic(),
        fallback_awaiting=lambda: bool(armed),
        fallback_cancel=lambda cut: armed.clear(),
    )
    typed_assistant(fake_vad, vad_settings, r)
    r.propose("alza il volume", "it")
    r.actions.system.apply("pause")
    assert not armed and not r.awaiting


async def test_a_local_command_does_not_revive_an_old_history(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    client, clock = FakeClient(["Ok."]), Monotonic()
    b = brain(pricing, client, clock)
    r = wired(matcher, b, clock)
    await say(r, "chi ha scritto la Divina Commedia")
    clock.t += ClaudeSettings().history_reset_s + 1
    await say(r, "che ore sono")
    assert [q for q, _ in b._history] == ["che ore sono"]


async def test_the_system_commands_leave_nothing_in_claudes_history(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    b = brain(pricing, FakeClient(), clock := Monotonic())
    r = wired(matcher, b, clock)
    for said in ("vai in pausa", "svegliati", "spegniti", "accenditi"):
        await say(r, said)
        assert list(b._history) == [], said
