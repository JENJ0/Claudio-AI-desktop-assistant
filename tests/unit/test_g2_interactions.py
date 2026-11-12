"""How the G2 tasks interact (group review): a question armed after the tray's off or pause,
rule 6 for the router's own question, and the name rule reading the wake audio too."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from claudio.app import State
from claudio.brain.claude import LOCAL_NOTE
from claudio.brain.prompts import OFFER
from claudio.cost.pricing import Pricing, load_pricing
from claudio.intake import Decision, Kind
from claudio.metrics import Interaction
from claudio.router.actions import VK_F4, VK_MENU
from claudio.router.matcher import CommandMatcher
from tests.unit import test_timeline
from tests.unit.test_actions import Recorder
from tests.unit.test_claude_brain import HANDOFF, Monotonic, ScriptedClient, ask, brain
from tests.unit.test_cost import PRICES
from tests.unit.test_local_question import local, say
from tests.unit.test_pc_tools import FakeDesktop, text, use
from tests.unit.test_timeline import INTO_CHROME, PLAY, WRITE, idle, typed_anything
from tests.unit.timeline import text_reply

make_scene = test_timeline.make_scene  # the fixtures of the timeline scenes, reused here
home = test_timeline.home


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


ASK_CALL = (
    ["Scrivo ciao. "],
    "tool_use",
    [text("Scrivo ciao."), use("type_text", text="ciao", **INTO_CHROME)],
)
CLOSE = (VK_MENU, VK_F4)


# --- I1: off or pause while Claude thinks -------------------------------------------------


async def test_a_question_is_not_spoken_when_the_tray_switched_off_meanwhile(
    make_scene, home: Path
) -> None:
    desk = FakeDesktop()
    scene = make_scene(ASK_CALL, sentence_frames=PLAY, claude_wait=20, desk=desk, home=home)
    await scene.say(WRITE)
    assert scene.assistant.state is State.RESPONDING
    scene.responder.actions.system.apply("off")
    await idle(scene, poll_s=0.002)
    assert not any("Confermi" in s for s in scene.speaker.spoken)
    assert not scene.brain.awaiting
    assert not scene.responder.needs_voice  # a typed message has no spoken question to cancel
    scene.responder.actions.system.apply("on")
    await scene.say("Claudio, sì")
    await idle(scene, poll_s=0.002)
    assert not typed_anything(desk)


async def test_a_question_is_not_spoken_when_the_tray_paused_meanwhile(
    make_scene, home: Path
) -> None:
    desk = FakeDesktop()
    scene = make_scene(
        ASK_CALL, text_reply("Scritto."), sentence_frames=PLAY, claude_wait=20, desk=desk, home=home
    )
    await scene.say(WRITE)
    assert scene.assistant.state is State.RESPONDING
    scene.responder.actions.system.apply("pause")
    await idle(scene, poll_s=0.002)
    assert not any("Confermi" in s for s in scene.speaker.spoken)
    assert not scene.brain.awaiting
    assert not scene.listening_again()  # no follow-up window in the pause
    await scene.say("sì")
    await idle(scene, poll_s=0.002)
    assert not typed_anything(desk)


async def test_the_actions_dropped_by_a_pause_are_noted_as_not_run(make_scene, home: Path) -> None:
    desk = FakeDesktop()
    scene = make_scene(ASK_CALL, sentence_frames=PLAY, claude_wait=20, desk=desk, home=home)
    await scene.say(WRITE)
    scene.responder.actions.system.apply("pause")
    await idle(scene, poll_s=0.002)
    assert any("type_text" in answer and "not" in answer for _, answer in scene.brain._history)


async def test_a_pause_ends_a_question_the_router_asked_before_it(
    matcher: CommandMatcher,
) -> None:
    rec, clock = Recorder(), Monotonic()
    r = local(matcher, rec, clock)
    await say(r, "chiudi la finestra")
    r.actions.system.apply("pause")
    assert await say(r, "sì") == ""  # the pause: only the system commands are answered
    r.actions.system.apply("resume")
    assert await say(r, "Claudio, sì") != "Fatto."
    assert rec.combos == []


async def test_the_sonnet_offer_is_not_made_while_paused(pricing: Pricing) -> None:
    client = ScriptedClient((["Domanda complessa."], "tool_use", HANDOFF))
    b = brain(pricing, client)
    b.silenced = lambda: True
    said = await ask(b, "spiegami il mutuo variabile")
    assert OFFER not in said
    assert b._offer is None


async def test_the_proposal_is_not_asked_while_off(make_scene) -> None:
    scene = make_scene(text_reply("x"))
    scene.responder.propose("apri chrome", "it")
    scene.responder.actions.system.apply("off")
    decision = Decision(Kind.ASK, "Hai detto: apri chrome?", "recovered")
    stream = await scene.assistant._reply_to(decision, np.zeros(1), "it", [], Interaction())
    assert [chunk async for chunk in stream] == []
    assert not scene.responder.awaiting


# --- I2: rule 6 for the router's own question ----------------------------------------------


async def test_a_yes_over_the_playing_local_confermi_runs_the_command_once(make_scene) -> None:
    scene = make_scene(text_reply("Chiudo la finestra?"), sentence_frames=PLAY)
    scene.responder.fallback_note = scene.brain.note_local
    await scene.say("chiudi la finestra")
    await scene.until(lambda: any("Confermi?" in p for p in scene.speaker.played))
    await scene.say_over("Claudio, sì")
    await idle(scene)
    assert scene.recorder.combos == [CLOSE]
    assert scene.client.calls == []  # the yes never went to Haiku


async def test_a_no_over_the_playing_local_confermi_cancels_it(make_scene) -> None:
    scene = make_scene(text_reply("x"), sentence_frames=PLAY)
    await scene.say("chiudi la finestra")
    await scene.until(lambda: any("Confermi?" in p for p in scene.speaker.played))
    await scene.say_over("Claudio, no")
    await idle(scene)
    assert scene.recorder.combos == []
    assert scene.client.calls == []
    assert not scene.responder.awaiting


async def test_a_yes_over_a_local_confermi_that_never_played_has_it_said_again(
    make_scene,
) -> None:
    scene = make_scene(text_reply("x"), sentence_frames=PLAY, start_frames=40)
    await scene.say("chiudi la finestra")
    assert scene.assistant.state is State.RESPONDING
    await scene.say_over("Claudio, sì")
    await idle(scene, poll_s=0.002)
    assert scene.recorder.combos == []  # nobody heard the question: no yes yet
    assert sum("Confermi?" in s for s in scene.speaker.spoken) == 2  # said again, once
    scene.speaker.start_frames = 0
    await scene.say("Claudio, sì")
    await idle(scene)
    assert scene.recorder.combos == [CLOSE]
    assert scene.client.calls == []


# --- I3: the name heard by the wake audio -----------------------------------------------------


async def test_a_lone_name_then_a_request_cancels_a_waiting_local_question(make_scene) -> None:
    scene = make_scene(text_reply("x"), sentence_frames=PLAY)
    await scene.say("chiudi la finestra")
    await idle(scene)
    assert scene.responder.awaiting  # silence: it waits
    await scene.say("Claudio.")
    await scene.say("che ore sono")
    await idle(scene)
    assert any("Sono le 14 e 5." in s for s in scene.speaker.spoken)
    assert scene.recorder.combos == []
    assert not scene.responder.awaiting


async def test_a_lone_name_then_a_request_cancels_a_waiting_claude_question(
    make_scene, home: Path
) -> None:
    desk = FakeDesktop()
    scene = make_scene(ASK_CALL, sentence_frames=PLAY, desk=desk, home=home)
    await scene.say(WRITE)
    await idle(scene, poll_s=0.002)
    assert scene.brain.awaiting
    await scene.say("Claudio.")
    await scene.say("che ore sono")
    await idle(scene, poll_s=0.002)
    assert any("Sono le 14 e 5." in s for s in scene.speaker.spoken)
    assert not typed_anything(desk)
    assert not scene.brain.awaiting


async def test_named_by_the_wake_audio_is_a_new_request_over_the_local_question(
    matcher: CommandMatcher,
) -> None:
    rec, clock = Recorder(), Monotonic()
    notes: list[tuple[str, str]] = []
    r = local(matcher, rec, clock, fallback_note=lambda q, a: notes.append((q, a)))
    await say(r, "chiudi la finestra")
    reply = "".join([c async for c in r.respond("che ore sono", "it", named=True)])
    assert "Va bene" in reply and "Sono le" in reply
    assert rec.combos == []
    assert not r.awaiting


async def test_an_unnamed_request_still_answers_the_question_as_unclear(
    matcher: CommandMatcher,
) -> None:
    rec, clock = Recorder(), Monotonic()
    r = local(matcher, rec, clock)
    await say(r, "chiudi la finestra")
    reply = "".join([c async for c in r.respond("che ore sono", "it")])
    assert "Sono le" not in reply
    assert r.awaiting  # asked once more


# --- minors ------------------------------------------------------------------------------------


async def test_a_local_question_cancelled_by_the_name_is_noted_as_not_done(
    matcher: CommandMatcher,
) -> None:
    rec, clock = Recorder(), Monotonic()
    notes: list[tuple[str, str]] = []

    async def claude(text: str) -> AsyncIterator[str]:
        yield "Domani piove."

    r = local(matcher, rec, clock, fallback=claude, fallback_note=lambda q, a: notes.append((q, a)))
    await say(r, "chiudi la finestra")
    await say(r, "Claudio, che tempo fa domani")
    assert ("Claudio, che tempo fa domani", "Va bene, non lo faccio.") in notes, notes
    assert rec.combos == []


@pytest.mark.parametrize("answer", ["sì"])
async def test_the_history_keeps_a_space_between_the_question_and_the_reply(
    make_scene, home: Path, answer: str
) -> None:
    desk = FakeDesktop()
    scene = make_scene(ASK_CALL, text_reply("Scritto."), sentence_frames=PLAY, desk=desk, home=home)
    await scene.say(WRITE)
    await scene.until(lambda: scene.brain.awaiting and scene.listening_again(), 0.002)
    await scene.say(answer)
    await idle(scene, poll_s=0.002)
    lines = [reply for _, reply in scene.brain._history]
    assert any("Confermi? Scritto." in line for line in lines), lines
    assert LOCAL_NOTE not in "".join(lines)


# --- a lone name counts as the name only when Whisper is sure of it ---------------------------


def _lone_name_scene(scene, name_logprob: float) -> None:
    from claudio.intake import Segment

    scene.assistant.stt_segments = lambda: [
        Segment(0.01, name_logprob if scene.said == "Ehi Claudio." else -0.2)
    ]


async def test_a_shaky_lone_name_does_not_turn_a_yes_into_a_new_request(make_scene) -> None:
    scene = make_scene(text_reply("x"), sentence_frames=PLAY)
    _lone_name_scene(scene, -0.95)
    await scene.say("chiudi la finestra")
    await idle(scene)
    assert scene.responder.awaiting
    await scene.say("Ehi Claudio.")
    await scene.say("Sì.")
    await idle(scene)
    assert scene.recorder.combos == [CLOSE]  # the question was answered and run
    assert scene.client.calls == []


async def test_a_shaky_lone_name_leaves_an_unclear_phrase_as_an_unclear_answer(
    make_scene,
) -> None:
    scene = make_scene(text_reply("x"), sentence_frames=PLAY)
    _lone_name_scene(scene, -0.95)
    await scene.say("chiudi la finestra")
    await idle(scene)
    await scene.say("Ehi Claudio.")
    await scene.say("che ore sono")
    await idle(scene)
    assert not any("Sono le 14 e 5." in s for s in scene.speaker.spoken)
    assert scene.responder.awaiting  # asked once more
    assert scene.recorder.combos == [] and scene.client.calls == []


async def test_a_sure_lone_name_still_cancels_the_question_by_name(make_scene) -> None:
    scene = make_scene(text_reply("x"), sentence_frames=PLAY)
    _lone_name_scene(scene, -0.5)
    await scene.say("chiudi la finestra")
    await idle(scene)
    await scene.say("Ehi Claudio.")
    await scene.say("che ore sono")
    await idle(scene)
    assert any("Sono le 14 e 5." in s for s in scene.speaker.spoken)
    assert scene.recorder.combos == [] and not scene.responder.awaiting


# --- a late answer after a lone name needs a sure name too ------------------------------------


async def _late_yes_after_a_lone_name(make_scene, name_logprob: float):
    scene = make_scene(text_reply("x"), sentence_frames=PLAY)
    _lone_name_scene(scene, name_logprob)
    await scene.say("chiudi la finestra")
    await idle(scene)
    scene.assistant.intake.verify_wake = True  # noise wakes are judged from here on
    await scene.say("Ehi Claudio.")
    await scene.say("Sì.")
    await idle(scene)
    return scene


async def test_a_shaky_lone_name_does_not_let_a_late_yes_run(make_scene) -> None:
    scene = await _late_yes_after_a_lone_name(make_scene, -1.0)
    assert scene.recorder.combos == [] and scene.client.calls == []


@pytest.mark.parametrize("name_logprob", [-0.5, -0.9])  # -0.9 is the threshold: it counts
async def test_a_sure_lone_name_lets_a_late_yes_run(make_scene, name_logprob: float) -> None:
    scene = await _late_yes_after_a_lone_name(make_scene, name_logprob)
    assert scene.recorder.combos == [CLOSE] and scene.client.calls == []
    assert scene.rows[-1]["relisten_logprob"] == name_logprob


async def test_the_row_rounds_the_lone_name_logprob_to_three_decimals(make_scene) -> None:
    scene = await _late_yes_after_a_lone_name(make_scene, -0.123456)
    assert scene.rows[-1]["relisten_logprob"] == -0.123


async def test_a_shaky_lone_name_with_no_question_open_is_still_answered(make_scene) -> None:
    scene = make_scene(text_reply("x"), sentence_frames=PLAY)
    _lone_name_scene(scene, -1.0)
    scene.assistant.intake.verify_wake = True
    await scene.say("Ehi Claudio.")
    await scene.say("che ore sono")
    await idle(scene)
    assert any("Sono le 14 e 5." in s for s in scene.speaker.spoken)
