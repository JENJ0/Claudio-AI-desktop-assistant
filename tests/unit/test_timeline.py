"""Timeline tests (G0 task 11): what cannot be replayed from WAVs, with the real orchestrator,
fake models and the microphone's frames as the clock (see timeline.py).

The checklist is LiveKit's tests/test_agent_session.py (no code taken from it). A test that is
red today asserts what the plan or JenJo wants and carries `xfail(strict=True)` with its group
of piani/IT-2-passo-0b-consegne.md, so the group that fixes it sees it turn green."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from claudio.app import Interaction, State, _fixed_reply
from claudio.brain.claude import CUT_BEFORE_NOTE, CUT_NOTE
from claudio.brain.prompts import BRIDGE
from claudio.memory.store import MemoryStore
from claudio.memory.voice import MemoryVoice
from claudio.resume import ACTION_STOPPED, DrainedStream, remainder
from claudio.tts.speaker import StreamingSpeaker
from tests.unit.test_pc_tools import FakeDesktop, text, use
from tests.unit.test_speaker import FakePlayer, FakeSynth
from tests.unit.timeline import GatedDesktop, Scene, inline_to_thread, text_reply

# --- the scenes ------------------------------------------------------------------------

STORY = [  # a long reply, one sentence per chunk (as Haiku streams it)
    "C'era una volta un gatto che viveva in un faro. ",
    "Ogni sera accendeva la grande lampada. ",
    "Le barche tornavano a casa seguendo quella luce. ",
    "Una notte arrivò una nebbia fitta. ",
    "Il gatto restò sveglio fino all'alba e tutte le barche arrivarono in porto.",
]
ASKING_STORY = [*STORY[:-1], STORY[-1] + " ", "Vuoi il seguito?"]  # Haiku's habit
SENTENCES = [s.strip() for s in STORY]
# H2 12/11: Claude's stories had CLAUDIO as their hero, so the name no longer told JenJo's voice
# from the reply's echo. Over HERO "Claudio, che ore sono?" is mostly the reply's words
# (claudio, che, ore); over MOON its long words sound like the reply's (claudio, sono ~ solo).
HERO = [
    "C'era una volta un eroe bellissimo di nome Claudio. ",
    "Era così bello che gli specchi chiedevano un autografo. ",
    "Trovò un socio che cercava da tre ore la porta che aveva davanti. ",
    "Claudio lo salvò, naturalmente. ",
    "Il robot esplose di ammirazione, letteralmente.",
]
MOON = [
    "Un'altra? Insaziabile, socio, ma il genio non si nega. ",
    "C'era una volta una luna abbandonata, dove un solo ristorante serviva zuppa di roccia. ",
    "Ci arrivò Claudio, ovviamente il cliente più bello della storia. ",
    "Il cuoco, un robot arrugginito, gli disse che nessuno finiva mai il piatto. ",
    "Claudio sorrise e lo finì in tre cucchiaiate.",
]
PLAY = 15  # frames a sentence lasts: 1.2 s
RESUME = "Dicevo"


@pytest.fixture
async def make_scene(
    fake_vad: Any, vad_settings: Any, matcher: Any, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[..., Scene]]:
    """Scenes whose Whisper runs inline (no worker thread: the same frames at every run)."""
    scenes: list[Scene] = []

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)

    def make(*replies: tuple[list[str], str, list[Any]], **kwargs: Any) -> Scene:
        scene = Scene(fake_vad, vad_settings, matcher, replies, **kwargs)
        scenes.append(scene)
        return scene

    yield make
    for scene in scenes:
        await scene.close()


async def idle(scene: Scene, poll_s: float = 0.0) -> None:
    await scene.until(lambda: scene.assistant.state is State.IDLE, poll_s)


def resumed(scene: Scene) -> bool:
    return any(said.startswith(RESUME) for said in scene.speaker.spoken)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    (tmp_path / "Downloads").mkdir()
    return tmp_path


# --- interruptions ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunk_wait", "through_cut"),
    [
        # The reply is all read before the cut: _remember stores it whole.
        pytest.param(0, False, id="reply-read-before-the-cut"),
        # Chunks come every 9 frames (read at 15, 24, 33, 42, 51), the stop is at frame 47: the
        # last chunk is still to come, so _remember_cut stores what was read, which is more
        # than what was heard (the stream goes on being read while the voice plays).
        pytest.param(9, True, id="cut-while-reading"),
        # Text arrives no faster than it is said: nothing to tell apart, protects the fix.
        pytest.param(30, False, id="read-as-heard"),
    ],
)
async def test_basta_while_speaking_keeps_what_was_heard(
    make_scene, chunk_wait: int, through_cut: bool
) -> None:
    scene = make_scene(
        text_reply(*STORY), sentence_frames=PLAY, chunk_wait=chunk_wait, window_s=4.0
    )
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("basta")
    await idle(scene)
    heard = scene.speaker.played
    assert 0 < len(heard) < len(SENTENCES)  # it was cut in the middle
    ((_, answer),) = scene.remembered
    if through_cut and CUT_NOTE not in answer:  # the case must stay on _remember_cut
        raise RuntimeError("the reply was not cut while it was read: wrong path")
    assert all(sentence in answer for sentence in heard)
    assert not any(sentence in answer for sentence in SENTENCES[len(heard) :])


async def test_basta_before_the_first_word_says_nothing_later(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, claude_wait=40)
    await scene.say("raccontami una storia")
    assert scene.assistant.state is State.RESPONDING  # Claude has not answered yet
    await scene.say_over("basta")
    await idle(scene)
    assert scene.speaker.played == []
    assert scene.row(0)["outcome"] == "stopped"
    assert scene.speaker.beeps.count("cancel") == 1
    assert scene.assistant._unsaid_rest is None  # nothing was heard: nothing to go on with
    assert CUT_BEFORE_NOTE in "".join(answer for _, answer in scene.remembered)


async def test_basta_before_the_first_word_pays_for_no_more_words(make_scene) -> None:
    """The stream is closed at the stop: the chunks that would arrive later are never read."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, claude_wait=40, chunk_wait=3)
    await scene.say("raccontami una storia")
    await scene.say_over("basta")
    await idle(scene)
    await scene.advance(8.0)  # well past the frame where Claude's first words would arrive
    assert scene.client.streams[0].chunks_read == 0
    assert not any(scene.speaker.spoken) and api_calls(scene) == 1  # (the fake logs "")
    assert scene.assistant._unsaid_rest is None


async def test_a_phrase_with_the_name_before_the_first_word_takes_the_request_place(
    make_scene,
) -> None:
    scene = make_scene(
        text_reply(*STORY),
        text_reply("Circa quattrocentomila chilometri."),
        sentence_frames=PLAY,
        claude_wait=40,
    )
    await scene.say("raccontami una storia")
    assert scene.assistant.state is State.RESPONDING
    await scene.say_over("Claudio, quanto dista la Luna dalla Terra")
    await idle(scene)
    assert [r["outcome"] for r in scene.rows[:2]] == ["interrupted", "ok"]
    assert [said for said in scene.speaker.spoken if said] == ["Circa quattrocentomila chilometri."]
    assert not resumed(scene) and scene.assistant._unsaid_rest is None  # nothing was heard
    assert scene.speaker.beeps.count("wake") == 1  # the person is still talking: no beep


async def test_tv_without_the_name_before_the_first_word_changes_nothing(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, claude_wait=40)
    assert scene.assistant.barge_in is not None
    scene.assistant.barge_in.needs_name = True
    await scene.say("raccontami una storia")
    assert scene.assistant.state is State.RESPONDING
    await scene.say_over("e poi mio fratello ha comprato la macchina nuova")
    await idle(scene)
    assert scene.speaker.played == SENTENCES
    assert scene.row(0)["outcome"] == "ok" and len(scene.rows) == 1


async def test_a_noise_while_the_beep_plays_does_not_stop_the_thinking(make_scene) -> None:
    """Claudio's own sound is out of the listener, as when it speaks."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, claude_wait=40)
    await scene.say("raccontami una storia")
    assert scene.assistant.state is State.RESPONDING
    scene.assistant.own_sound.started()  # a beep is playing
    await scene.say_over("basta")
    scene.assistant.own_sound.ended()
    await idle(scene)
    assert scene.speaker.played == SENTENCES and scene.row(0)["outcome"] == "ok"


def count_listener_passes(scene: Scene) -> list[int]:
    """The over-the-reply Whisper passes, one entry each."""
    passes: list[int] = []
    assert scene.assistant.barge_in is not None
    transcribe = scene.assistant.barge_in.transcribe

    def counting(audio: Any) -> str:
        passes.append(1)
        return transcribe(audio)

    counting.__self__ = transcribe.__self__  # type: ignore[attr-defined]  # keeps its duration
    scene.assistant.barge_in.transcribe = counting  # type: ignore[assignment,misc]
    return passes


async def silent_typed_hears_nothing(scene: Scene) -> None:
    """A typed message answered in the chat only: voice frames during it reach no listener."""
    passes = count_listener_passes(scene)
    typed = asyncio.create_task(scene.assistant.handle_typed("raccontami una storia"))
    await scene.until(lambda: scene.assistant.state is State.RESPONDING)
    await scene.say_over("basta")
    await scene.until(typed.done, poll_s=0.002)
    assert not passes and not scene.assistant._interrupted and scene.assistant._takeover is None
    assert scene.row(-1)["wake_word"] == "typed" and scene.row(-1)["outcome"] == "ok"


async def test_a_typed_request_stays_deaf_while_claude_thinks(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), claude_wait=40)
    await silent_typed_hears_nothing(scene)


async def test_a_request_left_by_a_timeout_does_not_make_typed_listen(make_scene) -> None:
    """ "Claudio" alone listens again; its window times out: nothing stays armed."""
    scene = make_scene(text_reply(*STORY), claude_wait=40, window_s=0.0)
    await scene.say("Claudio")
    await idle(scene)
    assert not scene.assistant._listens_while_thinking
    await silent_typed_hears_nothing(scene)


async def test_a_stop_judged_while_whisper_works_ends_the_request_before_claude(
    make_scene,
) -> None:
    """The check of "basta" can only end during the request's own Whisper when a later pass of
    the request waits behind it (the lock); here the verdict is set while TRANSCRIBING."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, whisper_frames=30)
    await scene.say("raccontami una storia")
    assert scene.assistant.state is State.TRANSCRIBING
    scene.assistant._stop_before_reply = True
    await idle(scene)
    assert scene.row(0)["outcome"] == "stopped" and scene.row(0)["text"] == "raccontami una storia"
    assert api_calls(scene) == 0 and not any(scene.speaker.spoken)
    assert scene.speaker.beeps.count("cancel") == 1


@pytest.mark.parametrize(("heard", "stops"), [("basta", True), ("che ore sono", False)])
async def test_only_a_stop_judged_while_whisper_works_counts(
    make_scene, heard: str, stops: bool
) -> None:
    """The verdict of a check that ends while the request is still being transcribed (the
    state is put there by hand: the lock makes the natural order end later)."""
    scene = make_scene(text_reply("Sono le tre."))
    assistant = scene.assistant
    assert assistant.barge_in is not None
    assistant._set_state(State.TRANSCRIBING)
    assistant._listen_while_thinking()
    scene.said = heard
    assistant._phrase_stop_only = True  # what the start of a phrase in TRANSCRIBING sets
    assistant._start_barge_check(np.zeros(16000, dtype=np.float32), assistant.barge_in, -20.0)
    await scene.until(lambda: not assistant._barge_checking)
    assert assistant._stop_before_reply is stops and assistant._takeover is None
    assistant._set_state(State.IDLE)


async def test_basta_said_during_whisper_and_judged_after_stops_the_thinking(make_scene) -> None:
    """The check ends only after the request reached Claude (the lock order): the stop still
    works, as one said while Claude thinks."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, whisper_frames=30, claude_wait=80)
    await scene.say("raccontami una storia")
    assert scene.assistant.state is State.TRANSCRIBING
    await scene.say_over("basta")
    await idle(scene)
    assert scene.row(0)["outcome"] == "stopped" and scene.speaker.played == []


async def test_words_while_whisper_works_leave_the_request_alone(make_scene) -> None:
    scene = make_scene(text_reply("Sono le tre."), whisper_frames=30)
    await scene.say("che ore sono")
    assert scene.assistant.state is State.TRANSCRIBING
    await scene.say_over("Claudio, quanto dista la Luna dalla Terra")
    await idle(scene)
    assert scene.row(0)["text"] == "che ore sono" and scene.row(0)["outcome"] == "ok"
    assert scene.speaker.spoken == ["Sono le 14 e 5."]


async def test_basta_while_whisper_works_survives_the_switch_to_claude(make_scene) -> None:
    """The phrase is still being said (its end-silence) when Whisper ends: the listener keeps
    hearing it."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, whisper_frames=9, claude_wait=40)
    await scene.say("raccontami una storia")
    await scene.say_over("basta")
    await idle(scene)
    assert scene.speaker.played == [] and scene.row(0)["outcome"] == "stopped"


@pytest.mark.parametrize("first_word", [8, 10])
async def test_basta_survives_the_first_word_arriving(make_scene, first_word: int) -> None:
    """ "basta" said at once, its end not detected yet when Claude's first sound starts."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, claude_wait=first_word)
    await scene.say("raccontami una storia")
    await scene.say_over("basta")
    await idle(scene)
    assert len(scene.speaker.played) < len(SENTENCES)
    assert scene.row(0)["outcome"] in ("stopped", "interrupted")


async def test_a_phrase_begun_in_whisper_is_no_request_when_it_ends_later(make_scene) -> None:
    """Said while Whisper worked, ended while Claude thinks: the check starts in RESPONDING,
    but the phrase began earlier, so only a "basta" in it would count."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, whisper_frames=9, claude_wait=40)
    await scene.say("raccontami una storia")
    assert scene.assistant.state is State.TRANSCRIBING
    await scene.say_over("Claudio, quanto dista la Luna dalla Terra")
    assert scene.assistant.state is State.RESPONDING
    await idle(scene)
    assert scene.speaker.played == SENTENCES
    assert scene.row(0)["outcome"] == "ok" and len(scene.rows) == 1


async def test_the_wait_at_the_reply_end_does_not_chain_new_checks(make_scene) -> None:
    """A TV that keeps talking: the reply's end waits for the checks running then, no more."""
    scene = make_scene(text_reply(*STORY[:2]), sentence_frames=PLAY, whisper_frames=20)
    assert scene.assistant.barge_in is not None
    scene.assistant.barge_in.needs_name = True  # the TV is no request
    waited: list[int] = []
    judge = scene.assistant._judge_stranded

    async def timed() -> None:
        start = scene.ticks.count
        await judge()
        waited.append(scene.ticks.count - start)

    scene.assistant._judge_stranded = timed  # type: ignore[method-assign]
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    for _ in range(8):  # one phrase after the other, each checked for 20 frames
        await scene.say_over("e poi mio fratello ha comprato la macchina nuova")
    await idle(scene)
    assert waited and waited[0] <= 80  # the running check and the one waiting, not 120 frames


async def test_a_request_waiting_behind_a_tv_check_is_not_lost_at_the_reply_end(
    make_scene,
) -> None:
    """The TV is being checked, the request is said meanwhile and waits for that check; the
    reply ends before the check does."""
    scene = make_scene(
        text_reply(*STORY[:2]),
        text_reply("Circa quattrocentomila chilometri."),
        sentence_frames=PLAY,
        whisper_frames=20,
    )
    assert scene.assistant.barge_in is not None
    scene.assistant.barge_in.needs_name = True  # the TV is no request
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.say_over("e poi mio fratello ha comprato la macchina nuova")
    await scene.say_over("Claudio, quanto dista la Luna dalla Terra")
    await idle(scene)
    assert "Circa quattrocentomila chilometri." in scene.speaker.spoken


async def test_a_request_heard_while_the_whisper_check_runs_is_not_lost(make_scene) -> None:
    """5.2 #5: the reply ends while the check of the phrase said over it is still running."""
    scene = make_scene(
        text_reply(*STORY[:2]),
        text_reply("Circa quattrocentomila chilometri."),
        sentence_frames=PLAY,
        whisper_frames=20,
    )
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.say_over("Claudio, quanto dista la Luna dalla Terra")
    await idle(scene)
    assert "Circa quattrocentomila chilometri." in scene.speaker.spoken


async def test_noise_over_the_reply_lets_it_finish(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.say_over("eh")
    await idle(scene)
    assert scene.speaker.ducks == [0.3, None]  # lowered, then back to full volume
    assert scene.speaker.played == SENTENCES
    assert scene.row(0)["outcome"] == "ok"


async def test_interrupted_row_keeps_its_first_audio(make_scene) -> None:
    scene = make_scene(
        text_reply(*STORY), sentence_frames=PLAY, claude_wait=10, start_frames=2, window_s=4.0
    )
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(0.5)
    await scene.say_over("basta")
    await idle(scene)
    row = scene.row(0)
    assert row["outcome"] == "interrupted"
    assert row["m1_ms"] >= 800  # the 10 frames Claude took, then the voice
    assert row["audible_ms"] >= row["m1_ms"]
    assert row["tts_first_ms"] == pytest.approx(160, abs=1)  # the 2 frames of the voice


@pytest.mark.parametrize("whisper_frames", [0, 5])
async def test_frames_arrive_while_whisper_works(make_scene, whisper_frames: int) -> None:
    """Whisper lasts `whisper_frames` microphone frames: at 0 it ends within the frame that
    started it, as before; above 0 the microphone goes on feeding TRANSCRIBING."""
    scene = make_scene(text_reply("Sono le tre."), whisper_frames=whisper_frames)
    during: list[int] = []
    tick = scene.ticks.tick

    def counting() -> None:
        tick()
        if scene.assistant.state is State.TRANSCRIBING:
            during.append(scene.ticks.count)

    scene.ticks.tick = counting  # type: ignore[method-assign]
    await scene.say("che ore sono")
    await scene.advance(1.0)
    assert len(during) == whisper_frames
    assert scene.row(0)["outcome"] == "ok"


async def test_whisper_hears_what_was_said_when_it_started(make_scene) -> None:
    """A phrase said while Whisper works is not the text of the request it is working on."""
    scene = make_scene(text_reply("Sono le tre."), whisper_frames=10)
    await scene.say("che ore sono")
    assert scene.assistant.state is State.TRANSCRIBING
    scene.said = "un'altra frase detta mentre Whisper lavora"
    await idle(scene)
    assert scene.row(0)["text"] == "che ore sono"


async def test_the_check_of_a_phrase_over_the_reply_lasts_whisper_frames(make_scene) -> None:
    """The over-the-reply listener is a Whisper too: frames arrive during its check, and a
    slice that comes meanwhile waits in `_barge_pending`."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, whisper_frames=20)
    checking: list[int] = []
    pending: list[int] = []
    tick = scene.ticks.tick

    def counting() -> None:
        tick()
        if scene.assistant._barge_checking:
            checking.append(scene.ticks.count)
        if scene.assistant._barge_pending is not None:
            pending.append(scene.ticks.count)

    scene.ticks.tick = counting  # type: ignore[method-assign]
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.say_over("aspetta un momento per favore")
    await scene.say_over("anzi no, continua pure")  # said while the first is still checked
    await idle(scene)
    assert len(checking) >= 10
    assert pending


# --- tools -----------------------------------------------------------------------------

ASK = "metti un po' di musica su spotify"  # no local command: it goes to Claude


async def test_tool_runs_once_and_claude_gets_its_result(make_scene, home: Path) -> None:
    desk = FakeDesktop()
    scene = make_scene(
        ([], "tool_use", [use("open_app", name="spotify")]),
        text_reply("Aperto."),
        desk=desk,
        home=home,
    )
    await scene.say(ASK)
    await idle(scene, poll_s=0.002)
    assert desk.done.count(("launch", "Spotify")) == 1
    _first, second = scene.client.calls
    assert second["messages"][-2]["content"][0]["type"] == "tool_use"
    result = second["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "tu_open_app"
    assert result["content"].startswith("Started Spotify. Its window is open.")
    assert scene.speaker.played == ["Aperto."]
    assert scene.row(0)["route"] == "claude"


async def test_slow_tool_keeps_the_first_sentence_said(make_scene, home: Path) -> None:
    desk = GatedDesktop()
    said = "Ci guardo, socio. Apro Spotify. Un attimo."  # three sentences: not held (Task 10)
    scene = make_scene(
        ([said], "tool_use", [text(said), use("open_app", name="spotify")]),
        text_reply("Aperto."),
        sentence_frames=PLAY,
        desk=desk,
        home=home,
    )
    await scene.say(ASK)
    await scene.until(desk.started.is_set, poll_s=0.002)
    await scene.advance(0.5)
    assert scene.speaker.played == ["Ci guardo, socio."]  # said while the tool is still running
    assert ("launch", "Spotify") not in desk.done
    desk.gate.set()
    await idle(scene, poll_s=0.002)
    assert scene.speaker.played[0] == "Ci guardo, socio." and scene.speaker.played[-1] == "Aperto."
    assert "".join(scene.speaker.played).count("Un attimo.") == 1
    assert desk.done.count(("launch", "Spotify")) == 1


async def test_tool_without_a_reply_goes_back_to_idle(make_scene, home: Path) -> None:
    desk = FakeDesktop()
    scene = make_scene(
        ([], "tool_use", [use("open_app", name="spotify")]),
        ([], "end_turn", []),
        desk=desk,
        home=home,
    )
    await scene.say(ASK)
    await idle(scene, poll_s=0.002)
    assert desk.done.count(("launch", "Spotify")) == 1
    assert scene.row(0)["outcome"] == "ok"
    assert scene.speaker.played == []  # nothing to say, and nobody waits for it


# --- a new request said over a reply that was about to act -------------------------------

WRITE = "scrivi il piano di domani su chrome"
PLAN = "Il piano di domani comincia presto. " * 90
INTO_CHROME = {"window": "chrome"}  # level 2: it asks "Confermi?" first
SAY_THEN_CALL = (
    STORY,
    "tool_use",
    [text("".join(STORY)), use("type_text", text=PLAN, **INTO_CHROME)],
)
# The first leg is cut at the token limit; the retry's own words come slowly and are never said.
SAY_THEN_CUT = (
    STORY[:1],
    "max_tokens",
    [text(STORY[0]), use("type_text", text="Il", **INTO_CHROME)],
)
SLOW_RETRY = (list("abcdefgh"), "tool_use", [use("type_text", text=PLAN, **INTO_CHROME)])
OVER_THE_LEG = pytest.param((SAY_THEN_CALL,), id="the-call-comes-after-the-request")
OVER_THE_RETRY = pytest.param((SAY_THEN_CUT, SLOW_RETRY), id="during-the-retry")


def typed_anything(desk: FakeDesktop) -> bool:
    return any(done[0] in ("keys", "paste") for done in desk.done)


@pytest.mark.parametrize("replies", [OVER_THE_LEG, OVER_THE_RETRY])
async def test_a_thanks_over_the_reply_is_no_yes_to_a_question_nobody_heard(
    make_scene, home: Path, replies: tuple[Any, ...]
) -> None:
    """The stream is kept for "Dicevo", so the old reply still reached "Confermi?": then "ok,
    grazie", meant for something else, was the yes and 3240 characters went into Chrome."""
    desk = FakeDesktop()
    scene = make_scene(
        *replies,
        text_reply("Prego."),
        sentence_frames=PLAY,
        chunk_wait=9,
        desk=desk,
        home=home,
    )
    await scene.say(WRITE)
    await scene.until_speaking()
    await scene.say_over("Claudio, ok grazie")
    await idle(scene, poll_s=0.002)
    assert scene.row(0)["outcome"] == "interrupted"
    assert not typed_anything(desk) and not scene.brain.awaiting
    assert "[Claudio did not run: type_text" in scene.remembered[0][1]


@pytest.mark.parametrize("replies", [OVER_THE_LEG, OVER_THE_RETRY])
async def test_a_request_over_the_reply_gets_its_own_answer(
    make_scene, home: Path, replies: tuple[Any, ...]
) -> None:
    """An armed "Confermi?" took the new request for its answer: "Non ho capito: non lo faccio"
    instead of the time."""
    desk = FakeDesktop()
    scene = make_scene(*replies, sentence_frames=PLAY, chunk_wait=9, desk=desk, home=home)
    await scene.say(WRITE)
    await scene.until_speaking()
    await scene.say_over("che ore sono")
    await idle(scene, poll_s=0.002)
    assert "Sono le 14 e 5." in scene.speaker.spoken
    assert not typed_anything(desk) and not scene.brain.awaiting
    assert not any("Confermi" in said for said in scene.speaker.spoken)


async def test_certo_hey_claudio_cancels_the_question_and_opens_the_window(
    make_scene, home: Path
) -> None:
    """A greeting after a yes word is a new call: the question is cancelled and the window
    listens for the request that follows."""
    desk = FakeDesktop()
    call = ([], "tool_use", [use("type_text", text="ciao", **INTO_CHROME)])
    scene = make_scene(call, desk=desk, home=home, window_s=5.0)
    await scene.say(WRITE)
    await scene.until(lambda: scene.brain.awaiting and scene.listening_again())
    await scene.say("Certo. Hey Claudio.")
    await scene.until(lambda: len(scene.rows) == 2 and scene.listening_again())
    assert scene.speaker.spoken[-1] == "Va bene, non lo faccio."
    assert scene.row(1)["outcome"] == "ok" and not scene.brain.awaiting
    assert not typed_anything(desk)
    assert scene.assistant.state is State.LISTENING  # the window is open for the request


async def test_a_local_request_over_a_dropped_action_says_so_and_does_not_resume(
    make_scene, home: Path
) -> None:
    """Resuming would repeat the promise of the action that was stopped: the answer, then one
    sentence saying the action was left, no "Dicevo"."""
    desk = FakeDesktop()
    scene = make_scene(SAY_THEN_CALL, sentence_frames=PLAY, chunk_wait=9, desk=desk, home=home)
    await scene.say(WRITE)
    await scene.until_speaking()
    await scene.say_over("che ore sono")
    await idle(scene, poll_s=0.002)
    spoken = scene.speaker.spoken
    assert not resumed(scene) and spoken.count(ACTION_STOPPED) == 1
    assert spoken.index("Sono le 14 e 5.") < spoken.index(ACTION_STOPPED)
    assert not typed_anything(desk) and not scene.brain.awaiting
    assert scene.assistant._unsaid_rest is None  # its rest would repeat the promise


async def test_basta_over_the_reply_runs_nothing_either(make_scene, home: Path) -> None:
    desk = FakeDesktop()
    scene = make_scene(SAY_THEN_CALL, sentence_frames=PLAY, chunk_wait=9, desk=desk, home=home)
    await scene.say(WRITE)
    await scene.until_speaking()
    await scene.say_over("basta")
    await idle(scene, poll_s=0.002)
    assert not typed_anything(desk) and not scene.brain.awaiting
    assert not resumed(scene)


async def test_the_next_request_acts_again(make_scene, home: Path) -> None:
    """The stop belongs to the reply that was cut: it never blocks the request after it."""
    desk = FakeDesktop()
    scene = make_scene(
        SAY_THEN_CALL,
        ([], "tool_use", [use("open_app", name="spotify")]),
        text_reply("Aperto."),
        sentence_frames=PLAY,
        chunk_wait=9,
        desk=desk,
        home=home,
    )
    await scene.say(WRITE)
    await scene.until_speaking()
    await scene.say_over(ASK)
    await idle(scene, poll_s=0.002)
    assert desk.done.count(("launch", "Spotify")) == 1
    assert not typed_anything(desk)


# --- the person talks to CLAUDIO in several ways ----------------------------------------


async def test_typed_message_waits_for_the_spoken_reply(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    typed = asyncio.create_task(scene.assistant.handle_typed("che ore sono"))
    await scene.advance(2.0)
    assert not typed.done() and scene.speaker.played != SENTENCES  # still speaking: it waits
    await scene.until(typed.done, poll_s=0.002)
    assert scene.speaker.played == SENTENCES  # the reply was not cut
    assert await typed == "Sono le 14 e 5."
    assert [r["wake_word"] for r in scene.rows] == ["hey_claudio", "typed"]
    assert scene.row(0)["outcome"] == "ok"


async def test_phrase_left_halfway_is_joined(make_scene) -> None:
    scene = make_scene(text_reply("Circa quattrocentomila chilometri."), continue_s=2.0)
    await scene.say("Quanto dista la Luna...")
    await scene.say_more("Quanto dista la Luna dalla Terra?")
    await idle(scene)
    (call,) = scene.client.calls  # one request, not two
    assert call["messages"][-1]["content"].endswith("Quanto dista la Luna dalla Terra?")
    assert scene.speaker.beeps.count("wake") == 1
    assert scene.row(0)["continued_from"] == "Quanto dista la Luna..."
    first, second = scene.stt.samples
    assert second > first + 0.6 * 16000  # the second pass hears the first part too
    assert len(scene.rows) == 1 and scene.row(0)["outcome"] == "ok"


async def test_stt_error_says_so_and_goes_back(make_scene) -> None:
    scene = make_scene()
    scene.stt.fail = True
    await scene.say("che ore sono")
    await idle(scene)
    assert scene.row(0)["outcome"] == "error" and "error" in scene.speaker.beeps
    await scene.say("che ore sono")  # the next request works
    await idle(scene)
    assert scene.row(1)["outcome"] == "ok" and scene.speaker.played == ["Sono le 14 e 5."]


async def test_voice_latency_excludes_claude_wait(make_scene) -> None:
    scene = make_scene(
        text_reply("Circa quattrocentomila chilometri."), claude_wait=25, start_frames=2
    )
    await scene.say("quanto dista la Luna dalla Terra")
    await idle(scene)
    row = scene.row(0)
    assert row["m1_ms"] >= 2000  # the whole wait is in the first-word time
    assert row["tts_first_ms"] == pytest.approx(160, abs=1)  # the voice's own 2 frames


class Player(FakePlayer):
    def queued_s(self) -> float:
        return 0.0


@pytest.mark.parametrize(
    ("voice", "path"),
    [
        ("timed", "typed"),
        ("timed", "spoken"),
        # The real speaker leaves the stream open too while _produce is rendering (a sentence
        # to synthesise: the inline to_thread hands over for one turn) or on a full queue.
        ("streaming", "typed"),
        ("streaming", "spoken"),
    ],
)
async def test_typed_reply_cut_short_keeps_what_was_said(make_scene, voice: str, path: str) -> None:
    """The reply is cut while its words are still being read (the stop comes with the first
    sound): its row keeps what was said. Typed and spoken replies share `_marked`."""
    speaker = StreamingSpeaker(FakeSynth(), Player()) if voice == "streaming" else None
    scene = make_scene(text_reply("Uno. ", "Due. ", "Tre. ", "Quattro."), speaker=speaker)
    hook = "_mark_first_audio" if path == "typed" else "_on_first_audio"
    first_audio = getattr(scene.assistant, hook)

    def cut_at_first_sound(*args: Any) -> None:
        first_audio(*args)
        scene.speaker.interrupt()

    setattr(scene.assistant, hook, cut_at_first_sound)
    if path == "typed":
        done = asyncio.create_task(
            scene.assistant.handle_typed("raccontami una storia", speak=True)
        ).done
    else:
        await scene.say("raccontami una storia")
        done = lambda: scene.assistant.state is State.IDLE  # noqa: E731
    await scene.until(done, poll_s=0.002)
    reply = scene.row(0).get("reply", "")
    assert scene.speaker.played and all(s in reply for s in scene.speaker.played)


# --- the real sequences of 05/11 (their shape, with generic phrases) --------------------


async def test_a_story_heard_in_two_parts_is_whole_in_the_history(make_scene) -> None:
    """Cut by a local answer and resumed to its end ("Dicevo"): all of it was heard, so all of
    it is in the history, once, with no cut note."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, window_s=4.0)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("che ore sono")
    await idle(scene)
    assert resumed(scene)
    ((_, answer),) = scene.remembered
    assert CUT_NOTE not in answer and all(sentence in answer for sentence in SENTENCES)
    assert answer.count(SENTENCES[1]) == 1 and RESUME not in answer


async def test_17_08_story_then_time_resumes(make_scene) -> None:
    """A story that ends asking "Vuoi il seguito?" is cut by "che ore sono": the time is a local
    answer, so "Dicevo" goes on with the story (JenJo's decision)."""
    scene = make_scene(text_reply(*ASKING_STORY), sentence_frames=PLAY, window_s=4.0)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("che ore sono")
    await idle(scene)
    assert [r["outcome"] for r in scene.rows[:2]] == ["interrupted", "ok"]
    assert scene.row(1)["route"] == "local" and "Sono le 14 e 5." in scene.speaker.spoken
    assert resumed(scene)


async def test_17_13_story_then_new_request_does_not_resume(make_scene) -> None:
    """A story cut by a request for Claude is not resumed by itself (JenJo's decision)."""
    scene = make_scene(
        text_reply(*STORY),
        text_reply("Circa quattrocentomila chilometri."),
        sentence_frames=PLAY,
        window_s=4.0,
    )
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("quanto dista la Luna dalla Terra")
    await idle(scene)
    assert [r["outcome"] for r in scene.rows[:2]] == ["interrupted", "ok"]
    assert scene.row(1)["route"] == "claude"
    assert not resumed(scene)


@pytest.mark.parametrize("story", [HERO, MOON], ids=["word-share", "echo-of"])
async def test_the_name_over_a_story_about_claudio_is_jenjo(make_scene, story: list[str]) -> None:
    """H2 12/11 steps 1-3: the name said over a reply that says it too was taken for the echo,
    so no request over a story about CLAUDIO ever stopped it."""
    scene = make_scene(text_reply(*story), sentence_frames=PLAY, window_s=4.0)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("Claudio, che ore sono?")
    await idle(scene)
    assert [r["outcome"] for r in scene.rows[:2]] == ["interrupted", "ok"]
    assert "Sono le 14 e 5." in scene.speaker.spoken
    assert resumed(scene)


@pytest.mark.parametrize(
    "echo",
    [
        "Claudio lo salvò naturalmente",
        "Claudio?",
        "un eroe bellissimo di nome Claudio",  # the first sentence, judged late
    ],
)
async def test_the_echo_of_a_sentence_with_the_name_is_still_echo(make_scene, echo: str) -> None:
    """The story has said the name: its echo, or the lone name while the sentence playing says
    it, does not stop the reply."""
    scene = make_scene(text_reply(*HERO), sentence_frames=PLAY, window_s=4.0)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.until(lambda: scene.speaker.played[-1:] == [HERO[3].strip()])
    await scene.say_over(echo)
    await idle(scene)
    assert scene.row(0)["outcome"] == "ok"


async def test_a_short_command_over_a_sentence_with_the_name_is_jenjo(make_scene) -> None:
    """The last sentences say "Claudio lo salvò": the name's letters alone made a short command
    look like them (review of 13/11)."""
    scene = make_scene(text_reply(*HERO), sentence_frames=PLAY, window_s=4.0)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.until(lambda: scene.speaker.played[-1:] == [HERO[4].strip()])
    await scene.say_over("Claudio, volume su")
    await idle(scene)
    assert [r["outcome"] for r in scene.rows[:2]] == ["interrupted", "ok"]
    assert scene.row(1)["route"] == "local"


async def test_dicevo_after_a_claude_answer_keeps_both_lines(make_scene) -> None:
    scene = make_scene(
        text_reply(*STORY), text_reply("Parigi."), sentence_frames=PLAY, window_s=4.0
    )
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("qual è la capitale della Francia")
    await idle(scene)
    story, capital = scene.remembered
    assert capital == ("qual è la capitale della Francia", "Parigi.")
    assert "Parigi" not in story[1] and SENTENCES[0] in story[1]


async def test_a_cut_question_of_the_guess_leaves_the_claude_line_alone(make_scene) -> None:
    scene = make_scene(text_reply("Circa quattrocentomila chilometri."))
    await scene.say("quanto dista la Luna dalla Terra")
    await idle(scene)
    before = scene.remembered
    scene.responder.propose("chiudi la finestra")
    scene.assistant._interrupted = True
    inter = Interaction(clock=scene.clock, info={})
    await scene.assistant._speak_reply(
        _fixed_reply("Hai detto: chiudi la finestra?"), inter, lambda: None
    )
    assert scene.remembered == before


async def test_typed_request_over_a_stale_stream_does_not_wake_it(make_scene, home: Path) -> None:
    desk = FakeDesktop()
    scene = make_scene(
        SAY_THEN_CALL,
        text_reply("Circa quattrocentomila chilometri."),
        claude_wait=5,
        desk=desk,
        home=home,
    )
    stale = DrainedStream(scene.brain.ask(WRITE))
    for _ in range(5):
        await asyncio.sleep(0)
    scene.brain.stop_acting()
    await scene.assistant.handle_typed("quanto dista la Luna dalla Terra")
    await scene.advance(1.0)
    await stale.finish(5.0)
    assert not scene.brain.awaiting and not typed_anything(desk)


# --- "continua": the missing text said again, for free (G2 task 7) ------------------------


def api_calls(scene: Scene) -> int:
    return len(scene.client.calls)


async def stop_the_story(scene: Scene) -> None:
    """The story is told, "basta" cuts it after its first sentence."""
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("basta")
    await idle(scene)


async def test_continua_after_basta_says_the_exact_rest_for_free(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, window_s=4.0)
    await stop_the_story(scene)
    assert scene.row(0)["outcome"] == "interrupted"
    heard = list(scene.speaker.played)
    assert 0 < len(heard) < len(SENTENCES)
    await scene.say("continua")
    await idle(scene)
    assert api_calls(scene) == 1
    assert scene.row(1)["route"] == "local" and scene.row(1)["command"] == "continue_reply"
    assert scene.speaker.spoken[-1] == f"Dicevo: {remainder(''.join(STORY), heard)}"
    ((_, answer),) = scene.remembered  # Claude knows all of it was said, once
    assert all(sentence in answer for sentence in SENTENCES)
    assert answer.count(SENTENCES[1]) == 1 and RESUME not in answer and CUT_NOTE not in answer


async def test_continua_after_basta_over_a_reply_still_written_goes_to_claude(make_scene) -> None:
    """The stream is cancelled at "basta": no money for words nobody hears. More than
    `resume_min_chars` have arrived, so only the unfinished stream keeps the rest away."""
    scene = make_scene(
        text_reply(*STORY * 3),
        text_reply("Eravamo al faro."),
        sentence_frames=PLAY,
        chunk_wait=3,
        window_s=4.0,
    )
    await stop_the_story(scene)
    assert scene.assistant._unsaid_rest is None
    await scene.say("continua")
    await idle(scene)
    assert api_calls(scene) == 2 and scene.row(1)["route"] == "claude"
    assert not resumed(scene)


async def test_continua_with_nothing_missing_goes_to_claude(make_scene) -> None:
    scene = make_scene(text_reply("Eccomi."))
    await scene.say("continua")
    await idle(scene)
    assert scene.row(0)["route"] == "claude" and api_calls(scene) == 1


async def test_continua_after_history_reset_goes_to_claude(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), text_reply("Dove?"), sentence_frames=PLAY)
    assert scene.assistant.barge_in is not None
    scene.assistant.barge_in.rest_s = 5.0
    await stop_the_story(scene)
    await scene.advance(6.0)
    await scene.say("continua")
    await idle(scene)
    assert scene.row(1)["route"] == "claude" and not resumed(scene)


async def test_continua_after_switching_off_and_on_goes_to_claude(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), text_reply("Dove?"), sentence_frames=PLAY)
    await stop_the_story(scene)
    await scene.say("spegniti")
    await idle(scene)
    await scene.say("accenditi")
    await idle(scene)
    await scene.say("continua")
    await idle(scene)
    assert scene.rows[-1]["route"] == "claude" and not resumed(scene)


async def test_a_later_claude_answer_clears_the_rest(make_scene) -> None:
    scene = make_scene(
        text_reply(*STORY), text_reply("Parigi."), text_reply("Dimmi."), sentence_frames=PLAY
    )
    await stop_the_story(scene)
    await scene.say("qual è la capitale della Francia")
    await idle(scene)
    assert scene.assistant._unsaid_rest is None
    await scene.say("continua")
    await idle(scene)
    assert scene.rows[-1]["route"] == "claude" and not resumed(scene)


async def test_memory_request_over_a_story_resumes_it(make_scene, tmp_path: Path) -> None:
    """A memory answer is local too (free): the story goes on after it."""
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, window_s=4.0)
    scene.responder.memory = MemoryVoice(MemoryStore(tmp_path / "memory.db"))
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("ricordati che mia sorella è Anna")
    await idle(scene)
    assert scene.row(1)["route"] == "memory" and resumed(scene) and api_calls(scene) == 1


async def test_a_reply_cut_after_a_bridge_keeps_no_bridge_in_the_history(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    scene.speaker.played[:0] = [BRIDGE]  # "Ci guardo." was said before the story
    scene.assistant._interrupted = True
    scene.responder.reply_cut(list(scene.speaker.played))
    ((_, answer),) = scene.remembered
    assert BRIDGE not in answer


async def test_a_newer_cut_claude_reply_clears_the_older_rest(make_scene) -> None:
    """Story A is whole when "basta" cuts it; story B is still being written when "basta" cuts
    it: "continua" is for B (Claude), never A's rest."""
    scene = make_scene(
        text_reply(*STORY),
        text_reply(*STORY * 3),
        text_reply("Eravamo al drago."),
        sentence_frames=PLAY,
    )
    await stop_the_story(scene)
    assert scene.assistant._unsaid_rest is not None
    scene.client.chunk_wait = 3
    await stop_the_story(scene)
    assert scene.assistant._unsaid_rest is None
    await scene.say("continua")
    await idle(scene)
    assert api_calls(scene) == 3 and scene.rows[-1]["route"] == "claude"
    assert not resumed(scene)


async def test_basta_over_a_whole_reply_with_an_unrun_action_keeps_no_rest(
    make_scene, home: Path
) -> None:
    """The story ends with a "Confermi?" nobody can answer after "basta": no rest to say."""
    desk = FakeDesktop()
    scene = make_scene(
        SAY_THEN_CALL, text_reply("Dimmi."), sentence_frames=PLAY, desk=desk, home=home
    )
    await scene.say(WRITE)
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("basta")
    await idle(scene, poll_s=0.002)
    assert scene.assistant._unsaid_rest is None
    await scene.say("continua")
    await idle(scene, poll_s=0.002)
    assert scene.rows[-1]["route"] == "claude" and not resumed(scene)


async def test_continua_after_the_tray_switched_it_off_and_on_goes_to_claude(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), text_reply("Dove?"), sentence_frames=PLAY)
    await stop_the_story(scene)
    system = scene.responder.actions.system
    system.apply("off")
    system.apply("on")
    await scene.say("continua")
    await idle(scene)
    assert scene.rows[-1]["route"] == "claude" and not resumed(scene)


async def test_typed_continua_shows_the_rest_in_the_chat(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY)
    await stop_the_story(scene)
    heard = list(scene.speaker.played)
    reply = await scene.assistant.handle_typed("continua")
    assert reply == remainder("".join(STORY), heard) and api_calls(scene) == 1
    assert scene.assistant._unsaid_rest is None


async def test_dropped_action_answers_for_the_handles_turn_not_the_newest(make_scene) -> None:
    from claudio.brain.claude import _Turn

    brain = make_scene().brain
    old = _Turn("old")
    old.note_not_run(["type_text"])
    brain._turn = _Turn("newer")
    assert brain.dropped_action(old) and not brain.dropped_action(brain._turn)
    assert not brain.dropped_action(None)


async def test_a_yes_over_the_playing_confermi_runs_the_action_once(make_scene, home: Path) -> None:
    """Rule 6 through the Assistant: "Confermi?" is playing, "Claudio, sì" takes over the reply
    and is the answer, so the action runs and Haiku goes on."""
    desk = FakeDesktop()
    words = "Scrivo ciao."
    call = (
        [words + " "],
        "tool_use",
        [text(words), use("type_text", text="ciao", **INTO_CHROME)],
    )
    scene = make_scene(call, text_reply("Scritto."), sentence_frames=PLAY, desk=desk, home=home)
    await scene.say(WRITE)
    await scene.until(lambda: any("Confermi?" in p for p in scene.speaker.played))
    await scene.say_over("Claudio, sì")
    await idle(scene, poll_s=0.002)
    assert [d[0] for d in desk.done if d[0] in ("keys", "paste")] == ["keys"]
    assert not scene.brain.awaiting
