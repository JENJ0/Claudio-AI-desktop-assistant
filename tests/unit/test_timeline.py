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

import pytest

from claudio.app import State
from claudio.brain.claude import CUT_NOTE
from claudio.tts.speaker import StreamingSpeaker
from tests.unit.test_pc_tools import FakeDesktop, text, use
from tests.unit.test_speaker import FakePlayer, FakeSynth
from tests.unit.timeline import GatedDesktop, Scene, text_reply

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
PLAY = 15  # frames a sentence lasts: 1.2 s
RESUME = "Dicevo"


@pytest.fixture
async def make_scene(
    fake_vad: Any, vad_settings: Any, matcher: Any, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[..., Scene]]:
    """Scenes whose Whisper runs inline (no worker thread: the same frames at every run)."""
    scenes: list[Scene] = []

    async def inline(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)  # a thread hands over: the caller does not run on at once
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)

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


HISTORY_BUG = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="G2 history item: the history gets the whole reply, not what was heard",
)


@pytest.mark.parametrize(
    ("chunk_wait", "through_cut"),
    [
        # The reply is all read before the cut: _remember stores it whole.
        pytest.param(0, False, marks=HISTORY_BUG, id="reply-read-before-the-cut"),
        # Chunks come every 9 frames (read at 15, 24, 33, 42, 51), the stop is at frame 47: the
        # last chunk is still to come, so _remember_cut stores what was read, which is more
        # than what was heard (the stream goes on being read while the voice plays).
        pytest.param(9, True, marks=HISTORY_BUG, id="cut-while-reading"),
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="discovery (not in the plan): while RESPONDING (Claude has not spoken yet)"
    " handle_frame ignores the microphone, so a 'basta' said then is lost and the reply"
    " is spoken in full",
)
async def test_basta_before_the_first_word_says_nothing_later(make_scene) -> None:
    scene = make_scene(text_reply(*STORY), sentence_frames=PLAY, claude_wait=40)
    await scene.say("raccontami una storia")
    assert scene.assistant.state is State.RESPONDING  # Claude has not answered yet
    await scene.say_over("basta")
    await idle(scene)
    assert scene.speaker.played == []
    assert scene.row(0)["outcome"] in ("stopped", "interrupted")


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
    said = "Ci guardo, socio."
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
    assert scene.speaker.played == [said]  # said while the tool is still running
    assert ("launch", "Spotify") not in desk.done
    desk.gate.set()
    await idle(scene, poll_s=0.002)
    assert scene.speaker.played == [said, "Aperto."]
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


async def test_the_words_of_the_cut_reply_still_come_back(make_scene, home: Path) -> None:
    """Only the action is stopped: the story it told before the call is resumed, as always."""
    desk = FakeDesktop()
    scene = make_scene(SAY_THEN_CALL, sentence_frames=PLAY, chunk_wait=9, desk=desk, home=home)
    await scene.say(WRITE)
    await scene.until_speaking()
    await scene.say_over("che ore sono")
    await idle(scene, poll_s=0.002)
    assert resumed(scene)
    assert not typed_anything(desk) and not scene.brain.awaiting


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
    typed = asyncio.create_task(scene.assistant.handle_typed("che ore sono", "it"))
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


LOSES_THE_TEXT = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="G2: text lost in the row after basta (5.2 #4): _marked saves the reply in its"
    " finally, which does not run while the speaker leaves the stream open",
)
TIMED_TYPED = pytest.param(
    "timed",
    "typed",
    marks=pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="G2: text lost in the row after basta (5.2 #4): as the spoken one, but no voice"
        " can reach a typed reply today: speaker.interrupt() has two callers, _stop_reply and"
        " _take_over, never on a typed reply",
    ),
)


@pytest.mark.parametrize(
    ("voice", "path"),
    [
        TIMED_TYPED,
        pytest.param("timed", "spoken", marks=LOSES_THE_TEXT),
        # The real speaker leaves the stream open too while _produce is rendering (a sentence
        # to synthesise: the inline to_thread hands over for one turn) or on a full queue.
        pytest.param("streaming", "typed", marks=LOSES_THE_TEXT),
        pytest.param("streaming", "spoken", marks=LOSES_THE_TEXT),
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
            scene.assistant.handle_typed("raccontami una storia", "it", speak=True)
        ).done
    else:
        await scene.say("raccontami una storia")
        done = lambda: scene.assistant.state is State.IDLE  # noqa: E731
    await scene.until(done, poll_s=0.002)
    reply = scene.row(0).get("reply", "")
    assert scene.speaker.played and all(s in reply for s in scene.speaker.played)


# --- the real sequences of 05/11 (their shape, with generic phrases) --------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="G2: worth_resuming discards a story ending with a question (Vuoi il seguito?),"
    " so Dicevo does not start after a local command",
)
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="G2: _resume_reply goes on after any answered request, also one to Claude;"
    " JenJo wants it only after a local command",
)
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
