"""Safety levels: the question, the answer, and the assistant listening for it."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from claudio.app import Assistant, State
from claudio.audio.vad import Endpointer
from claudio.router.actions import VK_F4, VK_MENU, ActionRunner
from claudio.router.matcher import Command, CommandMatcher, Match, load_commands
from claudio.router.responder import LocalResponder
from claudio.safety.confirmation import Answer, parse_answer, question
from tests.unit.test_actions import Recorder
from tests.unit.test_app import AudioClock, FakeSpeaker, FakeStt, FakeWake


@pytest.mark.parametrize(
    "text",
    [
        "Sì.",
        "si",
        "Sì, certo",
        "va bene",
        "ok",
        "Okay!",
        "yes",
        "Yes, go ahead",
        "procedi",
        "Hey Claudio, sì",
    ],
)
def test_yes(text: str) -> None:
    assert parse_answer(text) is Answer.YES


def test_go_alone_is_not_a_yes() -> None:
    assert parse_answer("go away") is not Answer.YES


@pytest.mark.parametrize(
    "text",
    [
        "No.",
        "no grazie",
        "annulla",
        "Non farlo",
        "lascia stare",
        "stop",
        "cancel",
        "don't",
        "sì, anzi no",
        "yes, no wait",
    ],
)
def test_no_wins(text: str) -> None:
    assert parse_answer(text) is Answer.NO


@pytest.mark.parametrize("text", ["", "apri chrome", "boh", "Claudio", "che ore sono"])
def test_anything_else_is_unclear(text: str) -> None:
    assert parse_answer(text) is Answer.UNCLEAR


@pytest.mark.parametrize("text", ["confermo", "Sì, confermo.", "I confirm", "confirm"])
def test_level_3_needs_an_explicit_confirmation(text: str) -> None:
    assert parse_answer(text, 3) is Answer.YES


@pytest.mark.parametrize("text", ["sì", "ok", "yes", "va bene", "go ahead"])
def test_level_3_does_not_take_a_plain_yes(text: str) -> None:
    assert parse_answer(text, 3) is Answer.UNCLEAR


def test_level_3_refusal() -> None:
    assert parse_answer("non confermo", 3) is Answer.NO


def test_question_falls_back_to_the_level(tmp_path: Path) -> None:
    plain2 = Command(id="x", phrases=("x",), action="close_window", level=2)
    plain3 = Command(id="y", phrases=("y",), action="close_window", level=3)
    assert question(Match(plain2, 100), "it") == "Confermi?"
    assert question(Match(plain2, 100), "en") == "Do you confirm?"
    assert "confermo" in question(Match(plain3, 100), "it")
    own = Command(id="z", phrases=("z",), action="x", level=2, confirm="Sicuro?")
    assert question(Match(own, 100), "en") == "Sicuro?"  # Italian when no confirm_en


def test_invalid_level_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "c.toml"
    bad.write_text(
        '[[command]]\nid = "a"\nphrases = ["a"]\naction = "close_window"\nlevel = 4\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"livello|level"):
        load_commands(bad)


def test_shipped_close_window_is_level_2(matcher: CommandMatcher) -> None:
    match = matcher.match("chiudi la finestra")
    assert match is not None and match.command.level == 2
    en = matcher.match("close the window", "en")
    assert en is not None and en.command.id == "close_window"


def responder(matcher: CommandMatcher, rec: Recorder) -> LocalResponder:
    return LocalResponder(matcher, ActionRunner(rec.ctx()))


async def say(r: LocalResponder, text: str, language: str = "it") -> str:
    return "".join([c async for c in r.respond(text, language)])


async def test_level_2_asks_then_runs_on_yes(matcher: CommandMatcher) -> None:
    rec = Recorder()
    r = responder(matcher, rec)
    assert await say(r, "chiudi la finestra") == "Chiudo la finestra in primo piano. Confermi?"
    assert r.awaiting and rec.combos == []
    assert r.last_route["route"] == "confirm"
    assert await say(r, "sì") == "Fatto."
    assert rec.combos == [(VK_MENU, VK_F4)]
    assert not r.awaiting and r.last_route["route"] == "confirmed"


async def test_level_2_refused_or_unclear_does_nothing(matcher: CommandMatcher) -> None:
    rec = Recorder()
    r = responder(matcher, rec)
    await say(r, "chiudi la finestra")
    assert await say(r, "no") == "Va bene, non lo faccio."
    await say(r, "chiudi la finestra")
    assert await say(r, "apri chrome") == "Non ho capito: non lo faccio."
    assert rec.combos == [] and rec.ran == []  # Chrome was not opened either
    assert r.last_route["answer"] == "unclear"


async def test_the_answer_is_in_the_language_of_the_question(matcher: CommandMatcher) -> None:
    rec = Recorder()
    r = responder(matcher, rec)
    assert await say(r, "close the window", "en") == (
        "I'll close the window in front. Do you confirm?"
    )
    assert await say(r, "sì", "it") == "Done."  # Whisper may hear the "yes" as Italian
    assert r.language == "en"


async def test_cancel_pending(matcher: CommandMatcher) -> None:
    rec = Recorder()
    r = responder(matcher, rec)
    await say(r, "chiudi la finestra")
    r.cancel_pending()
    assert not r.awaiting
    assert await say(r, "sì") == "Questo non lo so ancora fare."  # a later "sì" runs nothing
    assert rec.combos == []


async def test_paused_never_asks(matcher: CommandMatcher) -> None:
    rec = Recorder()
    r = responder(matcher, rec)
    r.actions.system.apply("pause")
    assert await say(r, "chiudi la finestra") == ""
    assert not r.awaiting


# The whole loop: question, then the answer heard without the wake word.


def build(fake_vad, vad_settings, matcher: CommandMatcher, texts: list[str]):
    rec = Recorder()
    speaker = FakeSpeaker()
    r = responder(matcher, rec)
    wake = FakeWake({3})
    assistant = Assistant(
        wake, Endpointer(fake_vad, vad_settings), FakeStt(texts), r, speaker, clock=AudioClock()
    )
    return assistant, speaker, rec, r


async def feed(assistant: Assistant, speaker: FakeSpeaker, make_frames, answer: bool) -> None:
    """The request after a wake word, then (if `answer`) the answer once the question is
    spoken, then enough silence for everything to finish (or time out)."""
    clock = assistant.clock
    assert isinstance(clock, AudioClock)
    silence = np.zeros(1280, dtype=np.int16)

    async def gen():
        for f in make_frames("silence", 240) + make_frames("speech", 800):
            clock.t += 0.08
            yield f
            await asyncio.sleep(0)
        for _ in range(200):  # until the question has been asked
            if speaker.spoken and assistant.state is State.LISTENING:
                break
            clock.t += 0.08
            yield silence
            await asyncio.sleep(0.001)
        if answer:
            for f in make_frames("speech", 480):
                clock.t += 0.08
                yield f
                await asyncio.sleep(0)
        for _ in range(60):  # 4.8 s: the answer's reply, or the 3 s timeout
            clock.t += 0.08
            yield silence
            await asyncio.sleep(0.001)

    await assistant.run(gen())


async def test_the_answer_needs_no_wake_word(fake_vad, vad_settings, make_frames, matcher) -> None:
    assistant, speaker, rec, _ = build(
        fake_vad, vad_settings, matcher, ["chiudi la finestra", "sì"]
    )
    await feed(assistant, speaker, make_frames, answer=True)
    assert speaker.spoken == ["Chiudo la finestra in primo piano. Confermi?", "Fatto."]
    assert speaker.beeps == ["wake", "wake"]  # the second beep: "I'm listening for the answer"
    assert rec.combos == [(VK_MENU, VK_F4)]
    rows = assistant.history
    assert [r["route"] for r in rows] == ["confirm", "confirmed"]
    assert rows[1]["wake_word"] == "follow_up"
    assert assistant.state is State.IDLE


async def test_silence_after_the_question_cancels(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, rec, r = build(fake_vad, vad_settings, matcher, ["chiudi la finestra"])
    await feed(assistant, speaker, make_frames, answer=False)
    assert speaker.beeps == ["wake", "wake", "cancel"]
    assert rec.combos == [] and not r.awaiting
    assert [row["outcome"] for row in assistant.history] == ["ok", "timeout"]
    assert assistant.state is State.IDLE
