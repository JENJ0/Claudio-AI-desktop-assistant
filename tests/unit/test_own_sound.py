"""CLAUDIO never hears itself (IT-0, 29/10): while one of its own sounds plays and for its
tail (what the speakers still had queued, then the room) nothing listens, whatever made the
sound (a reply, a beep, "Sono pronto") and whichever window opens after it."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from claudio.app import Assistant, State
from claudio.audio.own_sound import OwnSound
from claudio.audio.vad import Endpointer
from claudio.router.actions import VK_F4, VK_MENU
from claudio.router.matcher import Command, CommandMatcher
from tests.unit.test_actions import Recorder
from tests.unit.test_app import AudioClock, FakeResponder, FakeSpeaker, FakeWake
from tests.unit.test_safety import responder

FRAME = 1280  # 80 ms at 16 kHz
SILENCE = np.zeros(FRAME, dtype=np.int16)


def tone(amplitude: int, hz: float) -> np.ndarray:
    t = np.arange(FRAME) / 16000
    return (np.sin(2 * np.pi * hz * t) * amplitude).astype(np.int16)


ECHO = tone(6000, 440.0)  # CLAUDIO's own sound coming back from the speakers
VOICE = tone(16000, 220.0)  # JenJo at the microphone
ECHO_PEAK = 0.3  # float audio below this is the echo, above it JenJo


# --- the gate ----------------------------------------------------------------------------


def test_deaf_while_a_sound_plays_and_for_its_tail() -> None:
    own = OwnSound(16000, tail_ms=400)
    own.started()
    assert own.playing and own.in_frame(FRAME)
    own.ended(queued_s=0.16)  # the speakers still had 160 ms of it
    heard = [own.in_frame(FRAME) for _ in range(8)]
    assert heard == [True] * 7 + [False]  # 160 + 400 ms = 7 frames, then the room is clear


def test_the_tail_runs_while_the_next_sound_plays() -> None:
    own = OwnSound(16000, tail_ms=400)
    own.started()
    own.ended()  # the reply: 400 ms of tail
    own.started()  # the conversation window's beep, right after it
    assert all(own.in_frame(FRAME) for _ in range(3))
    own.ended()  # the beep's own 400 ms: longer than what is left of the reply's
    assert [own.in_frame(FRAME) for _ in range(6)] == [True] * 5 + [False]


def test_two_sounds_at_once_keep_it_deaf_until_both_ended() -> None:
    own = OwnSound(16000)
    own.started()
    own.started()
    own.ended()
    assert own.in_frame(FRAME)
    own.ended()
    assert not own.in_frame(FRAME)  # no tail configured: only while playing


# --- the player: how long its sound is still coming out ------------------------------------


class Stream:
    latency = 0.05  # what PortAudio reports for the device

    def __init__(self, **kwargs) -> None: ...

    def start(self) -> None: ...

    def write(self, data: np.ndarray) -> None: ...


def test_the_player_knows_how_long_its_sound_still_lasts(monkeypatch) -> None:
    import sounddevice as sd

    import claudio.audio.player as player_module

    monkeypatch.setattr(sd, "OutputStream", Stream)
    monkeypatch.setattr(player_module, "describe_device", lambda index, kind: "Odyssey")
    monkeypatch.setattr(player_module, "stream_settings", lambda index: None)
    now = [100.0]
    p = player_module.AudioPlayer(10000, device=0, clock=lambda: now[0])
    p.start()  # reads the device latency
    assert p.queued_s() == 0.0  # nothing played yet
    p.write(np.zeros(3000, np.int16))  # 0.3 s handed over at once
    assert p.queued_s() == pytest.approx(0.35)
    now[0] += 0.2
    assert p.queued_s() == pytest.approx(0.15)
    now[0] += 1.0
    assert p.queued_s() == 0.0
    p.write(np.zeros(1000, np.int16))  # after a pause it plays from now on
    assert p.queued_s() == pytest.approx(0.15)


# --- the assistant ----------------------------------------------------------------------


class EarWake(FakeWake):
    """Fires on call `fire_on` (JenJo's wake word) and on any loud frame after it: what the
    "claudio" model did with CLAUDIO's own beep (28/10, 0.6-0.8)."""

    def __init__(self, fire_on: set[int]) -> None:
        super().__init__(fire_on)
        self.loud = 0

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        self.calls += 1
        if self.calls in self.fire_on:
            return ("hey_claudio", 0.9)
        if int(np.abs(frame.astype(np.int32)).max()) > 1000:
            self.loud += 1
            return ("claudio", 0.7)
        return None


class EchoStt:
    """Whisper: JenJo's phrases in order; CLAUDIO's echo becomes `echo` (what Whisper wrote
    for the end of a reply)."""

    def __init__(self, texts: list[str], echo: str) -> None:
        self.texts, self.echo, self.language = iter(texts), echo, "it"
        self.heard: list[str] = []
        self.samples: list[int] = []  # how much audio each phrase had

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> str:
        text = self.echo if float(np.abs(audio).max()) < ECHO_PEAK else next(self.texts, "")
        self.heard.append(text)
        self.samples.append(len(audio))
        return text


def assistant_for(
    vad, vad_settings, matcher: CommandMatcher, stt: EchoStt, wake: FakeWake, **kwargs
) -> tuple[Assistant, FakeSpeaker, Recorder]:
    rec = Recorder()
    speaker = FakeSpeaker()
    speaker.queued = 0.16  # every sound still had 160 ms to come out when write() returned
    r = responder(matcher, rec)
    assistant = Assistant(
        wake,
        Endpointer(vad, vad_settings),
        stt,
        r,
        speaker,
        clock=AudioClock(),
        system=r.actions.system,
        echo_tail_ms=400,  # with the 160 ms: 7 frames after every sound
        **kwargs,
    )
    return assistant, speaker, rec


async def play(assistant: Assistant, steps: AsyncIterator[np.ndarray]) -> None:
    clock = assistant.clock
    assert isinstance(clock, AudioClock)

    async def gen() -> AsyncIterator[np.ndarray]:
        async for frame in steps:
            clock.t += 0.08
            yield frame
            for _ in range(3):
                await asyncio.sleep(0.001)  # Whisper and the reply run meanwhile

    await assistant.run(gen())


async def until(assistant: Assistant, done, limit: int = 200) -> AsyncIterator[np.ndarray]:
    """Silence until `done()`."""
    for _ in range(limit):
        if done():
            return
        yield SILENCE
        await asyncio.sleep(0.001)
    raise AssertionError("never happened")


def several(frame: np.ndarray, n: int) -> list[np.ndarray]:
    return [frame] * n


def asked(assistant: Assistant, speaker: FakeSpeaker) -> bool:
    """CLAUDIO has asked its question and listens for the answer."""
    return assistant.state is State.LISTENING and bool(speaker.spoken)


LEVEL_3 = Command(id="delete", phrases=("cancella tutto",), action="close_window", level=3)


async def test_claudio_never_confirms_its_own_question(fake_vad, vad_settings) -> None:
    """Level 3 ends with "Per procedere dì: confermo.": the window for the answer opened at
    once, and the end of the question coming out of the speakers was its own "confermo"."""
    stt = EchoStt(["cancella tutto"], echo="Confermo.")
    assistant, speaker, rec = assistant_for(
        fake_vad, vad_settings, CommandMatcher([LEVEL_3]), stt, FakeWake({3})
    )

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(SILENCE, 3) + several(VOICE, 10):
            yield f
        async for f in until(assistant, lambda: asked(assistant, speaker)):
            yield f
        for f in several(ECHO, 6) + several(SILENCE, 60):  # the question's end, then nothing
            yield f

    await play(assistant, steps())
    assert "confermo" in speaker.spoken[0]
    assert rec.combos == []  # nothing ran
    assert "Confermo." not in stt.heard  # the echo never reached Whisper
    assert [row["outcome"] for row in assistant.history] == ["ok", "timeout"]


async def test_the_answer_after_the_tail_is_heard(fake_vad, vad_settings, matcher) -> None:
    """The question's own end is ignored, JenJo's yes right after it counts: no answer is
    lost to the deafness ("Confermi?" heard back used to cancel the action)."""
    stt = EchoStt(["chiudi la finestra", "sì"], echo="Confermi?")
    assistant, speaker, rec = assistant_for(fake_vad, vad_settings, matcher, stt, FakeWake({3}))

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(SILENCE, 3) + several(VOICE, 10):
            yield f
        async for f in until(assistant, lambda: asked(assistant, speaker)):
            yield f
        # The question's end, a pause long enough to close a phrase, then JenJo's "sì".
        for f in several(ECHO, 6) + several(SILENCE, 6) + several(VOICE, 5) + several(SILENCE, 30):
            yield f

    await play(assistant, steps())
    assert speaker.spoken == ["Chiudo la finestra in primo piano. Confermi?", "Fatto."]
    assert rec.combos == [(VK_MENU, VK_F4)]
    assert stt.heard == ["chiudi la finestra", "sì"]


async def test_the_low_beep_after_a_dropped_phrase_does_not_wake_claudio(
    fake_vad, vad_settings, matcher
) -> None:
    """28/10: the beep that ends a conversation woke the "claudio" model. It is also played
    when a phrase in the window is not a request: the wake word heard that one in full."""
    stt = EchoStt(["che ore sono", "e poi gli ho detto di no"], echo="")
    wake = EarWake({3})
    assistant, speaker, _ = assistant_for(
        fake_vad, vad_settings, matcher, stt, wake, conversation_s=3.0
    )

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(SILENCE, 3) + several(VOICE, 10):
            yield f
        async for f in until(assistant, lambda: len(assistant.history) == 1):
            yield f
        for f in several(SILENCE, 10) + several(VOICE, 10):  # the TV, in the window
            yield f
        async for f in until(assistant, lambda: len(assistant.history) == 2):
            yield f
        for f in several(ECHO, 6) + several(SILENCE, 20):  # the low beep, from the speakers
            yield f

    await play(assistant, steps())
    assert [row["outcome"] for row in assistant.history] == ["ok", "conversation_end"]
    assert speaker.beeps == ["wake", "wake", "cancel"]
    assert wake.loud == 0  # the wake word never heard the beep


async def test_the_wake_word_does_not_hear_sono_pronto(fake_vad, vad_settings, matcher) -> None:
    wake = EarWake(set())
    assistant, _, _ = assistant_for(fake_vad, vad_settings, matcher, EchoStt([], ""), wake)
    await assistant.announce("Sono pronto.")

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(ECHO, 7) + several(SILENCE, 3):
            yield f

    await play(assistant, steps())
    assert wake.loud == 0 and wake.calls == 3  # only the frames after the tail


async def test_the_second_chance_never_hears_claudio(fake_vad, vad_settings) -> None:
    """A near miss right after a reply: Whisper gets the last 2.5 s, which held the end of the
    reply. If that said the name ("Sono Claudio"), CLAUDIO woke itself."""
    heard: list[np.ndarray] = []

    def second_chance(audio: np.ndarray) -> str:
        heard.append(audio)
        return ""

    wake = FakeWake(set(), miss_on={1})
    speaker = FakeSpeaker()
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        EchoStt([], ""),
        FakeResponder(),
        speaker,
        clock=AudioClock(),
        second_chance=second_chance,
        echo_tail_ms=400,
    )
    assistant.own_sound.started()  # the reply
    for _ in range(3):
        await assistant.handle_frame(ECHO)
    assistant.own_sound.ended()
    for _ in range(5):  # its tail
        await assistant.handle_frame(ECHO)
    assert wake.calls == 0
    await assistant.handle_frame(SILENCE)  # the wake word listens again: a near miss
    await assistant.wait_idle()
    assert wake.calls == 1 and wake.resets == 1  # it started clean
    assert len(heard) == 1 and float(np.abs(heard[0]).max()) == 0.0  # no echo in it


async def test_a_request_said_over_the_wake_beep_is_heard_whole(fake_vad, vad_settings) -> None:
    """After a wake word JenJo may already be talking ("Claudio che ore sono" in one breath):
    the beep's deafness is only for the windows that open after CLAUDIO spoke."""
    stt = EchoStt(["che ore sono"], echo="")
    speaker = FakeSpeaker()
    speaker.queued = 0.5
    wake = FakeWake({3})
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        stt,
        FakeResponder(),
        speaker,
        clock=AudioClock(),
        echo_tail_ms=400,
    )

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(SILENCE, 3) + several(VOICE, 10) + several(SILENCE, 30):
            yield f

    await play(assistant, steps())
    assert stt.heard == ["che ore sono"]
    assert stt.samples[0] >= 10 * FRAME  # all of it, beep or no beep


async def test_a_sound_whose_length_is_unknown_never_leaves_claudio_deaf(
    fake_vad, vad_settings
) -> None:
    class Unplugged(FakeSpeaker):
        def queued_s(self) -> float:
            raise RuntimeError("device gone")

    wake = FakeWake(set())
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        EchoStt([], ""),
        FakeResponder(),
        Unplugged(),
        clock=AudioClock(),
        echo_tail_ms=400,
    )
    await assistant.announce("Sono pronto.")
    for _ in range(6):
        await assistant.handle_frame(SILENCE)
    assert wake.calls == 1  # 400 ms of tail, then it listens again


async def test_a_phrase_said_once_the_room_is_quiet_is_never_the_echo(
    fake_vad, vad_settings, matcher
) -> None:
    """28/10: "apri la chat" after "Chiudo la chat." shared most of its words with the reply
    and was dropped as CLAUDIO's own voice. Said a second into the window, it cannot be."""
    stt = EchoStt(["chiudi la chat", "apri la chat"], echo="")
    assistant, speaker, _ = assistant_for(
        fake_vad, vad_settings, matcher, stt, FakeWake({3}), conversation_s=3.0
    )

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(SILENCE, 3) + several(VOICE, 10):
            yield f
        async for f in until(assistant, lambda: len(assistant.history) == 1):
            yield f
        for f in several(SILENCE, 20) + several(VOICE, 8) + several(SILENCE, 30):
            yield f

    await play(assistant, steps())
    rows = list(assistant.history)
    assert [row["outcome"] for row in rows][:2] == ["ok", "ok"]
    assert [row["command"] for row in rows][:2] == ["close_chat", "open_chat"]
    assert len(speaker.spoken) >= 2 and rows[1]["listened_ms"] > 300


async def test_a_long_phrase_said_over_the_end_of_the_reply_is_never_the_echo(
    fake_vad, vad_settings, matcher
) -> None:
    """29/10 01:30: JenJo started "non si dice sono sempre ragione..." over the end of the
    reply and went on for 3 s; heard at once, with the reply's words, it was dropped as its
    echo. What is left of the echo once the window listens is a word or two, not seconds."""
    stt = EchoStt(["chiudi la chat", "apri la chat"], echo="")
    assistant, speaker, _ = assistant_for(
        fake_vad, vad_settings, matcher, stt, FakeWake({3}), conversation_s=3.0
    )

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(SILENCE, 3) + several(VOICE, 10):
            yield f
        async for f in until(assistant, lambda: bool(speaker.spoken)):
            yield f
        for f in several(VOICE, 25) + several(SILENCE, 30):  # from the reply's last words
            yield f

    await play(assistant, steps())
    rows = list(assistant.history)
    assert [row["command"] for row in rows][:2] == ["close_chat", "open_chat"]
    assert rows[1]["listened_ms"] <= 300 and rows[1]["voice_ms"] > 1000
