"""v0.1.2 voice extras: media keys, "ripeti", "basta" (also while speaking) and the
conversation window without the wake word."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest

from claudio.app import Assistant, BargeIn, State
from claudio.audio.vad import Endpointer, EndpointEvent
from claudio.config import VadSettings, load_settings
from claudio.router.actions import MEDIA_KEYS
from claudio.router.matcher import CommandMatcher
from claudio.tts.speaker import StreamingSpeaker
from tests.unit.test_actions import Recorder, run
from tests.unit.test_app import AudioClock, FakeSpeaker, FakeStt, FakeWake
from tests.unit.test_safety import responder, say

# --- the commands ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "command", "language"),
    [
        ("Metti in pausa la musica", "media_play_pause", "it"),
        ("Riprendi la musica.", "media_play_pause", "it"),
        ("Play", "media_play_pause", "it"),
        ("Canzone successiva", "media_next", "it"),
        ("Prossima canzone!", "media_next", "it"),
        ("Salta 3 canzoni", "media_next", "it"),
        ("Canzone precedente", "media_previous", "it"),
        ("Pause the music", "media_play_pause", "en"),
        ("Next song", "media_next", "en"),
        ("Previous track", "media_previous", "en"),
        ("Ripeti", "repeat", "it"),
        ("Puoi ripetere?", "repeat", "it"),
        ("Non ho capito", "repeat", "it"),
        ("Say that again", "repeat", "en"),
        ("Basta!", "stop", "it"),
        ("Claudio, basta così.", "stop", "it"),
        ("Stai zitto", "stop", "it"),
        ("Stop.", "stop", "it"),
        ("That's enough", "stop", "en"),
    ],
)
def test_new_commands_match(matcher: CommandMatcher, text: str, command: str, language: str):
    match = matcher.match(text, language)
    assert match is not None and match.command.id == command


@pytest.mark.parametrize(
    "text",
    [
        "Mettiti in pausa",  # CLAUDIO's own pause, not the music's
        "Vai in pausa",
        "Basta aprire Chrome",  # "basta" must be all that was said
        "Stop the computer",
        "Salta",  # too short to be sure
    ],
)
def test_lookalikes_are_not_media_or_stop(matcher: CommandMatcher, text: str) -> None:
    match = matcher.match(text)
    assert match is None or match.command.id not in {
        "media_play_pause",
        "media_next",
        "media_previous",
        "stop",
    }


@pytest.mark.parametrize(
    ("text", "key", "presses"),
    [
        ("metti in pausa la musica", "play_pause", 1),
        ("canzone successiva", "next_track", 1),
        ("salta 3 canzoni", "next_track", 3),
        ("canzone precedente", "previous_track", 1),
    ],
)
def test_media_keys_press_the_windows_key(
    matcher: CommandMatcher, text: str, key: str, presses: int
) -> None:
    rec = Recorder()
    assert run(matcher, rec, text) == ""  # the music itself is the answer
    assert rec.keys == [MEDIA_KEYS[key]] * presses


def test_media_virtual_key_codes() -> None:
    # https://learn.microsoft.com/windows/win32/inputdev/virtual-key-codes
    assert (MEDIA_KEYS["play_pause"], MEDIA_KEYS["next_track"], MEDIA_KEYS["previous_track"]) == (
        0xB3,
        0xB0,
        0xB1,
    )


async def test_repeat_says_the_last_reply_in_its_language(matcher: CommandMatcher) -> None:
    r = responder(matcher, Recorder())
    assert await say(r, "ripeti") == "Non ho ancora detto niente."
    assert await say(r, "what time is it", "en") == "It's 2 oh 5 p.m."
    assert await say(r, "ripeti") == "It's 2 oh 5 p.m."
    assert r.language == "en"  # read by the English voice, as the first time
    assert await say(r, "ripeti") == "It's 2 oh 5 p.m."  # "ripeti" twice: still the same
    assert r.last_route["command"] == "repeat"


async def test_repeat_after_a_question_repeats_the_question(matcher: CommandMatcher) -> None:
    r = responder(matcher, Recorder())
    await say(r, "che ore sono")
    await say(r, "chiudi la finestra")
    await say(r, "no")  # "Annullato."
    assert await say(r, "ripeti") == "Va bene, non lo faccio."


async def test_stop_says_nothing(matcher: CommandMatcher) -> None:
    r = responder(matcher, Recorder())
    assert await say(r, "basta") == ""
    assert r.last_route["command"] == "stop"


@pytest.mark.parametrize(
    ("text", "stop"),
    [
        ("Basta.", True),
        ("Basta, basta!", True),
        ("Stop", True),
        ("Shut up.", True),
        ("Claudio basta", True),
        ("Apro Chrome.", False),  # CLAUDIO hearing itself through the speakers
        ("Basta aprire Chrome", False),
        ("", False),
    ],
)
def test_is_stop(matcher: CommandMatcher, text: str, stop: bool) -> None:
    assert responder(matcher, Recorder()).is_stop(text) is stop


def test_conversation_settings_are_shipped() -> None:
    s = load_settings()
    assert s.conversation.window_s == 6.0
    assert s.conversation.barge_in and s.conversation.barge_in_max_s == 1.2
    assert s.conversation.barge_in_mode == "duck"


def test_endpointer_timeout_can_be_changed_per_phrase(fake_vad, vad_settings) -> None:
    ep = Endpointer(fake_vad, vad_settings)  # 3 s by default
    ep.reset(timeout_s=0.5)
    silence = np.zeros(1280, dtype=np.int16)
    events = [ep.process(silence) for _ in range(8)]  # 0.64 s
    assert EndpointEvent.TIMEOUT in events
    ep.reset()
    assert EndpointEvent.TIMEOUT not in [ep.process(silence) for _ in range(8)]


# --- speaker and player: a reply cut short ----------------------------------------------


class SlowPlayer:
    sample_rate = 22050

    def __init__(self) -> None:
        self.played = 0

    def write(self, audio, sample_rate=None, cancel: threading.Event | None = None) -> bool:
        for _ in range(10):  # ten slices, like AudioPlayer with a cancel event
            if cancel is not None and cancel.is_set():
                return False
            threading.Event().wait(0.005)
        self.played += 1
        return True


class Synth:
    sample_rate = 22050

    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(100, dtype=np.int16)


async def test_interrupt_stops_the_reply_and_the_next_one_plays() -> None:
    player = SlowPlayer()
    speaker = StreamingSpeaker(Synth(), player)  # type: ignore[arg-type]

    async def text() -> AsyncIterator[str]:
        for i in range(20):
            yield f"Frase {i}. "

    async def basta() -> None:
        while player.played < 1:
            await asyncio.sleep(0.001)
        speaker.interrupt()

    await asyncio.gather(speaker.speak(text()), basta())
    assert 1 <= player.played < 5
    played = player.played

    async def one() -> AsyncIterator[str]:
        yield "Ciao."

    await speaker.speak(one())  # a new reply is not cut by the old "basta"
    assert player.played == played + 1


def test_player_stops_between_slices(monkeypatch) -> None:
    import claudio.audio.player as player_module

    written: list[int] = []

    class Stream:
        def write(self, data: np.ndarray) -> None:
            written.append(len(data))
            if len(written) == 2:
                cancel.set()

    p = player_module.AudioPlayer(10000, device=0)
    p._stream = Stream()
    cancel = threading.Event()
    assert p.write(np.zeros(10000, np.int16), cancel=cancel) is False
    assert written == [1000, 1000]  # 0.1 s slices: stopped after the second
    written.clear()
    assert p.write(np.zeros(10000, np.int16)) is True  # no event: one write, as before
    assert written == [10000]


# --- the assistant: conversation window and barge-in -----------------------------------


def build(
    fake_vad,
    vad_settings,
    matcher: CommandMatcher,
    texts: list[str],
    window_s: float = 1.0,
    speaker: FakeSpeaker | None = None,
    barge_in: BargeIn | None = None,
):
    rec = Recorder()
    speaker = speaker or FakeSpeaker()
    r = responder(matcher, rec)
    assistant = Assistant(
        FakeWake({3}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(texts),
        r,
        speaker,
        clock=AudioClock(),
        system=r.actions.system,  # shared, as in __main__
        conversation_s=window_s,
        barge_in=barge_in,
    )
    return assistant, speaker, rec, r


async def talk(
    assistant: Assistant,
    make_frames: Callable[[str, int], list[np.ndarray]],
    speaker: FakeSpeaker,
    requests: int,
    tail_frames: int = 40,
) -> None:
    """The wake word and a request, then `requests - 1` more requests each said once
    CLAUDIO listens again after its reply, then `tail_frames` of silence."""
    clock = assistant.clock
    assert isinstance(clock, AudioClock)
    silence = np.zeros(1280, dtype=np.int16)

    async def gen():
        for f in make_frames("silence", 240) + make_frames("speech", 640):
            clock.t += 0.08
            yield f
            await asyncio.sleep(0)
        for n in range(1, requests):
            for _ in range(200):  # until the reply is over and CLAUDIO listens again
                if len(speaker.beeps) > n and assistant.state is State.LISTENING:
                    break
                clock.t += 0.08
                yield silence
                await asyncio.sleep(0.001)
            for f in make_frames("speech", 640):
                clock.t += 0.08
                yield f
                await asyncio.sleep(0)
        for _ in range(tail_frames):
            clock.t += 0.08
            yield silence
            await asyncio.sleep(0.001)

    await assistant.run(gen())


async def test_conversation_needs_no_wake_word_then_times_out(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, rec, _ = build(
        fake_vad, vad_settings, matcher, ["che ore sono", "canzone successiva", "ripeti"]
    )
    await talk(assistant, make_frames, speaker, requests=3)
    assert speaker.spoken == ["Sono le 14 e 5.", "", "Sono le 14 e 5."]
    assert rec.keys == [MEDIA_KEYS["next_track"]]
    rows = assistant.history
    assert [r["wake_word"] for r in rows] == ["hey_claudio", *["conversation"] * 3]
    assert [r["outcome"] for r in rows] == ["ok", "ok", "ok", "timeout"]
    assert speaker.beeps == ["wake", "wake", "wake", "wake", "cancel"]
    assert assistant.state is State.IDLE


async def test_conversation_ends_on_what_is_not_a_request(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, _ = build(
        fake_vad, vad_settings, matcher, ["che ore sono", "e poi gli ho detto di no"]
    )
    await talk(assistant, make_frames, speaker, requests=2)
    assert speaker.spoken == ["Sono le 14 e 5."]  # the TV's line gets no "non lo so fare"
    assert [r["outcome"] for r in assistant.history] == ["ok", "conversation_end"]
    assert speaker.beeps[-1] == "cancel" and assistant.state is State.IDLE


async def test_basta_ends_the_conversation(fake_vad, vad_settings, make_frames, matcher) -> None:
    assistant, speaker, _, _ = build(fake_vad, vad_settings, matcher, ["che ore sono", "basta"])
    await talk(assistant, make_frames, speaker, requests=2)
    assert [r["outcome"] for r in assistant.history] == ["ok", "stopped"]
    assert speaker.beeps == ["wake", "wake", "cancel"]  # no third window
    assert assistant.state is State.IDLE


async def test_window_zero_is_the_old_behaviour(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, _ = build(fake_vad, vad_settings, matcher, ["che ore sono"], 0.0)
    await talk(assistant, make_frames, speaker, requests=1)
    assert [r["outcome"] for r in assistant.history] == ["ok"]
    assert speaker.beeps == ["wake"]


async def test_no_conversation_after_pausing(fake_vad, vad_settings, make_frames, matcher) -> None:
    assistant, speaker, _, _ = build(fake_vad, vad_settings, matcher, ["vai in pausa"])
    await talk(assistant, make_frames, speaker, requests=1)
    assert len(assistant.history) == 1 and speaker.beeps == ["wake"]


class LongSpeaker(FakeSpeaker):
    """Speaks for up to `frames` microphone frames, or until interrupted."""

    def __init__(self) -> None:
        super().__init__()
        self.cut = False
        self._stop = asyncio.Event()

    def interrupt(self) -> None:
        self.cut = True
        self._stop.set()

    async def speak(self, text_stream, on_first_audio=None, language="it") -> None:
        self._stop = asyncio.Event()
        text = "".join([c async for c in text_stream])
        if on_first_audio:
            on_first_audio()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=1.0)
        self.spoken.append(text)


def barge(vad_settings: VadSettings, heard: list[str], **kwargs) -> BargeIn:
    from tests.conftest import FakeVad

    texts = iter(heard)
    return BargeIn(
        Endpointer(FakeVad(), vad_settings.model_copy(update={"max_utterance_s": 2.0})),
        lambda audio: next(texts, ""),
        **kwargs,
    )


async def speak_over(assistant: Assistant, make_frames, speaker: LongSpeaker) -> None:
    """A request, then speech while CLAUDIO is speaking, then silence."""
    clock = assistant.clock
    assert isinstance(clock, AudioClock)
    silence = np.zeros(1280, dtype=np.int16)

    async def gen():
        for f in make_frames("silence", 240) + make_frames("speech", 640):
            clock.t += 0.08
            yield f
            await asyncio.sleep(0)
        for _ in range(200):
            if assistant.state is State.SPEAKING:
                break
            clock.t += 0.08
            yield silence
            await asyncio.sleep(0.001)
        for f in make_frames("speech", 480) + make_frames("silence", 800):
            clock.t += 0.08
            yield f
            await asyncio.sleep(0.001)
        for _ in range(60):
            clock.t += 0.08
            yield silence
            await asyncio.sleep(0.005)

    await assistant.run(gen())


async def test_basta_while_speaking_interrupts(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["Basta!"]),
    )
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut
    row = assistant.history[0]
    assert row["outcome"] == "interrupted" and row["barge_in"] == "Basta!"
    assert len(assistant.history) == 1  # "basta" also closes the conversation
    assert assistant.state is State.IDLE


async def test_own_voice_while_speaking_does_not_interrupt(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        window_s=0.0,
        speaker=speaker,
        barge_in=barge(vad_settings, ["Sono le 14 e 5."]),
    )
    await speak_over(assistant, make_frames, speaker)
    assert not speaker.cut
    assert assistant.history[0]["outcome"] == "ok"


async def test_basta_cancels_a_question_being_asked(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, rec, r = build(
        fake_vad,
        vad_settings,
        matcher,
        ["chiudi la finestra"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["basta"]),
    )
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut and not r.awaiting and rec.combos == []
    assert len(assistant.history) == 1


# --- v0.3.2: any request said over the reply stops it and is heard ----------------------


async def test_a_request_over_the_reply_stops_it_and_is_answered(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "che giorno è oggi"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["che giorno è oggi"]),
    )
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut
    first, second = list(assistant.history)[:2]
    assert first["outcome"] == "interrupted" and first["barge_in"] == "che giorno è oggi"
    assert isinstance(first["barge_dbfs"], float)  # the level barge_in_min_dbfs is tuned on
    assert second["wake_word"] == "barge_in" and second["text"] == "che giorno è oggi"
    assert second["outcome"] == "ok"
    assert speaker.ducks[:2] == [0.3, None]  # lowered at once, full volume for the next reply
    assert speaker.beeps[0] == "wake" and speaker.beeps.count("wake") >= 1
    # Then the cut reply goes on (JenJo, 25/10): nothing of it had played in the fake.
    assert speaker.spoken[-1] == "Dicevo: Sono le 14 e 5." and second["resumed"] is True
    assert second["reply"] == "Oggi è domenica 18 ottobre 2026."  # the log keeps the answer


async def test_resume_s_zero_never_resumes(fake_vad, vad_settings, make_frames, matcher) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "che giorno è oggi"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["che giorno è oggi"], resume_s=0.0),
    )
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut and not any(t.startswith("Dicevo") for t in speaker.spoken)


async def test_basta_does_not_resume(fake_vad, vad_settings, make_frames, matcher) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "che giorno è oggi"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["basta"]),
    )
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut and assistant._cut is None
    assert not any(t.startswith("Dicevo") for t in speaker.spoken)


async def test_noise_over_the_reply_only_lowers_it_for_a_moment(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        window_s=0.0,
        speaker=speaker,
        barge_in=barge(vad_settings, ["eh"]),
    )
    await speak_over(assistant, make_frames, speaker)
    assert not speaker.cut
    assert speaker.ducks == [0.3, None]
    assert [r["outcome"] for r in assistant.history] == ["ok"]


async def test_a_quiet_voice_over_the_reply_is_not_even_transcribed(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    heard: list[str] = []
    b = barge(vad_settings, ["che giorno è oggi"], min_dbfs=0.0)  # nothing is that loud
    transcribe = b.transcribe
    b.transcribe = lambda audio: heard.append("x") or transcribe(audio)
    assistant, _, _, _ = build(
        fake_vad, vad_settings, matcher, ["che ore sono"], window_s=0.0, speaker=speaker, barge_in=b
    )
    await speak_over(assistant, make_frames, speaker)
    assert not speaker.cut and heard == [] and speaker.ducks == []


async def test_stop_mode_cuts_the_reply_as_soon_as_a_voice_starts(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "che giorno è oggi"],
        speaker=speaker,
        barge_in=barge(vad_settings, [], mode="stop"),  # Whisper is not even asked
    )
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut and speaker.ducks == [None]
    rows = list(assistant.history)
    assert rows[0]["barge_in"] == "voice" and rows[1]["wake_word"] == "barge_in"
    assert rows[1]["text"] == "che giorno è oggi"


async def test_confirm_mode_does_not_lower_the_reply(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "che giorno è oggi"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["che giorno è oggi"], mode="confirm"),
    )
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut and speaker.ducks == [None]


def test_the_reply_echo_is_not_a_request(fake_vad, vad_settings, matcher) -> None:
    assistant, _, _, _ = build(fake_vad, vad_settings, matcher, ["x"])
    assistant._reply_parts = ["Sono le 14 ", "e 5, capo."]
    assert not assistant._is_talking_to_me("sono le 14 e 5")
    assert assistant._is_talking_to_me("aspetta, che giorno è oggi?")
    assert assistant._is_talking_to_me("Claudio")  # the name alone
    assert not assistant._is_talking_to_me("eh")
    assert not assistant._is_talking_to_me("")
    # JenJo's log, 25/10: Whisper wrote a noise as "Mm-hmm." and it stopped the story.
    assert not assistant._is_talking_to_me("Mm-hmm.")
    assert not assistant._is_talking_to_me("Ehm, mh.")


# --- a longer pause closes a longer phrase ------------------------------------------------


def test_a_long_phrase_gets_a_longer_pause(fake_vad, vad_settings, make_frames) -> None:
    settings = vad_settings.model_copy(update={"long_end_silence_ms": 800, "long_after_s": 1.5})
    ep = Endpointer(fake_vad, settings)
    events = [ep.process(f) for f in make_frames("speech", 640) + make_frames("silence", 480)]
    assert EndpointEvent.END in events  # a command: 0.4 s of pause is enough

    ep.reset()
    events = [ep.process(f) for f in make_frames("speech", 2000) + make_frames("silence", 560)]
    assert EndpointEvent.END not in events  # a question: a short pause to think
    events = [ep.process(f) for f in make_frames("speech", 400) + make_frames("silence", 960)]
    assert EndpointEvent.END in events
    assert len(ep.audio()) > 16000 * 2.4


def test_resume_continues_a_phrase_already_begun(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    start = np.concatenate(make_frames("speech", 480)).astype(np.float32) / 32768.0
    ep.resume(start)
    assert ep.started
    assert ep.stats()["speech_dbfs"] > -20  # its level counts (not "too far away")
    events = [ep.process(f) for f in make_frames("silence", 480)]
    assert EndpointEvent.END in events and EndpointEvent.TIMEOUT not in events
    assert len(ep.audio()) >= len(start)


def test_player_lowers_only_replies(monkeypatch) -> None:
    import claudio.audio.player as player_module

    written: list[np.ndarray] = []

    class Stream:
        def write(self, data: np.ndarray) -> None:
            written.append(data)

    p = player_module.AudioPlayer(10000, device=0)
    p._stream = Stream()
    p.gain = 0.5
    p.write(np.full(2000, 1000, np.int16), cancel=threading.Event())
    assert all(int(chunk.max()) == 500 for chunk in written)
    written.clear()
    p.write(np.full(2000, 1000, np.int16))  # a beep: never lowered
    assert int(written[0].max()) == 1000


async def test_a_phrase_over_the_reply_does_not_move_the_gain(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    """Said over the reply, louder and with its echo: the automatic gain ignores it (JenJo
    25/10: the gain fell to +0 dB after one and the next requests came in at -42 dBFS)."""

    class Level:
        gain_db = 0.0

        def __init__(self) -> None:
            self.seen: list[dict] = []

        def observe(self, stats) -> None:
            self.seen.append(dict(stats))

    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "che giorno è oggi"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["che giorno è oggi"]),
    )
    level = Level()
    assistant.level = level
    await speak_over(assistant, make_frames, speaker)
    wakes = [r["wake_word"] for r in assistant.history]
    assert "barge_in" in wakes and len(level.seen) == wakes.index("barge_in")


# --- 26/10: CLAUDIO's own voice from the speakers is not a request -----------------------


async def test_needs_name_ignores_a_request_without_the_name(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        window_s=0.0,
        speaker=speaker,
        barge_in=barge(vad_settings, ["aspetta, raccontami una barzelletta"], needs_name=True),
    )
    await speak_over(assistant, make_frames, speaker)
    assert not speaker.cut
    assert [r["outcome"] for r in assistant.history] == ["ok"]


async def test_needs_name_takes_a_request_with_the_name(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "Claudio, che giorno è oggi"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["Claudio, che giorno è oggi"], needs_name=True),
    )
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut
    second = list(assistant.history)[1]
    assert second["wake_word"] == "barge_in" and second["outcome"] == "ok"


def test_needs_name_still_lets_basta_and_commands_through(fake_vad, vad_settings, matcher) -> None:
    assistant, _, _, _ = build(
        fake_vad, vad_settings, matcher, ["x"], barge_in=barge(vad_settings, [], needs_name=True)
    )
    assistant._reply_parts = ["Sono le 14 ", "e 5, capo."]
    assert assistant._is_talking_to_me("Claudio, che giorno è oggi?")
    assert assistant._is_talking_to_me("canzone successiva")  # a local command
    assert not assistant._is_talking_to_me("aspetta, raccontami una barzelletta")


def test_a_misheard_echo_is_not_a_request(fake_vad, vad_settings, matcher) -> None:
    """JenJo's log 26/10 19:41: Whisper turned the reply's echo into new words."""
    speaker = FakeSpeaker()
    assistant, _, _, _ = build(
        fake_vad, vad_settings, matcher, ["x"], speaker=speaker, barge_in=barge(vad_settings, [])
    )
    speaker.played = ["Sono le 19 e 42.", "Il padrone dell'universo è in linea."]
    assistant._reply_parts = speaker.played[:]
    assert not assistant._is_talking_to_me("Se non le diciamo")
    assert not assistant._is_talking_to_me("dell'università e lì")
    assert assistant._is_talking_to_me("aspetta, che giorno è oggi?")


async def test_the_echo_in_the_conversation_window_is_dropped(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "Se non le quattordici e cinque"],
        barge_in=barge(vad_settings, []),
    )
    await talk(assistant, make_frames, speaker, requests=2)
    assert speaker.spoken == ["Sono le 14 e 5."]  # its own echo got no answer
    rows = list(assistant.history)
    assert [r["outcome"] for r in rows] == ["ok", "conversation_end"]


async def test_the_window_starts_deaf_but_still_hears_the_next_request(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, _ = build(
        fake_vad, vad_settings, matcher, ["che ore sono", "che giorno è oggi"]
    )
    assistant.echo_tail_ms = 300
    await talk(assistant, make_frames, speaker, requests=2)
    assert len(speaker.spoken) == 2
    assert [r["outcome"] for r in assistant.history][:2] == ["ok", "ok"]


def test_echo_settings_are_shipped() -> None:
    c = load_settings().conversation
    assert c.barge_in_needs_name and c.echo_similarity == 70.0 and c.echo_tail_ms == 300


async def test_once_its_echo_is_known_the_reply_stops_lowering_itself(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    """With speakers every echo lowered the reply until Whisper had listened: once a phrase
    was the echo, voices no louder than it wait for Whisper at full volume."""
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        window_s=0.0,
        speaker=speaker,
        barge_in=barge(vad_settings, ["Sono le 14 e 5."]),
    )
    await speak_over(assistant, make_frames, speaker)
    assert assistant._echo_dbfs is not None  # learnt from the first echo
    assert speaker.ducks == [0.3, None]  # that one still lowered it
    assert assistant._sounds_like_echo(assistant.barge_in)
    assistant._echo_dbfs = float("-inf")  # any voice is louder: lowered at once again
    assert not assistant._sounds_like_echo(assistant.barge_in)
