"""IT-1 listening (30/10 live test): a phrase said over the reply is judged whole (2.2), its
loudness is measured before the microphone's gain (2.3), and a phrase begun while a window
was still deaf keeps its start (2.4)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest

from claudio.app import Assistant, BargeIn, State
from claudio.audio.vad import Endpointer
from claudio.config import VadSettings
from claudio.router.actions import VK_F4, VK_MENU
from claudio.router.matcher import CommandMatcher
from tests.conftest import FakeVad
from tests.unit.test_app import AudioClock, FakeWake
from tests.unit.test_conversation import LongSpeaker, build
from tests.unit.test_own_sound import (
    FRAME,
    SILENCE,
    VOICE,
    EchoStt,
    assistant_for,
    play,
    several,
    tone,
    until,
)

SR = 16000


class DipVad(FakeVad):
    """FakeVad with a middle level: a softer sound is neither speech nor silence (0.4, between
    the fixture's 0.5 threshold and its 0.35 silence level), as Silero is on a weak syllable."""

    def __call__(self, window: np.ndarray) -> float:
        peak = float(np.abs(window).max())
        return 0.9 if peak > 0.1 else 0.4 if peak > 0.02 else 0.0


def slicing_barge(
    vad_settings: VadSettings,
    transcribe: Callable[[np.ndarray], str],
    vad: FakeVad | None = None,
    **kwargs,
) -> BargeIn:
    """A barge-in listener whose slices hold 0.6 s: a 1.6 s phrase comes in three pieces."""
    return BargeIn(
        Endpointer(vad or FakeVad(), vad_settings.model_copy(update={"max_utterance_s": 0.6})),
        transcribe,
        **kwargs,
    )


async def say_over_the_reply(
    assistant: Assistant,
    make_frames,
    speaker: LongSpeaker,
    speech_ms: int = 0,
    over: list[np.ndarray] | None = None,
    on_frame: Callable[[int], None] | None = None,
) -> None:
    """A request, then `speech_ms` of speech (or the frames `over`) while CLAUDIO answers it,
    then silence. `on_frame(n)` is called before the n-th frame said over the reply."""
    clock = assistant.clock
    assert isinstance(clock, AudioClock)
    said = over if over is not None else make_frames("speech", speech_ms)

    async def gen() -> AsyncIterator[np.ndarray]:
        for f in make_frames("silence", 240) + make_frames("speech", 640):
            clock.t += 0.08
            yield f
            await asyncio.sleep(0)
        for _ in range(200):
            if assistant.state is State.SPEAKING:
                break
            clock.t += 0.08
            yield SILENCE
            await asyncio.sleep(0.001)
        for n, f in enumerate(said + make_frames("silence", 800)):
            if on_frame is not None:
                on_frame(n)
            clock.t += 0.08
            yield f
            await asyncio.sleep(0.002)
        for _ in range(80):
            clock.t += 0.08
            yield SILENCE
            await asyncio.sleep(0.005)

    await assistant.run(gen())


# --- 2.2: the phrase said over the reply, whole -------------------------------------------


async def test_a_phrase_longer_than_a_slice_is_judged_whole(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """30/10: "Apri la chat." said over the reply, transcribed slice by slice, came out
    "apri la ciotola" and "Grazie a te" and was dropped. The slices are only checked for
    "basta"; the whole phrase decides."""
    heard: list[float] = []

    def transcribe(audio: np.ndarray) -> str:
        heard.append(len(audio) / SR)
        return "Claudio, apri Chrome." if len(audio) > 1.2 * SR else "apri la"

    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "Claudio, apri Chrome."],
        window_s=0.0,
        speaker=speaker,
        barge_in=slicing_barge(vad_settings, transcribe),
    )
    await say_over_the_reply(assistant, make_frames, speaker, speech_ms=1600)
    assert speaker.cut  # taken over by the whole phrase, never by a slice
    assert max(heard) > 1.5 and sum(h > 1.2 for h in heard) == 1
    assert assistant.history[0]["barge_in"] == "Claudio, apri Chrome."
    assert "Apro Chrome." in speaker.spoken  # then "Dicevo: ..." resumes the cut reply


async def test_basta_in_a_slice_stops_the_reply_at_once(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    def transcribe(audio: np.ndarray) -> str:
        return "Basta." if len(audio) <= 1.2 * SR else "basta basta, ho detto basta"

    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        window_s=0.0,
        speaker=speaker,
        barge_in=slicing_barge(vad_settings, transcribe),
    )
    await say_over_the_reply(assistant, make_frames, speaker, speech_ms=1600)
    assert speaker.cut and assistant.history[0]["outcome"] == "interrupted"
    assert assistant.history[0]["barge_in"] == "Basta."


async def test_what_is_not_for_claudio_is_kept_apart(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """With its text in the log: what was said over the reply and left alone (JenJo 30/10:
    keep it, with a cap, only on the PC)."""
    saved: list[tuple[str, float]] = []
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        window_s=0.0,
        speaker=speaker,
        barge_in=slicing_barge(
            vad_settings, lambda audio: "il gatto dorme sul divano", needs_name=True
        ),
    )
    assistant.rejected_sink = lambda audio, kind: saved.append((kind, len(audio) / SR)) or "x.wav"
    await say_over_the_reply(assistant, make_frames, speaker, speech_ms=480)
    assert not speaker.cut
    assert [kind for kind, _ in saved] == ["barge_ignored"] and saved[0][1] > 0.4


async def test_a_dip_of_the_voice_after_a_slice_does_not_split_the_phrase(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """Review of 31/10: after a slice was cut, the next frames had to show clear speech at
    once, so a weak syllable ended the phrase there and split it again. Only silence ends it
    now."""
    heard: list[float] = []

    def transcribe(audio: np.ndarray) -> str:
        heard.append(len(audio) / SR)
        return "Claudio, apri Chrome." if len(audio) > 1.2 * SR else "apri la"

    soft = [tone(1500, 220.0)] * 15  # ~0.05 peak, 1.2 s: weak syllables over any cut
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "Claudio, apri Chrome."],
        window_s=0.0,
        speaker=speaker,
        barge_in=slicing_barge(vad_settings, transcribe, vad=DipVad()),
    )
    over = make_frames("speech", 800) + soft + make_frames("speech", 640)
    await say_over_the_reply(assistant, make_frames, speaker, over=over)
    whole = [h for h in heard if h > 1.2]
    assert len(whole) == 1 and whole[0] > 1.6, heard  # one phrase, the soft part inside
    assert speaker.cut and assistant.history[0]["barge_in"] == "Claudio, apri Chrome."


async def test_a_stop_word_inside_a_request_does_not_stop_the_reply_alone(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """A slice is checked for "basta" alone: "apri Chrome e basta" is not a stop, the whole
    phrase decides (and it was a request)."""

    def transcribe(audio: np.ndarray) -> str:
        if len(audio) > 1.2 * SR:
            return "Claudio, apri Chrome e basta."
        return "apri Chrome e basta"

    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "Claudio, apri Chrome e basta."],
        window_s=0.0,
        speaker=speaker,
        barge_in=slicing_barge(vad_settings, transcribe),
    )
    await say_over_the_reply(assistant, make_frames, speaker, speech_ms=1600)
    assert assistant.history[0]["barge_in"] == "Claudio, apri Chrome e basta."
    assert assistant.history[1]["wake_word"] == "barge_in"  # taken over by the request


class CueSpeaker(LongSpeaker):
    """A first reply that ends when the test says so (`done`), whatever was said over it."""

    def __init__(self) -> None:
        super().__init__()
        self.done = asyncio.Event()

    async def speak(self, text_stream, on_first_audio=None, language="it") -> None:
        text = "".join([c async for c in text_stream])
        if on_first_audio:
            on_first_audio()
        if not self.spoken:
            stop = asyncio.ensure_future(self._stop.wait())
            cue = asyncio.ensure_future(self.done.wait())
            await asyncio.wait({stop, cue}, timeout=3.0, return_when=asyncio.FIRST_COMPLETED)
            stop.cancel()
            cue.cancel()
        self.spoken.append(text)


async def test_a_phrase_still_going_when_the_reply_ends_is_not_lost(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """Review of 31/10: the reply ended 1 s into "Claudio, apri Chrome": its first slice had
    only been checked for "basta" and was never judged. Now it is, at the reply's end, and
    the request goes on without a beep."""

    def transcribe(audio: np.ndarray) -> str:
        return "Claudio, apri Chrome." if len(audio) > 0.9 * SR else "Claudio, apri"

    speaker = CueSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", "Claudio, apri Chrome."],
        window_s=0.0,
        speaker=speaker,
        barge_in=slicing_barge(vad_settings, transcribe),
    )

    def end_the_reply(n: int) -> None:
        if n == 13:  # ~1 s into the 1.6 s phrase: one slice cut, the next one going on
            speaker.done.set()

    await say_over_the_reply(
        assistant, make_frames, speaker, speech_ms=1600, on_frame=end_the_reply
    )
    assert speaker.spoken[1] == "Apro Chrome."
    assert assistant.history[1]["wake_word"] == "barge_in"
    assert speaker.beeps == ["wake"]  # no beep for the request that went on


async def test_a_phrase_still_going_over_a_question_answers_it(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """The reply ended on "Confermi?" with a phrase still being said over it: the phrase goes
    on without the beep of a window for the answer, and the action stays armed for it."""
    speaker = CueSpeaker()
    assistant, _, rec, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["chiudi la finestra", "Claudio, sì."],
        window_s=0.0,
        speaker=speaker,
        barge_in=slicing_barge(vad_settings, lambda audio: "Claudio, sì."),
    )

    def end_the_reply(n: int) -> None:
        if n == 13:
            speaker.done.set()

    await say_over_the_reply(
        assistant, make_frames, speaker, speech_ms=1600, on_frame=end_the_reply
    )
    assert speaker.spoken[0].endswith("Confermi?") and speaker.beeps == ["wake"]
    assert assistant.history[0]["barge_in"] == "Claudio, sì."  # judged before its row is written
    assert assistant.history[1]["wake_word"] == "barge_in"
    assert assistant.history[1]["route"] == "confirmed" and rec.combos == [(VK_MENU, VK_F4)]


# --- 2.3: the barge-in threshold, before the gain -------------------------------------------


@pytest.mark.parametrize(("gain", "transcribed"), [(12.0, False), (0.0, True)])
async def test_the_barge_in_threshold_is_measured_before_the_gain(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher, gain: float, transcribed: bool
) -> None:
    """The speech tone peaks at -9 dBFS after the gain. With +12 dB of AutoGain it was -21
    at the microphone: under a -15 threshold, so not a voice to stop the reply for (the old
    rule compared -9 and let the echo through as the gain went up)."""
    heard: list[int] = []
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        window_s=0.0,
        speaker=speaker,
        barge_in=slicing_barge(
            vad_settings, lambda audio: heard.append(1) or "il gatto dorme", min_dbfs=-15.0
        ),
    )
    assistant.input_gain_db = gain
    await say_over_the_reply(assistant, make_frames, speaker, speech_ms=480)
    assert bool(heard) is transcribed


# --- 2.4: the window's deaf start -----------------------------------------------------------


def window_assistant(fake_vad, vad_settings, matcher, stt: EchoStt, **kwargs):
    return assistant_for(
        fake_vad, vad_settings, matcher, stt, FakeWake({3}), conversation_s=2.0, **kwargs
    )


def listening_again(assistant: Assistant, speaker) -> Callable[[], bool]:
    # own_sound.ended() runs in the beep task's done callback, a loop turn after the beep
    return lambda: (
        assistant.state is State.LISTENING
        and len(speaker.beeps) >= 2
        and not assistant.own_sound.playing
    )


async def test_a_phrase_begun_while_the_window_was_deaf_keeps_its_start(
    fake_vad, vad_settings, matcher: CommandMatcher
) -> None:
    """JenJo starts "apri Chrome" while the window's beep is still in the room: the frames the
    window did not listen to are the start of the phrase, not lost."""
    stt = EchoStt(["che ore sono", "apri chrome"], echo="")
    meter: list[int] = []
    assistant, speaker, _ = window_assistant(
        fake_vad, vad_settings, matcher, stt, voice_meter=lambda audio: meter.append(1) or 480
    )

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(SILENCE, 3) + several(VOICE, 10):
            yield f
        async for f in until(assistant, listening_again(assistant, speaker)):
            yield f
        for f in several(VOICE, 12) + several(SILENCE, 30):  # 7 of them while still deaf
            yield f

    await play(assistant, steps())
    assert stt.heard == ["che ore sono", "apri chrome"]
    assert stt.samples[1] >= 12 * FRAME  # all of it, the deaf start too
    row = assistant.history[1]
    assert row["deaf_joined_ms"] >= 400 and row["deaf_speech_ms"] == 480 and meter


async def test_the_echo_in_the_deaf_start_never_joins_the_phrase(
    fake_vad, vad_settings, matcher: CommandMatcher
) -> None:
    """The reply's end coming back from the speakers is quieter than any voice at the
    microphone: it stays out, even with JenJo talking right after it."""
    stt = EchoStt(["che ore sono", "apri chrome"], echo="")
    assistant, speaker, _ = window_assistant(fake_vad, vad_settings, matcher, stt)
    faint_echo = tone(150, 440.0)  # about -50 dBFS

    async def steps() -> AsyncIterator[np.ndarray]:
        for f in several(SILENCE, 3) + several(VOICE, 10):
            yield f
        async for f in until(assistant, listening_again(assistant, speaker)):
            yield f
        for f in several(faint_echo, 7) + several(VOICE, 6) + several(SILENCE, 30):
            yield f

    await play(assistant, steps())
    assert stt.heard == ["che ore sono", "apri chrome"]
    assert "deaf_joined_ms" not in assistant.history[1]
    assert stt.samples[1] < 7 * FRAME + 6 * FRAME
