import asyncio
import threading
import time

import numpy as np
import pytest

from claudio.audio.earcons import make_earcons
from claudio.audio.player import resample
from claudio.protocols import CUT_SHORT
from claudio.tts.speaker import StreamingSpeaker, leading_silence_s


class FakeSynth:
    sample_rate = 22050

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, float]] = []
        self.fail_on = fail_on

    def synthesize(self, text: str) -> np.ndarray:
        if text == self.fail_on:
            raise RuntimeError("piper")
        self.calls.append((text, time.perf_counter()))
        time.sleep(0.02)
        return np.full(100, len(self.calls), dtype=np.int16)


class FakePlayer:
    sample_rate = 22050

    def __init__(self) -> None:
        self.played: list[tuple[int, float]] = []
        self.lock = threading.Lock()

    def write(self, audio: np.ndarray, sample_rate: int | None = None, cancel=None) -> None:
        with self.lock:
            self.played.append((int(audio[0]), time.perf_counter()))
        time.sleep(0.05)

    def starts_in_s(self) -> float:
        return 0.0


async def stream(*chunks: str):
    for c in chunks:
        await asyncio.sleep(0)
        yield c


async def test_speaks_sentences_in_order_and_overlaps_synthesis() -> None:
    synth, player = FakeSynth(), FakePlayer()
    speaker = StreamingSpeaker(synth, player)
    firsts: list[float] = []
    await speaker.speak(stream("Uno. Due", ". Tre."), on_first_audio=lambda: firsts.append(1))
    assert [t for t, _ in synth.calls] == ["Uno.", "Due.", "Tre."]
    assert [n for n, _ in player.played] == [1, 2, 3]
    assert firsts == [1]
    # sentence 2 was synthesised while sentence 1 was still playing
    assert synth.calls[1][1] < player.played[0][1] + 0.05


async def test_empty_stream_plays_nothing() -> None:
    player = FakePlayer()
    await StreamingSpeaker(FakeSynth(), player).speak(stream(""))
    assert player.played == []


async def test_synth_error_propagates() -> None:
    speaker = StreamingSpeaker(FakeSynth(fail_on="Due."), FakePlayer())
    with pytest.raises(RuntimeError, match="piper"):
        await speaker.speak(stream("Uno. Due. Tre."))


async def test_beep_uses_player() -> None:
    player = FakePlayer()
    await StreamingSpeaker(FakeSynth(), player).beep("wake")
    assert len(player.played) == 1


def test_earcons_and_resample() -> None:
    tones = make_earcons(22050, 0.25)
    assert set(tones) == {"wake", "cancel", "error", "tick"}
    assert len(tones["tick"]) < len(tones["wake"]) / 2  # the tick while off: short and soft
    assert np.abs(tones["tick"]).max() < np.abs(tones["wake"]).max() / 2
    assert all(t.dtype == np.int16 and np.abs(t).max() <= 0.25 * 32767 + 1 for t in tones.values())
    audio = np.arange(1000, dtype=np.int16)
    assert len(resample(audio, 16000, 22050)) == 1378
    assert resample(audio, 22050, 22050) is audio


class StreamingSynth:
    """Like Jack: stream() yields the audio of a sentence in pieces."""

    sample_rate = 24000

    def __init__(self) -> None:
        self.streamed: list[str] = []
        self.closed = 0

    def synthesize(self, text: str) -> np.ndarray:
        raise AssertionError("a streaming synth is played through stream()")

    def stream(self, text: str):
        self.streamed.append(text)
        try:
            for k in (1, 2, 3):
                yield np.full(100, k, dtype=np.int16)
        finally:
            self.closed += 1


def test_a_streaming_voice_is_played_piece_by_piece() -> None:
    synth, player = StreamingSynth(), FakePlayer()
    speaker = StreamingSpeaker(synth, player)

    async def text():
        yield "Prima frase. Seconda frase."

    asyncio.run(speaker.speak(text()))
    assert synth.streamed == ["Prima frase.", "Seconda frase."]
    assert [first for first, _ in player.played] == [1, 2, 3, 1, 2, 3]
    assert synth.closed == 2


def test_played_lists_each_sentence_once_as_it_starts() -> None:
    """Where a reply cut by a new request resumes: the sentences that started playing."""
    synth, player = StreamingSynth(), FakePlayer()
    speaker = StreamingSpeaker(synth, player)

    async def text():
        yield "Prima frase. Seconda frase."

    asyncio.run(speaker.speak(text()))
    assert speaker.played == ["Prima frase.", "Seconda frase."]


async def test_played_stops_where_the_reply_was_cut() -> None:
    player = FakePlayer()
    speaker = StreamingSpeaker(FakeSynth(), player)
    original = player.write

    def write(audio, sample_rate=None, cancel=None):
        original(audio, sample_rate, cancel)
        speaker.interrupt()  # cut during the first sentence

    player.write = write
    await speaker.speak(stream("Uno. Due. Tre."))
    assert speaker.played == ["Uno."]


@pytest.mark.parametrize(
    ("group_chars", "expected"),
    [
        (0, ["C'era una volta un re.", "Mi fermo qui: se vuoi, continuo."]),
        (220, ["C'era una volta un re. Mi fermo qui: se vuoi, continuo."]),
    ],
)
async def test_the_fragment_of_a_reply_cut_at_the_token_limit_is_never_said(
    group_chars: int, expected: list[str]
) -> None:
    """30/10: a story stopped at the token limit inside "Attraverso" and Jack said "Attra"."""
    synth = FakeSynth()
    speaker = StreamingSpeaker(  # the [tts] defaults: a first clause, or Jack's grouping
        synth, FakePlayer(), first_clause_chars=25, group_chars=group_chars, group_first_chars=60
    )
    await speaker.speak(
        stream("C'era una volta un re. Attra", CUT_SHORT, " Mi fermo qui: se vuoi, continuo.")
    )
    assert [t for t, _ in synth.calls] == expected
    assert speaker.played == expected


async def test_grouped_reply_is_said_in_pieces_up_to_the_limit() -> None:
    synth, player = FakeSynth(), FakePlayer()
    speaker = StreamingSpeaker(synth, player, group_chars=20, group_first_chars=8)
    await speaker.speak(stream("Uno. Due", ". Tre lunghissimo davvero. Quattro."))
    assert [t for t, _ in synth.calls] == ["Uno. Due.", "Tre lunghissimo davvero.", "Quattro."]
    assert speaker.played == ["Uno. Due.", "Tre lunghissimo davvero.", "Quattro."]


async def test_only_a_change_of_volume_is_logged(caplog) -> None:
    """Nobody could tell when ducking fired. Back to full volume is logged when the reply
    was lowered, not at every duck(None): every reply starts with one."""
    player = FakePlayer()
    speaker = StreamingSpeaker(FakeSynth(), player)
    with caplog.at_level("INFO", logger="claudio.tts.speaker"):
        await speaker.speak(stream("Uno."))  # already at full volume: nothing to say
        speaker.duck(0.3)
        speaker.duck(0.3)  # a second voice while it is lowered
        assert player.gain == 0.3
        speaker.duck(None)  # "not for me"
        speaker.duck(None)
        speaker.duck(0.3)  # the reply ends while a voice is being checked...
        await speaker.speak(stream("Due."))  # ...and the next one starts at full volume
    back = [r for r in caplog.records if r.getMessage() == "Risposta di nuovo a volume pieno"]
    assert len(back) == 2 and player.gain == 1.0


async def test_basta_while_claude_runs_a_tool_stops_at_once() -> None:
    """Review of IT-1 5.1: the first piece now plays while the tools run, so "basta" can
    come while speak() waits for the next piece: it must not wait for the tools, nor count
    the next piece as played."""
    synth, player = FakeSynth(), FakePlayer()
    speaker = StreamingSpeaker(synth, player, group_chars=220, group_first_chars=40)
    speaker.group_pause_s = 0.05

    async def text():
        yield "Ci guardo, socio. "
        await asyncio.sleep(2)  # the tools and the next call
        yield "Trovata."

    loop = asyncio.get_running_loop()
    loop.call_later(0.3, speaker.interrupt)
    start = loop.time()
    await speaker.speak(text())
    assert loop.time() - start < 1.0
    assert speaker.played == ["Ci guardo, socio."]
    assert [t for t, _ in synth.calls] == ["Ci guardo, socio."]


async def test_basta_from_another_thread_also_wakes_it() -> None:
    synth, player = FakeSynth(), FakePlayer()
    speaker = StreamingSpeaker(synth, player)

    async def text():
        yield "Uno. "
        await asyncio.sleep(2)
        yield "Due."

    threading.Timer(0.3, speaker.interrupt).start()
    start = time.perf_counter()
    await speaker.speak(text())
    assert time.perf_counter() - start < 1.0 and speaker.played == ["Uno."]


class Voice(FakeSynth):
    """Like Jack: told when a reply begins and when it is stopped."""

    def __init__(self) -> None:
        super().__init__()
        self.began_after: list[int] = []  # the sentences it had said at each new reply
        self.cancelled = 0

    def begin_reply(self) -> None:
        self.began_after.append(len(self.calls))

    def cancel(self) -> None:
        self.cancelled += 1


async def test_the_voice_is_told_a_reply_begins_before_it_speaks() -> None:
    main = Voice()
    speaker = StreamingSpeaker(main, FakePlayer())
    await speaker.speak(stream("Uno."))
    await speaker.speak(stream("Due."))
    assert main.began_after == [0, 1]


def test_interrupt_between_two_replies_stops_the_voice_and_not_the_next_reply() -> None:
    main, player = Voice(), FakePlayer()
    speaker = StreamingSpeaker(main, player)
    asyncio.run(speaker.speak(stream("Uno.")))
    speaker.interrupt()  # its loop is closed: nothing is left to wake
    assert main.cancelled == 1
    asyncio.run(speaker.speak(stream("Due.")))
    assert speaker.played == ["Due."] and len(player.played) == 2


async def test_a_player_error_reaches_the_caller_ahead_of_a_voice_error() -> None:
    class BrokenPlayer(FakePlayer):
        def write(self, audio: np.ndarray, sample_rate: int | None = None, cancel=None) -> None:
            time.sleep(0.1)  # the voice has already failed on the next sentence
            raise RuntimeError("player")

    speaker = StreamingSpeaker(FakeSynth(fail_on="Due."), BrokenPlayer())
    with pytest.raises(RuntimeError, match="player"):
        await speaker.speak(stream("Uno. Due. Tre."))
    assert speaker.played == ["Uno."]  # counted as it started


async def test_cancelling_a_reply_stops_the_text_and_leaves_nothing_running() -> None:
    ended: list[str] = []

    async def text():
        try:
            yield "Uno. "
            await asyncio.sleep(30)
            yield "Due."
        finally:
            ended.append("text")

    first = asyncio.Event()
    speaker = StreamingSpeaker(FakeSynth(), FakePlayer())
    reply = asyncio.ensure_future(speaker.speak(text(), on_first_audio=first.set))
    await asyncio.wait_for(first.wait(), 2)
    reply.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(reply, 2)
    assert reply.cancelled() and ended == ["text"]
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_basta_while_a_voice_is_busy_in_its_thread_raises_nothing() -> None:
    """Its stream cannot be closed while a thread runs in it (ValueError): it ends by itself."""
    release = threading.Event()

    class BusySynth(StreamingSynth):
        def stream(self, text: str):
            yield np.full(100, 1, dtype=np.int16)
            release.wait(10)
            yield np.full(100, 2, dtype=np.int16)

    player = FakePlayer()
    speaker = StreamingSpeaker(BusySynth(), player)
    asyncio.get_running_loop().call_later(0.2, speaker.interrupt)
    try:
        await asyncio.wait_for(speaker.speak(stream("Uno.")), 2)
    finally:
        release.set()
    assert speaker.played == ["Uno."] and [first for first, _ in player.played] == [1]


async def test_basta_with_the_queue_full_stops_the_reply() -> None:
    """A long reply rendered ahead of the player: no room for the wake-up, none is needed."""

    class WaitingPlayer(FakePlayer):
        def write(self, audio: np.ndarray, sample_rate: int | None = None, cancel=None) -> bool:
            self.played.append((int(audio[0]), 0.0))
            cancel.wait(10)
            return False

    async def rendered_ahead() -> None:
        while len(synth.calls) < 18:  # 1 in the player, 16 queued, 1 waiting for room
            await asyncio.sleep(0.05)

    synth, player = FakeSynth(), WaitingPlayer()
    speaker = StreamingSpeaker(synth, player)
    loop = asyncio.get_running_loop()
    errors: list[dict] = []
    loop.set_exception_handler(lambda _, context: errors.append(context))
    reply = asyncio.ensure_future(speaker.speak(stream("Frase. " * 20 + "Fine.")))
    try:
        await asyncio.wait_for(rendered_ahead(), 5)
        speaker.interrupt()
        await asyncio.wait_for(reply, 2)
    finally:
        loop.set_exception_handler(None)
    assert speaker.played == ["Frase."] and len(player.played) == 1
    assert errors == []


async def test_basta_as_soon_as_the_text_starts_also_wakes_it() -> None:
    """speak() can be woken before the text is first read: no gap for a "basta" to fall into."""
    speaker = StreamingSpeaker(FakeSynth(), FakePlayer())

    async def text():
        speaker.interrupt()
        await asyncio.sleep(2)
        yield "Uno."

    loop = asyncio.get_running_loop()
    start = loop.time()
    await speaker.speak(text())
    assert loop.time() - start < 1.0 and speaker.played == []


def test_leading_silence_finds_the_first_audible_sample() -> None:
    audio = np.concatenate([np.zeros(800, np.int16), np.array([1000], np.int16)])
    assert leading_silence_s(audio, 16000) == pytest.approx(0.05)
    assert leading_silence_s(np.zeros(1600, np.int16), 16000) == pytest.approx(0.1)
    assert leading_silence_s(np.array([1000, 0], np.int16), 16000) == 0.0


class QueuedPlayer(FakePlayer):
    def starts_in_s(self) -> float:
        return 0.25  # what is queued plus the device


async def test_first_sound_delay_is_ready_when_first_audio_fires() -> None:
    class QuietStart(FakeSynth):
        def synthesize(self, text: str) -> np.ndarray:
            return np.concatenate([np.zeros(2205, np.int16), np.full(100, 1000, np.int16)])

    speaker = StreamingSpeaker(QuietStart(), QueuedPlayer())
    seen: list[float] = []
    await speaker.speak(
        stream("Uno."), on_first_audio=lambda: seen.append(speaker.first_sound_delay_s)
    )
    assert seen == [pytest.approx(0.35)]  # 0.25 s ahead, then 0.1 s of silence
    await speaker.speak(stream(""))
    assert speaker.first_sound_delay_s == 0.0  # a new reply starts from zero
