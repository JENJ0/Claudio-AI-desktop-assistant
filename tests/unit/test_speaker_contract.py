"""What every Speaker does, checked on the fake the tests use and on the real one, so the
fake cannot hide a bug (G0 task 6)."""

import asyncio
import threading
from collections.abc import AsyncIterator

import pytest

from claudio.protocols import Speaker
from claudio.tts.speaker import StreamingSpeaker
from tests.unit.test_app import FakeSpeaker
from tests.unit.test_speaker import FakePlayer, FakeSynth
from tests.unit.timeline import Ticks, TimedSpeaker

TIMEOUT_S = 1.0


@pytest.fixture(params=["fake", "timed", "streaming"])
def speaker(request: pytest.FixtureRequest) -> Speaker:
    if request.param == "fake":
        return FakeSpeaker()
    if request.param == "timed":  # sentences that last, on frames that pass by themselves
        return TimedSpeaker(Ticks(free_running=True), start_frames=2, sentence_frames=3)
    return StreamingSpeaker(FakeSynth(), FakePlayer())


async def run(speaker: Speaker, *args: object, **kwargs: object) -> None:
    await asyncio.wait_for(speaker.speak(*args, **kwargs), TIMEOUT_S)  # type: ignore[arg-type]


async def text(*chunks: str) -> AsyncIterator[str]:
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


async def test_first_audio_does_not_wait_for_the_rest_of_the_text(speaker: Speaker) -> None:
    started = asyncio.Event()

    async def stream() -> AsyncIterator[str]:
        yield "Uno. "
        await started.wait()  # a speaker that wants all the text first never gets past here
        yield "Due."

    await run(speaker, stream(), on_first_audio=started.set)
    assert speaker.played == ["Uno.", "Due."]


async def test_first_audio_fires_once(speaker: Speaker) -> None:
    fired: list[int] = []
    await run(speaker, text("Uno. Due. ", "Tre."), on_first_audio=lambda: fired.append(1))
    assert fired == [1]


async def test_no_text_no_first_audio(speaker: Speaker) -> None:
    fired: list[int] = []
    await run(speaker, text(""), on_first_audio=lambda: fired.append(1))
    assert fired == []
    assert speaker.played == []


async def test_played_lists_the_sentences_that_started(speaker: Speaker) -> None:
    await run(speaker, text("Uno. Due", ". Tre."))
    assert speaker.played == ["Uno.", "Due.", "Tre."]


async def test_interrupt_stops_the_reply(speaker: Speaker) -> None:
    await run(speaker, text("Uno. ", "Due."), on_first_audio=speaker.interrupt)
    assert speaker.played == ["Uno."]


async def stuck_after_first_sentence(
    speaker: Speaker,
) -> tuple["asyncio.Task[None]", asyncio.Event]:
    started = asyncio.Event()
    never = asyncio.Event()

    async def stream() -> AsyncIterator[str]:
        yield "Uno. "
        await never.wait()
        yield "Due."

    task = asyncio.create_task(speaker.speak(stream(), on_first_audio=started.set))
    await asyncio.wait_for(started.wait(), TIMEOUT_S)
    return task, started


async def test_interrupt_while_waiting_for_text_returns_at_once(speaker: Speaker) -> None:
    task, _ = await stuck_after_first_sentence(speaker)
    speaker.interrupt()
    await asyncio.wait_for(task, TIMEOUT_S)
    assert speaker.played == ["Uno."]


async def test_interrupt_from_another_thread_returns_at_once(speaker: Speaker) -> None:
    task, _ = await stuck_after_first_sentence(speaker)
    threading.Timer(0.05, speaker.interrupt).start()  # after the loop blocks in select
    await asyncio.wait_for(task, TIMEOUT_S)
    assert speaker.played == ["Uno."]


async def test_cancelled_while_waiting_for_text_leaves_nothing_running(speaker: Speaker) -> None:
    task, _ = await stuck_after_first_sentence(speaker)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, TIMEOUT_S)
    await asyncio.sleep(0)
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_a_reply_after_an_interrupt_starts_clean(speaker: Speaker) -> None:
    await run(speaker, text("Uno. ", "Due."), on_first_audio=speaker.interrupt)
    await run(speaker, text("Tre. Quattro."))
    assert speaker.played == ["Tre.", "Quattro."]


async def test_first_audio_waits_for_a_whole_sentence(speaker: Speaker) -> None:
    complete: list[int] = []
    fired_when: list[int] = []

    async def stream() -> AsyncIterator[str]:
        yield "Uno"
        await asyncio.sleep(0)
        complete.append(1)  # from here ". " ends the sentence
        yield ". Due."

    await run(speaker, stream(), on_first_audio=lambda: fired_when.append(len(complete)))
    assert fired_when == [1]


async def test_first_sound_delay_is_readable_when_first_audio_fires(speaker: Speaker) -> None:
    delays: list[float] = []
    await run(
        speaker,
        text("Uno."),
        on_first_audio=lambda: delays.append(speaker.first_sound_delay_s),
    )
    assert len(delays) == 1
    assert isinstance(delays[0], float) and delays[0] >= 0.0


async def test_an_error_of_the_stream_reaches_the_caller_after_the_sentences_that_started(
    speaker: Speaker,
) -> None:
    async def stream() -> AsyncIterator[str]:
        yield "Uno. "
        raise RuntimeError("la rete è caduta")

    with pytest.raises(RuntimeError, match="la rete è caduta"):
        await run(speaker, stream())
    assert speaker.played == ["Uno."]
    await run(speaker, text("Due."))  # the speaker is usable again
    assert speaker.played == ["Due."]


async def test_a_stream_cancelled_without_an_await_does_not_hang_the_speaker(
    speaker: Speaker,
) -> None:
    async def stream() -> AsyncIterator[str]:
        yield "Uno. "
        raise asyncio.CancelledError  # the producer ends cancelled, nothing was awaited

    await run(speaker, stream())
    assert speaker.played == ["Uno."]
    await asyncio.sleep(0)
    assert asyncio.all_tasks() == {asyncio.current_task()}
