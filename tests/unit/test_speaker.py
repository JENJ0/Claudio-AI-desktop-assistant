import asyncio
import threading
import time

import numpy as np
import pytest

from claudio.audio.earcons import make_earcons
from claudio.audio.player import resample
from claudio.tts.speaker import StreamingSpeaker


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
    assert set(tones) == {"wake", "cancel", "error"}
    assert all(t.dtype == np.int16 and np.abs(t).max() <= 0.25 * 32767 + 1 for t in tones.values())
    audio = np.arange(1000, dtype=np.int16)
    assert len(resample(audio, 16000, 22050)) == 1378
    assert resample(audio, 22050, 22050) is audio


async def test_effect_colours_every_sentence() -> None:
    seen: list[int] = []

    def effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        seen.append(sample_rate)
        return audio + 1000

    player = FakePlayer()
    speaker = StreamingSpeaker(FakeSynth(), player, effect=effect)
    await speaker.speak(stream("Uno. ", "Due."))
    assert [n for n, _ in player.played] == [1001, 1002]
    assert seen == [22050, 22050]


async def test_each_language_speaks_with_its_voice() -> None:
    class Voice(FakeSynth):
        def __init__(self, rate: int) -> None:
            super().__init__()
            self.sample_rate = rate

    class RatePlayer(FakePlayer):
        def write(self, audio: np.ndarray, sample_rate: int | None = None, cancel=None) -> None:
            self.played.append((sample_rate or 0, 0.0))

    italian, english, player = Voice(22050), Voice(16000), RatePlayer()
    speaker = StreamingSpeaker(italian, player, voices={"en": english})
    language = iter(["en", "en", "it"])
    await speaker.speak(stream("Opening. Chrome."), language=lambda: next(language))
    await speaker.speak(stream("Apro."), language="it")
    await speaker.speak(stream("Hallo."), language="de")  # no German voice: the main one
    assert [t for t, _ in english.calls] == ["Opening.", "Chrome."]
    assert [t for t, _ in italian.calls] == ["Apro.", "Hallo."]
    assert [rate for rate, _ in player.played] == [16000, 16000, 22050, 22050]
