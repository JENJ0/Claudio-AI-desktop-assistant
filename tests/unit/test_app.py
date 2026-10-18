"""State machine with fake components: normal path, timeout, STT error, unknown command."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest

from claudio.app import Assistant, State
from claudio.audio.vad import Endpointer
from claudio.metrics import MetricsWriter


class FakeWake:
    def __init__(self, fire_on: set[int], miss_on: set[int] | None = None) -> None:
        self.fire_on, self.miss_on, self.calls, self.resets = fire_on, miss_on or set(), 0, 0
        self.assistant: Assistant | None = None
        self.states: list[State] = []

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        self.calls += 1
        if self.assistant is not None:
            self.states.append(self.assistant.state)
        return ("hey_claudio", 0.9) if self.calls in self.fire_on else None

    def pop_near_miss(self) -> tuple[str, float] | None:
        return ("hey_claudio", 0.3) if self.calls in self.miss_on else None

    def reset(self) -> None:
        self.resets += 1


class FakeStt:
    def __init__(self, text: str = "apri chrome", error: bool = False, delay: float = 0) -> None:
        self.text, self.error, self.delay, self.audio = text, error, delay, None

    def transcribe(self, audio: np.ndarray) -> str:
        self.audio = audio
        if self.error:
            raise RuntimeError("CUDA esplosa")
        return self.text


class FakeResponder:
    def __init__(self) -> None:
        self.last_route: dict = {}
        self.seen: list[str] = []

    async def respond(self, text: str) -> AsyncIterator[str]:
        self.seen.append(text)
        if "chrome" in text:
            self.last_route = {"route": "local", "command": "open_chrome"}
            yield "Apro "
            yield "Chrome."
        else:
            self.last_route = {"route": "unknown"}
            yield "Questo non lo so ancora fare."


class FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.beeps: list[str] = []

    async def beep(self, kind: str = "wake") -> None:
        self.beeps.append(kind)

    async def speak(
        self, text_stream: AsyncIterator[str], on_first_audio: Callable[[], None] | None = None
    ) -> None:
        text = "".join([c async for c in text_stream])
        if on_first_audio:
            on_first_audio()
        self.spoken.append(text)


class AudioClock:
    """Advances 80 ms per frame, as a live microphone does."""

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def build(
    fake_vad, vad_settings, stt: FakeStt, wake_frames: set[int], tmp_path=None, audio_sink=None
):
    wake = FakeWake(wake_frames)
    speaker = FakeSpeaker()
    responder = FakeResponder()
    metrics = MetricsWriter(tmp_path / "metrics.jsonl") if tmp_path else None
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        stt,
        responder,
        speaker,
        metrics,
        clock=AudioClock(),
        audio_sink=audio_sink,
    )
    wake.assistant = assistant
    return assistant, wake, speaker, responder


async def run_frames(assistant: Assistant, frames: list[np.ndarray]) -> None:
    clock = assistant.clock
    assert isinstance(clock, AudioClock)

    async def gen():
        for f in frames:
            clock.t += 0.08
            yield f
            for _ in range(5):  # let the pipeline task progress, like real audio
                await asyncio.sleep(0)

    await assistant.run(gen())


async def test_normal_path(fake_vad, vad_settings, make_frames, tmp_path) -> None:
    stt = FakeStt()
    assistant, wake, speaker, responder = build(fake_vad, vad_settings, stt, {3}, tmp_path)
    frames = (
        make_frames("silence", 400)
        + make_frames("silence", 240)
        + make_frames("speech", 1000)
        + make_frames("silence", 800)
    )
    await run_frames(assistant, frames)

    assert speaker.beeps == ["wake"]
    assert speaker.spoken == ["Apro Chrome."]
    assert responder.seen == ["apri chrome"]
    assert stt.audio is not None and len(stt.audio) > 16000
    assert assistant.state is State.IDLE
    assert wake.resets == 1
    row = assistant.history[-1]
    assert row["outcome"] == "ok" and row["command"] == "open_chrome"
    assert row["wake_word"] == "hey_claudio"
    assert row["m1_ms"] is not None and row["m2_ms"] >= row["m1_ms"] >= 0
    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").count("\n") == 1


async def test_timeout_without_speech(fake_vad, vad_settings, make_frames) -> None:
    stt = FakeStt()
    assistant, _wake, speaker, _ = build(fake_vad, vad_settings, stt, {1})
    await run_frames(assistant, make_frames("silence", 3600))
    assert speaker.beeps == ["wake", "cancel"]
    assert stt.audio is None and speaker.spoken == []
    assert assistant.history[-1]["outcome"] == "timeout"
    assert assistant.state is State.IDLE


async def test_stt_error_plays_error_beep_and_recovers(fake_vad, vad_settings, make_frames) -> None:
    stt = FakeStt(error=True)
    assistant, _wake, speaker, _ = build(fake_vad, vad_settings, stt, {1, 40})
    utterance = make_frames("speech", 800) + make_frames("silence", 800)
    # first wake (frame 1) fails in STT; second wake later works again
    frames = make_frames("silence", 80) + utterance + make_frames("silence", 1600)
    await run_frames(assistant, frames)
    assert speaker.beeps[:2] == ["wake", "error"]
    assert assistant.history[0]["outcome"] == "error"
    assert assistant.state is State.IDLE


async def test_unknown_command_gets_fixed_reply(fake_vad, vad_settings, make_frames) -> None:
    stt = FakeStt(text="prenota un volo")
    assistant, _, speaker, _ = build(fake_vad, vad_settings, stt, {1})
    await run_frames(
        assistant,
        make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800),
    )
    assert speaker.spoken == ["Questo non lo so ancora fare."]
    assert assistant.history[-1]["route"] == "unknown"


async def test_empty_transcription_cancels(fake_vad, vad_settings, make_frames) -> None:
    stt = FakeStt(text="")
    assistant, _, speaker, _ = build(fake_vad, vad_settings, stt, {1})
    await run_frames(
        assistant,
        make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800),
    )
    assert speaker.spoken == [] and speaker.beeps == ["wake", "cancel"]
    assert assistant.history[-1]["outcome"] == "empty"


async def test_wake_word_ignored_while_busy(fake_vad, vad_settings, make_frames) -> None:
    stt = FakeStt()
    # wake fires on every frame it sees: frames must reach it only in IDLE
    assistant, wake, _speaker, _ = build(fake_vad, vad_settings, stt, set(range(1, 1000)))
    frames = make_frames("speech", 800) + make_frames("silence", 800)
    await run_frames(assistant, frames)
    assert wake.states and set(wake.states) == {State.IDLE}
    assert wake.calls < len(frames) / 4


async def test_heard_audio_is_saved_with_its_level(fake_vad, vad_settings, make_frames) -> None:
    saved: list[tuple[np.ndarray, str]] = []

    def sink(audio: np.ndarray, outcome: str) -> str:
        saved.append((audio, outcome))
        return f"{outcome}.wav"

    assistant, *_ = build(fake_vad, vad_settings, FakeStt(), {3, 25}, audio_sink=sink)
    frames = (
        make_frames("silence", 400)  # wake on frame 3
        + make_frames("speech", 1000)
        + make_frames("silence", 800)
        + make_frames("silence", 5000)  # second wake after 1.8 s, then nothing said
    )
    await run_frames(assistant, frames)

    ok, timeout = assistant.history
    assert (ok["outcome"], ok["audio_file"], ok["vad_max"]) == ("ok", "ok.wav", 0.9)
    assert ok["prewake_ms"] == 240 and ok["input_dbfs"] > -20
    assert timeout["audio_file"] == "timeout.wav" and timeout["vad_max"] == 0.0
    assert timeout["prewake_ms"] == 1500 and timeout["input_dbfs"] == -120.0
    audio, outcome = saved[0]
    assert outcome == "ok" and audio.dtype == np.float32
    assert len(audio) > 16000 * (0.24 + 1.0)  # 240 ms before the wake word + the phrase


async def test_failing_audio_sink_does_not_break_the_request(
    fake_vad, vad_settings, make_frames
) -> None:
    def sink(audio: np.ndarray, outcome: str) -> str:
        raise OSError("disk full")

    assistant, _, speaker, _ = build(fake_vad, vad_settings, FakeStt(), {1}, audio_sink=sink)
    await run_frames(
        assistant,
        make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800),
    )
    assert speaker.spoken == ["Apro Chrome."]
    assert assistant.history[-1]["outcome"] == "ok" and "audio_file" not in assistant.history[-1]
    assert assistant.state is State.IDLE


async def test_near_miss_is_logged_without_waking(fake_vad, vad_settings, make_frames, tmp_path):
    saved: list[str] = []

    def sink(audio: np.ndarray, outcome: str) -> str:
        saved.append(outcome)
        return f"{outcome}.wav"

    wake = FakeWake(set(), miss_on={20})
    speaker = FakeSpeaker()
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        FakeResponder(),
        speaker,
        MetricsWriter(tmp_path / "metrics.jsonl"),
        clock=AudioClock(),
        audio_sink=sink,
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert speaker.beeps == [] and assistant.state is State.IDLE and assistant.history == []
    assert saved == ["near_miss"]
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "near_miss" and row["wake_score"] == 0.3
    assert row["audio_file"] == "near_miss.wav" and row["m2_ms"] is None


async def test_quiet_microphone_is_reported_once(fake_vad, vad_settings, make_frames, caplog):
    assistant, *_ = build(fake_vad, vad_settings, FakeStt(), {1, 30})
    stats = assistant.endpointer.stats
    assistant.endpointer.stats = lambda: {**stats(), "speech_dbfs": -36.0}
    utterance = make_frames("speech", 800) + make_frames("silence", 800)
    with caplog.at_level("WARNING"):
        await run_frames(
            assistant,
            make_frames("silence", 80) + utterance + make_frames("silence", 1600) + utterance,
        )
    assert [r["outcome"] for r in assistant.history] == ["ok", "ok"]
    warnings = [r for r in caplog.records if "Microfono basso" in r.getMessage()]
    assert len(warnings) == 1 and "INPUT_GAIN_DB" in warnings[0].getMessage()


@pytest.mark.parametrize("state", list(State))
def test_states_are_strings(state: State) -> None:
    assert isinstance(state.value, str)
