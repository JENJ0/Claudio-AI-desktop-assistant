"""State machine with fake components: normal path, timeout, STT error, unknown command."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest

from claudio.app import Assistant, State
from claudio.audio.vad import Endpointer
from claudio.metrics import MetricsWriter
from claudio.system import SystemControl


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
    """Returns `text`, or the items of a list one call after another (the last one repeats)."""

    def __init__(
        self,
        text: str | list[str] = "apri chrome",
        error: bool = False,
        delay: float = 0,
        language: str = "it",
    ) -> None:
        self.texts = [text] if isinstance(text, str) else text
        self.error, self.delay, self.audio, self.language = error, delay, None, language
        self.heard: list[np.ndarray] = []

    def transcribe(self, audio: np.ndarray) -> str:
        self.audio = audio
        self.heard.append(audio)
        if self.delay:
            time.sleep(self.delay)  # runs in a worker thread, like Whisper
        if self.error:
            raise RuntimeError("CUDA esplosa")
        return self.texts[min(len(self.heard), len(self.texts)) - 1]


class FakeResponder:
    def __init__(self) -> None:
        self.last_route: dict = {}
        self.seen: list[str] = []
        self.language = "it"
        self.awaiting = False
        self.asked_question = False
        self.answer_expected = False
        self.prepared = 0

    @property
    def needs_voice(self) -> bool:
        return self.awaiting  # the fake's questions are all "Confermi?"

    def prepare(self) -> None:
        self.prepared += 1

    def understands(self, text: str, language: str = "it", memory: bool = True) -> bool:
        return True

    def is_stop(self, text: str) -> bool:
        return False

    def cancel_pending(self, silence: bool = False) -> None:
        pass

    async def respond(self, text: str, language: str = "it") -> AsyncIterator[str]:
        self.seen.append(text)
        self.language = "en" if "open" in text else language  # the language of the phrase
        if "chrome" in text:
            self.last_route = {"route": "local", "command": "open_chrome"}
            yield "Opening " if self.language == "en" else "Apro "
            yield "Chrome."
        else:
            self.last_route = {"route": "unknown"}
            yield (
                "I can't do that yet." if self.language == "en" else "Questo non lo so ancora fare."
            )


class FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.languages: list[str] = []
        self.beeps: list[str] = []
        self.ducks: list[float | None] = []
        self.played: list[str] = []
        self.queued = 0.0  # seconds still coming out of the speakers after a sound

    def queued_s(self) -> float:
        return self.queued

    def interrupt(self) -> None:
        pass

    def duck(self, volume: float | None) -> None:
        self.ducks.append(volume)

    async def beep(self, kind: str = "wake") -> None:
        self.beeps.append(kind)

    async def speak(
        self,
        text_stream: AsyncIterator[str],
        on_first_audio: Callable[[], None] | None = None,
        language: str | Callable[[], str] = "it",
    ) -> None:
        text = "".join([c async for c in text_stream])
        if on_first_audio:
            on_first_audio()
        self.spoken.append(text)
        self.languages.append(language() if callable(language) else language)


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
    assert responder.prepared == 1  # at the wake word: Claude's connection opens early
    assert stt.audio is not None and len(stt.audio) > 16000
    assert assistant.state is State.IDLE
    assert wake.resets >= 1  # at the end, and again once the beep it heard is over
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


async def test_a_metrics_file_held_by_another_program_does_not_wedge_the_assistant(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    # A directory where metrics.jsonl should be: opening it for append raises OSError, as a
    # file locked by another program does on Windows.
    (tmp_path / "metrics.jsonl").mkdir()
    assistant, _wake, _speaker, responder = build(
        fake_vad, vad_settings, FakeStt(), {1, 40}, tmp_path
    )
    utterance = make_frames("speech", 800) + make_frames("silence", 800)
    frames = make_frames("silence", 80) + utterance + make_frames("silence", 1600) + utterance
    await run_frames(assistant, frames + make_frames("silence", 800))
    assert assistant.state is State.IDLE
    assert responder.seen == ["apri chrome", "apri chrome"]  # the next request was heard
    assert len(assistant.history) == 2 and assistant.history[-1]["outcome"] == "ok"


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
    assert speaker.beeps == [] and assistant.state is State.IDLE and not assistant.history
    assert saved == ["near_miss"]
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "near_miss" and row["wake_score"] == 0.3
    assert row["audio_file"] == "near_miss.wav" and row["m2_ms"] is None


class SecondChance:
    """Stands for Whisper listening to the last 2.5 s after a near miss."""

    def __init__(
        self, text: str = "", error: bool = False, gate: threading.Event | None = None
    ) -> None:
        self.text, self.error, self.gate, self.heard = text, error, gate, []

    def __call__(self, audio: np.ndarray) -> str:
        self.heard.append(audio)
        if self.gate is not None:
            self.gate.wait(5)  # runs in a worker thread, like Whisper
        if self.error:
            raise RuntimeError("CUDA esplosa")
        return self.text


def with_second_chance(fake_vad, vad_settings, check: SecondChance, tmp_path):
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
        audio_sink=lambda audio, outcome: f"{outcome}.wav",
        second_chance=check,
    )
    return assistant, speaker


async def test_second_chance_wakes_when_whisper_hears_the_name(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    check = SecondChance("Hey Claudio.")
    assistant, speaker = with_second_chance(fake_vad, vad_settings, check, tmp_path)
    await run_frames(assistant, make_frames("silence", 2400))
    assert len(check.heard) == 1 and 0 < len(check.heard[0]) <= 40000  # the last 2.5 s
    assert speaker.beeps == ["wake"] and assistant.state is State.LISTENING
    info = assistant._interaction.info
    assert info["second_chance"] == "Hey Claudio." and info["wake_score"] == 0.3
    assert info["second_chance_ms"] >= 0
    await run_frames(assistant, make_frames("speech", 800) + make_frames("silence", 800))
    assert speaker.spoken == ["Apro Chrome."]
    assert assistant.history[-1]["outcome"] == "ok"
    assert assistant.history[-1]["second_chance"] == "Hey Claudio."
    assert not (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").count("near_miss")


async def test_second_chance_keeps_the_near_miss_when_it_hears_something_else(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    assistant, speaker = with_second_chance(
        fake_vad, vad_settings, SecondChance("il gatto dorme sul divano"), tmp_path
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert speaker.beeps == [] and assistant.state is State.IDLE
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "near_miss" and row["audio_file"] == "near_miss.wav"
    assert row["second_chance"] == "il gatto dorme sul divano" and row["second_chance_ms"] >= 0


class NoCommands(FakeResponder):
    def understands(self, text: str, language: str = "it") -> bool:
        return False


def second_chance_with(
    fake_vad, vad_settings, check: SecondChance, tmp_path, responder=None, **options
):
    """A near miss of "hey_claudio" (0.3) at frame 20, with the given Assistant options."""
    speaker = FakeSpeaker()
    assistant = Assistant(
        FakeWake(set(), miss_on={20}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        responder or FakeResponder(),
        speaker,
        MetricsWriter(tmp_path / "metrics.jsonl"),
        clock=AudioClock(),
        second_chance=check,
        **options,
    )
    return assistant, speaker


async def test_no_voice_no_whisper(fake_vad, vad_settings, make_frames, tmp_path) -> None:
    """A click, a key, the headset's gate opening: Silero hears no voice, Whisper is not even
    asked, and the audio is kept apart, named after why it was rejected."""
    saved: list[str] = []
    check = SecondChance("Claudio.")
    assistant, speaker = second_chance_with(
        fake_vad,
        vad_settings,
        check,
        tmp_path,
        voice_meter=lambda audio: 32,
        wake_min_voice_ms=96,
        rejected_sink=lambda audio, kind: saved.append(kind) or f"rejected/{kind}.wav",
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert check.heard == [] and speaker.beeps == [] and assistant.state is State.IDLE
    assert saved == ["no_voice"]
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "near_miss" and row["rejected"] == "no_voice"
    assert row["wake_voice_ms"] == 32 and row["audio_file"] == "rejected/no_voice.wav"


def strong_wake_with(fake_vad, vad_settings, tmp_path, **options):
    """A clear "hey_claudio" (0.9) at frame 20, no second chance needed."""
    speaker = FakeSpeaker()
    assistant = Assistant(
        FakeWake({20}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        FakeResponder(),
        speaker,
        MetricsWriter(tmp_path / "metrics.jsonl"),
        clock=AudioClock(),
        **options,
    )
    return assistant, speaker


async def test_a_clear_wake_on_silence_is_no_wake(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    """31/10 12:45:36: "hey_claudio" at 0.77 on the headset's gate (-87 dBFS), then "Nessuna
    richiesta": a clear score needs a voice too, as a weak one does. Fired for several frames
    in a row, it is logged and saved once."""
    saved: list[str] = []
    assistant, speaker = strong_wake_with(
        fake_vad,
        vad_settings,
        tmp_path,
        voice_meter=lambda audio: 0,
        wake_min_voice_ms=96,
        rejected_sink=lambda audio, kind: saved.append(kind) or f"rejected/{kind}.wav",
    )
    assistant.wake.fire_on = {20, 21, 22}
    await run_frames(assistant, make_frames("silence", 2400))
    assert speaker.beeps == [] and assistant.state is State.IDLE
    assert saved == ["no_voice"]
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "near_miss" and row["rejected"] == "no_voice"
    assert row["wake_score"] == 0.9 and row["wake_voice_ms"] == 0


async def test_a_clear_wake_with_a_voice_wakes_at_once(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    assistant, speaker = strong_wake_with(
        fake_vad, vad_settings, tmp_path, voice_meter=lambda audio: 480, wake_min_voice_ms=96
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert speaker.beeps == ["wake"]
    assert assistant._interaction is not None
    assert assistant._interaction.info["wake_voice_ms"] == 480


async def test_a_voice_goes_on_to_whisper(fake_vad, vad_settings, make_frames, tmp_path) -> None:
    check = SecondChance("Claudio.")
    assistant, speaker = second_chance_with(
        fake_vad, vad_settings, check, tmp_path, voice_meter=lambda audio: 480, wake_min_voice_ms=96
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert len(check.heard) == 1 and speaker.beeps == ["wake"]
    assert assistant._interaction is not None
    assert assistant._interaction.info["wake_voice_ms"] == 480


@pytest.mark.parametrize(
    ("heard", "voice", "wakes"),
    [
        ("", None, False),  # 30/10 14:00:43: too quiet for Whisper, then "Nessuna richiesta"
        ("", 480, True),  # a voice Whisper could not write: the request will be verified
        ("Grazie.", None, True),  # Whisper's guess on a lone word
    ],
)
async def test_an_unclear_name_wakes_only_on_a_voice(
    fake_vad, vad_settings, make_frames, tmp_path, heard: str, voice: int | None, wakes: bool
) -> None:
    assistant, speaker = second_chance_with(
        fake_vad,
        vad_settings,
        SecondChance(heard),
        tmp_path,
        responder=NoCommands(),
        voice_meter=None if voice is None else (lambda audio: voice),
        wake_min_voice_ms=96,
        verify_wake=True,
        always_confirm=frozenset({"hey_claudio"}),
        unclear_min=0.2,
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert (speaker.beeps == ["wake"]) is wakes
    assert (assistant.state is State.LISTENING) is wakes


async def test_a_zero_voice_floor_still_never_wakes_on_silence(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    """min_voice_ms = 0 lets Whisper hear everything, but an empty text on 0 ms of voice is
    still silence, never an unclear name (review 31/10)."""
    assistant, speaker = second_chance_with(
        fake_vad,
        vad_settings,
        SecondChance(""),
        tmp_path,
        responder=NoCommands(),
        voice_meter=lambda audio: 0,
        wake_min_voice_ms=0,
        verify_wake=True,
        always_confirm=frozenset({"hey_claudio"}),
        unclear_min=0.2,
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert speaker.beeps == [] and assistant.state is State.IDLE


async def test_a_noisy_models_misfires_leave_only_their_audio(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    """ "claudio" alone fires often on noise: its rejected detections are saved apart (the
    negatives to retrain it on), without a metrics row each."""
    saved: list[str] = []
    assistant, speaker = second_chance_with(
        fake_vad,
        vad_settings,
        SecondChance("il gatto dorme"),
        tmp_path,
        responder=NoCommands(),
        always_confirm=frozenset({"hey_claudio"}),
        rejected_sink=lambda audio, kind: saved.append(kind) or f"{kind}.wav",
    )
    await run_frames(assistant, make_frames("silence", 2400))
    assert saved == ["no_name"] and speaker.beeps == []
    assert not (tmp_path / "metrics.jsonl").exists()


async def test_a_failing_second_chance_is_a_plain_near_miss(
    fake_vad, vad_settings, make_frames, tmp_path, caplog
) -> None:
    assistant, speaker = with_second_chance(
        fake_vad, vad_settings, SecondChance(error=True), tmp_path
    )
    with caplog.at_level("ERROR"):
        await run_frames(assistant, make_frames("silence", 2400))
    assert speaker.beeps == [] and assistant.state is State.IDLE
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "near_miss" and row["second_chance"] == ""
    assert "CUDA esplosa" in caplog.text


async def test_second_chance_never_wakes_a_busy_assistant(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    """Whisper answers after the model itself woke CLAUDIO: no second wake, no second beep."""
    gate = threading.Event()
    check = SecondChance("Hey Claudio", gate=gate)
    assistant, speaker = with_second_chance(fake_vad, vad_settings, check, tmp_path)
    assistant.wake.fire_on = {22}
    for frame in make_frames("silence", 2400):
        assistant.clock.t += 0.08
        await assistant.handle_frame(frame)
        await asyncio.sleep(0)  # lets the second chance start (it waits on the gate)
    assert assistant.state is State.LISTENING and speaker.beeps == ["wake"]
    gate.set()
    await assistant.wait_idle()
    assert speaker.beeps == ["wake"] and assistant.state is State.LISTENING
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["outcome"] == "near_miss" and row["second_chance"] == "Hey Claudio"
    # Not a rejection: no WAV kept apart as a negative to retrain the wake word on.
    assert row["superseded"] is True and "audio_file" not in row and "rejected" not in row


async def test_going_busy_counts_ahead_of_no_voice(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    """Busy by the time the voice meter answers: a row for the record, and no WAV kept apart as
    a negative to retrain the wake word on, even if the meter heard no voice."""
    reached, gate = threading.Event(), threading.Event()
    saved: list[str] = []

    def meter(audio: np.ndarray) -> int:
        reached.set()
        gate.wait(5)
        return 0

    assistant, _ = second_chance_with(
        fake_vad,
        vad_settings,
        SecondChance("Hey Claudio"),
        tmp_path,
        voice_meter=meter,
        wake_min_voice_ms=96,
        rejected_sink=lambda audio, kind: saved.append(kind) or f"{kind}.wav",
    )
    for frame in make_frames("silence", 1600):  # the near miss is on the last one
        await assistant.handle_frame(frame)
    for _ in range(500):
        if reached.is_set():
            break
        await asyncio.sleep(0.01)
    assert reached.is_set()
    assistant.state = State.RESPONDING
    gate.set()
    await assistant.wait_idle()
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["superseded"] is True and "rejected" not in row and saved == []


async def test_a_second_chance_over_a_phrase_already_begun_keeps_to_its_lead_in(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    """The name heard with the voice still going: the phrase is the 1.5 s of lead-in and what
    follows. The 2.4 s the second chance heard replace them only for a phrase already over."""
    assistant, _ = second_chance_with(
        fake_vad, vad_settings, SecondChance("Claudio, apri Chrome."), tmp_path
    )
    assistant.wake.miss_on = {30}
    for frame in make_frames("speech", 2400):  # the near miss is on the last one
        await assistant.handle_frame(frame)
    await assistant.wait_idle()
    for frame in make_frames("silence", 1200):
        await assistant.handle_frame(frame)
    await assistant.wait_idle()
    assert 1.4 < len(assistant.transcriber.audio) / 16000 < 1.8


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


async def test_one_breath_request_keeps_the_words_before_the_wake_word(
    fake_vad, vad_settings, make_frames
) -> None:
    stt = FakeStt()
    # speech from the first frame, wake word fires on frame 10 while the request goes on
    assistant, *_ = build(fake_vad, vad_settings, stt, {10})
    await run_frames(assistant, make_frames("speech", 1600) + make_frames("silence", 800))
    assert stt.audio is not None
    # 800 ms heard before the wake word + 800 ms after it, and ~100 ms of the final silence
    assert 1.55 < len(stt.audio) / 16000 < 1.75
    assert assistant.history[-1]["outcome"] == "ok"


async def test_only_the_wake_word_listens_again(fake_vad, vad_settings, make_frames) -> None:
    # "Hey Claudio." [the beep] "apri Chrome": the request arrives while the first phrase is
    # being transcribed, and must not be lost
    stt = FakeStt(text=["Hey Claudio.", "apri chrome"], delay=0.2)
    saved: list[np.ndarray] = []
    assistant, wake, speaker, responder = build(
        fake_vad, vad_settings, stt, {1}, audio_sink=lambda audio, _: saved.append(audio)
    )
    frames = (
        make_frames("silence", 80)
        + make_frames("speech", 400)
        + make_frames("silence", 480)  # end of the first phrase
        + make_frames("speech", 800)  # the request, said during the transcription
        + make_frames("silence", 800)
    )
    await run_frames(assistant, frames)
    assert len(stt.heard) == 2 and len(stt.heard[1]) / 16000 > 0.8
    assert responder.seen == ["apri chrome"] and speaker.spoken == ["Apro Chrome."]
    assert speaker.beeps == ["wake"]
    (row,) = assistant.history
    assert row["outcome"] == "ok" and row["relisten"] == "Hey Claudio."
    assert row["text"] == "apri chrome" and row["m1_ms"] is not None
    assert row["prewake_ms"] == 80 and len(saved[0]) / 16000 > 1.9  # both phrases and pauses
    assert assistant.state is State.IDLE and wake.resets == 1


async def test_what_was_held_stops_at_the_timeout(fake_vad, vad_settings, make_frames) -> None:
    """ "Hey Claudio." took Whisper longer than the timeout, and only silence was held: the
    window times out in the middle of those frames, and the rest are not listened to."""
    assistant, _, speaker, _ = build(fake_vad, vad_settings, FakeStt("Hey Claudio."), {1})
    phrase = make_frames("silence", 80) + make_frames("speech", 400) + make_frames("silence", 480)
    for frame in phrase + make_frames("silence", 3600):  # no await that suspends: all held
        await assistant.handle_frame(frame)
    await assistant.wait_idle()
    assert assistant.history[-1]["outcome"] == "timeout" and assistant.state is State.IDLE
    assert speaker.beeps == ["wake", "cancel"]
    assert assistant._task is not None and assistant._task.exception() is None


async def test_wake_word_twice_then_gives_up(fake_vad, vad_settings, make_frames) -> None:
    stt = FakeStt(text="Hey Claudio.")
    assistant, _, speaker, responder = build(fake_vad, vad_settings, stt, {1})
    phrase = make_frames("speech", 400) + make_frames("silence", 480)
    await run_frames(
        assistant, make_frames("silence", 80) + phrase + make_frames("silence", 80) + phrase
    )
    assert len(stt.heard) == 2 and responder.seen == []
    assert speaker.beeps == ["wake", "cancel"]
    assert assistant.history[-1]["outcome"] == "empty"


async def test_microphone_level_is_adjusted_after_speech_only(
    fake_vad, vad_settings, make_frames
) -> None:
    class Level:
        gain_db = 6.0

        def __init__(self) -> None:
            self.seen: list[dict] = []

        def observe(self, stats) -> None:
            self.seen.append(dict(stats))
            self.gain_db += 3

    level = Level()
    wake = FakeWake({1, 30})
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        FakeResponder(),
        FakeSpeaker(),
        clock=AudioClock(),
        level=level,
    )
    utterance = make_frames("speech", 800) + make_frames("silence", 800)
    # a request, then a wake word followed by nothing (timeout): only the first one counts
    await run_frames(
        assistant, make_frames("silence", 80) + utterance + make_frames("silence", 6000)
    )
    assert [r["outcome"] for r in assistant.history] == ["ok", "timeout"]
    assert len(level.seen) == 1 and level.seen[0]["speech_dbfs"] > -12
    assert [r["gain_db"] for r in assistant.history] == [6.0, 9.0]


async def test_a_phrase_without_words_does_not_move_the_microphone_level(
    fake_vad, vad_settings, make_frames
) -> None:
    """26/10 19:50: an empty phrase at -50 dBFS raised the gain, the next request saturated."""

    class Level:
        gain_db = 6.0

        def __init__(self) -> None:
            self.seen: list[dict] = []

        def observe(self, stats) -> None:
            self.seen.append(dict(stats))

    level = Level()
    assistant = Assistant(
        FakeWake({1}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(""),
        FakeResponder(),
        FakeSpeaker(),
        clock=AudioClock(),
        level=level,
    )
    await run_frames(
        assistant,
        make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800),
    )
    assert assistant.history[0]["outcome"] not in {"ok", "timeout"}
    assert level.seen == []


class SeenLevel:
    gain_db = 6.0
    microphone = "Cuffie"

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def observe(self, stats) -> None:
        self.seen.append(dict(stats))


class FailingResponder(FakeResponder):
    async def respond(self, text: str, language: str = "it") -> AsyncIterator[str]:
        self.seen.append(text)
        raise RuntimeError("Claude non risponde")
        yield ""  # an async generator


class StopResponder(FakeResponder):
    async def respond(self, text: str, language: str = "it") -> AsyncIterator[str]:
        self.seen.append(text)
        self.last_route = {"route": "local", "command": "stop"}
        yield "Ok."


class SwitchingResponder(FakeResponder):
    """The capture moves to the webcam while the reply is still going."""

    def __init__(self, level: SeenLevel) -> None:
        super().__init__()
        self.level = level

    async def respond(self, text: str, language: str = "it") -> AsyncIterator[str]:
        self.level.microphone = "Webcam"
        async for part in super().respond(text, language):
            yield part


@pytest.mark.parametrize(
    ("responder", "outcome", "learns"),
    [(FailingResponder, "error", False), (StopResponder, "stopped", True)],
)
async def test_only_the_outcomes_of_a_real_request_move_the_microphone_level(
    fake_vad, vad_settings, make_frames, responder, outcome: str, learns: bool
) -> None:
    level = SeenLevel()
    assistant = Assistant(
        FakeWake({1}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        responder(),
        FakeSpeaker(),
        clock=AudioClock(),
        level=level,
    )
    await run_frames(
        assistant,
        make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800),
    )
    assert assistant.history[0]["outcome"] == outcome and assistant.history[0]["text"]
    assert bool(level.seen) is learns


async def test_a_phrase_heard_on_another_microphone_does_not_move_this_ones_level(
    fake_vad, vad_settings, make_frames
) -> None:
    level = SeenLevel()
    assistant = Assistant(
        FakeWake({1}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        SwitchingResponder(level),
        FakeSpeaker(),
        clock=AudioClock(),
        level=level,
    )
    await run_frames(
        assistant,
        make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800),
    )
    assert assistant.history[0]["outcome"] == "ok" and level.seen == []


async def test_an_error_beep_that_cannot_play_still_closes_the_request(
    fake_vad, vad_settings, make_frames
) -> None:
    class Unplugged(FakeSpeaker):
        async def beep(self, kind: str = "wake") -> None:
            if kind == "error":
                raise RuntimeError("device gone")
            await super().beep(kind)

    assistant = Assistant(
        FakeWake({1}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        FailingResponder(),
        Unplugged(),
        clock=AudioClock(),
    )
    await run_frames(
        assistant,
        make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800),
    )
    assert assistant.history[0]["outcome"] == "error" and assistant.state is State.IDLE
    assert assistant._task is not None and assistant._task.exception() is None


async def test_a_request_cancelled_during_its_last_beep_keeps_its_outcome(
    fake_vad, vad_settings, make_frames
) -> None:
    class EndlessBeep(FakeSpeaker):
        def __init__(self) -> None:
            super().__init__()
            self.beeping = asyncio.Event()

        async def beep(self, kind: str = "wake") -> None:
            await super().beep(kind)
            if kind == "cancel":
                self.beeping.set()
                await asyncio.Event().wait()

    speaker = EndlessBeep()
    assistant = Assistant(
        FakeWake({1}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        StopResponder(),
        speaker,
        clock=AudioClock(),
    )
    frames = make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800)
    for frame in frames:
        await assistant.handle_frame(frame)
    await asyncio.wait_for(speaker.beeping.wait(), 5)
    assert assistant._task is not None
    assistant._task.cancel()
    await assistant.wait_idle()
    assert assistant.history[0]["outcome"] == "stopped" and assistant.state is State.IDLE


@pytest.mark.parametrize("state", list(State))
def test_states_are_strings(state: State) -> None:
    assert isinstance(state.value, str)


class SystemResponder(FakeResponder):
    """The real system commands on top of the fake router: pause, resume, shut down."""

    def __init__(self, system: SystemControl) -> None:
        super().__init__()
        self.system = system

    async def respond(self, text: str, language: str = "it") -> AsyncIterator[str]:
        self.seen.append(text)
        command = {"dormi": "pause", "svegliati": "resume", "spegniti": "shutdown"}.get(text)
        if command:
            self.system.apply(command)
            self.last_route = {"route": "local", "command": command}
            yield "Ok."
        elif self.system.paused:
            self.last_route = {"route": "paused"}
        else:
            async for chunk in super().respond(text):
                yield chunk


class OnsetWake(FakeWake):
    """Fires at the start of every burst of speech heard while idle."""

    def __init__(self) -> None:
        super().__init__(set())
        self._loud = False

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        self.calls += 1
        loud, was = bool(np.abs(frame).max() > 1000), self._loud
        self._loud = loud
        return ("claudio", 0.9) if loud and not was else None


def build_system(fake_vad, vad_settings, texts: list[str]):
    system = SystemControl()
    wake, speaker, responder = OnsetWake(), FakeSpeaker(), SystemResponder(system)
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        FakeStt(texts),
        responder,
        speaker,
        clock=AudioClock(),
        system=system,
    )
    return assistant, speaker, system


async def run_requests(assistant: Assistant, make_frames, n: int) -> None:
    """n requests (a wake word at the start of 800 ms of speech), each sent only once the
    previous one is over, with silence in between: no race with the STT worker thread."""
    clock = assistant.clock
    assert isinstance(clock, AudioClock)
    request = make_frames("silence", 160) + make_frames("speech", 800)
    silence = np.zeros(1280, dtype=np.int16)

    async def gen():
        for _ in range(n):
            for f in request:
                clock.t += 0.08
                yield f
                await asyncio.sleep(0)
            for _ in range(100):  # listening ends, then the reply
                clock.t += 0.08
                yield silence
                await asyncio.sleep(0.001)
                # Over: back to idle, and no second chance still deciding whether to wake.
                if (
                    assistant.state is State.IDLE
                    and assistant.endpointer.started
                    and not assistant._checks
                ):
                    break

    await assistant.run(gen())


async def test_pause_ignores_requests_until_woken(fake_vad, vad_settings, make_frames) -> None:
    texts = ["dormi", "apri chrome", "svegliati", "apri chrome"]
    assistant, speaker, system = build_system(fake_vad, vad_settings, texts)
    await run_requests(assistant, make_frames, 4)
    assert speaker.spoken == ["Ok.", "", "Ok.", "Apro Chrome."]
    assert speaker.beeps.count("cancel") == 1  # the request heard while paused
    assert [row["outcome"] for row in assistant.history] == ["ok", "paused", "ok", "ok"]
    assert not system.paused


async def test_shutdown_ends_the_run_after_the_reply(fake_vad, vad_settings, make_frames) -> None:
    assistant, speaker, system = build_system(fake_vad, vad_settings, ["spegniti", "apri chrome"])
    await run_requests(assistant, make_frames, 3)
    assert speaker.spoken == ["Ok."] and system.stopping
    assert len(assistant.history) == 1  # the second wake word never came: CLAUDIO had closed
    assert assistant.state is State.IDLE


async def test_reply_is_spoken_in_the_language_heard(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    for heard, text, spoken, voice in [
        ("en", "close the door", "I can't do that yet.", "en"),
        ("it", "open chrome", "Opening Chrome.", "en"),  # the phrase is English
        ("it", "apri chrome", "Apro Chrome.", "it"),
    ]:
        stt = FakeStt(text, language=heard)
        assistant, _, speaker, _ = build(fake_vad, vad_settings, stt, {3}, tmp_path)
        frames = make_frames("silence", 640) + make_frames("speech", 1000)
        await run_frames(assistant, frames + make_frames("silence", 800))
        assert speaker.spoken == [spoken] and speaker.languages == [voice]
        assert assistant.history[-1]["stt_language"] == heard


def test_history_is_bounded(fake_vad, vad_settings) -> None:
    from claudio.app import HISTORY_MAX

    assistant, *_ = build(fake_vad, vad_settings, FakeStt("apri chrome"), {1})
    for i in range(HISTORY_MAX + 50):
        assistant.history.append({"n": i})
    assert len(assistant.history) == HISTORY_MAX
    assert assistant.history[-1] == {"n": HISTORY_MAX + 49}


# --- v0.2.2: messages written in the chat window --------------------------------------


async def test_typed_message_is_answered_in_silence(fake_vad, vad_settings, tmp_path) -> None:
    assistant, _, speaker, responder = build(fake_vad, vad_settings, FakeStt(), set(), tmp_path)
    reply = await assistant.handle_typed("apri chrome", "it")
    assert reply == "Apro Chrome." and responder.seen == ["apri chrome"]
    assert speaker.spoken == [] and speaker.beeps == []  # JenJo's choice: written reply only
    row = assistant.history[-1]
    assert row["wake_word"] == "typed" and row["text"] == "apri chrome"
    assert row["reply"] == "Apro Chrome." and row["outcome"] == "ok"
    assert assistant.state is State.IDLE


async def test_typed_message_can_be_spoken(fake_vad, vad_settings) -> None:
    assistant, _, speaker, _ = build(fake_vad, vad_settings, FakeStt(), set())
    await assistant.handle_typed("apri chrome", "it", speak=True)
    assert speaker.spoken == ["Apro Chrome."]


async def test_typed_message_waits_for_the_spoken_request(fake_vad, vad_settings) -> None:
    assistant, _, _, responder = build(fake_vad, vad_settings, FakeStt(), set())
    assistant.state = State.SPEAKING
    task = asyncio.create_task(assistant.handle_typed("apri chrome", "it"))
    await asyncio.sleep(0.12)
    assert responder.seen == []  # not while Claudio speaks
    assistant.state = State.IDLE
    await task
    assert responder.seen == ["apri chrome"]


async def test_typed_actions_to_confirm_need_the_voice(fake_vad, vad_settings) -> None:
    assistant, _, _, responder = build(fake_vad, vad_settings, FakeStt(), set())
    cancelled: list[bool] = []
    responder.awaiting = True  # the router asked "Confermi?"
    responder.cancel_pending = lambda: cancelled.append(True)  # type: ignore[method-assign]
    reply = await assistant.handle_typed("chiudi la finestra", "it")
    assert "a voce" in reply and cancelled
    assert assistant.history[-1]["outcome"] == "needs_voice"


async def test_serve_inbox_answers_the_window(fake_vad, vad_settings, tmp_path) -> None:
    from claudio.__main__ import serve_inbox
    from claudio.chat.store import ChatStore
    from claudio.config import Settings

    store = ChatStore(tmp_path / "claudio.db", "s")
    assistant, _, _, responder = build(fake_vad, vad_settings, FakeStt(), set())
    store.enqueue("apri chrome")
    settings = Settings(chat={"poll_s": 0.05})
    task = asyncio.create_task(serve_inbox(assistant, store, settings))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if responder.seen:
            break
    task.cancel()
    assert responder.seen == ["apri chrome"]


# --- startup: "Sono pronto" once Jack has loaded, between two requests -------------------


async def test_announce_waits_for_the_request_in_progress(fake_vad, vad_settings) -> None:
    assistant, _, speaker, _ = build(fake_vad, vad_settings, FakeStt(), set())
    assistant.state = State.SPEAKING
    task = asyncio.create_task(assistant.announce("Sono pronto."))
    await asyncio.sleep(0.12)
    assert speaker.spoken == []  # never over a reply
    assistant.state = State.IDLE
    await task
    assert speaker.spoken == ["Sono pronto."]
    assert assistant.state is State.IDLE


class LiteralResponder(FakeResponder):
    """No local command understands the phrase: it is for Claude."""

    def understands(self, text: str, language: str = "it", memory: bool = True) -> bool:
        return False


def build_continuing(fake_vad, vad_settings, stt: FakeStt, continue_s: float = 2.0):
    assistant, _wake, speaker, _ = build(fake_vad, vad_settings, stt, {1})
    responder = LiteralResponder()
    assistant.responder = responder
    assistant.continue_s = continue_s
    assistant.intake.continue_incomplete = continue_s > 0
    return assistant, speaker, responder


async def test_a_phrase_left_halfway_waits_for_the_rest(
    fake_vad, vad_settings, make_frames
) -> None:
    # 26/10: "Apri Chrome e cerca..." [JenJo thinks] "Dainese Moto" was answered at the pause
    stt = FakeStt(text=["Apri Chrome e cerca...", "Apri Chrome e cerca Dainese Moto."], delay=0.1)
    assistant, speaker, responder = build_continuing(fake_vad, vad_settings, stt)
    frames = (
        make_frames("silence", 80)
        + make_frames("speech", 800)
        + make_frames("silence", 1200)  # the phrase ends, then JenJo thinks
        + make_frames("speech", 640)
        + make_frames("silence", 1200)  # a long phrase now: it ends after long_end_silence_ms
    )
    await run_frames(assistant, frames)
    assert responder.seen == ["Apri Chrome e cerca Dainese Moto."]
    assert len(stt.heard) == 2 and len(stt.heard[1]) > len(stt.heard[0]) + 0.6 * 16000
    assert speaker.beeps == ["wake"]  # no beep in between: the same request
    (row,) = assistant.history
    assert row["outcome"] == "ok" and row["continued_from"] == "Apri Chrome e cerca..."
    assert row["m1_ms"] is not None
    assert assistant.state is State.IDLE


async def test_nothing_more_said_answers_the_first_part(
    fake_vad, vad_settings, make_frames
) -> None:
    stt = FakeStt(text="Dimmi...")
    assistant, _speaker, responder = build_continuing(fake_vad, vad_settings, stt)
    frames = make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 3200)
    await run_frames(assistant, frames)
    assert responder.seen == ["Dimmi..."]
    assert len(stt.heard) == 1  # the same audio is not transcribed twice
    assert assistant.history[-1]["outcome"] == "ok" and assistant.state is State.IDLE


async def test_a_finished_phrase_is_answered_at_once(fake_vad, vad_settings, make_frames) -> None:
    stt = FakeStt(text="Apri Chrome e cerca Dainese Moto.")
    assistant, _speaker, responder = build_continuing(fake_vad, vad_settings, stt)
    frames = make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800)
    await run_frames(assistant, frames)
    assert responder.seen == ["Apri Chrome e cerca Dainese Moto."]
    assert "continued_from" not in assistant.history[-1]


async def test_a_phrase_left_over_from_another_one_is_not_the_request(
    fake_vad, vad_settings, make_frames
) -> None:
    assistant, _, _, responder = build(fake_vad, vad_settings, FakeStt("apri chrome"), {1})
    assistant._continued = "Dimmi..."  # its phrase ended another way
    frames = make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800)
    await run_frames(assistant, frames)
    assert responder.seen == ["apri chrome"]


async def test_a_wake_word_reset_failing_once_still_goes_back_to_idle(
    fake_vad, vad_settings, make_frames
) -> None:
    # "Hey Claudio." listens again and the window times out while the held frames are replayed.
    # That _finish fails after clearing the interaction, so the request closes with none left.
    assistant, wake, _, _ = build(fake_vad, vad_settings, FakeStt("Hey Claudio."), {1})
    real_reset = wake.reset
    failures = [RuntimeError("onnx hiccup")]

    def reset_failing_once() -> None:
        if failures and assistant.history:  # its row is written just before the reset
            raise failures.pop()
        real_reset()

    wake.reset = reset_failing_once
    phrase = make_frames("silence", 80) + make_frames("speech", 400) + make_frames("silence", 480)
    for frame in phrase + make_frames("silence", 3600):
        await assistant.handle_frame(frame)
    await assistant.wait_idle()
    assert not failures  # it did fail
    assert assistant.state is State.IDLE


async def test_continuing_is_off_with_zero(fake_vad, vad_settings, make_frames) -> None:
    stt = FakeStt(text="Dimmi...")
    assistant, _speaker, responder = build_continuing(fake_vad, vad_settings, stt, continue_s=0)
    frames = make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800)
    await run_frames(assistant, frames)
    assert responder.seen == ["Dimmi..."]
