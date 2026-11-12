"""A scene for timeline tests (G0 task 11): the real Assistant, LocalResponder and ClaudeBrain
with fake models, and a clock that is the microphone's own frames (80 ms each).

What happens depends on frames, not on time: the voice and Claude's wait last a number of
frames, and every frame is followed by enough loop turns for the pipeline to settle, so the
same scene gives the same result alone or after other tests. Real time enters in two places
that never change a result: the tools' own threads (PcTools' pool), which `until(poll_s=)`
polls for, and `handle_typed`, which polls for an idle CLAUDIO with a real 50 ms sleep.

Whisper (the fixture's `inline_to_thread`) takes `whisper_frames` frames, 0 by default, both for
the request and for the check of a phrase said over the reply: with 0 no frame arrives while it
works, with more the microphone goes on feeding TRANSCRIBING and the barge-in check (the `_held`
frames, `_barge_pending`, the `_since_cut` slice and `_judge_stranded` are reached). It hears
what was said when it started, as the audio it was given.

Not collected by pytest (the name does not start with test_)."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from claudio.app import Assistant, BargeIn, State
from claudio.audio.vad import Endpointer
from claudio.brain.claude import ClaudeBrain
from claudio.config import ClaudeSettings, PcSettings, VadSettings
from claudio.cost.pricing import load_pricing
from claudio.pc.tools import PcTools
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import LocalResponder
from tests.unit.test_actions import Recorder
from tests.unit.test_app import AudioClock, FakeSpeaker, FakeWake
from tests.unit.test_claude_brain import USAGE, Monotonic, ScriptedClient, ScriptedStream
from tests.unit.test_conversation import barge
from tests.unit.test_cost import PRICES, tracker
from tests.unit.test_own_sound import SILENCE, VOICE
from tests.unit.test_pc_tools import FakeDesktop

FRAME_S = 0.08
SETTLE_TURNS = 30  # loop turns after a frame: enough for Whisper, Claude and the voice to move
SPEECH_FRAMES = 8  # a request: 640 ms of voice
END_FRAMES = 7  # then 560 ms of silence: the phrase ends (400 ms) with margin
OVER_FRAMES = 6  # a phrase said over the reply: 480 ms of voice
OVER_END_FRAMES = 10  # then 800 ms of silence
LIMIT_FRAMES = 400  # a wait that never ends is an error (32 s of microphone)


class Ticks:
    """Counts the microphone's frames; what lasts "n frames" waits for them. `free_running`
    (for tests with no microphone): n frames last n loop turns."""

    def __init__(self, free_running: bool = False) -> None:
        self.free_running = free_running
        self.count = 0
        self._waiting: list[tuple[int, asyncio.Future[None]]] = []

    def tick(self) -> None:
        self.count += 1
        due = [w for w in self._waiting if w[0] <= self.count]
        self._waiting = [w for w in self._waiting if w[0] > self.count]
        for _, future in due:
            if not future.done():
                future.set_result(None)

    async def after(self, frames: int) -> None:
        if self.free_running:
            for _ in range(frames):
                await asyncio.sleep(0)
            return
        if frames <= 0:
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiting.append((self.count + frames, future))
        await future


class TimedSpeaker(FakeSpeaker):
    """FakeSpeaker whose voice takes time, counted in frames: `start_frames` before the first
    sentence is audible (the synthesis), then each sentence lasts `sentence_frames`. With no
    `ticks` nothing lasts and it is FakeSpeaker."""

    def __init__(
        self, ticks: Ticks | None = None, start_frames: int = 0, sentence_frames: int = 0
    ) -> None:
        super().__init__()
        self.ticks, self.start_frames, self.sentence_frames = ticks, start_frames, sentence_frames

    async def _last(self, frames: int) -> None:
        """Lasts `frames` frames, or until interrupted."""
        for _ in range(frames):
            if self._cancel.is_set() or self.ticks is None:
                return
            await self.ticks.after(1)

    async def _before_first_sound(self) -> None:
        await self._last(self.start_frames)

    async def _during_sentence(self) -> None:
        await self._last(self.sentence_frames)


class PacedStream(ScriptedStream):
    """A scripted reply whose first words arrive `wait` frames after the request, and each
    next chunk `chunk_wait` frames after the one before."""

    def __init__(self, *args: Any, ticks: Ticks, wait: int, chunk_wait: int) -> None:
        super().__init__(*args)
        self.ticks, self.wait, self.chunk_wait = ticks, wait, chunk_wait
        self.chunks_read = 0  # what the consumer took: the words that are paid for

    @property
    def text_stream(self) -> AsyncIterator[str]:
        inner = super().text_stream

        async def gen() -> AsyncIterator[str]:
            await self.ticks.after(self.wait)
            first = True
            async for chunk in inner:
                if not first:
                    await self.ticks.after(self.chunk_wait)
                first = False
                self.chunks_read += 1
                yield chunk

        return gen()


class PacedClient(ScriptedClient):
    """ScriptedClient whose first call waits `wait` frames before its first word; every
    chunk then comes `chunk_wait` frames after the one before."""

    def __init__(
        self, *replies: tuple[list[str], str, list[Any]], ticks: Ticks, wait: int, chunk_wait: int
    ):
        super().__init__(*replies)
        self.ticks, self.wait, self.chunk_wait = ticks, wait, chunk_wait
        self.streams: list[PacedStream] = []

    def stream(self, **params: Any) -> PacedStream:
        self.calls.append(params)
        chunks, stop, content = self.replies.pop(0)
        wait = self.wait if len(self.calls) == 1 else 0
        stream = PacedStream(
            chunks,
            stop,
            content,
            dict(USAGE),
            ticks=self.ticks,
            wait=wait,
            chunk_wait=self.chunk_wait,
        )
        self.streams.append(stream)
        return stream


class SceneWake(FakeWake):
    """Fires on the next frame once `armed`."""

    def __init__(self) -> None:
        super().__init__(set())
        self.armed = False

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        self.calls += 1
        if self.armed:
            self.armed = False
            return ("hey_claudio", 0.9)
        return None


async def inline_to_thread(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Stands in for `asyncio.to_thread`: no worker thread, so the same frames at every run.
    The scene's Whisper first lets its `whisper_frames` frames go by."""
    await asyncio.sleep(0)  # a thread hands over: the caller does not run on at once
    owner = getattr(fn, "__self__", None)
    if isinstance(owner, SceneStt):
        owner.heard = owner.scene.said  # the audio it was given, not what is said meanwhile
        await owner.scene.ticks.after(owner.scene.whisper_frames)
    return fn(*args, **kwargs)


class SceneStt:
    """Whisper hears what the scene says; `fail` makes the next pass blow up; `samples`
    is the length of the audio of each pass."""

    def __init__(self, scene: Scene) -> None:
        self.scene, self.language, self.fail = scene, "it", False
        self.samples: list[int] = []
        self.heard = ""  # what the pass in progress hears, fixed when it starts

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> str:
        if self.fail:
            self.fail = False
            raise RuntimeError("CUDA esplosa")
        self.samples.append(len(audio))
        return self.heard

    def over(self, audio: np.ndarray) -> str:
        """The listener over the reply: the same Whisper."""
        return self.heard


class GatedDesktop(FakeDesktop):
    """FakeDesktop whose `launch` waits for `gate`: a slow tool, in the tools' own thread."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.started = threading.Event()

    def launch(self, app: Any) -> None:
        self.started.set()
        self.gate.wait(timeout=10)  # safety: a test that forgets to open it still ends
        super().launch(app)


class Scene:
    """The Assistant on a timeline. Time advances only when the scene feeds a frame.

    scene = Scene(vad, settings, matcher, replies=[...])
    await scene.say("che ore sono")
    await scene.until_speaking()
    await scene.say_over("basta")
    await scene.advance(2.0)
    assert scene.row(0)["outcome"] == "interrupted"
    """

    def __init__(
        self,
        vad: Any,
        vad_settings: VadSettings,
        matcher: CommandMatcher,
        replies: tuple[tuple[list[str], str, list[Any]], ...] = (),
        *,
        sentence_frames: int = 0,
        start_frames: int = 0,
        whisper_frames: int = 0,
        claude_wait: int = 0,
        chunk_wait: int = 0,
        window_s: float = 0.0,
        continue_s: float = 0.0,
        desk: FakeDesktop | None = None,
        home: Path | None = None,
        speaker: Any = None,
    ) -> None:
        self.whisper_frames = whisper_frames
        self.said = ""  # what Whisper hears, for the request and for what is said over a reply
        self.ticks = Ticks()
        self.clock = AudioClock()
        self.recorder = Recorder()
        self.wake = SceneWake()
        self.stt = SceneStt(self)
        self.speaker = speaker or TimedSpeaker(self.ticks, start_frames, sentence_frames)
        self.tools = PcTools(desk, PcSettings(), home=home) if desk and home else None
        self.client = PacedClient(
            *replies, ticks=self.ticks, wait=claude_wait, chunk_wait=chunk_wait
        )
        self.brain = ClaudeBrain(
            self.client,
            tracker(load_pricing(PRICES), 1.0),
            ClaudeSettings(tell_time=False),
            clock=Monotonic(),
            tools=self.tools,
            pc=PcSettings(),
        )
        actions = ActionRunner(self.recorder.ctx())
        self.brain.silenced = lambda: actions.system.off or actions.system.paused
        self.responder = LocalResponder(
            matcher,
            actions,
            fallback=self.brain.ask,
            fallback_info=lambda: self.brain.last_call,
            fallback_awaiting=lambda: self.brain.awaiting,
            fallback_cancel=self.brain.cancel_pending,
            fallback_repeats=self.brain.repeats_waiting,
            fallback_stop=self.brain.stop_acting,
            fallback_cut=self.brain.reply_cut,
            fallback_dropped=self.brain.dropped_action,
            fallback_heard=self.brain.heard,
        )
        self.assistant = Assistant(
            self.wake,
            Endpointer(vad, vad_settings),
            self.stt,
            self.responder,
            self.speaker,
            clock=self.clock,
            system=self.responder.actions.system,
            conversation_s=window_s,
            barge_in=self._listener(vad_settings),
        )
        self.wake.assistant = self.assistant
        self.assistant.intake.claude_min_words = 3
        if continue_s:
            self.assistant.continue_s = continue_s
            self.assistant.intake.continue_incomplete = True
        self._frames: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._asked = asyncio.Event()  # the microphone wants the next frame
        self._run: asyncio.Task[None] | None = None

    def _listener(self, vad_settings: VadSettings) -> BargeIn:
        over = barge(vad_settings, [])
        over.transcribe = self.stt.over  # type: ignore[assignment,misc]
        return over

    # --- the microphone ---------------------------------------------------------------

    async def _microphone(self) -> AsyncIterator[np.ndarray]:
        while True:
            self._asked.set()
            frame = await self._frames.get()
            self._asked.clear()
            if frame is None:
                return
            self.clock.t += FRAME_S
            self.ticks.tick()
            yield frame

    async def _feed(self, frame: np.ndarray) -> None:
        """One frame of microphone, handled and settled."""
        if self._run is None:
            self._run = asyncio.create_task(self.assistant.run(self._microphone()))
            await asyncio.sleep(0)
        self._asked.clear()
        self._frames.put_nowait(frame)
        waiting = asyncio.create_task(self._asked.wait())
        await asyncio.wait({waiting, self._run}, return_when=asyncio.FIRST_COMPLETED)
        waiting.cancel()
        if self._run.done():
            self._run.result()  # raises what stopped the assistant
            raise RuntimeError("the assistant stopped listening")
        for _ in range(SETTLE_TURNS):
            await asyncio.sleep(0)

    async def _feed_many(self, frame: np.ndarray, frames: int) -> None:
        for _ in range(frames):
            await self._feed(frame)

    async def until(self, done: Callable[[], bool], poll_s: float = 0.0) -> None:
        """Silence, a frame at a time, until `done()`. `poll_s`: real time to give a thread
        (a tool's) between two frames; the result never depends on how many frames it took."""
        for _ in range(LIMIT_FRAMES):
            if done():
                return
            await self._feed(SILENCE)
            if poll_s:
                await asyncio.sleep(poll_s)
        raise RuntimeError(f"never happened; the assistant is {self.assistant.state}")

    async def advance(self, seconds: float) -> None:
        await self._feed_many(SILENCE, round(seconds / FRAME_S))

    async def close(self) -> None:
        """Silence until CLAUDIO is idle again (windows time out, voices end), then stop."""
        with contextlib.suppress(RuntimeError):
            await self.until(lambda: self.assistant.state is State.IDLE)
        self.speaker.interrupt()
        if self._run is not None and not self._run.done():
            self._frames.put_nowait(None)
            await asyncio.wait({self._run}, timeout=5)
            if not self._run.done():
                self._run.cancel()
        if self.tools is not None:
            self.tools.close()

    # --- what JenJo does --------------------------------------------------------------

    def listening_again(self) -> bool:
        """A window is open and the beep that opened it is over (the beep's task ends one
        loop turn after it is appended to `speaker.beeps`: the beeps list is not enough)."""
        return self.assistant.state is State.LISTENING and not self.assistant.own_sound.playing

    async def say(self, text: str) -> None:
        """A request: with the wake word when CLAUDIO is idle, else into the open window."""
        self.said = text
        if self.assistant.state is State.IDLE:
            self.wake.armed = True
            await self._feed(SILENCE)
        await self.until(self.listening_again)
        await self._feed_many(VOICE, SPEECH_FRAMES)
        await self._feed_many(SILENCE, END_FRAMES)

    async def say_more(self, text: str) -> None:
        """The rest of a phrase left halfway: `text` is the whole phrase, as Whisper hears it."""
        self.said = text
        await self._feed_many(VOICE, SPEECH_FRAMES)
        await self._feed_many(SILENCE, END_FRAMES)

    async def until_speaking(self) -> None:
        await self.until(lambda: self.assistant.state is State.SPEAKING)

    async def say_over(self, text: str) -> None:
        """A phrase said while CLAUDIO answers (speaking, or still thinking)."""
        self.said = text
        await self._feed_many(VOICE, OVER_FRAMES)
        await self._feed_many(SILENCE, OVER_END_FRAMES)

    # --- what happened ----------------------------------------------------------------

    def row(self, index: int) -> dict[str, Any]:
        return list(self.assistant.history)[index]

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self.assistant.history)

    @property
    def remembered(self) -> list[tuple[str, str]]:
        """Claude's conversation memory: (question, answer) pairs."""
        return list(self.brain._history)


def text_reply(*chunks: str) -> tuple[list[str], str, list[Any]]:
    """A scripted Claude reply made of words only."""
    return list(chunks), "end_turn", [SimpleNamespace(type="text", text="".join(chunks))]
