"""Replay gate: saved WAVs through the production wiring, compared with a manifest. Also the
determinism run (does one WAV give the same result every time?).

    uv run python scripts/replay_gate.py                      # the gate, from cassettes: no API
    uv run python scripts/replay_gate.py --record --report    # record what is missing (paid)
    (--report writes samples/gate/report.md; the manifest, cassettes and corpus are in samples/)
    uv run python scripts/replay_gate.py --determinism 100 --clips <a.wav> <b.wav> ...

What is real: Silero VAD, Whisper, the router and the Assistant, built by
`claudio.__main__.build_assistant`, the same function production uses, and the responder, brain
and PC tools built by `build_responder`. What is not:
- Speaker and actions are fakes (FakeSpeaker, Recorder): nothing is heard, no key is pressed.
- The PC is a FakeDesktop with a home inside the work dir. Claude is a CassetteClient: a reply
  saved by --record, found by the SHA-256 of model, system, tools and messages. A key that was
  never recorded is a miss: it raises, counts and exits 1; nothing reaches the network. A row
  with `claude_error` fails its clip, and with --record the first API error stops the recording:
  no further request leaves, what finished stays saved.
- Hybrid wake word. Pass 1 (every clip): `ReplayWake` fires where the live wake fired (the row's
  `prewake_ms`), so VAD, Whisper, router and tools are tried on every clip. LIMIT: its score
  (1.0) skips the second chance, and intake skips `_false_wake` only after one: pass 1 can run
  a check the live run did not. Any `name_not_heard` must be read against the live row. Pass 2
  (clips marked `real_wake`): the real openWakeWord, through the real second chance, with
  Claude off: a clip that stops firing fails.
- A turn-on cannot be replayed: its WAV holds no name by design (nothing said while off is
  kept), Whisper cannot read what is left and the intake drops it as a false wake.
  tests/unit/test_off_mode.py covers that behaviour.
- Scenes: clips with the same `scene` run one after another on one Assistant, so the
  conversation window opens for real after a reply (`conversation`, `follow_up` rows have no
  wake word). A window's WAV starts at the first frame it listened to, not at its opening: the
  deaf frames are not saved. So after a clip digital zeros are fed until its phrase has ended
  and the window is no longer deaf, then the next WAV starts. A later clip with a live wake
  point first waits for idle and quiet: the wake word is never asked while a window is open, so
  live its window had closed. A phrase or a window that outlasts its limit fails its clip and
  stops the scene. Barge-in is not tried (no frame is fed while the Assistant speaks).
- A fresh memory store, speaker, recorder and clock for every scene.
- The feed WAITS for the Assistant between frames, which production never does. LIMIT: a
  detection of `claudio` (confirm = true) goes through Whisper's second chance while production
  keeps feeding frames (the `_checking` guard drops the repeats, `_prewake` is cut about 250 ms
  after the detection, `_held` fills). Here the repeats each get their own second chance (more
  chances to hear the name than live), `_prewake` is cut at the detection, `_held` stays empty
  and the "superseded" branch cannot happen. `--paced` is a control for this wait (determinism
  mode only) and no more: one frame per 80 ms by the wall clock, no wait, the Assistant's real
  clock. Both modes still share digital zeros around the WAV, a system that starts on, no pending
  state and FakeSpeaker, so neither is "as live".
- Time is logical: the clock moves by one block per frame and the feed waits for the Assistant
  to settle (transcribing, replying, speaking, wake checks) before the next frame. Latency
  figures in a replay row mean nothing.
- Whisper uses the shipped prompt, unless the WAV has a note (<stem>.json): then it is frozen.
  A clip without a note is listed in the report.

A clip is 3 s of silence, the WAV, then silence long enough for a wake with nothing said to time
out ([vad] no_speech_timeout_s + 1 s). With the shipped config the request leaves one row (the
conversation window it opens, [conversation] window_s, outlasts the tail and closes no row); if
there were more, the Outcome is read from the first row that holds text, else from the first.
`fired` is the model crossing its threshold (counted around `wake.process`; a detection Whisper
refuses leaves no row); `woke` is CLAUDIO opening an interaction (a row, or the one still open at
the end).

Manifest (samples/gate/manifest.jsonl), one JSON object per line: clip (path under
samples/corpus), expect (any of woke, route, command, tools, args, safety; absent keys are not
compared; `actions` are the Recorder's, `args` the tool_use inputs that ran (not a call cut); a
tool that ended in error fails the clip unless `tool_errors` names it), known_bug with known_diff
(the expect keys
wrong today: only a difference listed there is `known`, an error never is), why, scene,
real_wake, no_wake (the clip runs with the wake unarmed; without it a clip with no live
wake point fails). An unknown key is an error. A known_bug that now passes is `fixed` and fails
the gate until the label is removed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import dataclasses
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import astuple, dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # `tests` (FakeSpeaker, Recorder) is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))  # `benchmark_latency` (ReplayWake)

import httpx2  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from tests.unit.test_actions import Recorder  # noqa: E402
from tests.unit.test_app import FakeSpeaker  # noqa: E402
from tests.unit.test_pc_tools import FakeDesktop  # noqa: E402

import claudio.__main__ as entry  # noqa: E402
from benchmark_latency import ReplayWake  # noqa: E402
from claudio.app import State  # noqa: E402
from claudio.config import Settings, load_settings  # noqa: E402
from claudio.intake import BARGE_IN, CONVERSATION, FOLLOW_UP  # noqa: E402

if TYPE_CHECKING:
    from claudio.app import Assistant
    from claudio.pc.tools import App
    from claudio.router.responder import LocalResponder
    from claudio.system import SystemControl

SILENCE_S = 3
BUSY = (State.TRANSCRIBING, State.RESPONDING, State.SPEAKING)
SETTLE_LIMIT_S = (
    120.0  # real seconds one frame may wait for the Assistant: then the clip is an error
)
CLIP_LIMIT_S = 900.0  # real seconds for a whole clip
NOT_A_WAKE_WORD = {CONVERSATION, FOLLOW_UP, BARGE_IN, "typed"}
GATE_DIR = PROJECT_ROOT / "samples" / "gate"
MANIFEST = GATE_DIR / "manifest.jsonl"
CASSETTES = GATE_DIR / "cassettes"
REPORT = GATE_DIR / "report.md"
CORPUS = PROJECT_ROOT / "samples" / "corpus"
GATE_TIME = (
    1_800_000_000.0  # the memory store's fixed wall clock: a memory's date reaches the prompt
)
BIG_BUDGET = 1000.0  # USD a day: the daily block is not what the gate tries
# what makes a request the same request (tool_choice: a forced retry repeats the rest)
KEY_FIELDS = ("model", "system", "tools", "messages", "tool_choice")
TOKEN_CHARS_FLOOR = 2  # chars per token at the very worst: --record's estimate is an upper bound


class RealActionBlocked(RuntimeError):
    """A real key, program or file was asked for inside the gate."""


TRIPS: list[str] = []  # every time the tripwire fired, in this process


class GateTimeout(RuntimeError):
    pass


class GateRefused(RuntimeError):
    """The gate cannot start safely (a message for JenJo, not a crash)."""


class CassetteMiss(RuntimeError):
    """A request nobody recorded: the gate never sends it to the API."""


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Clip:
    wav: Path
    prompt: str | None  # Whisper's prompt when the WAV was saved (note); None: the shipped one
    language: str | None
    gain_db: float | None  # not applied: the WAV is already amplified
    session: str
    problem: str | None = None  # the note could not be read: the clip is an error


@dataclass(frozen=True)
class Outcome:
    fired: bool  # the wake word model crossed its threshold
    woke: bool  # CLAUDIO opened an interaction (Whisper's check passed)
    wake_word: str | None
    text: str
    route: str | None
    command: str | None
    tools: tuple[str, ...]
    actions: tuple[str, ...]
    safety: int | None
    outcome: str | None
    error: bool  # the row says the command failed, or the clip did (a tripwire has its own reason)
    args: tuple[str, ...] = ()  # the tool_use inputs Claude sent, `name {json}` each
    tool_errors: tuple[
        str, ...
    ] = ()  # tools that ended in error or were refused (rules, or a "no")
    claude_error: str = ""  # the brain caught it and spoke it: the row's outcome says `ok`

    def gate_key(self) -> tuple[Any, ...]:
        """Everything but the text (Whisper may punctuate differently without changing what
        the gate checks)."""
        return tuple(
            v for f, v in zip(fields(self), astuple(self), strict=True) if f.name != "text"
        )


def load_clip(wav: Path) -> Clip:
    note: dict[str, Any] = {}
    problem = None
    note_file = wav.with_suffix(".json")
    if note_file.is_file():
        try:
            note = json.loads(note_file.read_text(encoding="utf-8"))
            if not isinstance(note, dict):
                raise ValueError("not an object")
        except (OSError, ValueError) as exc:
            note, problem = {}, f"nota illeggibile ({type(exc).__name__})"
    return Clip(
        wav,
        note.get("whisper_prompt"),
        note.get("language"),
        note.get("gain_db"),
        wav.parent.parent.name,
        problem,
    )


def cassette_key(params: Mapping[str, Any]) -> str:
    """SHA-256 of model, system, tools, messages and tool_choice as sorted JSON: not max_tokens,
    cache_control or timeout (they do not change what Claude is asked); as max_tokens is left out,
    after changing a token limit the cassettes must be deleted and recorded again, because a
    replay keeps the old cut. tool_choice is in because
    a forced retry repeats the other four of the call it follows."""
    keyed = {name: params.get(name) for name in KEY_FIELDS}
    text = json.dumps(keyed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plain(value: Any) -> Any:
    """JSON-ready copy of an SDK object (pydantic), a fake (SimpleNamespace) or plain data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, SimpleNamespace):
        return {k: plain(v) for k, v in vars(value).items()}
    return value


def blocks(content: list[dict[str, Any]]) -> list[SimpleNamespace]:
    """The saved blocks as objects with the attributes the brain reads (type, text, id, name,
    input); `input` stays a dict."""
    return [SimpleNamespace(**block) for block in content]


def args_line(name: str, args: Mapping[str, Any]) -> str:
    return f"{name} {json.dumps(args, sort_keys=True, ensure_ascii=False)}"


class ReplayStream:
    """The saved reply shaped as the SDK's stream manager (the brain uses `async with`)."""

    def __init__(self, saved: dict[str, Any], note: Callable[[dict[str, Any]], None]):
        self.saved, self._note = saved, note

    async def __aenter__(self) -> ReplayStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def chunks() -> AsyncIterator[str]:
            for chunk in self.saved["chunks"]:
                yield chunk
                await asyncio.sleep(0)

        return chunks()

    @property
    def current_message_snapshot(self) -> Any:
        return SimpleNamespace(usage=self.saved["usage"])

    async def get_final_message(self) -> Any:
        self._note(self.saved)
        return rebuilt(self.saved)


class UnfinishedReply(httpx2.TransportError):
    """A transport error because the brain ends that as a Claude error ("network") on the row."""


class RecordingStream:
    """The real stream, copied into the cassette once its final message is read. An error on the
    way is told to `failed` and goes on to the brain, as without a cassette."""

    def __init__(
        self,
        manager: Any,
        record: Callable[[dict[str, Any]], None],
        failed: Callable[[Exception], None],
    ) -> None:
        self.manager, self._record, self._failed = manager, record, failed
        self.inner: Any = None
        self.chunks: list[str] = []

    @contextlib.contextmanager
    def _telling(self) -> Iterator[None]:
        try:
            yield
        except Exception as exc:
            self._failed(exc)
            raise

    async def __aenter__(self) -> RecordingStream:
        with self._telling():
            self.inner = await self.manager.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self.manager.__aexit__(*exc)

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def chunks() -> AsyncIterator[str]:
            with self._telling():
                async for chunk in self.inner.text_stream:
                    self.chunks.append(chunk)
                    yield chunk

        return chunks()

    @property
    def current_message_snapshot(self) -> Any:
        return self.inner.current_message_snapshot

    async def get_final_message(self) -> Any:
        with self._telling():
            final = await self.inner.get_final_message()
        if final.stop_reason is None:  # the body ended with no message_stop: half a reply
            unfinished = UnfinishedReply("risposta non terminata (nessun message_stop)")
            self._failed(unfinished)
            raise unfinished
        self._record(saved_reply(final, self.chunks))
        return final


def saved_reply(final: Any, chunks: list[str]) -> dict[str, Any]:
    return {
        "chunks": chunks,
        "stop_reason": final.stop_reason,
        "content": plain(final.content),
        "usage": plain(final.usage),
    }


def rebuilt(saved: dict[str, Any]) -> Any:
    return SimpleNamespace(
        usage=saved["usage"], stop_reason=saved["stop_reason"], content=blocks(saved["content"])
    )


class CassetteClient:
    """The Anthropic client's shape (`messages.stream`, `messages.create`, `models.retrieve`)
    answering from `folder/<key>.json`. With `real`, a request not saved yet goes to it and its
    successful reply is saved; without, it is a miss. Only chunks, stop reason, content and usage
    are saved: never a header or the key."""

    def __init__(
        self,
        folder: Path,
        real: Any | None = None,
        pricing: Any = None,
        budget: float | None = None,
    ) -> None:
        self.folder, self.real = folder, real
        self.pricing, self.budget = pricing, budget  # the brake: stop recording at this spend
        self.spent = 0.0  # what the replies recorded by THIS client cost (a replay costs nothing)
        self.braked: list[str] = []  # requests not recorded: the brake refused, or the one failed
        self.failure: str | None = None  # the first API error: nothing more leaves
        self.first_error: tuple[str, Exception] | None = None  # answers the brain's one retry
        self.saved: list[str] = []  # keys written by this client, in order
        self.misses: list[str] = []
        self.sizes: dict[str, int] = {}  # miss -> characters of its request (--record's estimate)
        self.tool_uses: list[tuple[str, dict[str, Any]]] = []  # calls that ran, not cut ones
        self.messages = self
        self.models = SimpleNamespace(retrieve=self._retrieve)

    async def _retrieve(self, model: str) -> Any:
        return SimpleNamespace(id=model)  # the warm-up: no network, no cost

    def stream(self, **params: Any) -> Any:
        key = cassette_key(params)
        saved = self._load(key)
        if saved is not None:
            return ReplayStream(saved, self._note)
        if self.first_error and self.first_error[0] == key:
            raise self.first_error[1]  # the brain's retry of a broken stream gets the same error
        if self.real is None or self._brake(key):
            raise self._miss(key, params)
        return RecordingStream(
            self.real.messages.stream(**params),
            lambda reply: self._saved_and_noted(key, reply, params["model"]),
            lambda exc: self._failed(key, params, exc),
        )

    async def create(self, **params: Any) -> Any:
        key = cassette_key(params)
        saved = self._load(key)
        if saved is None:
            if self.real is None or self._brake(key):
                raise self._miss(key, params)
            try:
                saved = saved_reply(await self.real.messages.create(**params), [])
            except Exception as exc:
                self._failed(key, params, exc)
                raise
            self._save(key, saved)
            self._charge(params["model"], saved)
        self._note(saved)
        return rebuilt(saved)

    def _load(self, key: str) -> dict[str, Any] | None:
        try:
            saved = json.loads((self.folder / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):  # missing, or half written: recorded again
            return None
        return saved if isinstance(saved, dict) else None

    def _save(self, key: str, saved: dict[str, Any]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.folder / f"{key}.json"
        partial = target.with_name(target.name + ".part")
        partial.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        os.replace(partial, target)
        self.saved.append(key)

    def discard(self, since: int) -> None:
        """Delete what was saved after the first `since` (a scene in which the tripwire fired:
        its replies answered a false failure). The money is spent; the reply is not kept."""
        for key in self.saved[since:]:
            (self.folder / f"{key}.json").unlink(missing_ok=True)
        del self.saved[since:]

    def _saved_and_noted(self, key: str, saved: dict[str, Any], model: str) -> None:
        self._save(key, saved)
        self._charge(model, saved)
        self._note(saved)

    def _charge(self, model: str, saved: dict[str, Any]) -> None:
        if self.pricing is not None:
            from claudio.brain.claude import CACHE_TTL
            from claudio.cost.pricing import Usage

            self.spent += self.pricing.cost(model, Usage.from_api(saved["usage"], CACHE_TTL))

    def _brake(self, key: str) -> bool:
        """Real money keeps a brake: once the recorded replies cost the budget, or a request has
        failed, stop sending."""
        if self.failure is None and (self.budget is None or self.spent < self.budget):
            return False
        self.braked.append(key)
        return True

    def _failed(self, key: str, params: Mapping[str, Any], exc: Exception) -> None:
        """The first API error stops the recording: the SDK has already retried, and what would
        follow goes out with a damaged history. Type and status only: its message may quote the
        request. A cut request costs money but no final message: `spent` does not count it."""
        if self.failure is None:
            self.first_error = (key, exc)
            status = getattr(exc, "status_code", None)
            self.failure = type(exc).__name__ + (f" {status}" if status else "")
        self.braked.append(key)
        self._miss(key, params)  # a hard reason on its clip, whatever the brain makes of the error

    def _note(self, saved: dict[str, Any]) -> None:
        if saved["stop_reason"] == "max_tokens":  # a cut call never runs (`_Leg.read`)
            return
        self.tool_uses.extend(
            (b["name"], dict(b.get("input") or {}))
            for b in saved["content"]
            if b.get("type") == "tool_use"
        )

    def _miss(self, key: str, params: Mapping[str, Any]) -> CassetteMiss:
        self.misses.append(key)  # every miss, so a clip can be told how many it caused
        self.sizes[key] = len(json.dumps({n: params.get(n) for n in KEY_FIELDS}, default=str))
        return CassetteMiss(key)


def install_tripwire(setter: Callable[[Any, str, Any], None] = setattr) -> None:
    """Whatever presses a key, closes a window or starts a program raises, in this process only.
    `setter` lets a test undo it (monkeypatch.setattr). Whisper, ONNX and the wake word must be
    imported first: a library that starts a process while loading would trip it."""
    from claudio import gpu

    def blocked(*args: object, **kwargs: object) -> None:
        TRIPS.append("blocked")  # PcTools.run turns the raise into a tool error: count it here
        raise RealActionBlocked("azione reale bloccata dal gate")

    setter(ctypes.windll.user32, "keybd_event", blocked)
    setter(ctypes.windll.user32, "SendInput", blocked)
    setter(os, "startfile", blocked)
    setter(os, "system", blocked)
    setter(subprocess, "Popen", blocked)
    # read_gpu bound subprocess.Popen as a default when it was defined: patching the module
    # later does not reach it.
    setter(gpu.read_gpu, "__defaults__", (blocked,))
    for module, name in (("win32gui", "PostMessage"), ("pywinauto.keyboard", "send_keys")):
        try:
            setter(importlib.import_module(module), name, blocked)
        except ImportError:
            continue


class GateWake(ReplayWake):
    """ReplayWake for the gate: re-armed for each clip, with the attribute the Assistant reads."""

    confirmed: frozenset[str] = frozenset()  # score 1.0: no second chance (see the docstring)

    def __init__(self) -> None:
        super().__init__(0)  # frames start at 1: it never fires until armed

    def arm(self, fire_at: int | None) -> None:
        """Fire on the `fire_at`-th frame from now (None: not at all in this clip)."""
        self.fire_at, self.frames = fire_at or 0, 0


class Detections:
    """What the wake word model did, seen around `wake.process` (the Assistant's history does
    not hold a detection Whisper refused)."""

    def __init__(self) -> None:
        self.count = 0  # detections: frames where the model crossed its threshold
        self.second_chances = 0  # times Whisper was asked to check the name
        self.peak = 0.0  # the highest score over the stream, any frame

    def count_second_chances(self, assistant: Assistant) -> None:
        check = getattr(assistant, "second_chance", None)
        if check is None:
            return

        def counted(audio: np.ndarray) -> str:
            self.second_chances += 1  # runs in a worker thread: one int, only read at the end
            return check(audio)

        assistant.second_chance = counted

    def watch(self, wake: Any) -> None:
        process = wake.process

        def counted(frame: np.ndarray) -> Any:
            detection = process(frame)
            self.count += detection is not None
            return detection

        wake.process = counted
        model = getattr(wake, "_model", None)  # shortcut: openWakeWord's own object, for the peak
        if model is not None:
            predict = model.predict

            def seen(frame: np.ndarray, *args: Any, **kwargs: Any) -> Any:
                scores = predict(frame, *args, **kwargs)
                self.peak = max([self.peak, *map(float, scores.values())])
                return scores

            model.predict = seen


class LogicalClock:
    """Time that moves only when a frame is fed."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class GateDesktop(FakeDesktop):
    """A launched app shows an empty document: the fixed text reads as an old one, and Haiku
    then refuses to write. One desktop serves the whole run, so each scene starts with reset()."""

    def __init__(self) -> None:
        super().__init__()
        self.first_text = self.text

    def launch(self, app: App) -> None:
        super().launch(app)
        self.text = ""

    def reset(self) -> None:
        """The window text of a new scene: the original one, with nothing typed."""
        self.text, self.typed = self.first_text, ""


@dataclass
class Harness:
    settings: Settings
    work_dir: Path
    wake: Any
    stt: Any
    vad: Any
    wake_vad: Any
    tracker: Any
    client: Any = None  # a CassetteClient: Claude and the PC tools are on; None: both are off
    desktop: Any = None  # the FakeDesktop of the gate (never a WindowsDesktop)
    real_wake: Any = None  # the real detector, for the clips marked real_wake
    home: Path = None  # type: ignore[assignment]  # the fake home, inside work_dir
    command_levels: dict[str, int] = field(default_factory=dict)  # local command -> its level
    brain: bool = True  # False: this scene runs with Claude off
    levels: list[int] = field(default_factory=list)  # levels asked before the tools ran
    running: bool = False  # a tool is running: PcTools.run asks its own level for the log
    scenes: int = 0
    # Rebuilt for every scene by fresh():
    recorder: Recorder = None  # type: ignore[assignment]
    speaker: FakeSpeaker = None  # type: ignore[assignment]
    system: SystemControl = None  # type: ignore[assignment]
    responder: LocalResponder = None  # type: ignore[assignment]
    assistant: Assistant = None  # type: ignore[assignment]
    clock: LogicalClock = None  # type: ignore[assignment]
    paced: bool = False
    detections: Detections = field(default_factory=lambda: Detections())

    def scene_settings(self) -> Settings:
        """A memory store of its own for each scene: what one scene saves must not reach the
        prompt of the next. With `brain` off, Claude and the PC tools are off too."""
        s = self.settings
        memory = s.memory.model_copy(update={"db_file": s.data_dir / f"memory-{self.scenes}.db"})
        update: dict[str, Any] = {"memory": memory}
        if not self.brain:
            update["claude"] = s.claude.model_copy(update={"enabled": False})
            update["pc"] = s.pc.model_copy(update={"enabled": False})
        return s.model_copy(update=update)

    def fresh(self) -> None:
        """A new Assistant, speaker, recorder, memory store and clock; the loaded models start
        over."""
        from claudio.system import SystemControl

        self.scenes += 1
        if isinstance(self.desktop, GateDesktop):
            self.desktop.reset()
        self.recorder = Recorder()
        self.speaker = FakeSpeaker()
        self.system = SystemControl()
        self.clock = LogicalClock()
        settings = self.scene_settings()
        self.responder = entry.build_responder(
            settings,
            self.recorder.ctx(),
            self.system,
            tracker=self.tracker,
            client=self.client,
            desktop=self.desktop,
            home=self.home,
            memory_clock=lambda: GATE_TIME,
        )
        self.levels = []
        if (tools := pc_tools(self)) is not None:
            ask = tools.level

            def recorded(call: Any) -> int:  # the level when the reply is judged
                level: int = ask(call)
                if not self.running:  # one asked by run() is read after earlier tools ran
                    self.levels.append(level)
                return level

            run = tools.run

            async def running(call: Any, *args: Any, **kwargs: Any) -> Any:
                self.running = True
                try:
                    return await run(call, *args, **kwargs)
                finally:
                    self.running = False

            tools.level = recorded
            tools.run = running
        self.wake.reset()
        self.vad.reset()
        self.wake_vad.reset()
        self.detections.count, self.detections.second_chances = 0, 0
        self.detections.peak = 0.0
        # shortcut: Whisper's private state, no public reset; upgrade if WhisperTranscriber gets one
        self.stt._encoded = None
        self.assistant = entry.build_assistant(
            settings,
            wake=self.wake,
            vad=self.vad,
            wake_vad=self.wake_vad,
            transcriber=self.stt,
            responder=self.responder,
            speaker=self.speaker,
            system=self.system,
            clock=None if self.paced else self.clock,
        )
        self.detections.count_second_chances(self.assistant)


def isolated(settings: Settings, work_dir: Path, claude: bool = False) -> Settings:
    """Everything that writes goes to work_dir (an empty memory too). Claude and the PC tools
    are off, unless the gate gives them a client: then they are on, with no clock in the prompt
    (it would change the cassette key every minute) and a budget that never blocks."""
    data = work_dir / "data"
    return settings.model_copy(
        update={
            "data_dir": data,
            "logs_dir": work_dir / "logs",
            "claude": settings.claude.model_copy(update={"enabled": claude, "tell_time": False}),
            # roots pinned: a move_file must never reach further than the throw-away home
            "pc": settings.pc.model_copy(update={"enabled": claude, "roots": ["~"]}),
            "memory": settings.memory.model_copy(update={"db_file": data / "memory.db"}),
            "cost": settings.cost.model_copy(
                update={
                    "db_file": data / "spend.db",
                    **({"daily_budget": BIG_BUDGET} if claude else {}),
                }
            ),
        }
    )


def build_harness(
    settings: Settings,
    work_dir: Path,
    wake: Any = None,
    stt: Any = None,
    vad: Any = None,
    paced: bool = False,
    client: Any = None,
    desktop: Any = None,
    real_wake: Any = None,
) -> Harness:
    from claudio.audio.vad import SileroVad
    from claudio.pc.files import NEAR_HOME
    from claudio.router.matcher import load_commands
    from claudio.stt.transcriber import WhisperTranscriber
    from claudio.wakeword.detector import OpenWakeWordDetector

    settings = isolated(settings, work_dir, claude=client is not None)
    (work_dir / "data").mkdir(parents=True, exist_ok=True)
    home = work_dir / "home"  # in the work dir, and never in a request: see the gate's tests
    for sub in NEAR_HOME:  # the folders a user has: a tool that looks for them finds them
        (home / sub).mkdir(parents=True, exist_ok=True)
    for folder in (home, *(p for p in home.rglob("*") if p.is_dir())):
        os.utime(folder, (GATE_TIME, GATE_TIME))  # search_files tells Claude `modified <date>`
    if client is not None and desktop is None:
        desktop = GateDesktop()  # the gate never builds a WindowsDesktop
    models = settings.resolve(settings.models_dir)
    rate = settings.audio.sample_rate
    wake = wake or OpenWakeWordDetector(
        settings.wakeword, models / "wakeword", settings.audio.block_size
    )
    stt = stt or WhisperTranscriber(settings.stt, models / "whisper")
    vad_file = models / "vad" / "silero_vad.onnx"
    # Production gives the voice meter a VAD of its own (own state); a given fake serves both.
    wake_vad = vad if vad is not None else SileroVad(vad_file, rate)
    vad = vad if vad is not None else SileroVad(vad_file, rate)
    commands = load_commands(settings.resolve(settings.router.commands_file))
    harness = Harness(
        settings,
        work_dir,
        wake,
        stt,
        vad,
        wake_vad,
        entry.build_tracker(settings),
        client=client,
        desktop=desktop,
        real_wake=real_wake,
        home=home,
        command_levels={c.id: c.level for c in commands},
        paced=paced,
    )
    harness.detections.watch(wake)
    if real_wake is not None:
        harness.detections.watch(real_wake)
    harness.fresh()
    return harness


def tail_s(settings: Settings) -> float:
    """Long enough for a wake with nothing said to close its row (no_speech_timeout_s)."""
    return settings.vad.no_speech_timeout_s + 1.0


def phrase_limit_s(settings: Settings) -> float:
    """The most silence fed to let a phrase end before the next clip of a scene: the longest
    pause that closes one (a phrase left halfway waits continue_s), plus a second."""
    return max(settings.vad.long_end_silence_ms / 1000, settings.vad.continue_s) + 1.0


def idle_limit_s(settings: Settings) -> float:
    """The most silence fed to let a window close by itself: the longest one (the conversation's,
    or a follow-up's no_speech_timeout_s), the deaf time after its closing beep, plus a second."""
    conversation = settings.conversation
    longest = max(conversation.window_s, settings.vad.no_speech_timeout_s)
    return longest + conversation.echo_tail_ms / 1000 + 1.0


def clip_parts(
    clip: Clip,
    settings: Settings,
    prewake_ms: float | None = None,
    lead_s: float | None = None,
    trail_s: float | None = None,
) -> tuple[np.ndarray, int | None]:
    """The audio of a clip (lead silence, WAV, trailing silence, whole blocks) and the frame
    where the live wake fired: with `prewake_ms` the lead grows so that the detection falls on a
    frame boundary, as benchmark_latency.replay() does. None: no wake in this clip."""
    audio, rate = sf.read(clip.wav, dtype="int16")
    if rate != settings.audio.sample_rate:
        raise ValueError(f"{clip.wav.name}: {rate} Hz, expected {settings.audio.sample_rate}")
    if audio.ndim > 1:
        audio = audio[:, 0]
    block = settings.audio.block_size
    lead = round((SILENCE_S if lead_s is None else lead_s) * rate)
    fire_at = None
    if prewake_ms is not None:
        prewake = round(prewake_ms * rate / 1000)
        lead += (-(lead + prewake)) % block
        fire_at = (lead + prewake) // block
    trail = math.ceil((tail_s(settings) if trail_s is None else trail_s) * rate)
    padded = np.concatenate((np.zeros(lead, np.int16), audio, np.zeros(trail, np.int16)))
    return np.concatenate((padded, np.zeros(-len(padded) % block, np.int16))), fire_at


def clip_audio(clip: Clip, settings: Settings) -> np.ndarray:
    return clip_parts(clip, settings)[0]


def is_busy(assistant: Assistant) -> bool:
    # shortcut: the wake checks and beeps are tasks the Assistant keeps in private sets; they run
    # Whisper in a thread, so the next frame must not race them. Upgrade: a public `busy`.
    return assistant.state in BUSY or bool(assistant._checks) or bool(assistant._beeps)


async def settle(assistant: Assistant) -> None:
    deadline = time.monotonic() + SETTLE_LIMIT_S
    while is_busy(assistant):
        if time.monotonic() > deadline:
            raise GateTimeout(f"l'assistente non si ferma: stato {assistant.state}")
        await asyncio.sleep(0.001)


async def frames(harness: Harness, audio: np.ndarray) -> Any:
    """The frames as the microphone's callback delivers them, one at a time (its queue drops
    old frames when full)."""
    from claudio.audio.capture import AudioCapture

    rate, block = harness.settings.audio.sample_rate, harness.settings.audio.block_size
    capture = AudioCapture(rate, block, device="replay")  # a device: no Windows default; no stream
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    callback = capture._frame_callback(asyncio.get_running_loop(), queue)
    next_at = time.monotonic()
    for start in range(0, len(audio), block):
        if harness.paced:  # the microphone's rhythm: a frame per block, whatever the Assistant does
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
            next_at += block / rate
        indata = audio[start : start + block].reshape(-1, 1)
        callback(indata, len(indata), None, None)  # gain 1: the WAV is already amplified
        frame = await queue.get()
        harness.clock.advance(block / rate)
        yield frame
        if not harness.paced:
            await settle(harness.assistant)


@contextlib.contextmanager
def frozen_prompt(stt: Any, prompt: str | None) -> Iterator[None]:
    if prompt is None:
        yield
        return
    stt.prompt = lambda language: prompt
    try:
        yield
    finally:
        del stt.prompt


@dataclass(frozen=True)
class Mark:
    """How far the Assistant, the actions and the client had got: a clip's Outcome is what
    happened between two marks."""

    rows: int = 0
    ran: int = 0
    opened: int = 0
    keys: int = 0
    combos: int = 0
    fired: int = 0
    tool_uses: int = 0
    misses: int = 0
    levels: int = 0
    trips: int = 0
    saved: int = 0


ZERO_MARK = Mark()


def take_mark(harness: Harness) -> Mark:
    rec, client = harness.recorder, harness.client
    return Mark(
        len(harness.assistant.history),
        len(rec.ran),
        len(rec.opened),
        len(rec.keys),
        len(rec.combos),
        harness.detections.count,
        len(client.tool_uses) if client is not None else 0,
        len(client.misses) if client is not None else 0,
        len(harness.levels),
        len(TRIPS),
        len(client.saved) if client is not None else 0,
    )


def request_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if row.get("text")), rows[0] if rows else {})


def pc_tools(harness: Harness) -> Any:
    # shortcut: the brain is reached through the responder's bound `ask`; upgrade if the
    # responder exposes it.
    brain = getattr(harness.responder.fallback, "__self__", None)
    return getattr(brain, "tools", None)


def declared_safety(harness: Harness, row: dict[str, Any], since: Mark, until: Mark) -> int | None:
    """The level the command or the tools declared when the run asked (the row holds one only
    when a yes was asked)."""
    levels = harness.levels[since.levels : until.levels]
    if row.get("level"):
        levels.append(row["level"])
    if row.get("command") in harness.command_levels:
        levels.append(harness.command_levels[row["command"]])
    return max(levels, default=None)


def tool_notes(row: dict[str, Any]) -> list[tuple[str, str]]:
    """(tool, how it ended) from the row: `name(args) ok|error|refused|ask|not_run_cut`."""
    notes = (str(t) for t in row.get("tools") or ())
    return [(t.split("(", 1)[0], t.rsplit(" ", 1)[-1]) for t in notes]


def read_outcome(
    harness: Harness, since: Mark = ZERO_MARK, until: Mark | None = None, scene: bool = False
) -> Outcome:
    """The Outcome of what happened between two marks (the whole run by default). In a `scene`
    the conversation window and a follow-up count as the interaction opening."""
    assistant = harness.assistant
    end = until or take_mark(harness)
    rows = list(assistant.history)[since.rows : end.rows]
    row = request_row(rows)
    not_wake = NOT_A_WAKE_WORD - ({CONVERSATION, FOLLOW_UP} if scene else set())
    # shortcut: the interaction still open at the end (a wake that never closed) is private.
    open_row = (
        assistant._interaction.info if until is None and assistant._interaction is not None else {}
    )
    opened = next(
        (r for r in (*rows, open_row) if r.get("wake_word") and r["wake_word"] not in not_wake),
        None,
    )
    if not row and opened is not None:
        row = {**opened, "outcome": "open"}
    rec = harness.recorder
    actions = (
        *(" ".join(a) for a in rec.ran[since.ran : end.ran]),
        *(f"open {p}" for p in rec.opened[since.opened : end.opened]),
        *(f"key {k}" for k in rec.keys[since.keys : end.keys]),
        *(f"combo {c}" for c in rec.combos[since.combos : end.combos]),
    )
    client = harness.client
    uses = client.tool_uses[since.tool_uses : end.tool_uses] if client is not None else []
    claude_error = next((r["claude_error"] for r in rows if r.get("claude_error")), "")
    return Outcome(
        fired=end.fired > since.fired,
        woke=opened is not None,
        wake_word=opened.get("wake_word") if opened else None,
        text=str(row.get("text") or ""),
        route=row.get("route"),
        command=row.get("command"),
        tools=tuple(name for name, how in tool_notes(row) if how != "not_run_cut"),
        actions=actions,
        safety=declared_safety(harness, row, since, end),
        outcome=row.get("outcome"),
        error=bool(row.get("error") or claude_error) or row.get("outcome") == "error",
        args=tuple(args_line(name, a) for name, a in uses),
        tool_errors=tuple(name for name, how in tool_notes(row) if how in ("error", "refused")),
        claude_error=str(claude_error),
    )


def error_outcome(harness: Harness, exc: Exception) -> Outcome:
    detail = f" {exc}" if isinstance(exc, GateTimeout) else ""
    message = f"gate_error: {type(exc).__name__}{detail}"
    fired = harness.detections.count > 0
    return Outcome(fired, False, None, "", None, None, (), (), None, message, True)


async def _replay(harness: Harness, clip: Clip) -> Outcome:
    harness.fresh()
    try:
        if clip.problem:
            raise ValueError(clip.problem)
        audio = clip_audio(clip, harness.settings)
        with frozen_prompt(harness.stt, clip.prompt):
            await asyncio.wait_for(harness.assistant.run(frames(harness, audio)), CLIP_LIMIT_S)
    except Exception as exc:  # the clip is an error, the run goes on
        return error_outcome(harness, exc)
    return read_outcome(harness)


def replay_clip(harness: Harness, clip: Clip) -> Outcome:
    return asyncio.run(_replay(harness, clip))


def determinism_report(outcomes: list[Outcome]) -> dict[str, int]:
    """Each Outcome field -> how many distinct values it took."""
    return {f.name: len({getattr(o, f.name) for o in outcomes}) for f in fields(Outcome)}


def whisper_line(stt: Any) -> str:
    return f"Whisper: {stt.model_name} su {stt.device}"


def live_row(clip: Clip) -> dict[str, Any]:
    metrics = clip.wav.parent.parent / "metrics.jsonl"
    try:
        lines = metrics.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return {}
    for line in lines:
        try:
            row = json.loads(line) if line.strip() else {}
        except ValueError:  # one bad line only loses the comparison with the live row
            continue
        if isinstance(row, dict) and row.get("audio_file") == clip.wav.name:
            return row
    return {}


def run_clip(harness: Harness, clip: Clip, repetitions: int, show_text: bool) -> list[Outcome]:
    live = live_row(clip)
    outcomes: list[Outcome] = []
    counts: list[tuple[int, int]] = []  # per repetition: detections, second chances
    started = time.perf_counter()
    for _ in range(repetitions):
        outcomes.append(replay_clip(harness, clip))
        counts.append((harness.detections.count, harness.detections.second_chances))
    seconds = (time.perf_counter() - started) / repetitions
    report = determinism_report(outcomes)
    first = outcomes[0]
    print(f"\n== {clip.wav.name}")
    print(
        f"  model scattato {sum(o.fired for o in outcomes)}/{repetitions}, "
        f"Claudio sveglio {sum(o.woke for o in outcomes)}/{repetitions}, "
        f"picco (ultima ripetizione) {harness.detections.peak:.3f}, replay {first.wake_word} | "
        f"dal vivo {live.get('wake_word')} (punteggio {live.get('wake_score')})"
    )
    detections, checks = [c[0] for c in counts], [c[1] for c in counts]
    print(
        f"  rilevazioni {min(detections)}-{max(detections)}, "
        f"seconde chance di Whisper {min(checks)}-{max(checks)} per ripetizione"
    )
    errors = sorted({o.outcome for o in outcomes if o.error})
    if errors:
        print(f"  ERRORI ({sum(o.error for o in outcomes)}/{repetitions}): {errors}")
    print(f"  valori distinti per campo: {report}")
    print(f"  campi che variano: {[k for k, n in report.items() if n > 1] or 'nessuno'}")
    print(
        f"  route replay {first.route} / dal vivo {live.get('route')}; "
        f"esito {first.outcome} / {live.get('outcome')}"
    )
    same = first.text.strip() == str(live.get("text", "")).strip()
    print(f"  testo uguale al dal vivo: {'si' if same else 'no'}")
    print(f"  {seconds:.2f} s per ripetizione")
    if show_text:
        print(f"  testo replay: {first.text!r}\n  testo dal vivo: {live.get('text')!r}")
    return outcomes


# --- the gate: a manifest of clips and what each must do ------------------------------------

ENTRY_KEYS = {
    "clip", "expect", "known_bug", "known_diff", "why", "scene", "real_wake", "no_wake",
}  # fmt: skip
EXPECT_KEYS = {
    "woke", "route", "command", "tools", "args", "safety", "actions", "tool_errors",
}  # fmt: skip
Reason = tuple[str, str]  # (safe to print, detail: may hold what was said)


@dataclass(frozen=True)
class Entry:
    clip: str  # path under samples/corpus
    expect: dict[str, Any]
    known_bug: str | None = None
    why: str = ""
    scene: str | None = None  # consecutive entries with the same scene run on one Assistant
    real_wake: bool | None = None  # pass 2: True must wake, False must not, None: no pass 2
    known_diff: tuple[str, ...] = ()  # with known_bug: the expect keys that are wrong today
    no_wake: bool = False  # the clip is meant to run with no wake armed (a discarded phrase)

    def __post_init__(self) -> None:
        if not self.expect:
            raise ManifestError(f"{self.clip}: expect vuoto: passerebbe su qualsiasi esito")
        if unknown := set(self.expect) - EXPECT_KEYS:
            raise ManifestError(f"{self.clip}: chiavi sconosciute in expect {sorted(unknown)}")
        args = self.expect.get("args", [])
        if not isinstance(args, list) or not all(
            isinstance(i, list) and len(i) == 2 and isinstance(i[1], dict) for i in args
        ):
            raise ManifestError(f"{self.clip}: args vuole coppie [nome, {{argomenti}}]")
        if bool(self.known_bug) != bool(self.known_diff):
            raise ManifestError(f"{self.clip}: known_bug e known_diff vanno insieme")
        if unknown := set(self.known_diff) - EXPECT_KEYS:
            raise ManifestError(f"{self.clip}: known_diff con chiavi sconosciute {sorted(unknown)}")


def parse_entry(line: str, where: str) -> Entry:
    try:
        raw = json.loads(line)
    except ValueError as exc:
        raise ManifestError(f"{where}: JSON non valido") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: non è un oggetto")
    if unknown := set(raw) - ENTRY_KEYS:
        raise ManifestError(f"{where}: chiavi sconosciute {sorted(unknown)}")
    if not isinstance(raw.get("clip"), str) or not isinstance(raw.get("expect"), dict):
        raise ManifestError(f"{where}: servono `clip` (testo) e `expect` (oggetto)")
    kinds = {"known_bug": str, "scene": str, "real_wake": bool, "no_wake": bool, "why": str}
    kinds |= {"known_diff": list}
    for key, kind in kinds.items():
        if raw.get(key) is not None and not isinstance(raw[key], kind):
            raise ManifestError(f"{where}: {key} ha il tipo sbagliato")
    if not all(isinstance(k, str) for k in raw.get("known_diff") or ()):
        raise ManifestError(f"{where}: known_diff vuole una lista di chiavi di expect")
    try:
        return Entry(
            raw["clip"],
            raw["expect"],
            raw.get("known_bug"),
            raw.get("why") or "",
            raw.get("scene"),
            raw.get("real_wake"),
            tuple(raw.get("known_diff") or ()),
            bool(raw.get("no_wake", False)),
        )
    except ManifestError as exc:
        raise ManifestError(f"{where}: {exc}") from exc


def load_manifest(path: Path) -> list[Entry]:
    entries = [
        parse_entry(line, f"{path.name}:{n}")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip()
    ]
    if not entries:
        raise ManifestError(f"{path.name}: nessuna clip: un gate vuoto non prova nulla")
    group_scenes(entries)  # a scene split in two is an error here
    return entries


def group_scenes(entries: Sequence[Entry]) -> list[list[Entry]]:
    """Consecutive entries with the same `scene` run together; one without a scene is alone."""
    scenes: list[list[Entry]] = []
    seen: set[str] = set()
    for item in entries:
        if item.scene is not None and scenes and scenes[-1][0].scene == item.scene:
            scenes[-1].append(item)
            continue
        if item.scene is not None:
            if item.scene in seen:
                raise ManifestError(f"la scena {item.scene} non è di righe consecutive")
            seen.add(item.scene)
        scenes.append([item])
    return scenes


@dataclass
class GateResult:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)  # a known_bug that still fails
    fixed: list[str] = field(default_factory=list)  # a known_bug that passes: take the label off
    misses: list[str] = field(default_factory=list)  # cassette keys nobody recorded
    no_note: list[str] = field(default_factory=list)  # clips without a note: Whisper unfrozen
    missing: list[str] = field(default_factory=list)  # manifest clips that are not on disk
    reasons: dict[str, list[Reason]] = field(default_factory=dict)  # clip -> what differs
    text_diffs: list[tuple[str, str, str]] = field(default_factory=list)  # clip, live, replay

    def exit_code(self) -> int:
        return 1 if self.failed or self.fixed or self.misses or self.missing else 0


def require_recorder(harness: Harness) -> None:
    """Refuse to start unless every action goes to the Recorder and the PC to the fake desktop."""
    ctx, rec = harness.responder.actions.ctx, harness.recorder
    if not (
        ctx.run == rec.ran.append
        and ctx.press_key == rec.keys.append
        and ctx.press_combo == rec.combos.append
        and ctx.open_path == rec.opened.append
    ):
        raise RealActionBlocked("il gate non è sul Recorder: non parto")
    tools = pc_tools(harness)
    if tools is not None and tools.desktop is not harness.desktop:
        raise RealActionBlocked("gli strumenti del PC non sono sul desktop finto: non parto")
    if not hasattr(harness.wake, "arm"):
        raise RealActionBlocked("il gate vuole GateWake come rilevatore della passata 1")


def expected_value(key: str, want: Any) -> Any:
    return [args_line(name, args) for name, args in want] if key == "args" else want


def found_value(key: str, outcome: Outcome) -> Any:
    value = getattr(outcome, key)
    return list(value) if isinstance(value, tuple) else value


def differences(expect: Mapping[str, Any], outcome: Outcome) -> list[Reason]:
    """(key, detail) for each expected key the outcome does not match. A tool that ended in
    error is a difference too, unless `tool_errors` names it."""
    out = []
    for key in (*expect, *(["tool_errors"] if "tool_errors" not in expect else [])):
        want = expected_value(key, expect.get(key, []))
        got = found_value(key, outcome)
        if got != want:
            out.append((key, f"{key}: atteso {want!r}, trovato {got!r}"))
    return out


@dataclass(frozen=True)
class Planned:
    entry: Entry
    clip: Clip
    live: dict[str, Any]
    audio: np.ndarray
    fire_at: int | None  # the frame where the live wake fired; None: no wake in this clip


def plan_scene(harness: Harness, entries: Sequence[Entry], corpus: Path) -> list[Planned]:
    """The audio of each clip of a scene: only the first has a lead of silence, only the last a
    tail. `no_wake`: the wake stays unarmed."""
    planned = []
    for n, item in enumerate(entries):
        clip = load_clip(corpus / item.clip)
        live = live_row(clip)
        audio, fire_at = clip_parts(
            clip,
            harness.settings,
            None if item.no_wake else live.get("prewake_ms"),
            lead_s=None if n == 0 else 0.0,
            trail_s=0.0 if n < len(entries) - 1 else None,
        )
        planned.append(Planned(item, clip, live, audio, fire_at))
    return planned


@dataclass(frozen=True)
class Ran:
    """A clip of a scene after the run: its Outcome, whether it hit a missing cassette, whether
    the scene reached it at all (it does not after "chiuditi"), whether the tripwire fired, and
    why the scene stopped on it, if it did."""

    outcome: Outcome
    missed: bool
    reached: bool
    tripped: bool
    stuck: Reason | None = None


def unfed_outcome() -> Outcome:
    message = "gate_error: scena interrotta prima di questa clip"
    return Outcome(False, False, None, "", None, None, (), (), None, message, True)


def own_sound_on(assistant: Assistant) -> bool:
    # shortcut: asking OwnSound publicly uses up a frame; upgrade if it gets a read-only property.
    return assistant.own_sound.playing or assistant.own_sound._left > 0


def phrase_open(assistant: Assistant) -> bool:
    return assistant.state is State.LISTENING and assistant.endpointer.started


def window_deaf(assistant: Assistant) -> bool:
    listening = assistant.state is State.LISTENING and not assistant.endpointer.started
    return listening and own_sound_on(assistant)


def idle_and_quiet(assistant: Assistant) -> bool:
    return assistant.state is State.IDLE and not own_sound_on(assistant)


async def silence_while(harness: Harness, waiting: Callable[[], bool], limit_s: float) -> Any:
    """Silent blocks while `waiting()`, no longer than `limit_s`: the caller reads how it ended."""
    settings = harness.settings
    block = settings.audio.block_size
    for _ in range(math.ceil(limit_s * settings.audio.sample_rate / block)):
        if not waiting():
            return
        async for frame in frames(harness, np.zeros(block, np.int16)):
            yield frame


async def end_phrase(harness: Harness) -> Any:
    """Silent blocks until the phrase just fed has ended and the window it opened listens. The
    next WAV starts at the first frame its window listened to (`_on_listen` returns on a deaf
    frame before the endpointer, and `_finish` saves only what the endpointer heard): fed during
    the deaf time it would lose its start; any silence after that would count twice."""
    assistant, limit_s = harness.assistant, phrase_limit_s(harness.settings)
    async for frame in silence_while(harness, lambda: phrase_open(assistant), limit_s):
        yield frame
    async for frame in silence_while(harness, lambda: window_deaf(assistant), limit_s):
        yield frame


async def until_idle(harness: Harness) -> Any:
    """Silent blocks until the window still open has closed by itself and its beep is over."""
    assistant, limit_s = harness.assistant, idle_limit_s(harness.settings)
    async for frame in silence_while(harness, lambda: not idle_and_quiet(assistant), limit_s):
        yield frame


async def _replay_scene(
    harness: Harness,
    planned: Sequence[Planned],
    before: Mark,
    stop_when: Callable[[], bool] | None = None,
) -> list[Ran]:
    """`stop_when`: asked before each clip after the first; True ends the scene there (the
    clips left are not fed)."""
    marks = [before]
    begun: list[int] = []  # the clips whose frames started to be fed
    stuck: dict[int, Reason] = {}  # the clip the scene stopped on, and why
    assistant = harness.assistant

    async def feed() -> Any:
        for n, part in enumerate(planned):
            if harness.system.stopping:  # "chiuditi" was said: this clip is never heard
                return
            if n > 0 and stop_when is not None and stop_when():
                return
            begun.append(1)
            if n > 0 and part.fire_at is not None:
                async for frame in until_idle(harness):
                    yield frame
                if not idle_and_quiet(assistant):
                    stuck[n] = WINDOW_STUCK
                    return
                marks[-1] = take_mark(harness)  # the window's closing row is not this clip's
            harness.wake.arm(part.fire_at)  # the wake fires where it fired live, or not at all
            with frozen_prompt(harness.stt, part.clip.prompt):
                async for frame in frames(harness, part.audio):
                    yield frame
                if part is not planned[-1]:
                    async for frame in end_phrase(harness):
                        yield frame
                    if phrase_open(assistant) or window_deaf(assistant):
                        stuck[n] = PHRASE_STUCK
                        return
            marks.append(take_mark(harness))

    await asyncio.wait_for(harness.assistant.run(feed()), CLIP_LIMIT_S * len(planned))
    reached, end = len(marks) - 1, take_mark(harness)  # the feed or run() may stop early
    rans = []
    for n in range(len(planned)):
        if n >= len(begun):
            rans.append(Ran(unfed_outcome(), False, False, False))
            continue
        last = n == len(planned) - 1  # the last one also reads what closed after its tail
        until = None if last or n == reached else marks[n + 1]
        stop = until or end
        outcome = read_outcome(harness, marks[n], until, scene=True)
        missed, tripped = stop.misses > marks[n].misses, stop.trips > marks[n].trips
        rans.append(Ran(outcome, missed, True, tripped, stuck.get(n)))
    return rans


def window_problem(n: int, fire_at: int | None, outcome: Outcome) -> Reason | None:
    """A clip after the first needs the conversation window open for real: a continuation must
    land in it (and be heard). A re-armed wake starts from idle and must fire."""
    if n == 0:
        return None
    if fire_at is None:
        if outcome.wake_word not in (CONVERSATION, FOLLOW_UP):
            return "finestra", "la finestra di conversazione non si è aperta"
        if outcome.outcome in ("open", "timeout"):  # the window of the clip before, nobody spoke
            return "finestra", "la finestra era aperta ma la clip non è stata sentita"
        return None
    if outcome.wake_word != "replay":
        return "wake", "la wake (replay) non è scattata, anche con la finestra chiusa"
    return None


def run_scene(harness: Harness, entries: Sequence[Entry], corpus: Path, result: GateResult) -> None:
    for item in entries:
        wav = corpus / item.clip
        if not wav.with_suffix(".json").is_file() and wav.is_file():
            result.no_note.append(item.clip)
    absent = [item.clip for item in entries if not (corpus / item.clip).is_file()]
    if absent:
        result.missing.extend(absent)
        for item in entries:
            if item.clip not in absent:
                result.failed.append(item.clip)
                msg = "scena incompleta: manca una clip"
                result.reasons[item.clip] = [(msg, msg)]
        return
    harness.fresh()
    before = take_mark(harness)
    planned: list[Planned] = []
    try:
        planned = plan_scene(harness, entries, corpus)
        if any(p.clip.problem for p in planned):
            raise ValueError(next(p.clip.problem for p in planned if p.clip.problem))
        rans = asyncio.run(_replay_scene(harness, planned, before))
    except Exception as exc:  # the scene is an error, the run goes on
        now = take_mark(harness)
        lost, tripped = now.misses > before.misses, now.trips > before.trips
        rans = [
            Ran(error_outcome(harness, exc), lost and n == 0, False, tripped and n == 0)
            for n in range(len(entries))
        ]
    if any(r.tripped for r in rans) and harness.client is not None:
        harness.client.discard(before.saved)  # what answered a false failure is not kept
    new_misses = harness.client.misses[before.misses :] if harness.client is not None else []
    result.misses.extend(k for k in dict.fromkeys(new_misses) if k not in result.misses)
    for n, (item, ran) in enumerate(zip(entries, rans, strict=True)):
        fire_at = planned[n].fire_at if planned else None
        hard: list[Reason] = []
        if planned and n == 0 and fire_at is None and not item.no_wake:
            msg = (
                "senza punto di wake nella riga dal vivo: non prova nulla (no_wake: true se voluto)"
            )
            hard.append(("no_wake", msg))
        if ran.missed:
            hard.append(CASSETTE_MISS)
        if ran.tripped:  # no tool_errors and no known_diff can absorb a real action tried
            hard.append(TRIPPED)
        if ran.stuck:
            hard.append(ran.stuck)
        elif ran.reached and (problem := window_problem(n, fire_at, ran.outcome)):
            hard.append(problem)
        judge(item, ran.outcome, result, hard, planned[n].live if planned else {})


PHRASE_STUCK: Reason = (
    "frase",
    "la frase non è finita entro il limite: la scena si ferma, le clip dopo non sono date",
)
WINDOW_STUCK: Reason = (
    "finestra",
    "la finestra non si è chiusa entro il limite: la wake di questa clip non è armata",
)
TRIPPED: Reason = (
    "tripwire",
    "il tripwire è scattato: un'azione reale è stata bloccata (con --record le cassette "
    "registrate in questa scena sono scartate)",
)
CASSETTE_MISS: Reason = (
    "cassetta mancante: rilancia con --record",
    "cassetta mancante: rilancia con --record",
)


def judge(
    item: Entry, outcome: Outcome, result: GateResult, hard: list[Reason], live: Mapping[str, Any]
) -> None:
    """`hard` are reasons no known_bug may absorb (an error, a miss, a window that stayed shut);
    only a difference listed in `known_diff` becomes `known`."""
    diffs = differences(item.expect, outcome)
    if outcome.claude_error:
        message = f"errore di Claude: {outcome.claude_error}"
        hard = [*hard, (message, message)]
    elif outcome.error:
        hard = [*hard, ("errore", f"errore: {outcome.outcome}")]
    outside = [d for d in diffs if d[0] not in item.known_diff]
    if item.known_bug and (hard or outside):
        changed = f"il fallimento è cambiato rispetto a {item.known_bug}"
        reasons = [*hard, *outside, *([("cambiato", changed)] if outside else [])]
        result.failed.append(item.clip)
    elif item.known_bug:
        reasons = diffs
        (result.known if diffs else result.fixed).append(item.clip)
    else:
        reasons = [*hard, *diffs]
        (result.failed if reasons else result.passed).append(item.clip)
    if reasons:
        result.reasons[item.clip] = reasons
    said, heard = str(live.get("text") or ""), outcome.text
    if live and said.strip() != heard.strip():
        result.text_diffs.append((item.clip, said, heard))


def check_real_wake(
    harness: Harness, entries: Sequence[Entry], corpus: Path, result: GateResult
) -> None:
    """Pass 2: the real wake word model, through the real second chance, with Claude off."""
    wanted = [e for e in entries if e.real_wake is not None and (corpus / e.clip).is_file()]
    if not wanted:
        return
    if harness.real_wake is None:
        raise RealActionBlocked("clip real_wake ma nessun rilevatore vero")
    replay_wake, harness.brain = harness.wake, False
    harness.wake = harness.real_wake
    try:
        for item in wanted:
            outcome = replay_clip(harness, load_clip(corpus / item.clip))
            if outcome.woke != item.real_wake or outcome.error:
                name = f"{item.clip} [real_wake]"
                result.failed.append(name)
                if outcome.error:
                    why = f"errore: {outcome.outcome}"
                elif item.real_wake:
                    why = "la wake word vera non scatta più"
                else:
                    why = "la wake word vera ha svegliato una frase che doveva restare scartata"
                result.reasons[name] = [(why, why)]
    finally:
        harness.wake, harness.brain = replay_wake, True


def run_gate(manifest: Sequence[Entry], harness: Harness, corpus: Path = CORPUS) -> GateResult:
    if not manifest:
        raise ManifestError("il manifest è vuoto: un gate senza clip non prova nulla")
    require_recorder(harness)
    result = GateResult()
    for scene in group_scenes(manifest):
        run_scene(harness, scene, corpus, result)
    check_real_wake(harness, manifest, corpus, result)
    return result


def render_report(result: GateResult, stt_line: str, model: str) -> str:
    """The report for samples/gate/report.md (ignored by git): it holds what was said."""
    rows = [
        ("superata", result.passed),
        ("fallita", result.failed),
        ("nota (known_bug)", result.known),
        ("corretta (togli il segno)", result.fixed),
        ("clip mancante", result.missing),
    ]
    lines = [
        "# Gate di replay",
        "",
        f"{stt_line}; Claude: {model}",
        "",
        "| Esito | Clip |",
        "|---|---|",
    ]
    lines += [f"| {name} | {clip} |" for name, clips in rows for clip in clips]
    lines += [f"| cassetta mancante | {key} |" for key in result.misses]
    if result.reasons:
        lines += [
            "",
            "## Differenze di esito",
            *(f"- {c}: {'; '.join(d for _, d in r)}" for c, r in result.reasons.items()),
        ]
    if result.text_diffs:
        lines += ["", "## Testo diverso dal vivo (non blocca)"]
        lines += [f"- {c}: dal vivo {a!r}, replay {b!r}" for c, a, b in result.text_diffs]
    if result.no_note:
        lines += [
            "",
            "## Clip senza nota (Whisper non congelato)",
            *(f"- {c}" for c in result.no_note),
        ]
    lines += ["", release_line(result)]
    return "\n".join(lines) + "\n"


def release_line(result: GateResult) -> str:
    """One line for the release notes: counts only."""
    return (
        f"Gate di replay: {len(result.passed)} superate, {len(result.failed)} fallite, "
        f"{len(result.known)} note, {len(result.fixed)} corrette da togliere, "
        f"{len(result.misses)} cassette mancanti, {len(result.missing)} clip mancanti."
    )


def print_result(result: GateResult) -> None:
    """Clip names, the keys that differ and fixed messages: nothing that was said."""
    for clip, reasons in result.reasons.items():
        print(f"  {clip}: {'; '.join(public for public, _ in reasons)}")
    for clip in result.missing:
        print(f"  {clip}: clip mancante")
    for clip in result.no_note:
        print(f"  {clip}: senza nota (Whisper non congelato)")
    for clip in result.fixed:
        print(f"  {clip}: era un bug noto e ora passa, togli known_bug")
    print(release_line(result))


def upper_bound_usd(requests: int, max_chars: int, settings: Settings, pricing: Any) -> float:
    """A ceiling for `requests` requests of `max_chars` characters: each priced at the dearest
    model it can reach (the main model, or Sonnet when it hands over) with the output limit
    the brain really sends (`reply_cap`, with tools), at the dearest cache write and 2
    characters per token, for every round of the request ([pc] max_rounds)."""
    from claudio.brain.claude import reply_cap
    from claudio.cost.pricing import PricingError, Usage

    claude = settings.claude
    models = [(claude.model, reply_cap(claude, settings.pc, with_tools=True))]
    if claude.smart_model and claude.escalate and claude.smart_model != claude.model:
        models.append((claude.smart_model, claude.smart_max_tokens))
    tokens = math.ceil(max_chars / TOKEN_CHARS_FLOOR)
    dearest = 0.0
    for model, cap in models:
        try:
            dearest = max(
                dearest, pricing.cost(model, Usage(cache_write_1h_tokens=tokens, output_tokens=cap))
            )
        except PricingError:
            continue  # a model without prices is never called
    return float(requests * settings.pc.max_rounds * dearest)


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("deve essere almeno 1")
    return value


def run_clips(harness: Harness, wavs: list[Path], repetitions: int, show_text: bool) -> bool:
    """True if any clip ended in an error."""
    failed = False
    for wav in wavs:
        outcomes = run_clip(harness, load_clip(wav), repetitions, show_text)
        failed = failed or any(o.error for o in outcomes)
    return failed


def real_client(settings: Settings) -> Any:
    from claudio.brain.claude import make_client

    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    if not key.strip():
        raise SystemExit("Manca ANTHROPIC_API_KEY nel file .env: non posso registrare.")
    claude = settings.claude
    return make_client(key.strip(), claude.timeout_s, claude.keepalive_s)


def recording_requests(manifest: Sequence[Entry], result: GateResult) -> int:
    """Requests a first pass may hide: a miss makes its clip an error, no window opens, and
    the later clips of its scene send nothing."""
    total = 0
    for scene in group_scenes(manifest):
        for n, item in enumerate(scene):
            if any(r == CASSETTE_MISS for r in result.reasons.get(item.clip, [])):
                total += len(scene) - n
                break
    return total


def confirm_recording(
    harness: Harness, manifest: Sequence[Entry], cassettes: Path, corpus: Path, budget: float
) -> bool:
    """A first pass with no key finds the requests nobody recorded; then JenJo says `si`."""
    probe = CassetteClient(cassettes)
    harness.client = probe
    result = run_gate([dataclasses.replace(e, real_wake=None) for e in manifest], harness, corpus)
    keys = list(dict.fromkeys(probe.misses))
    if not keys:
        print("Niente da registrare.")
        return True
    requests = recording_requests(manifest, result)
    biggest = max(probe.sizes.values())
    bound = upper_bound_usd(requests, biggest, harness.settings, harness.tracker.pricing)
    print(
        f"Da registrare: almeno {len(keys)} cassette; fino a {requests} richieste, ognuna con "
        f"fino a {harness.settings.pc.max_rounds} giri. Costo massimo: {bound:.2f} $, al prezzo "
        f"del modello più caro raggiungibile e con la richiesta più grande vista ({biggest} "
        "caratteri). Non copre: il testo che gli strumenti riportano nei giri successivi "
        "(finestre, file), la cronologia che cresce, le chiamate di ipotesi non viste. Il freno "
        f"smette di inviare a {budget:.2f} $ per corsa: controlla prima di ogni invio, quindi "
        "l'ultima richiesta lo puo' superare, e una richiesta tagliata non e' contata; riparte "
        "da zero a ogni corsa e non conosce i giorni. Al primo errore dell'API la registrazione "
        "si ferma: quello che e' finito resta salvato."
    )
    prompt = 'Scrivi "si" per spendere (la spesa non entra in data/spend.db): '
    return input(prompt).strip() == "si"


def print_spend(client: CassetteClient) -> None:
    print(
        f"Speso davvero in questa corsa (solo le risposte registrate; non entra in "
        f"data/spend.db): {client.spent:.4f} $"
    )
    if client.failure:
        print(
            f"Registrazione fermata dal primo errore dell'API ({client.failure}), dopo i tentativi "
            "dell'SDK: nessun'altra richiesta e' partita. Una richiesta tagliata non e' contata: "
            "la spesa vera puo' essere un po' piu' alta."
        )
    if client.braked:
        print(
            f"{'Errore' if client.failure else 'Freno di spesa'}: {len(client.braked)} richieste "
            f"non registrate ({', '.join(k[:12] for k in client.braked)}); rilancia --record: "
            "quello che e' finito resta salvato."
        )


def remove_work_dir(work_dir: Path) -> None:
    """What was heard stays out of %TEMP%: all that can go, goes, and what cannot is named."""
    shutil.rmtree(work_dir, ignore_errors=True)
    if work_dir.exists():
        print(
            f"ATTENZIONE: non riesco a cancellare {work_dir}: puo' contenere quello che e' stato "
            "sentito. Cancellala a mano."
        )


def require_personality(settings: Settings) -> None:
    """The gate reads data/CLAUDIO.md and never writes it: personality.py would create it."""
    claude = settings.claude
    path = settings.resolve(claude.personality_file)
    if claude.personality_max_chars > 0 and not path.exists():
        raise GateRefused(f"{path} non esiste: il gate non la crea (copia config/CLAUDIO.md).")


def arm_process() -> None:
    """The libraries first, then the tripwire: one that starts a process while loading must not
    meet it."""
    import ctranslate2  # noqa: F401
    import faster_whisper  # noqa: F401
    import onnxruntime  # noqa: F401
    import openwakeword  # noqa: F401

    install_tripwire()


def gate_main(record: bool, report: bool, manifest_path: Path = MANIFEST) -> int:
    try:
        manifest = load_manifest(manifest_path)
        settings = load_settings()
        require_personality(settings)
    except (OSError, ManifestError, GateRefused) as exc:
        print(f"manifest non valido: {exc}" if isinstance(exc, ManifestError) else exc)
        return 1
    arm_process()
    work_dir = Path(tempfile.mkdtemp(prefix="claudio-gate-"))
    harness: Harness | None = None
    client = CassetteClient(CASSETTES)
    try:
        real_wake = None
        if any(e.real_wake is not None for e in manifest):
            from claudio.wakeword.detector import OpenWakeWordDetector

            models = settings.resolve(settings.models_dir)
            real_wake = OpenWakeWordDetector(
                settings.wakeword, models / "wakeword", settings.audio.block_size
            )
        harness = build_harness(
            settings, work_dir, wake=GateWake(), client=client, real_wake=real_wake
        )
        print(whisper_line(harness.stt))
        if record:
            if not confirm_recording(
                harness, manifest, CASSETTES, CORPUS, settings.cost.daily_budget
            ):
                return 1
            client = CassetteClient(
                CASSETTES,
                real=real_client(settings),
                pricing=harness.tracker.pricing,
                budget=settings.cost.daily_budget,  # real money keeps a brake
            )
        harness.client = client
        result = run_gate(manifest, harness)
    except (RealActionBlocked, ManifestError) as exc:
        print(exc)
        return 1
    finally:  # the spend is told even on Ctrl+C
        if record:
            print_spend(client)
        if harness is not None:
            harness.tracker.ledger.close()
        remove_work_dir(work_dir)
    print_result(result)
    if report:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            render_report(result, whisper_line(harness.stt), settings.claude.model),
            encoding="utf-8",
        )
        print(f"Rapporto: {REPORT}")
    return result.exit_code()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--determinism", type=positive, default=1, metavar="N")
    parser.add_argument("--clips", type=Path, nargs="+", help="determinism mode: these WAVs")
    parser.add_argument("--show-text", action="store_true", help="print what was said (private)")
    parser.add_argument("--paced", action="store_true", help="control: frames every 80 ms, no wait")
    parser.add_argument("--record", action="store_true", help="record the missing replies (paid)")
    parser.add_argument(
        "--report",
        action="store_true",
        help="write samples/gate/report.md (it holds what was said)",
    )
    args = parser.parse_args(argv)
    if args.clips is None:
        return gate_main(args.record, args.report)
    arm_process()
    settings = load_settings()
    work_dir = Path(tempfile.mkdtemp(prefix="claudio-gate-"))
    harness = build_harness(settings, work_dir, paced=args.paced)
    print(whisper_line(harness.stt))
    print(f"cartella di lavoro: {work_dir}{', paced' if args.paced else ''}")
    try:
        failed = run_clips(harness, args.clips, args.determinism, args.show_text)
    finally:
        harness.tracker.ledger.close()
        remove_work_dir(work_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
