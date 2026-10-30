"""Jack's voice: a worker process (Qwen3-TTS + RVC, see jack_worker.py) behind the same
synthesize()/sample_rate interface as Kokoro and Piper.

The worker has its own venv (torch, Applio), so Claudio never imports them. If it cannot
start, or dies, or is too slow for a sentence, that sentence is spoken by `fallback` and the
worker is restarted in the background; Claudio always answers.

With a `judge`, every take is checked before it is played and a bad one is generated again
with the next seed: a piece that opens a reply on its first FIRST_CHECK_S, the next ones
whole while the piece before them plays (held back at most until it is about to end). With
`takes`, a text keeps the seed of its good take, and a fixed reply saved in advance is
played from its WAV with no TTS at all."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import queue
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from claudio.i18n import tr
from claudio.tts.spoken_text import to_spoken_it
from claudio.tts.takes import TakeStore
from claudio.tts.voice_check import TakeJudge, TakeVerdict

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
RESTART_COOLDOWN_S = 30.0
# rvc "auto": a sentence asked for after this long without speaking opens a reply, so it is
# streamed in blocks (first audio ~0.3 s); the next ones are generated while it plays, so
# they can go through RVC whole (clearer) without a pause.
REPLY_GAP_S = 1.0
CREATE_NO_WINDOW = 0x08000000  # Windows: no console window for the worker
# The take check. A piece with less than OPENING_S still to play before it opens a reply:
# it is judged on its first FIRST_CHECK_S (one Qwen3 chunk), up to FIRST_CHECK_MAX_S when
# that has too little voice to tell, and generated again at most once (+0.3-0.5 s, JenJo
# 30/10). A later piece is held until it is whole, or until what plays before it has
# RELEASE_MARGIN_S left, and generated again only with RETRY_MIN_S still to play.
OPENING_S = 0.3
FIRST_CHECK_S = 0.6
FIRST_CHECK_MAX_S = 1.3
RELEASE_MARGIN_S = 0.4
RETRY_MIN_S = 1.0


@dataclass
class _Take:
    """One generation of a piece: what was played of it and how it was judged."""

    seed: int | None
    played: list[np.ndarray] = field(default_factory=list)
    raw: dict[str, float] = field(default_factory=dict)
    verdict: TakeVerdict | None = None
    finished: bool = False  # the worker said "done"
    cancelled: bool = False  # ...after an interrupt
    rejected: bool = False  # thrown away before any of it was played

    @property
    def released(self) -> bool:
        return bool(self.played)


def _raw(event: dict[str, Any]) -> dict[str, float]:
    """The take's own level as the worker reports it (absent with the level guard off)."""
    return {k: v for k, v in event.items() if k.startswith("raw_") and v is not None}


class WorkerError(RuntimeError):
    pass


def worker_command(models: Path, tts: Any, **overrides: Any) -> list[str]:
    """The worker's command line from the [tts] settings (paths relative to `models`);
    `overrides` replace single settings, e.g. jack_checkpoint=... for a comparison."""

    def get(name: str) -> Any:
        return overrides.get(name, getattr(tts, name))

    return [
        str(models / get("jack_python")),
        str(Path(__file__).with_name("jack_worker.py")),
        "--checkpoint",
        str(models / get("jack_checkpoint")),
        "--rvc-model",
        str(models / get("jack_rvc_model")),
        "--applio",
        str(models / get("jack_applio")),
        "--speed",
        str(get("jack_speed")),
        "--temperature",
        str(get("jack_temperature")),
        "--seed",
        str(get("jack_seed")),
        "--top-p",
        str(get("jack_top_p")),
        "--lufs",
        str(get("jack_lufs")),
        *(["--pitch-fix"] if get("jack_pitch_fix") else []),
    ]


class JackSynth:
    """`command` starts the worker; `fallback` (any synth) speaks when the worker cannot."""

    def __init__(
        self,
        command: list[str],
        fallback: Any,
        startup_timeout_s: float = 90.0,
        sentence_timeout_s: float = 15.0,
        language: str = "Italian",
        rvc: str = "blocks",
        words: dict[str, str] | None = None,
        first_wait_s: float = 30.0,
        monitor: Any = None,
        judge: TakeJudge | None = None,
        takes: TakeStore | None = None,
        seed: int = -1,
        retries: int = 2,
    ) -> None:
        self.monitor = monitor  # VoiceMonitor: every piece said is saved and its pitch logged
        self.judge = judge  # None: every take is played as it comes
        self.takes = takes  # None: no seed per text, no saved takes
        self.seed = seed  # the first seed tried for a new text (-1: a random one)
        self.retries = retries  # more takes of a piece at most (the opening one: 1)
        # When the audio handed out so far ends, at the earliest (it plays after it is
        # handed out): how long a held take may wait without leaving a gap.
        self._horizon = 0.0
        self.words = dict(words or {})  # respellings: "file" -> "fàil" (see spoken_text)
        self.command = command
        # Told to Qwen3. Always Italian (JenJo, 24/10): "Auto" guessed the language per
        # sentence and drifted to Spanish/English accents and stresses.
        self.language = language
        self.rvc = rvc  # "blocks", "sentence" or "auto" (see REPLY_GAP_S)
        self._last_done = 0.0
        self.fallback = fallback
        self.startup_timeout_s = startup_timeout_s
        # A sentence spoken while the worker is still loading waits this long for it, so
        # that Claudio speaks with Jack's voice from the first reply (0 = fallback at once).
        self.first_wait_s = first_wait_s
        self.sentence_timeout_s = sentence_timeout_s
        self.language = language  # told to Qwen3: "Auto" or "Italian"
        # Read by StreamingSpeaker right after synthesize(): the rate of the voice that spoke.
        self.sample_rate = SAMPLE_RATE
        self.load_s = 0.0
        self.vram_mb = 0
        self.last_first_ms: int | None = None  # worker-side timing of the last sentence
        self.last_total_ms = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._ready = False  # models loaded: only then does synthesize() use the worker
        self._events: queue.Queue[tuple[dict[str, Any], bytes] | None] = queue.Queue()
        self._send_lock = threading.Lock()
        self._next_id = 0
        self._restart_at = 0.0
        self._restarting = False
        self._started = threading.Event()  # set = no first start in progress
        self._started.set()
        self._abort_wait = threading.Event()  # "basta" while waiting for the first start
        atexit.register(self.close)

    # -- process ----------------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker and wait until its models are loaded. Raises WorkerError."""
        self._ready = False
        self._events = queue.Queue()
        flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=flags,
            )
        except OSError as exc:
            raise WorkerError(tr("worker Jack non avviabile: {error}", error=exc)) from exc
        self._proc = proc
        threading.Thread(target=self._read_stdout, args=(proc, self._events), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True).start()
        try:
            event = self._next_event(self.startup_timeout_s)
        except WorkerError:
            self.close()
            raise
        if event.get("event") != "ready":
            self.close()
            raise WorkerError(tr("worker Jack: {error}", error=event.get("message", event)))
        self.sample_rate = int(event.get("sample_rate", SAMPLE_RATE))
        self.load_s = float(event.get("load_s", 0.0))
        self.vram_mb = int(event.get("vram_mb", 0))
        self._ready = True
        log.info("Worker Jack pronto in %.1f s (VRAM GPU %d MB)", self.load_s, self.vram_mb)

    def start_in_background(
        self, on_done: Callable[[WorkerError | None], None] | None = None
    ) -> None:
        """Start the worker without waiting for it: its models load (8-30 s) while Claudio
        loads the rest and listens. `on_done` gets None once it is ready, or the error. A
        worker that fails here is not retried: its fallback speaks, as with start()."""
        self._restarting = True  # a sentence meanwhile must not start a second worker
        self._started.clear()

        def run() -> None:
            error: WorkerError | None = None
            try:
                self.start()
            except WorkerError as exc:
                error = exc
                self._restart_at = float("inf")
            finally:
                self._restarting = False
                self._started.set()
            if on_done is not None:
                on_done(error)

        threading.Thread(target=run, daemon=True, name="jack-start").start()

    @property
    def starting(self) -> bool:
        return not self._started.is_set()

    def wait_started(self, timeout: float | None = None) -> bool:
        """Wait until a background start has ended (ready or failed); False on timeout."""
        return self._started.wait(timeout)

    def close(self) -> None:
        proc, self._proc, self._ready = self._proc, None, False
        if proc is None:
            return
        with contextlib.suppress(Exception):
            self._send({"op": "quit"}, proc)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.stdin:
            with contextlib.suppress(OSError):
                proc.stdin.close()  # flushing into a dead worker's pipe raises

    def alive(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    def _read_stdout(self, proc: subprocess.Popen[bytes], events: queue.Queue[Any]) -> None:
        """Header lines, each `audio` one followed by its PCM bytes. None = worker is gone."""
        assert proc.stdout is not None
        try:
            while line := proc.stdout.readline():
                header = json.loads(line)
                payload = b""
                if header.get("event") == "audio":
                    payload = proc.stdout.read(int(header["bytes"]))
                events.put((header, payload))
        except (OSError, ValueError) as exc:
            log.warning("Worker Jack: protocollo rotto (%s)", exc)
        finally:
            events.put(None)

    def _read_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            log.debug("%s", "jack-worker: " + raw.decode("utf-8", "replace").rstrip())

    def _send(self, message: dict[str, Any], proc: subprocess.Popen[bytes] | None = None) -> None:
        proc = proc or self._proc
        if proc is None or proc.stdin is None:
            raise WorkerError("worker Jack non attivo")
        try:
            with self._send_lock:
                proc.stdin.write(json.dumps(message).encode() + b"\n")
                proc.stdin.flush()
        except OSError as exc:
            raise WorkerError(f"worker Jack: {exc}") from exc

    def _next_event(self, timeout: float) -> dict[str, Any]:
        """Next event as a dict (audio events carry their bytes under "pcm")."""
        event = self._poll(timeout)
        if event is None:
            raise WorkerError(
                tr("worker Jack: nessuna risposta in {seconds} s", seconds=round(timeout))
            )
        return event

    def _poll(self, timeout: float) -> dict[str, Any] | None:
        """Like _next_event(), but None when nothing came within `timeout`."""
        try:
            item = self._events.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return None
        if item is None:
            raise WorkerError(tr("worker Jack terminato"))
        header, payload = item
        return {**header, "pcm": payload} if payload else header

    # -- synth interface --------------------------------------------------------------------

    def cancel(self, upto: int | None = None) -> None:
        """ "basta": stop the sentences sent so far, or only up to `upto` (the audio already
        sent is dropped). A sentence asked for after this call is never cancelled by it, even
        when the message reaches the worker late."""
        if upto is None:  # a late cancel of one old sentence must not abort the next one
            self._abort_wait.set()
        if self.alive():
            with contextlib.suppress(WorkerError):
                self._send({"op": "interrupt", "upto": self._next_id if upto is None else upto})

    def synthesize(self, text: str) -> np.ndarray:
        """The whole sentence at once (see stream() for audio as it is generated)."""
        chunks = list(self.stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """int16 chunks as the worker produces them; the first one arrives in ~0.3 s.
        A worker that fails before any audio leaves the sentence to the fallback voice."""
        spoken = False
        saved = self._saved(text)
        if saved is not None:  # no worker needed, not even a loaded one
            yield from self._play_saved(text, *saved)
            return
        if self.starting and not self._wait_for_start():
            return  # "basta" while waiting: nothing to say
        if self.alive():
            try:
                self.sample_rate = SAMPLE_RATE
                for chunk in self._speak(text):
                    spoken = True
                    yield chunk
                return
            except WorkerError as exc:
                log.warning("Jack non ha risposto (%s): uso la voce di ripiego", exc)
                self.close()
        self._schedule_restart()
        if spoken:  # cut in the middle: a second voice would repeat what was already said
            return
        audio = self.fallback.synthesize(text)
        self.sample_rate = int(self.fallback.sample_rate)
        yield audio

    def _wait_for_start(self) -> bool:
        """Wait up to first_wait_s for the worker still loading. False if cancelled."""
        self._abort_wait.clear()
        deadline = time.monotonic() + self.first_wait_s
        while not self._started.wait(0.05):
            if self._abort_wait.is_set():
                return False
            if time.monotonic() >= deadline:
                log.info("Jack si sta ancora caricando: questa frase la dice la voce di riserva")
                break
        return True

    # -- takes ----------------------------------------------------------------------------

    def _spoken(self, text: str) -> str:
        # Digits, symbols, acronyms and English words become Italian spelling: left alone,
        # the model guesses them, sometimes in Spanish or English.
        return to_spoken_it(text, self.words)

    def _saved(self, text: str) -> tuple[np.ndarray, int] | None:
        """The take of this fixed reply saved in advance (scripts/jack_takes.py), if any."""
        if self.takes is None:
            return None
        return self.takes.audio(self.takes.key(self._spoken(text), self._rvc_for_next()))

    def _play_saved(self, text: str, pcm: np.ndarray, rate: int) -> Iterator[np.ndarray]:
        self.sample_rate = rate
        step = max(1, rate // 2)  # half a second at a time, like the worker's chunks
        for start in range(0, len(pcm), step):
            yield self._handed_out(pcm[start : start + step])
        self._last_done = time.monotonic()
        if self.monitor is not None:
            self.monitor.check(text, pcm, rate, spoken=self._spoken(text))

    def _handed_out(self, chunk: np.ndarray) -> np.ndarray:
        """Every chunk given to the speaker: it plays after what came before it."""
        now = time.monotonic()
        self._horizon = max(self._horizon, now) + len(chunk) / self.sample_rate
        return chunk

    def _seeds(self, key: str | None, count: int) -> list[int | None]:
        """The seeds to try, in order: the one of this text's good take if there is one, then
        the configured one and the ones after it (random ones with seed -1)."""
        if self.judge is None and self.takes is None:
            return [None]  # the worker's own --seed, as before
        known = self.takes.seed(key) if self.takes is not None and key is not None else None
        if self.seed >= 0:
            fresh = [self.seed + i for i in range(count + 1)]
        else:
            fresh = [random.randrange(2**31) for _ in range(count + 1)]
        seeds = [known] if known is not None else []
        seeds += [s for s in fresh if s != known]
        return list(seeds[:count])

    def _speak(self, text: str) -> Iterator[np.ndarray]:
        spoken = self._spoken(text)
        rvc = self._rvc_for_next()
        key = self.takes.key(spoken, rvc) if self.takes is not None else None
        opening = self._horizon - time.monotonic() < OPENING_S
        retries = min(self.retries, 1) if opening else self.retries
        seeds = self._seeds(key, 1 + (retries if self.judge is not None else 0))
        take = _Take(None)
        for attempt, seed in enumerate(seeds):
            last = attempt == len(seeds) - 1
            take = yield from self._take(spoken, rvc, seed, opening, last)
            if not take.rejected:
                break
            reasons = ", ".join(take.verdict.reasons) if take.verdict else ""
            log.info("Take di Jack scartata (seme %s): %s - %s", seed, reasons, text[:50])
        if take.cancelled or not take.played:
            return
        verdict = take.verdict
        if verdict is not None and not verdict.ok:
            reasons = ", ".join(verdict.reasons)
            log.info("Take di Jack suonata anche se non buona (%s) - %s", reasons, text[:50])
        elif verdict is not None and key is not None and take.seed is not None and self.takes:
            self.takes.remember(key, take.seed, text)
        if self.monitor is not None:
            audio = np.concatenate(take.played)
            self.monitor.check(text, audio, self.sample_rate, raw=take.raw, spoken=spoken)

    def _take(
        self, spoken: str, rvc: str, seed: int | None, opening: bool, last: bool
    ) -> Generator[np.ndarray, None, _Take]:
        """One generation of the piece. Its audio is held while it is judged and handed out
        once it passes (or there is no time, or no seed, left to do better); a bad take is
        dropped before anything of it is played (`rejected`)."""
        self._next_id += 1
        rid = self._next_id
        request: dict[str, Any] = {
            "op": "speak",
            "id": rid,
            "text": spoken,
            "language": self.language,
            "rvc": rvc,
        }
        if seed is not None:
            request["seed"] = seed
        self._send(request)
        take = _Take(seed)
        held: list[np.ndarray] = []
        # Held back only while another take could still replace it: with no check, no seed
        # left or no time left, every chunk goes out as it comes.
        hold = self.judge is not None and self._may_retry(opening, last)
        judged = not hold  # the early look at its start is done
        deadline = time.monotonic() + self.sentence_timeout_s
        try:
            while True:
                release_at = self._horizon - RELEASE_MARGIN_S
                if held and not opening and time.monotonic() >= release_at:
                    # what plays before it is about to end: out it goes, judged or not
                    yield from self._release(held, take)
                now = time.monotonic()
                wait = deadline - now
                if held and not opening:
                    wait = min(wait, release_at - now)
                event = self._poll(wait)
                if event is None:
                    if time.monotonic() >= deadline:
                        limit = round(self.sentence_timeout_s)
                        raise WorkerError(
                            tr("worker Jack: nessuna risposta in {seconds} s", seconds=limit)
                        )
                    continue
                if event.get("id") != rid:
                    continue  # leftovers of an interrupted sentence
                kind = event.get("event")
                if kind == "audio":
                    # A reply said in one piece can last longer than the timeout: it counts
                    # from the last audio, so only a worker that stops producing fails.
                    deadline = time.monotonic() + self.sentence_timeout_s
                    take.raw = _raw(event) or take.raw
                    chunk = np.frombuffer(event["pcm"], dtype="<i2").astype(np.int16)
                    if take.released or not hold:
                        take.played.append(chunk)
                        yield self._handed_out(chunk)
                        continue
                    held.append(chunk)
                    seconds = sum(len(c) for c in held) / self.sample_rate
                    if judged or seconds < FIRST_CHECK_S:
                        continue
                    verdict = self._judge(held, spoken, take.raw, complete=False)
                    if not verdict.ok and self._may_retry(opening, last):
                        take.verdict, take.rejected = verdict, True
                        return take
                    if not opening or verdict.stats.known or seconds >= FIRST_CHECK_MAX_S:
                        judged = True
                    if opening and judged:
                        take.verdict = verdict
                        yield from self._release(held, take)
                elif kind == "done":
                    self.last_first_ms = event.get("first_ms")
                    self.last_total_ms = int(event.get("total_ms", 0))
                    take.finished = True
                    self._last_done = time.monotonic()
                    if event.get("cancelled"):  # "basta": nothing more of it is played
                        take.cancelled = True
                        return take
                    take.raw = _raw(event) or take.raw
                    whole = held if not take.released else take.played
                    if whole and self.judge is not None:
                        take.verdict = self._judge(whole, spoken, take.raw, complete=True)
                        bad = not take.released and not take.verdict.ok
                        if bad and self._may_retry(opening, last):
                            take.rejected = True
                            return take
                    yield from self._release(held, take)
                    return take
                elif kind == "error":
                    raise WorkerError(event.get("message", "errore"))
        finally:
            if not take.finished:
                # Abandoned half way (barge-in, error, a bad take): stop generating. Only this
                # sentence: an abandoned stream may be closed late, after the next reply has
                # started (JenJo 25/10: "che ore sono?" after a takeover was never heard).
                self.cancel(upto=rid)

    def generate_take(
        self, text: str, seed: int, rvc: str | None = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        """One whole take of `text` with this seed, not judged, and its level before the
        level guard (scripts/jack_takes.py picks the best of several). Raises WorkerError."""
        rvc = rvc or ("blocks" if self.rvc == "auto" else self.rvc)
        self.sample_rate = SAMPLE_RATE
        take = self._take(self._spoken(text), rvc, seed, opening=True, last=True)
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(next(take))
            except StopIteration as done:
                raw = done.value.raw
                break
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return audio, raw

    def _judge(
        self, chunks: list[np.ndarray], spoken: str, raw: dict[str, float], complete: bool
    ) -> TakeVerdict:
        assert self.judge is not None
        audio = np.concatenate(chunks)
        return self.judge.judge(audio, self.sample_rate, spoken, raw, complete)

    def _may_retry(self, opening: bool, last: bool) -> bool:
        """Another take can still be tried without a pause the listener would hear."""
        if last:
            return False
        return opening or self._horizon - time.monotonic() > RETRY_MIN_S

    def _release(self, held: list[np.ndarray], take: _Take) -> Iterator[np.ndarray]:
        while held:
            chunk = held.pop(0)
            take.played.append(chunk)
            yield self._handed_out(chunk)

    def _rvc_for_next(self) -> str:
        if self.rvc != "auto":
            return self.rvc
        return "blocks" if time.monotonic() - self._last_done > REPLY_GAP_S else "sentence"

    def _schedule_restart(self) -> None:
        """After a failure, try the worker again in the background, at most every 30 s."""
        if self._restarting or time.monotonic() < self._restart_at:
            return
        self._restarting = True

        def restart() -> None:
            try:
                self.start()
            except WorkerError as exc:
                log.warning("Riavvio del worker Jack fallito: %s", exc)
            finally:
                self._restart_at = time.monotonic() + RESTART_COOLDOWN_S
                self._restarting = False

        threading.Thread(target=restart, daemon=True).start()
