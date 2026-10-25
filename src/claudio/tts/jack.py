"""Jack's voice: a worker process (Qwen3-TTS + RVC, see jack_worker.py) behind the same
synthesize()/sample_rate interface as Kokoro and Piper.

The worker has its own venv (torch, Applio), so Claudio never imports them. If it cannot
start, or dies, or is too slow for a sentence, that sentence is spoken by `fallback` and the
worker is restarted in the background; Claudio always answers."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from claudio.i18n import tr
from claudio.tts.number_words import digits_to_words_it

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
RESTART_COOLDOWN_S = 30.0
CREATE_NO_WINDOW = 0x08000000  # Windows: no console window for the worker


class WorkerError(RuntimeError):
    pass


class JackSynth:
    """`command` starts the worker; `fallback` (any synth) speaks when the worker cannot."""

    def __init__(
        self,
        command: list[str],
        fallback: Any,
        startup_timeout_s: float = 90.0,
        sentence_timeout_s: float = 15.0,
        language: str = "Auto",
        first_wait_s: float = 30.0,
    ) -> None:
        self.command = command
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
        try:
            item = self._events.get(timeout=timeout)
        except queue.Empty:
            raise WorkerError(
                tr("worker Jack: nessuna risposta in {seconds} s", seconds=round(timeout))
            ) from None
        if item is None:
            raise WorkerError(tr("worker Jack terminato"))
        header, payload = item
        return {**header, "pcm": payload} if payload else header

    # -- synth interface --------------------------------------------------------------------

    def cancel(self) -> None:
        """ "basta": stop the sentence being generated (the audio already sent is dropped)."""
        self._abort_wait.set()
        if self.alive():
            with contextlib.suppress(WorkerError):
                self._send({"op": "interrupt"})

    def synthesize(self, text: str) -> np.ndarray:
        """The whole sentence at once (see stream() for audio as it is generated)."""
        chunks = list(self.stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """int16 chunks as the worker produces them; the first one arrives in ~0.3 s.
        A worker that fails before any audio leaves the sentence to the fallback voice."""
        spoken = False
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

    def _speak(self, text: str) -> Iterator[np.ndarray]:
        self._next_id += 1
        rid = self._next_id
        # Digits are spelled out: with language "Auto" a bare "22" or "51" is sometimes read in
        # Spanish.
        spoken_text = digits_to_words_it(text)
        self._send({"op": "speak", "id": rid, "text": spoken_text, "language": self.language})
        deadline = time.monotonic() + self.sentence_timeout_s
        finished = False
        try:
            while True:
                event = self._next_event(max(deadline - time.monotonic(), 0.01))
                if event.get("id") != rid:
                    continue  # leftovers of an interrupted sentence
                kind = event.get("event")
                if kind == "audio":
                    yield np.frombuffer(event["pcm"], dtype="<i2").astype(np.int16)
                elif kind == "done":
                    self.last_first_ms = event.get("first_ms")
                    self.last_total_ms = int(event.get("total_ms", 0))
                    finished = True
                    return
                elif kind == "error":
                    raise WorkerError(event.get("message", "errore"))
        finally:
            if not finished:
                self.cancel()  # abandoned half way (barge-in, error): stop generating

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
