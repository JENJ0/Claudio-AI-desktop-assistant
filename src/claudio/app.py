"""Orchestrator and state machine:
IDLE -> LISTENING -> TRANSCRIBING -> RESPONDING -> SPEAKING -> IDLE."""

from __future__ import annotations

import asyncio
import enum
import functools
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np

from claudio.audio.ring_buffer import RingBuffer
from claudio.audio.vad import Endpointer, EndpointEvent
from claudio.metrics import Interaction, MetricsWriter
from claudio.protocols import LevelControl, Responder, Speaker, Transcriber, WakeWordDetector
from claudio.router.normalizer import heard_wake_word, only_wake_words
from claudio.system import SystemControl

log = logging.getLogger(__name__)

PREWAKE_S = 1.5  # audio kept before the wake word: one-breath phrases and the saved WAV
HELD_FRAMES = 100  # frames kept while transcribing (~8 s), replayed if we listen again
LEAD_IN_S = PREWAKE_S  # one breath: how much of the audio before the detection Whisper gets
QUIET_SPEECH_DBFS = -30.0  # loudest 32 ms of a phrase below this: the microphone is too quiet


class State(enum.StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    RESPONDING = "responding"
    SPEAKING = "speaking"


class Assistant:
    def __init__(
        self,
        wake: WakeWordDetector,
        endpointer: Endpointer,
        transcriber: Transcriber,
        responder: Responder,
        speaker: Speaker,
        metrics: MetricsWriter | None = None,
        clock: Callable[[], float] = time.perf_counter,
        audio_sink: Callable[[np.ndarray, str], str | None] | None = None,
        level: LevelControl | None = None,
        system: SystemControl | None = None,
        second_chance: Callable[[np.ndarray], str] | None = None,
        retry_language: str | None = None,
    ) -> None:
        self.wake = wake
        self.endpointer = endpointer
        self.transcriber = transcriber
        self.responder = responder
        self.speaker = speaker
        self.metrics = metrics
        self.clock = clock
        self.audio_sink = audio_sink  # saves what was heard; returns the file name
        self.level = level  # automatic microphone gain; None = only warn when it is too quiet
        self.system = system or SystemControl()  # pause and "spegniti", shared with the router
        # Hears the last 1.5 s after a near miss (a score just under the threshold): if the
        # text has the wake word, CLAUDIO wakes anyway. None: near misses are only logged.
        self.second_chance = second_chance
        # A request heard in another language that no command understands is transcribed
        # again in this one (fast Italian often sounds English). None: never.
        self.retry_language = retry_language
        self._checking = False
        self._checks: set[asyncio.Task[None]] = set()
        self._stt_lock = asyncio.Lock()  # Whisper is used by one thing at a time
        self._recent = RingBuffer(int(endpointer.sample_rate * PREWAKE_S))
        self._prewake = np.zeros(0, dtype=np.float32)
        self._held: deque[np.ndarray] = deque(maxlen=HELD_FRAMES)
        self._warned_quiet = False
        self.state = State.IDLE
        self._interaction: Interaction | None = None
        self._task: asyncio.Task[None] | None = None
        self._beeps: set[asyncio.Task[None]] = set()
        self.history: list[dict[str, Any]] = []  # finished interactions (for tests/benchmarks)

    def _set_state(self, state: State) -> None:
        if state != self.state:
            log.debug("Stato: %s -> %s", self.state, state)
            self.state = state

    async def run(self, frames: AsyncIterator[np.ndarray]) -> None:
        log.info("Pronto: in ascolto della wake word")
        try:
            async for frame in frames:
                await self.handle_frame(frame)
                if self.system.stopping and self.state is State.IDLE:  # after "A presto."
                    log.info("Spegnimento richiesto a voce: chiudo")
                    break
        finally:
            await self.wait_idle()

    async def handle_frame(self, frame: np.ndarray) -> None:
        if self.state == State.IDLE:
            self._recent.append(frame.astype(np.float32) / 32768.0)
            detection = self.wake.process(frame)
            if detection is not None:
                self._on_wake(*detection)
            elif (miss := self.wake.pop_near_miss()) is not None:
                self._on_near_miss(*miss)
        elif self.state == State.LISTENING:
            self._on_listen(frame)
        elif self.state == State.TRANSCRIBING:
            self._held.append(frame)  # the request itself, if the phrase was only the wake word
        # While answering/speaking the microphone is ignored (no barge-in yet).

    def _on_wake(self, name: str, score: float, **info: Any) -> None:
        self._interaction = Interaction(
            clock=self.clock, info={"wake_word": name, "wake_score": round(score, 3), **info}
        )
        self._interaction.mark("wake")
        log.info("Wake word rilevata: %s (%.2f)", name, score)
        self._beep("wake")
        self._prewake = self._recent.get()
        self._recent.clear()
        if self.audio_sink is not None:
            sr = self.endpointer.sample_rate
            self._interaction.info["prewake_ms"] = round(len(self._prewake) * 1000 / sr)
        # The wake word may fire late, while the request has already begun ("hey claudio che
        # ore sono" in one breath): the endpointer can prepend the speech heard just before.
        self.endpointer.reset(lead_in=self._lead_in())
        self._set_state(State.LISTENING)

    def _lead_in(self) -> np.ndarray:
        """The last LEAD_IN_S before the detection, wake word included. Cutting the wake word
        out was worse on JenJo's 22 recordings (10 commands understood instead of 13): the
        command is glued to it, so its start went too; Whisper's prompt spells "Hey Jarvis"."""
        keep = int(LEAD_IN_S * self.endpointer.sample_rate)
        return self._prewake[-keep:] if keep else self._prewake[:0]

    def _on_near_miss(self, name: str, score: float) -> None:
        heard = self._recent.get()
        if self.second_chance is None or self._checking:
            self._log_near_miss(name, score, heard)
            return
        self._checking = True
        task = asyncio.create_task(self._second_chance(name, score, heard))
        self._checks.add(task)
        task.add_done_callback(self._checks.discard)

    async def _second_chance(self, name: str, score: float, heard: np.ndarray) -> None:
        assert self.second_chance is not None
        text, start = "", self.clock()
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(self.second_chance, heard)
        except Exception:
            log.exception("Seconda possibilità non riuscita")
        finally:
            self._checking = False
        ms = round((self.clock() - start) * 1000)
        if self.state is State.IDLE and heard_wake_word(text):
            log.info(
                "Seconda possibilità: Whisper ha sentito %r (punteggio %.2f): attivo", text, score
            )
            self._on_wake(name, score, second_chance=text, second_chance_ms=ms)
            return
        log.info("Seconda possibilità: Whisper ha sentito %r: niente da fare", text)
        self._log_near_miss(name, score, heard, second_chance=text, second_chance_ms=ms)

    def _log_near_miss(self, name: str, score: float, heard: np.ndarray, **info: Any) -> None:
        inter = Interaction(
            clock=self.clock,
            info={"wake_word": name, "wake_score": round(score, 3), "outcome": "near_miss", **info},
        )
        if self.audio_sink is not None:
            try:
                inter.info["audio_file"] = self.audio_sink(heard, "near_miss")
            except Exception:
                log.exception("Impossibile salvare l'audio della wake word mancata")
        if self.metrics is not None:
            self.metrics.write(inter)

    def _on_listen(self, frame: np.ndarray) -> None:
        assert self._interaction is not None
        event = self.endpointer.process(frame)
        if event is EndpointEvent.START:
            self._interaction.mark("speech_start")
        elif event is EndpointEvent.TIMEOUT:
            log.info("Nessuna richiesta dopo la wake word: torno in attesa")
            self._beep("cancel")
            self._cancel_pending()
            self._finish(outcome="timeout")
        elif event is EndpointEvent.END:
            now = self.clock()
            # The phrase really ended when the trailing silence began.
            self._interaction.marks["speech_end"] = now - self.endpointer.trailing_silence_s
            self._interaction.marks["endpoint"] = now
            self._held.clear()
            self._set_state(State.TRANSCRIBING)
            self._task = asyncio.create_task(self._handle_utterance(self.endpointer.audio()))

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        inter = self._interaction
        assert inter is not None
        outcome = "ok"
        follow_up = False
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(self.transcriber.transcribe, audio)
                heard = self.transcriber.language
                text, heard = await self._retry(audio, text, heard, inter)
            inter.mark("stt_done")
            inter.info["text"] = text
            inter.info["stt_language"] = heard
            log.info("Trascritto (%s): %r", heard, text)
            if only_wake_words(text) and "relisten" not in inter.info:
                outcome = "relisten"
                self._relisten(inter)
                return
            if not text or only_wake_words(text):
                outcome = "empty"
                await self.speaker.beep("cancel")
                return
            self._set_state(State.RESPONDING)
            await self.speaker.speak(
                self._marked(self.responder.respond(text, heard), inter),
                on_first_audio=lambda: self._on_first_audio(inter),
                language=lambda: self.responder.language,
            )
            if getattr(self.responder, "last_route", {}).get("route") == "paused":
                outcome = "paused"  # only "svegliati" and "spegniti" are answered
                await self.speaker.beep("cancel")
            # A confirmation question ("Confermi?"): the answer comes without the wake word.
            follow_up = bool(getattr(self.responder, "awaiting", False))
        except Exception:
            outcome = "error"
            log.exception("Errore durante la richiesta")
            try:
                await self.speaker.beep("error")
            except Exception:
                log.exception("Impossibile riprodurre il segnale di errore")
        finally:
            if outcome != "relisten":
                route = getattr(self.responder, "last_route", None)
                if route:
                    inter.info.update(route)
                if not follow_up:
                    self._cancel_pending()  # an error or silence never leaves an action armed
                self._finish(outcome)
                if follow_up:
                    self._listen_for_answer()

    def _cancel_pending(self) -> None:
        cancel = getattr(self.responder, "cancel_pending", None)
        if cancel is not None:
            cancel()

    def _listen_for_answer(self) -> None:
        """Listen again at once, no wake word needed: the answer to the question just asked."""
        self._interaction = Interaction(
            clock=self.clock, info={"wake_word": "follow_up", "wake_score": 0.0}
        )
        self._interaction.mark("wake")
        self._beep("wake")
        self._prewake = np.zeros(0, dtype=np.float32)
        self._recent.clear()
        self.endpointer.reset()
        self._set_state(State.LISTENING)

    async def _retry(
        self, audio: np.ndarray, text: str, heard: str, inter: Interaction
    ) -> tuple[str, str]:
        """`text` again in retry_language when it was heard in another one and no command
        understands it; the first transcription when the retry is not understood either."""
        first = self.retry_language
        understands = getattr(self.responder, "understands", None)
        if (
            first is None
            or understands is None
            or heard == first
            or not text
            or only_wake_words(text)
            or getattr(self.responder, "awaiting", False)  # a yes/no: no command to match
            or understands(text, heard)
        ):
            return text, heard
        start = self.clock()
        again = await asyncio.to_thread(
            functools.partial(self.transcriber.transcribe, audio, language=first)
        )
        inter.info["retry_ms"] = round((self.clock() - start) * 1000)
        if again and understands(again, first):
            log.info("Riprovato in %s: %r -> %r", first, text, again)
            inter.info["retried_from"] = text
            return again, first
        log.info("Riprovato in %s, non capito neanche così: %r", first, again)
        return text, heard

    def _relisten(self, inter: Interaction) -> None:
        """The phrase was only the wake word (said again, or its tail after an early detection):
        listen for the request, starting from what was heard while transcribing."""
        log.info("Solo la wake word: ascolto la richiesta")
        inter.info["relisten"] = inter.info.pop("text")
        for stage in ("speech_start", "speech_end", "endpoint", "stt_done"):
            inter.marks.pop(stage, None)
        self._prewake = np.concatenate((self._prewake, self.endpointer.listened()))  # saved WAV
        self.endpointer.reset()
        self._set_state(State.LISTENING)
        held = list(self._held)
        self._held.clear()
        for frame in held:
            if self.state is not State.LISTENING:  # the request already ended (or timed out)
                break
            self._on_listen(frame)

    async def _marked(self, stream: AsyncIterator[str], inter: Interaction) -> AsyncIterator[str]:
        async for chunk in stream:
            inter.mark("first_text")
            yield chunk

    def _on_first_audio(self, inter: Interaction) -> None:
        inter.mark("first_audio")
        self._set_state(State.SPEAKING)

    def _finish(self, outcome: str) -> None:
        inter = self._interaction
        if inter is not None:
            inter.mark("done")
            inter.info["outcome"] = outcome
            stats = self.endpointer.stats()
            inter.info.update(stats)
            if self.level is not None:
                inter.info["gain_db"] = self.level.gain_db
            if self.endpointer.started:
                if self.level is not None:
                    self.level.observe(stats)
                else:
                    self._check_level(stats["speech_dbfs"])
            if self.audio_sink is not None:
                heard = np.concatenate((self._prewake, self.endpointer.listened()))
                try:
                    inter.info["audio_file"] = self.audio_sink(heard, outcome)
                except Exception:
                    log.exception("Impossibile salvare l'audio della richiesta")
            row = self.metrics.write(inter) if self.metrics else inter.summary()
            self.history.append(row)
        self._interaction = None
        self.wake.reset()
        self._set_state(State.IDLE)

    def _check_level(self, speech_dbfs: float) -> None:
        if speech_dbfs < QUIET_SPEECH_DBFS and not self._warned_quiet:
            self._warned_quiet = True
            log.warning(
                "Microfono basso: la voce arriva al massimo a %.0f dBFS (meglio tra -20 e -10). "
                "Aumenta il guadagno, per esempio CLAUDIO_AUDIO__INPUT_GAIN_DB=12 in .env",
                speech_dbfs,
            )

    def _beep(self, kind: str) -> None:
        task = asyncio.create_task(self.speaker.beep(kind))
        self._beeps.add(task)
        task.add_done_callback(self._beeps.discard)

    async def wait_idle(self) -> None:
        if self._checks:
            await asyncio.gather(*self._checks, return_exceptions=True)
        # A request that listened again has started a new task: wait for that one too.
        while self._task is not None and not self._task.done():
            await asyncio.gather(self._task, return_exceptions=True)
        if self._beeps:
            await asyncio.gather(*self._beeps, return_exceptions=True)
