"""Orchestrator and state machine:
IDLE -> LISTENING -> TRANSCRIBING -> RESPONDING -> SPEAKING -> IDLE."""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np

from claudio.audio.ring_buffer import RingBuffer
from claudio.audio.vad import Endpointer, EndpointEvent
from claudio.metrics import Interaction, MetricsWriter
from claudio.protocols import Responder, Speaker, Transcriber, WakeWordDetector

log = logging.getLogger(__name__)

PREWAKE_S = 1.5  # audio before the wake word that goes into the saved WAV


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
    ) -> None:
        self.wake = wake
        self.endpointer = endpointer
        self.transcriber = transcriber
        self.responder = responder
        self.speaker = speaker
        self.metrics = metrics
        self.clock = clock
        self.audio_sink = audio_sink  # saves what was heard; returns the file name
        self._recent = RingBuffer(int(endpointer.sample_rate * PREWAKE_S) if audio_sink else 0)
        self._prewake = np.zeros(0, dtype=np.float32)
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
        finally:
            await self.wait_idle()

    async def handle_frame(self, frame: np.ndarray) -> None:
        if self.state == State.IDLE:
            if self.audio_sink is not None:
                self._recent.append(frame.astype(np.float32) / 32768.0)
            detection = self.wake.process(frame)
            if detection is not None:
                self._on_wake(*detection)
            elif (miss := self.wake.pop_near_miss()) is not None:
                self._on_near_miss(*miss)
        elif self.state == State.LISTENING:
            self._on_listen(frame)
        # While transcribing/answering/speaking the microphone is ignored (no barge-in yet).

    def _on_wake(self, name: str, score: float) -> None:
        self._interaction = Interaction(
            clock=self.clock, info={"wake_word": name, "wake_score": round(score, 3)}
        )
        self._interaction.mark("wake")
        log.info("Wake word rilevata: %s (%.2f)", name, score)
        self._beep("wake")
        self._prewake = self._recent.get()
        self._recent.clear()
        self.endpointer.reset()
        self._set_state(State.LISTENING)

    def _on_near_miss(self, name: str, score: float) -> None:
        inter = Interaction(
            clock=self.clock,
            info={"wake_word": name, "wake_score": round(score, 3), "outcome": "near_miss"},
        )
        if self.audio_sink is not None:
            try:
                inter.info["audio_file"] = self.audio_sink(self._recent.get(), "near_miss")
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
            self._finish(outcome="timeout")
        elif event is EndpointEvent.END:
            now = self.clock()
            # The phrase really ended when the trailing silence began.
            self._interaction.marks["speech_end"] = now - self.endpointer.trailing_silence_s
            self._interaction.marks["endpoint"] = now
            self._set_state(State.TRANSCRIBING)
            self._task = asyncio.create_task(self._handle_utterance(self.endpointer.audio()))

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        inter = self._interaction
        assert inter is not None
        outcome = "ok"
        try:
            text = await asyncio.to_thread(self.transcriber.transcribe, audio)
            inter.mark("stt_done")
            inter.info["text"] = text
            log.info("Trascritto: %r", text)
            if not text:
                outcome = "empty"
                await self.speaker.beep("cancel")
                return
            self._set_state(State.RESPONDING)
            await self.speaker.speak(
                self._marked(self.responder.respond(text), inter),
                on_first_audio=lambda: self._on_first_audio(inter),
            )
        except Exception:
            outcome = "error"
            log.exception("Errore durante la richiesta")
            try:
                await self.speaker.beep("error")
            except Exception:
                log.exception("Impossibile riprodurre il segnale di errore")
        finally:
            route = getattr(self.responder, "last_route", None)
            if route:
                inter.info.update(route)
            self._finish(outcome)

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
            inter.info.update(self.endpointer.stats())
            if self.audio_sink is not None:
                sr = self.endpointer.sample_rate
                inter.info["prewake_ms"] = round(len(self._prewake) * 1000 / sr)
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

    def _beep(self, kind: str) -> None:
        task = asyncio.create_task(self.speaker.beep(kind))
        self._beeps.add(task)
        task.add_done_callback(self._beeps.discard)

    async def wait_idle(self) -> None:
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._beeps:
            await asyncio.gather(*self._beeps, return_exceptions=True)
