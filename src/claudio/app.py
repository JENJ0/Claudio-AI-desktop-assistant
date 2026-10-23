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
from dataclasses import dataclass
from typing import Any

import numpy as np

from claudio.audio.ring_buffer import RingBuffer
from claudio.audio.vad import Endpointer, EndpointEvent
from claudio.config import in_language
from claudio.metrics import Interaction, MetricsWriter
from claudio.protocols import LevelControl, Responder, Speaker, Transcriber, WakeWordDetector
from claudio.router.normalizer import heard_wake_word, normalize, only_wake_words
from claudio.system import SystemControl
from claudio.wakeword.verifier import wake_confirmed

log = logging.getLogger(__name__)

PREWAKE_S = 1.5  # audio kept before the wake word: one-breath phrases and the saved WAV
HELD_FRAMES = 100  # frames kept while transcribing (~8 s), replayed if we listen again
LEAD_IN_S = PREWAKE_S  # one breath: how much of the audio before the detection Whisper gets
QUIET_SPEECH_DBFS = -30.0  # loudest 32 ms of a phrase below this: the microphone is too quiet
HISTORY_MAX = 200  # finished interactions kept in memory
WHILE_SPEAKING_S = 3600.0  # the barge-in listener never times out while CLAUDIO speaks


class State(enum.StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    RESPONDING = "responding"
    SPEAKING = "speaking"


@dataclass
class BargeIn:
    """Listens while CLAUDIO speaks: its own endpointer (its own VAD, so the request's
    stats stay intact) and a quick Whisper pass on each phrase, looking for "basta"."""

    endpointer: Endpointer
    transcribe: Callable[[np.ndarray], str]


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
        conversation_s: float = 0.0,
        barge_in: BargeIn | None = None,
        verify_wake: bool = False,
        closed_list: Callable[[np.ndarray, str], list[str]] | None = None,
        closed_list_max_words: int = 0,
        claude_min_words: int | None = None,
        only_addressed: bool = True,
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
        # After a reply, listen this long for another request without the wake word (0: off).
        self.conversation_s = conversation_s
        self.barge_in = barge_in  # None: the microphone is ignored while speaking
        # Drop a wake whose transcript has no name at the start and no command (verifier.py).
        self.verify_wake = verify_wake
        # (audio, language) -> Whisper's n-best: tried on a request no command understands.
        self.closed_list = closed_list
        # Longer requests skip the closed list: a question for Claude, not a misheard command.
        self.closed_list_max_words = closed_list_max_words  # 0: no limit
        # Claude answers what no command understands. In the conversation window (no wake
        # word) a request needs at least this many words: shorter is the TV or a remark.
        # None: Claude is off and the window keeps only the local commands.
        self.claude_min_words = claude_min_words
        # True: in the window only phrases that sound addressed to Claudio go to Claude (the TV
        # costs nothing). False: every phrase of claude_min_words or more is answered, until
        # "basta" or silence (JenJo's choice, 23/10).
        self.only_addressed = only_addressed
        self._barge_checking = False
        self._interrupted = False  # the reply being spoken was cut short by "basta"
        self._checking = False
        self._checks: set[asyncio.Task[None]] = set()
        self._stt_lock = asyncio.Lock()  # Whisper is used by one thing at a time
        self._recent = RingBuffer(int(endpointer.sample_rate * PREWAKE_S))
        self._prewake = np.zeros(0, dtype=np.float32)
        self._wake_audio = np.zeros(0, dtype=np.float32)
        self._held: deque[np.ndarray] = deque(maxlen=HELD_FRAMES)
        self._warned_quiet = False
        self.state = State.IDLE
        self._interaction: Interaction | None = None
        self._task: asyncio.Task[None] | None = None
        self._beeps: set[asyncio.Task[None]] = set()
        # Finished interactions (for tests/benchmarks). Bounded: always-on runs last for weeks.
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_MAX)

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
        elif self.state == State.SPEAKING and self.barge_in is not None:
            self._on_barge_frame(frame, self.barge_in)

    def _on_wake(self, name: str, score: float, **info: Any) -> None:
        self._interaction = Interaction(
            clock=self.clock, info={"wake_word": name, "wake_score": round(score, 3), **info}
        )
        self._interaction.mark("wake")
        log.info("Wake word rilevata: %s (%.2f)", name, score)
        self._beep("wake")
        self._prewake = self._recent.get()
        self._recent.clear()
        self._wake_audio = self._lead_in()  # the wake word itself, for the verifier
        if self.audio_sink is not None:
            sr = self.endpointer.sample_rate
            self._interaction.info["prewake_ms"] = round(len(self._prewake) * 1000 / sr)
        # The wake word may fire late, while the request has already begun ("hey claudio che
        # ore sono" in one breath): the endpointer can prepend the speech heard just before.
        self.endpointer.reset(lead_in=self._lead_in())
        self._set_state(State.LISTENING)
        self.responder.prepare()

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
            if self._interaction.info.get("wake_word") == "conversation":
                self.responder.prepare()  # no wake word in the conversation window
        elif event is EndpointEvent.TIMEOUT:
            if self._interaction.info.get("wake_word") == "conversation":
                log.info("Fine della conversazione: torno in attesa della wake word")
            else:
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
        self._interrupted = False
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(self.transcriber.transcribe, audio)
                heard = self.transcriber.language
                text, heard = await self._retry(audio, text, heard, inter)
                text = await self._closed_list(audio, text, heard, inter)
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
            if self._false_wake(text, heard, inter) and not await self._name_before(inter):
                # Woke on something that only starts like the name: silence, nothing else.
                outcome = "false_wake"
                log.info("Wake word non confermata da Whisper, ignoro: %r", text)
                return
            if inter.info.get("wake_word") == "conversation" and not self._is_request(text, heard):
                # Heard without the wake word: the TV or someone else talking, not a request.
                outcome = "conversation_end"
                log.info("Non è una richiesta, fine della conversazione: %r", text)
                await self.speaker.beep("cancel")
                return
            self._set_state(State.RESPONDING)
            await self.speaker.speak(
                self._marked(self.responder.respond(text, heard), inter),
                on_first_audio=lambda: self._on_first_audio(inter),
                language=lambda: self.responder.language,
            )
            route = self.responder.last_route
            if route.get("route") == "paused":
                outcome = "paused"  # only "svegliati" and "spegniti" are answered
                await self.speaker.beep("cancel")
            elif route.get("command") == "stop":
                outcome = "stopped"  # "basta": the conversation is over
                await self.speaker.beep("cancel")
            elif self._interrupted:
                outcome = "interrupted"
            # A confirmation question ("Confermi?"): the answer comes without the wake word.
            follow_up = self.responder.awaiting and not self._interrupted
        except Exception:
            outcome = "error"
            log.exception("Errore durante la richiesta")
            try:
                await self.speaker.beep("error")
            except Exception:
                log.exception("Impossibile riprodurre il segnale di errore")
        finally:
            if outcome != "relisten":
                route = self.responder.last_route
                if route:
                    inter.info.update(route)
                if not follow_up:
                    self._cancel_pending()  # an error or silence never leaves an action armed
                self._finish(outcome)
                if follow_up:
                    self._listen_for_answer()
                elif self._keeps_conversation(outcome):
                    self._listen_for_answer(conversation=True)

    def _understood(self, text: str, language: str) -> bool:
        return self.responder.understands(text, language)

    def _is_request(self, text: str, language: str) -> bool:
        """Heard in the conversation window: a command, or (with Claude) a phrase that sounds
        addressed to Claudio. The TV's "e poi gli ho detto di no" is not: it costs nothing."""
        if self._understood(text, language):
            return True
        if self.claude_min_words is not None and self.responder.asked_question:
            return True  # Claude asked "da dove parti?": "Milano." is the answer
        if self.claude_min_words is None or _words(text) < self.claude_min_words:
            return False
        return not self.only_addressed or addressed_to_claudio(text)

    async def _name_before(self, inter: Interaction) -> bool:
        """The request was said after a pause ("Hey Claudio" ... beep ... "come ti chiami?"):
        its transcript has no name, so Whisper listens to the wake word itself (~100 ms, only
        on requests that are not commands). "Hey, ciao" there is still a false wake."""
        if self.second_chance is None or not self._wake_audio.size:
            return False
        start = self.clock()
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(self.second_chance, self._wake_audio)
        except Exception:
            log.exception("Verifica della wake word non riuscita")
            return False
        inter.info["wake_check_ms"] = round((self.clock() - start) * 1000)
        inter.info["wake_check"] = text
        if heard_wake_word(text):
            log.info("Nome sentito prima della richiesta: %r", text)
            return True
        return False

    def _false_wake(self, text: str, heard: str, inter: Interaction) -> bool:
        """Only a real wake word detection is judged: not the conversation window, not a
        yes/no answer, not a second chance (Whisper already heard the name) and not the
        request after a lone wake word (the name was in the phrase before)."""
        if (
            not self.verify_wake
            or inter.info.get("wake_word") == "conversation"
            or self.responder.awaiting
            or "second_chance" in inter.info
            or "relisten" in inter.info
        ):
            return False
        return not wake_confirmed(text, self._understood(text, heard))

    async def _closed_list(
        self, audio: np.ndarray, text: str, heard: str, inter: Interaction
    ) -> str:
        """The first of Whisper's n-best transcriptions that a command understands, when the
        best one is not understood; `text` when none is (or there is nothing to fix)."""
        understands = self.responder.understands
        if (
            self.closed_list is None
            or not text
            or only_wake_words(text)
            or self.responder.awaiting
            or understands(text, heard)
            or (self.closed_list_max_words and _words(text) > self.closed_list_max_words)
        ):
            return text
        start = self.clock()
        candidates = await asyncio.to_thread(self.closed_list, audio, heard)
        inter.info["closed_list_ms"] = round((self.clock() - start) * 1000)
        found = next((c for c in candidates if c != text and understands(c, heard)), None)
        if found is None:
            return text
        log.info("Lista chiusa: %r -> %r", text, found)
        inter.info["closed_list_from"] = text
        return found

    def _keeps_conversation(self, outcome: str) -> bool:
        return (
            outcome == "ok"
            and self.conversation_s > 0
            and not self.system.paused
            and not self.system.stopping
        )

    def _cancel_pending(self) -> None:
        self.responder.cancel_pending()

    def _listen_for_answer(self, conversation: bool = False) -> None:
        """Listen again at once, no wake word needed: the answer to the question just asked,
        or (`conversation`) another request within conversation_s of the reply."""
        kind = "conversation" if conversation else "follow_up"
        self._interaction = Interaction(
            clock=self.clock, info={"wake_word": kind, "wake_score": 0.0}
        )
        self._interaction.mark("wake")
        self._beep("wake")
        self._prewake = np.zeros(0, dtype=np.float32)
        self._recent.clear()
        self.endpointer.reset(timeout_s=self.conversation_s if conversation else None)
        self._set_state(State.LISTENING)

    def _on_barge_frame(self, frame: np.ndarray, barge: BargeIn) -> None:
        event = barge.endpointer.process(frame)
        if event is EndpointEvent.TIMEOUT:
            barge.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
        elif event is EndpointEvent.END:
            audio = barge.endpointer.audio()
            barge.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
            if self._barge_checking:
                return  # still looking at the last phrase: this one is skipped
            self._barge_checking = True
            task = asyncio.create_task(self._check_barge_in(audio, barge))
            self._checks.add(task)
            task.add_done_callback(self._checks.discard)

    async def _check_barge_in(self, audio: np.ndarray, barge: BargeIn) -> None:
        """Whisper on what was heard while speaking: "basta" interrupts the reply."""
        text = ""
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(barge.transcribe, audio)
        except Exception:
            log.exception("Ascolto durante la risposta non riuscito")
        finally:
            self._barge_checking = False
        if not text or not self.responder.is_stop(text):
            log.debug("Sentito mentre parlo: %r", text)
            return
        if self.state is not State.SPEAKING:
            return  # the reply had already ended
        log.info("Interrotto a voce: %r", text)
        self._interrupted = True
        if self._interaction is not None:
            self._interaction.info["barge_in"] = text
        self.speaker.interrupt()

    async def _retry(
        self, audio: np.ndarray, text: str, heard: str, inter: Interaction
    ) -> tuple[str, str]:
        """`text` again in retry_language when it was heard in another one and no command
        understands it; the first transcription when the retry is not understood either."""
        first = self.retry_language
        understands = self.responder.understands
        if (
            first is None
            or heard == first
            or not text
            or only_wake_words(text)
            or self.responder.awaiting  # a yes/no: no command to match
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

    async def handle_typed(self, text: str, language: str, speak: bool = False) -> str:
        """A message written in the chat window: the same router and Claude as a spoken
        request, between two spoken ones (never during). Silent unless `speak`: the reply is
        shown in the chat. Returns the reply."""
        while self.state is not State.IDLE:  # a spoken request goes first
            await asyncio.sleep(0.05)
        self._set_state(State.RESPONDING)  # the wake word waits until this is answered
        inter = Interaction(clock=self.clock, info={"wake_word": "typed", "text": text})
        inter.mark("stt_done")
        outcome = "ok"
        log.info("Scritto nella finestra: %r", text)
        try:
            stream = self._marked(self.responder.respond(text, language), inter)
            if speak:
                await self.speaker.speak(
                    stream,
                    on_first_audio=lambda: inter.mark("first_audio"),
                    language=lambda: self.responder.language,
                )
            else:
                async for _ in stream:
                    pass
        except Exception:
            outcome = "error"
            log.exception("Errore rispondendo al messaggio scritto")
        finally:
            if self.responder.awaiting:  # level 2-3: confirmed by voice only, never armed here
                self._cancel_pending()
                note = in_language(TYPED_NEEDS_VOICE, self.responder.language)
                inter.info["reply"] = note
                outcome = "needs_voice"
            inter.info.update(self.responder.last_route)
            inter.mark("done")
            inter.info["outcome"] = outcome
            row = self.metrics.write(inter) if self.metrics else inter.summary()
            self.history.append(row)
            self.wake.reset()
            self._set_state(State.IDLE)
        return str(inter.info.get("reply", ""))

    async def _marked(self, stream: AsyncIterator[str], inter: Interaction) -> AsyncIterator[str]:
        parts: list[str] = []
        try:
            async for chunk in stream:
                inter.mark("first_text")
                parts.append(chunk)
                yield chunk
        finally:
            if parts:  # what was said, also when "basta" cut it short (session log, metrics)
                inter.info["reply"] = "".join(parts).strip()

    def _on_first_audio(self, inter: Interaction) -> None:
        inter.mark("first_audio")
        if self.barge_in is not None:
            self.barge_in.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
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


# A level 2-3 action written in the chat window: it is confirmed by voice only.
TYPED_NEEDS_VOICE = {
    "it": "Questa azione va confermata a voce: chiedimela parlando.",
    "en": "This action needs a spoken confirmation: ask me out loud.",
}


# How a follow-up request to Claudio usually starts, when it is not a question.
REQUEST_STARTS = frozenset(
    [
        "dimmi",
        "spiegami",
        "raccontami",
        "descrivimi",
        "dammi",
        "fammi",
        "consigliami",
        "suggeriscimi",
        "traduci",
        "calcola",
        "elenca",
        "riassumi",
        "continua",
        "aiutami",
        "scrivi",
        "perche",
        "come",
        "cosa",
        "chi",
        "dove",
        "quando",
        "quale",
        "quanto",
        "tell",
        "explain",
        "describe",
        "give",
        "show",
        "recommend",
        "suggest",
        "translate",
        "calculate",
        "list",
        "summarize",
        "continue",
        "help",
        "write",
        "why",
        "how",
        "what",
        "who",
        "where",
        "when",
        "which",
    ]
)


def addressed_to_claudio(text: str) -> bool:
    """A question (Whisper writes the "?"), a request verb at the start ("dimmi di più") or
    the name: worth asking Claude without the wake word."""
    words = normalize(text, drop_fillers=False).split()
    return (
        text.rstrip().endswith("?")
        or (bool(words) and words[0] in REQUEST_STARTS)
        or heard_wake_word(text)
    )


def _words(text: str) -> int:
    return len(normalize(text, drop_fillers=False).split())
