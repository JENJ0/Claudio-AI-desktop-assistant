"""Orchestrator and state machine:
IDLE -> LISTENING -> TRANSCRIBING -> RESPONDING -> SPEAKING -> IDLE."""

from __future__ import annotations

import asyncio
import enum
import functools
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from claudio.audio.ring_buffer import RingBuffer
from claudio.audio.vad import Endpointer, EndpointEvent
from claudio.config import NoiseSettings, in_language
from claudio.intake import BARGE_IN, Decision, Heard, Kind, RequestFilter, Segment, content_words
from claudio.intake import addressed_to_claudio as addressed_to_claudio  # re-exported
from claudio.metrics import Interaction, MetricsWriter
from claudio.protocols import LevelControl, Responder, Speaker, Transcriber, WakeWordDetector
from claudio.resume import RESUME_PREFIX, DrainedStream, ends_the_reply, remainder
from claudio.router.normalizer import heard_wake_word, normalize
from claudio.system import SystemControl

log = logging.getLogger(__name__)

PREWAKE_S = 1.5  # audio kept before the wake word: one-breath phrases and the saved WAV
HELD_FRAMES = 100  # frames kept while transcribing (~8 s), replayed if we listen again
LEAD_IN_S = PREWAKE_S  # one breath: how much of the audio before the detection Whisper gets
QUIET_SPEECH_DBFS = -30.0  # loudest 32 ms of a phrase below this: the microphone is too quiet
HISTORY_MAX = 200  # finished interactions kept in memory
WHILE_SPEAKING_S = 3600.0  # the barge-in listener never times out while CLAUDIO speaks
CUT_FINISH_S = 5.0  # a cut reply still arriving from Claude: wait this long for the rest
ECHO_SHARE = 0.6  # words heard while speaking that are mostly in the reply: CLAUDIO's own voice


class State(enum.StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    RESPONDING = "responding"
    SPEAKING = "speaking"


@dataclass
class BargeIn:
    """Listens while CLAUDIO speaks: its own endpointer (its own, stricter VAD, so the
    request's stats stay intact) and a quick Whisper pass on each phrase. "basta" stops the
    reply; other words said close to the microphone stop it too and become the next request.

    `mode`: "duck" lowers the reply as soon as a loud voice starts and stops it when Whisper
    hears words; "stop" stops it at once; "confirm" waits for Whisper at full volume.
    `min_dbfs`: the loudest 32 ms of the voice must reach this (the TV and the echo do not)."""

    endpointer: Endpointer
    transcribe: Callable[[np.ndarray], str]
    mode: str = "duck"
    min_dbfs: float = -30.0
    duck_volume: float = 0.3
    # A reply cut by a new request goes on ("Dicevo: ...") after that request is answered,
    # if it took less than this (0: never).
    resume_s: float = 60.0


@dataclass(frozen=True)
class CutReply:
    """A reply a new request cut short: the stream still reading it, the sentences that had
    started playing, when it was cut, its language, and (once read) the part not said."""

    stream: DrainedStream
    played: list[str]
    at: float
    language: str
    rest: str = ""


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
        noise: NoiseSettings | None = None,
        confirm_below: float = 0.0,
        always_confirm: frozenset[str] = frozenset(),
        stt_segments: Callable[[], Sequence[Segment]] | None = None,
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
        # Whether a phrase is a request: the rules live in intake.py. Their settings are also
        # attributes of the Assistant (properties below).
        self.intake = RequestFilter(
            verify_wake=verify_wake,
            claude_min_words=claude_min_words,
            only_addressed=only_addressed,
            closed_list_max_words=closed_list_max_words,
            retry_language=retry_language,
            noise=noise,
        )
        # Whisper's confidence in each segment of the last transcription (the noise filter).
        # None: not known, only the speech level is judged.
        self.stt_segments = stt_segments
        # After a reply, listen this long for another request without the wake word (0: off).
        self.conversation_s = conversation_s
        self.barge_in = barge_in  # None: the microphone is ignored while speaking
        # A detection scoring below this wakes CLAUDIO only if Whisper hears the name (the
        # second chance): a 0.34 on Claudio's own voice or the TV no longer beeps. 0: off.
        self.confirm_below = confirm_below
        # Wake word models whose every detection Whisper checks ("claudio" alone: noisy).
        self.always_confirm = always_confirm
        # (audio, language) -> Whisper's n-best: tried on a request no command understands.
        self.closed_list = closed_list
        self._barge_checking = False
        self._interrupted = False  # the reply being spoken was cut short by "basta"
        # A voice took over while CLAUDIO spoke: the start of the next request, then the frames
        # heard until the reply has stopped. None: nobody interrupted.
        self._takeover: tuple[np.ndarray, list[np.ndarray]] | None = None
        self._reply_parts: list[str] = []  # the reply being spoken (to recognise its echo)
        # A reply cut by a new request, still read to its end: resumed once that request is
        # answered. None: no reply was cut.
        self._cut: CutReply | None = None
        # Frames heard since the last phrase went to Whisper: if it was someone talking, they
        # are the rest of the request.
        self._since_check: deque[np.ndarray] = deque(maxlen=HELD_FRAMES)
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

    # The request filter's settings, as attributes of the Assistant.

    @property
    def verify_wake(self) -> bool:
        """Drop a wake whose transcript has no name at the start and no command."""
        return self.intake.verify_wake

    @verify_wake.setter
    def verify_wake(self, value: bool) -> None:
        self.intake.verify_wake = value

    @property
    def retry_language(self) -> str | None:
        """A request heard in another language that no command understands is transcribed
        again in this one (fast Italian often sounds English). None: never."""
        return self.intake.retry_language

    @retry_language.setter
    def retry_language(self, value: str | None) -> None:
        self.intake.retry_language = value

    @property
    def closed_list_max_words(self) -> int:
        """Longer requests skip the closed list: a question for Claude (0: no limit)."""
        return self.intake.closed_list_max_words

    @closed_list_max_words.setter
    def closed_list_max_words(self, value: int) -> None:
        self.intake.closed_list_max_words = value

    @property
    def claude_min_words(self) -> int | None:
        """In the conversation window a phrase for Claude needs this many words (shorter is
        the TV or a remark). None: Claude is off, the window keeps only local commands."""
        return self.intake.claude_min_words

    @claude_min_words.setter
    def claude_min_words(self, value: int | None) -> None:
        self.intake.claude_min_words = value

    @property
    def only_addressed(self) -> bool:
        """True: in the window only phrases that sound addressed to Claudio go to Claude.
        False: every phrase of claude_min_words or more is answered (JenJo's choice, 23/10)."""
        return self.intake.only_addressed

    @only_addressed.setter
    def only_addressed(self, value: bool) -> None:
        self.intake.only_addressed = value

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
            if detection is not None and self._weak(*detection):
                if not self._checking:
                    log.info("Wake word debole (%.2f): Whisper controlla il nome", detection[1])
                self._on_near_miss(*detection)
            elif detection is not None:
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

    def _weak(self, name: str, score: float) -> bool:
        return self.second_chance is not None and (
            score < self.confirm_below or name in self.always_confirm
        )

    def _lead_in(self) -> np.ndarray:
        """The last LEAD_IN_S before the detection, wake word included. Cutting the wake word
        out was worse on JenJo's 22 recordings (10 commands understood instead of 13): the
        command is glued to it, so its start went too; Whisper's prompt spells "Hey Jarvis"."""
        keep = int(LEAD_IN_S * self.endpointer.sample_rate)
        return self._prewake[-keep:] if keep else self._prewake[:0]

    def _on_near_miss(self, name: str, score: float) -> None:
        heard = self._recent.get()
        if self._checking and self._weak(name, score):
            return  # the same phrase fires for many frames: Whisper is already listening to it
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
        # A weak detection said fast often loses the name ("apri cancolatrice"): a command counts.
        language = self.transcriber.language
        weak = self._weak(name, score) and bool(text) and self.responder.understands(text, language)
        if self.state is State.IDLE and (heard_wake_word(text) or weak):
            log.info(
                "Seconda possibilità: Whisper ha sentito %r (punteggio %.2f): attivo", text, score
            )
            self._on_wake(name, score, second_chance=text, second_chance_ms=ms)
            return
        log.info("Seconda possibilità: Whisper ha sentito %r: niente da fare", text)
        if name in self.always_confirm:
            return  # a noisy model: its rejected detections are not worth a WAV each
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
        self._takeover = None
        self._reply_parts = []
        cut, self._cut = self._cut, None
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(self.transcriber.transcribe, audio)
                heard = self.transcriber.language
                # Whisper's confidence in the first pass: a retry only replaces it when understood.
                segments = list(self.stt_segments()) if self.stt_segments is not None else []
                text, heard = await self._retry(audio, text, heard, inter)
                text = await self._closed_list(audio, text, heard, inter)
            inter.mark("stt_done")
            inter.info["text"] = text
            inter.info["stt_language"] = heard
            log.info("Trascritto (%s): %r", heard, text)
            if "retried_from" in inter.info or "closed_list_from" in inter.info:
                segments = []  # a known command now: the first pass's doubts no longer apply
            decision = await self._decide(text, heard, inter, segments)
            outcome = decision.kind.value
            if decision.kind is Kind.RELISTEN:
                self._cut, cut = cut, None  # still the same request
                self._relisten(inter)
                return
            if decision.dropped:
                self._log_drop(decision, inter)
                if decision.kind is not Kind.FALSE_WAKE:  # a false wake ends in silence
                    await self.speaker.beep("cancel")
                return
            self._set_state(State.RESPONDING)
            resume = await self._unsaid(cut)  # the reply this request cut, read to its end
            cut = None
            await self._speak_reply(
                self.responder.respond(decision.text, heard),
                inter,
                on_first_audio=lambda: self._on_first_audio(inter),
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
            if resume and outcome == "ok" and not follow_up and self._takeover is None:
                await self._resume_reply(resume, decision.text, inter)
        except Exception:
            outcome = "error"
            log.exception("Errore durante la richiesta")
            try:
                await self.speaker.beep("error")
            except Exception:
                log.exception("Impossibile riprodurre il segnale di errore")
        finally:
            if cut is not None:  # the request was dropped: the cut reply is not resumed
                cut.stream.cancel()
            if outcome != "relisten":
                route = self.responder.last_route
                if route:
                    inter.info.update(route)
                if not follow_up:
                    self._cancel_pending()  # an error or silence never leaves an action armed
                self._finish(outcome)
                if self._takeover is not None:
                    self._listen_to_takeover()
                elif follow_up:
                    self._listen_for_answer()
                elif self._keeps_conversation(outcome):
                    self._listen_for_answer(conversation=True)

    async def _decide(
        self, text: str, language: str, inter: Interaction, segments: Sequence[Segment]
    ) -> Decision:
        """The request filter's decision. A false wake gets one more chance: the request may
        have been said after a pause, with the name only in the wake word audio."""
        info = inter.info
        speech_dbfs = self.endpointer.stats()["speech_dbfs"] if self.endpointer.started else None
        heard = Heard(
            wake_word=str(info.get("wake_word", "")),
            relistened="relisten" in info,
            second_chance="second_chance" in info,
            speech_dbfs=speech_dbfs,
            segments=segments,
        )
        if segments:  # for tuning the noise filter on real logs
            info["stt_no_speech"] = round(max(s.no_speech_prob for s in segments), 3)
            info["stt_logprob"] = round(min(s.avg_logprob for s in segments), 3)
        decision = self.intake.decide(text, language, heard, self.responder)
        if decision.kind is Kind.FALSE_WAKE and await self._name_before(inter):
            heard = replace(heard, name_heard=True)
            decision = self.intake.decide(text, language, heard, self.responder)
        if decision.reason and not decision.dropped:
            info["intake"] = decision.reason  # e.g. quiet_answer: the "sì" was not counted
            log.info("Risposta troppo lontana dal microfono, non la conto: %r", text)
        return decision

    def _log_drop(self, decision: Decision, inter: Interaction) -> None:
        text = decision.text
        if decision.kind is Kind.FALSE_WAKE:
            log.info("Wake word non confermata da Whisper, ignoro: %r", text)
        elif decision.kind is Kind.NOT_REQUEST:
            # Heard without the wake word: the TV or someone else talking, not a request.
            log.info("Non è una richiesta, fine della conversazione: %r", text)
        elif decision.kind is Kind.NOISE:
            inter.info["drop_reason"] = "noise"
            inter.info["noise"] = decision.reason
            log.info("Rumore scambiato per parole (%s), ignoro: %r", decision.reason, text)

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

    async def _closed_list(
        self, audio: np.ndarray, text: str, heard: str, inter: Interaction
    ) -> str:
        """The first of Whisper's n-best transcriptions that a command understands, when the
        best one is not understood; `text` when none is (or there is nothing to fix)."""
        understands = self.responder.understands
        if self.closed_list is None or not self.intake.wants_closed_list(
            text, heard, self.responder
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
        if self._takeover is not None:  # the reply is stopping: keep the rest of the request
            self._takeover[1].append(frame)
            return
        self._since_check.append(frame)
        event = barge.endpointer.process(frame)
        if event is EndpointEvent.START:
            if not self._barge_loud(barge):
                return
            if barge.mode == "stop":
                # Everything heard so far is in the phrase: nothing more to add.
                self._take_over(barge.endpointer.partial(), "", rest=[])
            elif barge.mode == "duck":
                self.speaker.duck(barge.duck_volume)
        elif event is EndpointEvent.TIMEOUT:
            barge.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
        elif event is EndpointEvent.END:
            audio = barge.endpointer.audio()
            dbfs = barge.endpointer.stats()["speech_dbfs"]
            barge.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
            if dbfs < barge.min_dbfs:
                # Logged with its level: the data barge_in_min_dbfs is tuned on.
                log.info("Voce troppo debole mentre parlo (%.0f dBFS): ignorata", dbfs)
                return
            if self._barge_checking:
                return  # still looking at the last phrase: this one is skipped
            self._barge_checking = True
            self._since_check.clear()
            task = asyncio.create_task(self._check_barge_in(audio, barge, dbfs))
            self._checks.add(task)
            task.add_done_callback(self._checks.discard)

    @staticmethod
    def _barge_loud(barge: BargeIn) -> bool:
        return barge.endpointer.stats()["speech_dbfs"] >= barge.min_dbfs

    async def _check_barge_in(
        self, audio: np.ndarray, barge: BargeIn, dbfs: float | None = None
    ) -> None:
        """Whisper on what was heard while speaking: "basta" stops the reply, other words
        stop it and become the next request, noise or CLAUDIO's own echo change nothing."""
        text = ""
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(barge.transcribe, audio)
        except Exception:
            log.exception("Ascolto durante la risposta non riuscito")
        finally:
            self._barge_checking = False
        if self.state is not State.SPEAKING or self._takeover is not None:
            return  # the reply had already ended, or someone already took over
        if self._interaction is not None and dbfs is not None:
            self._interaction.info["barge_dbfs"] = round(dbfs, 1)
        if text and self.responder.is_stop(text):
            log.info("Interrotto a voce: %r", text)
            self._interrupted = True
            if self._interaction is not None:
                self._interaction.info["barge_in"] = text
            self.speaker.interrupt()
            self.speaker.duck(None)
        elif self._is_talking_to_me(text):
            self._take_over(audio, text)
        else:
            log.info("Sentito mentre parlo, non è per me: %r", text)
            self.speaker.duck(None)

    def _is_talking_to_me(self, text: str) -> bool:
        """Words said over the reply: two or more, the name or a command, and not mostly the
        reply's own words (its echo, with speakers instead of headphones)."""
        if not text or not (
            content_words(text) >= 2 or heard_wake_word(text) or self.responder.understands(text)
        ):
            return False
        heard = normalize(text, drop_fillers=False).split()
        reply = set(normalize("".join(self._reply_parts), drop_fillers=False).split())
        echo = sum(word in reply for word in heard) / len(heard)
        if echo >= ECHO_SHARE:
            log.info("La mia voce nel microfono, la ignoro: %r", text)
            return False
        return True

    def _take_over(
        self, audio: np.ndarray, text: str, rest: list[np.ndarray] | None = None
    ) -> None:
        """Stop the reply: what was said over it (`audio`, then `rest`, by default the frames
        heard while Whisper listened) is the start of the next request."""
        log.info("Interrotto da una nuova richiesta: %r", text)
        self._takeover = (audio, list(self._since_check) if rest is None else rest)
        self._since_check.clear()
        self._interrupted = True
        if self._interaction is not None:
            self._interaction.info["barge_in"] = text or "voice"
        self.speaker.interrupt()
        self.speaker.duck(None)

    def _listen_to_takeover(self) -> None:
        """Listen to the rest of the request that interrupted the reply: no wake word, no beep
        (the person is still talking), the audio heard so far counts."""
        assert self._takeover is not None
        start, frames = self._takeover
        self._takeover = None
        self._interaction = Interaction(
            clock=self.clock, info={"wake_word": BARGE_IN, "wake_score": 0.0}
        )
        self._interaction.mark("wake")
        self._interaction.mark("speech_start")
        self._prewake = np.zeros(0, dtype=np.float32)
        self._recent.clear()
        self.endpointer.resume(start, timeout_s=self.conversation_s or None)
        self._set_state(State.LISTENING)
        self.responder.prepare()
        for frame in frames:  # heard while the reply was stopping
            if self.state is not State.LISTENING:
                break
            self._on_listen(frame)

    async def _retry(
        self, audio: np.ndarray, text: str, heard: str, inter: Interaction
    ) -> tuple[str, str]:
        """`text` again in retry_language when it was heard in another one and no command
        understands it; the first transcription when the retry is not understood either."""
        first = self.retry_language
        understands = self.responder.understands
        if first is None or not self.intake.wants_retry(text, heard, self.responder):
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

    async def _speak_reply(
        self,
        stream: AsyncIterator[str],
        inter: Interaction,
        on_first_audio: Callable[[], None],
        language: str | None = None,
        resumed: bool = False,
    ) -> None:
        """Speak a reply. The stream is read in a task of its own: when a new request cuts the
        reply, the rest still arrives and is kept to be resumed after it. A `resumed` reply
        is not the request's own (the session log keeps the answer)."""
        lang = (lambda: self.responder.language) if language is None else language
        drained = DrainedStream(stream)
        text = self._echoed(drained.read()) if resumed else self._marked(drained.read(), inter)
        try:
            await self.speaker.speak(text, on_first_audio=on_first_audio, language=lang)
        finally:
            if self._takeover is not None and self.barge_in and self.barge_in.resume_s > 0:
                language_now = lang() if callable(lang) else lang
                self._cut = CutReply(drained, list(self.speaker.played), self.clock(), language_now)
            else:
                drained.cancel()

    async def _unsaid(self, cut: CutReply | None) -> CutReply | None:
        """The reply this request cut, with its last words (Claude is usually done by now):
        `cut` with only the part that was not said, None when nothing is left."""
        if cut is None:
            return None
        await cut.stream.finish(CUT_FINISH_S)
        if self.responder.awaiting:
            return None  # it asked something ("Lo uso?"): not a story to go on with
        rest = remainder(cut.stream.text, cut.played)
        return replace(cut, rest=rest) if rest else None

    async def _resume_reply(self, cut: CutReply, request: str, inter: Interaction) -> None:
        """After answering the request that cut a reply: "Dicevo: ..." and the rest of it."""
        assert self.barge_in is not None
        if ends_the_reply(request) or self.clock() - cut.at > self.barge_in.resume_s:
            return
        rest, language = cut.rest, cut.language
        log.info("Riprendo la risposta interrotta: %r", rest[:60])
        inter.info["resumed"] = True
        self._reply_parts = []

        async def text() -> AsyncIterator[str]:
            yield f"{in_language(RESUME_PREFIX, language)} {rest}"

        await self._speak_reply(text(), inter, self._on_resume_audio, language, resumed=True)

    def _on_resume_audio(self) -> None:
        """The resumed reply can be cut again, like any other."""
        if self.barge_in is not None:
            self.barge_in.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
            self._since_check.clear()
        self._set_state(State.SPEAKING)

    async def _echoed(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        async for chunk in stream:
            self._reply_parts.append(chunk)  # its echo is recognised like any reply's
            yield chunk

    async def _marked(self, stream: AsyncIterator[str], inter: Interaction) -> AsyncIterator[str]:
        parts: list[str] = []
        try:
            async for chunk in stream:
                inter.mark("first_text")
                parts.append(chunk)
                self._reply_parts.append(chunk)
                yield chunk
        finally:
            if parts:  # what was said, also when "basta" cut it short (session log, metrics)
                inter.info["reply"] = "".join(parts).strip()

    def _on_first_audio(self, inter: Interaction) -> None:
        inter.mark("first_audio")
        if self.barge_in is not None:
            self.barge_in.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
            self._since_check.clear()
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
