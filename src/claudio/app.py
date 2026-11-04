"""Orchestrator and state machine:
IDLE -> LISTENING -> TRANSCRIBING -> RESPONDING -> SPEAKING -> IDLE."""

from __future__ import annotations

import asyncio
import enum
import functools
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from claudio.audio.own_sound import OwnSound
from claudio.audio.ring_buffer import RingBuffer
from claudio.audio.vad import Endpointer, EndpointEvent
from claudio.audio.vad import dbfs as level_dbfs
from claudio.config import NoiseSettings, in_language
from claudio.echo import is_echo, recent_reply
from claudio.intake import (
    BARGE_IN,
    CONVERSATION,
    FOLLOW_UP,
    GUESSED_DROPS,
    NEVER_GUESSED,
    Decision,
    Doubt,
    Heard,
    Hearing,
    Kind,
    Recovery,
    RequestFilter,
    Segment,
    Verdict,
    content_words,
    echo_of,
    unsure,
)
from claudio.intake import addressed_to_claudio as addressed_to_claudio  # re-exported
from claudio.metrics import Interaction, MetricsWriter
from claudio.protocols import (
    LevelControl,
    Responder,
    Speaker,
    StrictWake,
    Transcriber,
    WakeWordDetector,
)
from claudio.resume import (
    RESUME_PREFIX,
    DrainedStream,
    ends_the_reply,
    remainder,
    worth_resuming,
)
from claudio.router.normalizer import heard_wake_word, normalize, only_wake_words
from claudio.stt.recovery import CommandRecovery
from claudio.system import SystemControl
from claudio.wakeword.verifier import clipped_name, unclear

log = logging.getLogger(__name__)

PREWAKE_S = 1.5  # audio kept before the wake word: one-breath phrases and the saved WAV
ECHO_MARGIN_DB = 3.0  # a voice must be this much louder than the echo to lower the reply
HELD_FRAMES = 100  # frames kept while transcribing (~8 s), replayed if we listen again
LEAD_IN_S = PREWAKE_S  # one breath: how much of the audio before the detection Whisper gets
# What Whisper hears to check the name. The "claudio" model fires late, on the next word when
# JenJo speaks in one breath: in 1.5 s the "C" was cut off and Whisper wrote "l'audio" (25/10).
WAKE_CHECK_S = 2.5
QUIET_SPEECH_DBFS = -30.0  # loudest 32 ms of a phrase below this: the microphone is too quiet
HISTORY_MAX = 200  # finished interactions kept in memory
WHILE_SPEAKING_S = 3600.0  # the barge-in listener never times out while CLAUDIO speaks
CUT_FINISH_S = 5.0  # a cut reply still arriving from Claude: wait this long for the rest
# The outcomes of a request that was really said to Claudio: the microphone level learns from
# these only (a false wake, noise, the TV in the conversation window or a timeout are not it).
LEVEL_OUTCOMES = frozenset({"ok", "interrupted", "stopped", "paused"})
# The conversation window is deaf while the reply can still be in the room: a phrase heard
# later than this once it listens cannot be the reply's echo, whatever words it shares with
# it (28-29/10: "apri la chat" after "Chiudo la chat." was dropped as CLAUDIO's own voice).
ECHO_GRACE_MS = 300
# Nor can a phrase with more voice than this: the echo left after the reply and its tail is a
# fading word or two (29/10: 3 s of "non si dice sono sempre ragione..." said over the end of
# the reply were dropped as its echo).
ECHO_VOICE_MS = 1000
OTHER_HEARINGS = 3  # of Whisper's other guesses, given to Claude with a phrase to guess
# A phrase said over the reply is judged whole: its slices are only checked for "basta" while
# the voice goes on, up to this long (then it is judged anyway: the TV never pauses).
BARGE_PHRASE_MAX_S = 8.0
# After a slice, no clear voice and no pause for this long: the phrase is judged as it is.
BARGE_RESTART_S = 1.5
# What a window heard while it was deaf (the reply's tail, the beep) is kept this long, and
# put before a phrase whose voice was already there when the window began to listen.
DEAF_FRAMES = 12  # ~1 s of 80 ms frames
DEAF_JOIN_S = 0.3


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
    `min_dbfs`: the loudest 32 ms of the voice must reach this before the microphone's gain
    (the TV and the echo do not): AutoGain moves by up to 12 dB, the room does not.
    `needs_name`: a new request said over the reply must have the name ("Claudio, che ore
    sono") or be a local command; "basta" always works. With speakers the reply's echo is
    never taken for a request (JenJo 26/10: it answered its own voice).
    `echo_similarity`: 0-100, a phrase whose letters are this close to what was just said
    is the echo (echo.py); 0 = only the word check."""

    endpointer: Endpointer
    transcribe: Callable[[np.ndarray], str]
    mode: str = "duck"
    min_dbfs: float = -42.0
    duck_volume: float = 0.3
    needs_name: bool = False
    echo_similarity: float = 70.0
    # A reply cut by a new request goes on ("Dicevo: ...") after that request is answered,
    # if it took less than this (0: never) and the reply was long enough to be worth it.
    resume_s: float = 60.0
    resume_min_chars: int = 200


@dataclass(frozen=True)
class CutReply:
    """A reply a new request cut short: the stream still reading it, the sentences that had
    started playing, when it was cut, its language, whether it answered a request Whisper was
    unsure of, whether it was already a resumed one ("Dicevo: ...") and (once read) the part
    not said."""

    stream: DrainedStream
    played: list[str]
    at: float
    language: str
    unsure: bool = False
    resumed: bool = False
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
        unclear_min: float = 1.0,
        stt_segments: Callable[[], Sequence[Segment]] | None = None,
        unsure_logprob: float | None = None,
        alternatives: int = 0,
        on_heard: Callable[[Hearing], None] | None = None,
        continue_s: float = 0.0,
        echo_tail_ms: int = 0,
        guess: Callable[[Doubt], Awaitable[Recovery | None]] | None = None,
        command_recovery: CommandRecovery | None = None,
        voice_meter: Callable[[np.ndarray], int] | None = None,
        wake_min_voice_ms: int = 0,
        rejected_sink: Callable[[np.ndarray, str], str | None] | None = None,
        input_gain_db: float = 0.0,
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
        # Milliseconds of voice in that audio (Silero): under wake_min_voice_ms Whisper is not
        # even asked, and a clear detection does not wake. None: no check.
        self.voice_meter = voice_meter
        self.wake_min_voice_ms = wake_min_voice_ms
        self._silent_wake = False  # the detection firing now was already rejected as silence
        # Saves what a second chance rejected, apart from the requests, with a cap of its own
        # (the negatives to retrain "claudio" on), and what was heard over a reply that was
        # not for CLAUDIO. None: near misses in audio_sink, as before.
        self.rejected_sink = rejected_sink
        # The microphone's digital boost when there is no AutoGain (level None).
        self.input_gain_db = input_gain_db
        # Whether a phrase is a request: the rules live in intake.py. Their settings are also
        # attributes of the Assistant (properties below).
        self.intake = RequestFilter(
            verify_wake=verify_wake,
            claude_min_words=claude_min_words,
            only_addressed=only_addressed,
            closed_list_max_words=closed_list_max_words,
            retry_language=retry_language,
            noise=noise,
            continue_incomplete=continue_s > 0,
        )
        # Whisper's confidence in each segment of the last transcription (the noise filter).
        # None: not known, only the speech level is judged.
        self.stt_segments = stt_segments
        # After a reply, listen this long for another request without the wake word (0: off).
        self.conversation_s = conversation_s
        # A phrase left halfway ("apri Chrome e cerca...") waits this long for its end (0: off).
        self.continue_s = continue_s
        self._continued: str | None = None  # the text of the phrase that is going on
        self.barge_in = barge_in  # None: the microphone is ignored while speaking
        # A detection scoring below this wakes CLAUDIO only if Whisper hears the name (the
        # second chance): a 0.34 on Claudio's own voice or the TV no longer beeps. 0: off.
        self.confirm_below = confirm_below
        # Wake word models whose every detection Whisper checks ("claudio" alone: noisy).
        self.always_confirm = always_confirm
        self.unclear_min = unclear_min
        # (audio, language) -> Whisper's n-best: tried on a request no command understands.
        self.closed_list = closed_list
        # A request for Claude that Whisper doubted (a segment below `unsure_logprob`; None:
        # never judged) is passed on with up to `alternatives` other transcriptions, through
        # `on_heard`, called before every reply (None: Claude only gets the text).
        self.unsure_logprob = unsure_logprob
        self.alternatives = alternatives
        self.on_heard = on_heard
        # A phrase about to be dropped that may be a misheard command: Claude's guess
        # (LocalResponder.guess) answers it, asks "Hai detto: ...?" or lets it drop.
        self.guess = guess
        # A short phrase no command understands, about to go to Claude or to be dropped:
        # Whisper's second opinion, local and free, before Claude's guess (stt/recovery.py).
        self.command_recovery = command_recovery
        self._candidates: list[str] | None = None  # the closed list's n-best, when it ran
        self._barge_checking = False
        # A phrase heard over the reply while Whisper was busy (audio, loudest dBFS).
        self._barge_pending: tuple[np.ndarray, float] | None = None
        # The slices of a phrase still being said over the reply (audio, loudest dBFS).
        self._barge_slices: list[tuple[np.ndarray, float]] = []
        # What the window heard while deaf, and what it was for this window (metrics).
        self._deaf: deque[np.ndarray] = deque(maxlen=DEAF_FRAMES)
        self._deaf_heard = np.zeros(0, dtype=np.float32)
        self._interrupted = False  # the reply being spoken was cut short by "basta"
        # A voice took over while CLAUDIO spoke: the start of the next request, then the frames
        # heard until the reply has stopped. None: nobody interrupted.
        self._takeover: tuple[np.ndarray, list[np.ndarray]] | None = None
        self._reply_parts: list[str] = []  # the reply being spoken (to recognise its echo)
        self._said = ""  # the reply before the request being handled
        self._heard_echo = False  # the last phrase over the reply was CLAUDIO's own voice
        # The loudest echo of CLAUDIO's voice heard so far (None: none, e.g. headphones).
        self._echo_dbfs: float | None = None
        # When the microphone may be hearing CLAUDIO itself (a beep, a reply, their tail):
        # the wake word and the windows that open after CLAUDIO spoke do not listen then.
        self.own_sound = OwnSound(endpointer.sample_rate, echo_tail_ms)
        self._wake_stale = False  # the wake word model heard CLAUDIO: reset it before use
        # A reply cut by a new request, still read to its end: resumed once that request is
        # answered. None: no reply was cut.
        self._cut: CutReply | None = None
        self._kept_after_empty = False  # the window already outlived one empty phrase
        self._listened = 0  # samples the window heard before its phrase began
        # Frames heard since the last phrase went to Whisper: if it was someone talking, they
        # are the rest of the request.
        self._since_check: deque[np.ndarray] = deque(maxlen=HELD_FRAMES)
        self._checking = False
        self._checks: set[asyncio.Task[None]] = set()
        self._stt_lock = asyncio.Lock()  # Whisper is used by one thing at a time
        self._recent = RingBuffer(int(endpointer.sample_rate * WAKE_CHECK_S))
        self._prewake = np.zeros(0, dtype=np.float32)
        self._wake_audio = np.zeros(0, dtype=np.float32)
        self._held: deque[np.ndarray] = deque(maxlen=HELD_FRAMES)
        self._warned_quiet = False
        self.state = State.IDLE
        self._interaction: Interaction | None = None
        # The microphone the request is heard on: the capture may move to another one before
        # the reply ends, and the phrase's level must not be credited to that one.
        self._heard_on: str | None = None
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

    @property
    def echo_tail_ms(self) -> int:
        """How long the room and the microphone keep a sound of CLAUDIO's own after it came
        out of the speakers: nothing listens to it ([conversation] echo_tail_ms)."""
        return self.own_sound.tail_ms

    @echo_tail_ms.setter
    def echo_tail_ms(self, value: int) -> None:
        self.own_sound.tail_ms = value

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
        own = self.own_sound.in_frame(len(frame))  # CLAUDIO's own sound may be in it
        if self.state == State.IDLE:
            self._on_idle_frame(frame, own)
        elif self.state == State.LISTENING:
            self._on_listen(frame, own)
        elif self.state == State.TRANSCRIBING:
            self._held.append(frame)  # the request itself, if the phrase was only the wake word
        elif self.state == State.SPEAKING and self.barge_in is not None:
            self._on_barge_frame(frame, self.barge_in)

    def _on_idle_frame(self, frame: np.ndarray, own: bool) -> None:
        """The wake word listens: a clear detection wakes CLAUDIO, a weak one or a near miss
        goes to Whisper's second chance."""
        if self._hears_itself(own):
            return
        if isinstance(self.wake, StrictWake):
            self.wake.strict = self.system.off
        self._recent.append(frame.astype(np.float32) / 32768.0)
        detection = self.wake.process(frame)
        if self.system.off and self.second_chance is None:
            # Off without Whisper's check: only a clear wake (the strict threshold) gets the
            # phrase after it transcribed, to look for "accenditi".
            self.wake.pop_near_miss()
            if detection is not None:
                self._on_wake(*detection)
        elif detection is not None and self._weak(*detection):
            if not self._checking:
                log.info("Wake word debole (%.2f): Whisper controlla il nome", detection[1])
            self._on_near_miss(*detection)
        elif detection is not None:
            name, score = detection
            voice = self._wake_voice(name, score)
            if voice is None:
                self._on_wake(name, score)
            elif voice >= self.wake_min_voice_ms:
                self._on_wake(name, score, wake_voice_ms=voice)
        else:
            self._silent_wake = False
            if (miss := self.wake.pop_near_miss()) is not None:
                self._on_near_miss(*miss)

    def _on_wake(self, name: str, score: float, **info: Any) -> None:
        self._interaction = Interaction(
            clock=self.clock, info={"wake_word": name, "wake_score": round(score, 3), **info}
        )
        self._interaction.mark("wake")
        self._heard_on = self._microphone()
        log.info("Wake word rilevata: %s (%.2f)", name, score)
        if not self.system.off:
            self._beep("wake")
        elif "second_chance" in info:
            # Off, a soft tick once Whisper heard the name: "I heard you, say accenditi"
            # (JenJo 30/10). Never on the model's score alone, which fires on noise.
            self._beep("tick")
        recent = self._recent.get()
        self._prewake = recent[-int(PREWAKE_S * self.endpointer.sample_rate) :]
        self._recent.clear()
        self._wake_audio = recent  # the wake word itself, for the verifier
        if self.audio_sink is not None:
            sr = self.endpointer.sample_rate
            self._interaction.info["prewake_ms"] = round(len(self._prewake) * 1000 / sr)
        # The wake word may fire late, while the request has already begun ("hey claudio che
        # ore sono" in one breath): the endpointer can prepend the speech heard just before.
        self.endpointer.reset(lead_in=self._lead_in())
        if self.endpointer.started:  # begun before the wake word: its END may come first
            self._interaction.mark("speech_start")
        self._set_state(State.LISTENING)
        self.responder.prepare()

    def _wake_voice(self, name: str, score: float) -> int | None:
        """Milliseconds of voice around a clear detection (None: no voice meter). A clear
        score needs a voice too, as a weak one does: 31/10 the headset's gate (-87 dBFS) woke
        CLAUDIO twice at 0.64 and 0.77 with nobody talking. Fired for several frames in a
        row on the same silence, it is logged and saved once (~6 ms a check)."""
        if self.voice_meter is None:
            return None
        heard = self._recent.get()
        voice = self.voice_meter(heard)
        if voice < self.wake_min_voice_ms and not self._silent_wake:
            self._silent_wake = True
            log.info("Wake word su silenzio (%.2f, %d ms di voce): la ignoro", score, voice)
            self._log_near_miss(name, score, heard, rejected="no_voice", wake_voice_ms=voice)
        return voice

    def _weak(self, name: str, score: float) -> bool:
        """Whisper checks the name before CLAUDIO wakes: a low score, a noisy model, and
        every detection while off (only the name, or "accenditi", wakes it then)."""
        return self.second_chance is not None and (
            self.system.off or score < self.confirm_below or name in self.always_confirm
        )

    def _unsure_wake(self, name: str, score: float, text: str, voiced: bool) -> bool:
        """A "claudio" detection Whisper did not spell as the name: a clipped "l'audio" or
        "fraudio" (the model fires late, the "C" falls out), or nothing / "Grazie." at a strong
        score (a lone name is too short for Whisper: 0.86-0.95 lost on 26/10). It wakes, but
        the request must have the name or be a command: a "l'audio" at 0.33 trusted outright
        sent a stray "Ciao." to Haiku (26/10, 19:41). Only with verify on, never while off,
        and never on silence: an empty text counts only if the voice meter heard a voice
        (`voiced`; 30/10 14:00:43: a wake on audio too quiet for Whisper, "Nessuna richiesta")."""
        if not self.verify_wake or name not in self.always_confirm or self.system.off:
            return False
        if clipped_name(text):
            return True
        return score >= self.unclear_min and unclear(text) and (bool(text.strip()) or voiced)

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
        """One second chance at a time: the next waits until this one has decided (the
        closed list while off included)."""
        try:
            await self._hear_second_chance(name, score, heard)
        finally:
            self._checking = False

    async def _hear_name(self, heard: np.ndarray) -> tuple[str, int | None, dict[str, Any]]:
        """What Whisper wrote of `heard` ("" if it failed or there was no voice to ask it
        about), the voice meter's reading and the row's fields for both."""
        assert self.second_chance is not None
        text, start, voice = "", self.clock(), None
        try:
            if self.voice_meter is not None:
                voice = await asyncio.to_thread(self.voice_meter, heard)
            if voice is None or voice >= self.wake_min_voice_ms:
                async with self._stt_lock:
                    text = await asyncio.to_thread(self.second_chance, heard)
        except Exception:
            log.exception("Seconda possibilità non riuscita")
        info: dict[str, Any] = {"second_chance_ms": round((self.clock() - start) * 1000)}
        if voice is not None:
            info["wake_voice_ms"] = voice
        return text, voice, info

    async def _hear_second_chance(self, name: str, score: float, heard: np.ndarray) -> None:
        text, voice, info = await self._hear_name(heard)
        # A weak detection said fast often loses the name ("apri cancolatrice"): a command
        # counts. Off, the only command understood is "accenditi".
        language = self.transcriber.language
        weak = self._weak(name, score) and bool(text) and self.responder.understands(text, language)
        if self.state is State.IDLE and text and not (heard_wake_word(text) or weak):
            found = await self._turns_on(heard, text, language)
            if found is not None:
                log.info("Da spento, lista chiusa: %r -> %r", text, found)
                info["second_chance_from"] = text
                text, weak = found, True
        if self.state is not State.IDLE:
            # The model itself woke CLAUDIO meanwhile: a row for the record, no WAV (it was
            # not rejected, and it is no negative to retrain the wake word on).
            self._log_near_miss(name, score, heard, second_chance=text, superseded=True, **info)
            return
        if voice is not None and voice < self.wake_min_voice_ms:
            log.info("Seconda possibilità: nessuna voce (%d ms), niente da fare", voice)
            self._log_near_miss(name, score, heard, rejected="no_voice", **info)
            return
        if heard_wake_word(text) or weak:
            log.info(
                "Seconda possibilità: Whisper ha sentito %r (punteggio %.2f): attivo", text, score
            )
            self._on_wake(name, score, second_chance=text, **info)
            if (
                not self.endpointer.started
                and not only_wake_words(text)
                and self.responder.understands(text, language)
            ):
                # A command said whole before the wake word fired, too long ago for the
                # lead-in ("Claudio, accenditi." heard here, then a timeout): what the second
                # chance heard is the request; it ends with the usual pause.
                self.endpointer.resume(heard)
                assert self._interaction is not None
                self._interaction.mark("speech_start")
            return
        if self._unsure_wake(name, score, text, voiced=bool(voice)):
            # Not a second chance: the request must have the name or be a command (verify).
            log.info("Whisper non ha capito %r (punteggio %.2f): ascolto e verifico", text, score)
            self._on_wake(name, score, wake_unclear=text, **info)
            return
        log.info("Seconda possibilità: Whisper ha sentito %r: niente da fare", text)
        self._log_near_miss(name, score, heard, second_chance=text, rejected="no_name", **info)

    async def _turns_on(self, heard: np.ndarray, text: str, language: str) -> str | None:
        """Off, a second chance that heard words but not the name: Whisper's other
        transcriptions, searched for "accenditi" (30/10: "Grazie a te." was heard, "Dai,
        accenditi." was the second guess). None: not off, nothing heard, or not found."""
        if not (self.system.off and text and self.closed_list is not None):
            return None
        try:
            async with self._stt_lock:
                candidates = await asyncio.to_thread(self.closed_list, heard, language)
        except Exception:
            log.exception("Lista chiusa da spento non riuscita")
            return None
        understands = self.responder.understands
        return next((c for c in candidates if understands(c, language)), None)

    def _log_near_miss(self, name: str, score: float, heard: np.ndarray, **info: Any) -> None:
        """An attempt that did not wake CLAUDIO: its audio, apart from the requests when there
        is a place for it (rejected_sink), and a metrics row. A noisy model (always_confirm)
        leaves only the audio: a row for each of its misfires would fill the session log."""
        noisy = name in self.always_confirm
        sink, kind = self.audio_sink, "near_miss"
        if info.get("superseded"):
            sink = None  # the wake's own WAV has this audio: no copy, and no negative
        elif self.rejected_sink is not None:  # named after why: no_voice, no_name
            sink, kind = self.rejected_sink, str(info.get("rejected", kind))
        elif noisy:
            sink = None
        audio_file = None
        if sink is not None:
            try:
                audio_file = sink(heard, kind)
            except Exception:
                log.exception("Impossibile salvare l'audio della wake word mancata")
        if noisy:
            return
        inter = Interaction(
            clock=self.clock,
            info={"wake_word": name, "wake_score": round(score, 3), "outcome": "near_miss", **info},
        )
        if sink is not None:
            inter.info["audio_file"] = audio_file
        if self.metrics is not None:
            self.metrics.write(inter)

    def _on_listen(self, frame: np.ndarray, own: bool = False) -> None:
        """`own`: CLAUDIO's own sound may be in the frame. A window that opened after CLAUDIO
        spoke (the conversation, the answer to "Confermi?") does not hear it: the end of the
        reply and the beep are still in the room. After a wake word JenJo may already be
        talking over the beep: that phrase is heard whole."""
        assert self._interaction is not None
        if own and self._interaction.info.get("wake_word") in (CONVERSATION, FOLLOW_UP):
            if not self.endpointer.started:
                self._deaf.append(frame)  # kept: the phrase may have begun in it
            return
        if not self.endpointer.started:
            self._listened += len(frame)
        event = self.endpointer.process(frame)
        if event is EndpointEvent.START:
            self._interaction.mark("speech_start")
            self._join_deaf(self._interaction)
            if self._interaction.info.get("wake_word") == CONVERSATION:
                self.responder.prepare()  # no wake word in the conversation window
                sr = self.endpointer.sample_rate
                self._interaction.info["listened_ms"] = round(self._listened * 1000 / sr)
        elif event is EndpointEvent.TIMEOUT:
            if self._interaction.info.get("wake_word") == CONVERSATION:
                log.info("Fine della conversazione: torno in attesa della wake word")
            else:
                log.info("Nessuna richiesta dopo la wake word: torno in attesa")
            self._beep("cancel")
            # Silence: a Claude action keeps waiting (ACTION_TTL_S) for "Hey Claudio, confermo"
            # (24/10: JenJo answered 30 s later and the command had already been dropped).
            self.responder.cancel_pending(silence=True)
            self._finish(outcome="timeout")
        elif event is EndpointEvent.END:
            now = self.clock()
            # The phrase really ended when the trailing silence began.
            self._interaction.marks["speech_end"] = now - self.endpointer.trailing_silence_s
            self._interaction.marks["endpoint"] = now
            self._held.clear()
            self._set_state(State.TRANSCRIBING)
            self._task = asyncio.create_task(self._handle_utterance(self.endpointer.audio()))

    def _join_deaf(self, inter: Interaction) -> None:
        """The phrase began while the conversation window was still deaf (the reply's tail,
        the beep: echo_tail_ms goes mostly on the speakers' delay, 0.17-0.41 s on 30/10).
        When the voice was there as soon as the window listened, the loud end of the deaf
        frames is its start. Only a voice louder than CLAUDIO's echo can be: the reply's end
        never goes to Whisper. Never in the window that answers "Confermi?": there a "sì"
        lost to the deafness only leaves the action undone."""
        if not self._deaf:
            return
        frames = list(self._deaf)
        self._deaf.clear()
        self._deaf_heard = np.concatenate(frames).astype(np.float32) / 32768.0
        sr = self.endpointer.sample_rate
        if inter.info.get("wake_word") != CONVERSATION or self._listened > DEAF_JOIN_S * sr:
            return  # an answer, or a voice that began later: the deafness hid nothing of it
        voice = self._loud_end(frames)
        if not voice.size:
            return
        self.endpointer.prepend(voice)
        inter.info["deaf_joined_ms"] = round(len(voice) * 1000 / sr)
        log.info("La frase era cominciata mentre non ascoltavo: tengo anche quell'audio")

    def _loud_end(self, frames: list[np.ndarray]) -> np.ndarray:
        """The last frames, back to the first that is not a voice at the microphone: louder,
        before the gain, than barge_in_min_dbfs and the echo heard so far (float32)."""
        floor = self.barge_in.min_dbfs if self.barge_in is not None else BargeIn.min_dbfs
        if self._echo_dbfs is not None:
            floor = max(floor, self._echo_dbfs + ECHO_MARGIN_DB)
        kept: list[np.ndarray] = []
        for frame in reversed(frames):
            audio = frame.astype(np.float32) / 32768.0
            win = self.endpointer.vad.window_size
            loudest = max(
                (
                    float(np.dot(w, w)) / len(w)
                    for w in np.split(audio, range(win, len(audio), win))
                ),
                default=0.0,
            )
            if self._before_gain(level_dbfs(loudest)) < floor:
                break
            kept.append(audio)
        return np.concatenate(kept[::-1]) if kept else np.zeros(0, dtype=np.float32)

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        inter = self._interaction
        assert inter is not None
        outcome = "ok"
        follow_up = False
        self._interrupted = False
        self._takeover = None
        # The reply before: the conversation window may be hearing its echo.
        self._said, self._reply_parts = "".join(self._reply_parts), []
        cut, self._cut = self._cut, None
        try:
            text, heard, segments = await self._hear_phrase(audio, inter)
            decision = await self._decide(audio, text, heard, inter, segments)
            outcome = decision.kind.value
            if decision.kind is Kind.RELISTEN:
                self._cut, cut = cut, None  # still the same request
                self._relisten(inter)
                return
            if decision.kind is Kind.CONTINUE:
                self._cut, cut = cut, None  # still the same request
                self._continue(inter, audio, text)
                return
            if decision.dropped:
                self._log_drop(decision, inter)
                if decision.kind is not Kind.FALSE_WAKE:  # a false wake ends in silence
                    await self._beep_now("cancel")
                return
            reply = await self._reply_to(decision, audio, heard, segments, inter)
            self._set_state(State.RESPONDING)
            resume = await self._unsaid(cut)  # the reply this request cut, read to its end
            cut = None
            await self._speak_reply(
                reply, inter, on_first_audio=lambda: self._on_first_audio(inter)
            )
            outcome = self._reply_outcome(outcome)  # before the beep: it may be cancelled
            if outcome in ("paused", "stopped"):
                await self._beep_now("cancel")
            # A confirmation question ("Confermi?"): the answer comes without the wake word.
            follow_up = self.responder.awaiting and not self._interrupted
            if resume and outcome == "ok" and not follow_up and self._takeover is None:
                await self._resume_reply(resume, decision.text, inter)
        except Exception:
            outcome = "error"
            log.exception("Errore durante la richiesta")
            try:
                await self._beep_now("error")
            except Exception:
                log.exception("Impossibile riprodurre il segnale di errore")
        finally:
            if cut is not None:  # the request was dropped: the cut reply is not resumed
                cut.stream.cancel()
            if outcome not in (Kind.RELISTEN.value, Kind.CONTINUE.value):
                await self._close_request(inter, outcome, follow_up)

    async def _hear_phrase(
        self, audio: np.ndarray, inter: Interaction
    ) -> tuple[str, str, list[Segment]]:
        """The phrase as Whisper heard it: its text, its language and Whisper's confidence in
        it. The Whisper lock is free again on return (_decide's passes take it themselves)."""
        self._candidates = None
        continued, self._continued = self._continued, None
        if not self.endpointer.extended:
            continued = None  # left over from a phrase that ended another way
        async with self._stt_lock:
            if continued is not None and not self.endpointer.spoke_again:
                # Nothing more was said: the same phrase, no second Whisper pass.
                text, heard = continued, self.transcriber.language
                segments = list(self.stt_segments()) if self.stt_segments is not None else []
            else:
                text = await asyncio.to_thread(self.transcriber.transcribe, audio)
                heard = self.transcriber.language
                # Whisper's confidence in the first pass: a retry only replaces it when
                # understood.
                segments = list(self.stt_segments()) if self.stt_segments is not None else []
                # Off, the only command understood is "accenditi": the retry and the
                # closed list look for that one (30/10: "Dai, accenditi." was the second
                # transcription of an attempt dropped as "Grazie a te.").
                text, heard = await self._retry(audio, text, heard, inter)
                text = await self._closed_list(audio, text, heard, inter)
        inter.mark("stt_done")
        inter.info["text"] = text
        inter.info["stt_language"] = heard
        log.info("Trascritto (%s): %r", heard, text)
        if "retried_from" in inter.info or "closed_list_from" in inter.info:
            segments = []  # a known command now: the first pass's doubts no longer apply
        return text, heard, segments

    async def _reply_to(
        self,
        decision: Decision,
        audio: np.ndarray,
        heard: str,
        segments: Sequence[Segment],
        inter: Interaction,
    ) -> AsyncIterator[str]:
        """The reply to the phrase, not started yet: the router only runs once it is read. The
        one place where a request reaches the router."""
        if decision.kind is Kind.ASK:  # "Hai detto: ...?": the answer comes next
            return _fixed_reply(decision.text)
        if not self.system.off:  # off: nothing goes to Claude, no n-best pass for it
            await self._tell_hearing(audio, decision.text, heard, segments, inter)
        return self.responder.respond(decision.text, heard)

    def _reply_outcome(self, outcome: str) -> str:
        """`outcome`, unless the route or a barge-in says the reply ended another way."""
        route = self.responder.last_route
        if route.get("route") == "off":
            return "off"  # only "accenditi" is answered, without a sound otherwise
        if route.get("route") == "paused":
            return "paused"  # only the system commands are answered
        if route.get("command") == "stop":
            return "stopped"  # "basta": the conversation is over
        return "interrupted" if self._interrupted else outcome

    async def _close_request(self, inter: Interaction, outcome: str, follow_up: bool) -> None:
        """The request's row, then what listens next. It also runs on a cancel. Nothing may wait
        between _finish and _listen_*: for that moment CLAUDIO looks IDLE to announce,
        handle_typed and close_when_idle."""
        await self._judge_stranded()
        route = self.responder.last_route
        if route:
            inter.info.update(route)
        if not follow_up:
            self.responder.cancel_pending()  # an error or silence never leaves an action armed
        self._finish(outcome)
        if self._takeover is not None:
            self._listen_to_takeover()
        elif follow_up:
            self._listen_for_answer()
        elif self._keeps_conversation(outcome, inter):
            self._listen_for_answer(conversation=True)

    async def _decide(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
        inter: Interaction,
        segments: Sequence[Segment],
    ) -> Decision:
        """The request filter's decision. A false wake gets one more chance: the request may
        have been said after a pause, with the name only in the wake word audio. A short
        phrase no command understands gets Whisper's second opinion, then Claude's guess."""
        info = inter.info
        speech_dbfs = self.endpointer.stats()["speech_dbfs"] if self.endpointer.started else None
        heard = Heard(
            wake_word=str(info.get("wake_word", "")),
            relistened="relisten" in info,
            continued="continued_from" in info,
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
        echo = self._window_echo(text, decision, heard, info)
        if echo and not decision.dropped:
            log.info("La mia voce nel microfono dopo la risposta, la ignoro: %r", text)
            return Decision(Kind.NOT_REQUEST, text, "echo")
        if decision.reason and not decision.dropped and decision.kind is not Kind.CONTINUE:
            info["intake"] = decision.reason  # e.g. quiet_answer: the "sì" was not counted
            log.info("Risposta troppo lontana dal microfono, non la conto: %r", text)
        if not echo and self._may_recover(decision, heard, language):
            recovered = await self._recover_command(audio, decision.text, language, inter)
            if recovered is not None:
                return recovered
        if decision.dropped:
            return await self._recover(decision, language, heard, inter) or decision
        return decision

    def _window_echo(
        self, text: str, decision: Decision, heard: Heard, info: dict[str, Any]
    ) -> bool:
        """The end of the reply, still in the room when the conversation window opened (a
        speaker slower than measured): only a short phrase heard at once."""
        return (
            heard.wake_word == CONVERSATION
            and decision.kind is not Kind.CONTINUE
            and not self.responder.awaiting  # "sì" to "Confermi?" may be in the question
            and info.get("listened_ms", 0) <= ECHO_GRACE_MS
            and self.endpointer.stats()["voice_ms"] <= ECHO_VOICE_MS
            and self._is_echo(text, self._said)
        )

    def _may_recover(self, decision: Decision, heard: Heard, language: str) -> bool:
        """Worth Whisper's second opinion: a phrase about to go to Claude or to be dropped as
        no request, said close to the microphone, with Claudio on and awake and no question
        waiting for its answer. Never a voice far away (the TV), nor silence."""
        if (
            self.command_recovery is None
            or self.system.off  # off: one Whisper pass, for "accenditi" only
            or self.system.paused  # paused: only the system commands, as they were heard
            or self.responder.awaiting  # a yes or a no: no command to look for
            # Claude just asked something: "Pasta." is the answer, not a misheard "basta".
            or self.responder.answer_expected
            # A memory request ("dimenticalo", "cosa ricordi di Anna?") is the router's, not
            # a misheard command: the recovery only knows the commands.
            or self.responder.understands(decision.text, language)
        ):
            return False
        if decision.kind is not Kind.ANSWER and (
            decision.kind not in GUESSED_DROPS or decision.reason in NEVER_GUESSED
        ):
            return False
        noise, dbfs = self.intake.noise, heard.speech_dbfs
        return dbfs is None or noise is None or dbfs >= noise.answer_min_dbfs

    async def _recover_command(
        self, audio: np.ndarray, text: str, language: str, inter: Interaction
    ) -> Decision | None:
        """Whisper's second opinion on a short phrase no command understands: the command it
        proposes, done at once (ANSWER) or asked about first ("Hai detto: ...?", ASK), None
        when there is none (stt/recovery.py, ~0.2 s, free). Any phrase can come through here,
        e.g. one said over Claudio's reply once put back together (IT-1, 2.2). It takes the
        Whisper lock itself: never call it holding it. Whether the phrase is worth it (not the
        echo, not an answer, not while paused or off) is the caller's: see _may_recover()."""
        recovery = self.command_recovery
        if recovery is None or not recovery.wants(text, language):
            return None
        start = self.clock()
        try:
            async with self._stt_lock:
                found = await asyncio.to_thread(recovery, audio, text, language)
        except Exception:
            log.exception("Seconda opinione di Whisper non riuscita")
            return None
        if found is None:
            return None
        inter.info.update(found.info, recovery_ms=round((self.clock() - start) * 1000))
        if found.verdict is Verdict.ACT:
            log.info("Seconda opinione: %r era %r", text, found.text)
            inter.info.update(recovered_from=text, text=found.text)
            return Decision(Kind.ANSWER, found.text, "recovered")
        if found.verdict is Verdict.ASK:
            log.info("Seconda opinione incerta: %r, forse %r", text, found.text)
            inter.info["recovered_from"] = text  # not Haiku's guess, in the session log too
            return Decision(Kind.ASK, self.responder.propose(found.text, language), "recovered")
        log.info("Seconda opinione: %r, Whisper propone %r: scartata", text, found.text)
        return None

    async def _recover(
        self, decision: Decision, language: str, heard: Heard, inter: Interaction
    ) -> Decision | None:
        """A phrase about to be dropped may be a command Whisper misheard ("Spenditi." in the
        conversation window, "Audio Bust." after a reply): Claude's guess turns it into the
        command (sure enough), into "Hai detto: ...?" (not sure) or leaves the drop. The
        echo of Claudio's own voice, a voice far away and silence are never guessed."""
        if (
            self.guess is None
            or decision.kind not in GUESSED_DROPS
            or decision.reason in NEVER_GUESSED
        ):
            return None
        text = decision.text
        noise = self.intake.noise
        doubt = Doubt(
            text=text,
            language=language,
            reason=decision.reason,
            alternatives=self._other_hearings(text),
            wake_word=str(inter.info.get("wake_word", "")),
            wake_score=float(inter.info.get("wake_score", 0.0)),
            name_heard=heard.name_heard or heard_wake_word(text) or clipped_name(text),
            close=(
                None
                if heard.speech_dbfs is None or noise is None
                else heard.speech_dbfs >= noise.answer_min_dbfs
            ),
            logprob=min((s.avg_logprob for s in heard.segments), default=None),
            said_before=self._said,
        )
        try:
            recovery = await self.guess(doubt)
        except Exception:
            log.exception("Ipotesi sulla frase incerta non riuscita")
            return None
        if recovery is None:
            return None
        inter.info.update(recovery.info)
        if recovery.verdict is Verdict.ACT:
            inter.info["guessed_from"] = text
            return Decision(Kind.ANSWER, recovery.said, "guessed")
        if recovery.verdict is Verdict.ASK:
            inter.info["guessed_from"] = text
            return Decision(Kind.ASK, recovery.question, "guessed")
        return None

    def _other_hearings(self, text: str) -> tuple[str, ...]:
        """Whisper's other guesses from the closed list, when it ran (no extra pass)."""
        return _distinct(text, self._candidates or (), OTHER_HEARINGS)

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
        if self.second_chance is None or not self._wake_audio.size or self.system.off:
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
        try:
            candidates = await asyncio.to_thread(self.closed_list, audio, heard)
        except Exception:
            log.exception("Lista chiusa non riuscita")
            return text
        self._candidates = candidates
        inter.info["closed_list_ms"] = round((self.clock() - start) * 1000)
        found = next((c for c in candidates if c != text and understands(c, heard)), None)
        if found is None:
            return text
        log.info("Lista chiusa: %r -> %r", text, found)
        inter.info["closed_list_from"] = text
        return found

    async def _tell_hearing(
        self,
        audio: np.ndarray,
        text: str,
        heard: str,
        segments: Sequence[Segment],
        inter: Interaction,
    ) -> None:
        """Before the reply: how sure Whisper was. Only a request for Claude that Whisper
        doubted gets the note and the alternatives ("apri la cartella desi" -> also heard
        "apri la cartella Tesi"); a local command or a yes/no never pays the extra pass.
        Also whether it was said to Claudio: in the conversation window, as an answer or over
        the reply, without the name, it may be the TV (the memory then asks first)."""
        if self.on_heard is None:
            return
        addressed = inter.info.get("wake_word") not in (CONVERSATION, FOLLOW_UP, BARGE_IN) or (
            heard_wake_word(text)
        )
        hearing = Hearing(addressed=addressed)
        if (
            self.unsure_logprob is not None
            and text
            and unsure(segments, self.unsure_logprob)
            and not self.responder.awaiting
            and not self.responder.understands(text, heard)
        ):
            alternatives = await self._alternatives(audio, text, heard, inter)
            hearing = Hearing(True, alternatives, addressed)
            inter.info["stt_unsure"] = True
            if hearing.alternatives:
                inter.info["stt_alternatives"] = list(hearing.alternatives)
            log.info("Whisper insicuro: %r, forse %s", text, list(hearing.alternatives))
        self.on_heard(hearing)

    async def _alternatives(
        self, audio: np.ndarray, text: str, heard: str, inter: Interaction
    ) -> tuple[str, ...]:
        """Whisper's other guesses, different from `text`: the closed list's, or one more
        n-best pass when it did not run (a long request, a question)."""
        if self.alternatives <= 0 or self.closed_list is None:
            return ()
        candidates = self._candidates
        if candidates is None:
            start = self.clock()
            try:
                async with self._stt_lock:
                    candidates = await asyncio.to_thread(self.closed_list, audio, heard)
            except Exception:
                log.exception("Trascrizioni alternative non riuscite")
                return ()
            inter.info["alternatives_ms"] = round((self.clock() - start) * 1000)
        return _distinct(text, candidates, self.alternatives)

    def _keeps_conversation(self, outcome: str, inter: Interaction) -> bool:
        """After a reply the window stays open. An empty phrase in the window (noise Whisper
        threw away as "Grazie a tutti") keeps it open once more while Claude's question waits
        for its answer (26/10: the window closed and the "Confermo." after it was lost)."""
        system = self.system
        if self.conversation_s <= 0 or system.paused or system.off or system.stopping:
            return False
        if outcome == "ok":
            self._kept_after_empty = False
            return True
        if (
            outcome == Kind.EMPTY.value
            and inter.info.get("wake_word") == CONVERSATION
            and self.responder.answer_expected
            and not self._kept_after_empty
        ):
            self._kept_after_empty = True
            log.info("Frase vuota mentre aspetto una risposta: continuo ad ascoltare")
            return True
        return False

    def _listen_for_answer(self, conversation: bool = False) -> None:
        """Listen again at once, no wake word needed: the answer to the question just asked,
        or (`conversation`) another request within conversation_s of the reply."""
        kind = CONVERSATION if conversation else FOLLOW_UP
        self._interaction = Interaction(
            clock=self.clock, info={"wake_word": kind, "wake_score": 0.0}
        )
        self._interaction.mark("wake")
        self._heard_on = self._microphone()
        self._beep("wake")  # deaf until it and the reply before it are out of the room
        self._listened = 0
        self._deaf.clear()
        self._deaf_heard = np.zeros(0, dtype=np.float32)
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
            self._on_barge_voice(barge)
        elif event is EndpointEvent.TIMEOUT:
            self._restart_barge(barge.endpointer)
        elif event is EndpointEvent.END:
            ep = barge.endpointer
            audio, dbfs, cut = ep.audio(), ep.stats()["speech_dbfs"], ep.cut
            self._restart_barge(ep)
            self._barge_slices.append((audio, dbfs))
            heard = sum(len(a) for a, _ in self._barge_slices)
            if cut and heard < BARGE_PHRASE_MAX_S * ep.sample_rate:
                # The voice goes on: this slice is only checked for "basta", the phrase is
                # judged whole when it ends (30/10: transcribed slice by slice, without the
                # prompt, "Apri la chat." came out "apri la ciotola" and "Grazie a te").
                if not self._barge_checking and self._before_gain(dbfs) >= barge.min_dbfs:
                    self._start_barge_check(audio, barge, dbfs, slice_only=True)
                return
            self._judge_barge_phrase(barge)
        elif self._barge_slices and not barge.endpointer.started:
            ep = barge.endpointer
            since_cut = len(ep.listened())
            if (
                ep.quiet_s * 1000 >= ep.settings.end_silence_ms
                or since_cut >= BARGE_RESTART_S * ep.sample_rate
            ):
                # After a slice the voice stopped (a pause, not a dip of the VAD), or never came
                # back clearly: the phrase is over, with what was said since the cut.
                self._barge_slices.append((self._since_cut(ep), ep.stats()["speech_dbfs"]))
                self._restart_barge(ep)
                self._judge_barge_phrase(barge)

    def _on_barge_voice(self, barge: BargeIn) -> None:
        """A voice starts over the reply: `barge.mode` says what to do before Whisper hears it."""
        if not self._barge_loud(barge):
            return
        if self._sounds_like_echo(barge):
            return  # no louder than CLAUDIO's own echo: wait for Whisper
        if barge.mode == "stop":
            # Everything heard so far is in the phrase (its slices too): nothing to add.
            self._take_over(self._barge_audio(barge.endpointer.partial()), "", rest=[])
        elif barge.mode == "duck":
            self.speaker.duck(barge.duck_volume)
            # Before the gain too: the level [conversation] barge_in_min_dbfs is tuned on.
            level = barge.endpointer.stats()["speech_dbfs"]
            log.info(
                "Abbasso la risposta: voce a %.0f dBFS (%.0f prima del guadagno)",
                level,
                self._before_gain(level),
            )

    def _barge_audio(self, more: np.ndarray) -> np.ndarray:
        """The phrase said over the reply so far: its slices, then `more` (float32)."""
        return np.concatenate([*(a for a, _ in self._barge_slices), more])

    @staticmethod
    def _since_cut(ep: Endpointer) -> np.ndarray:
        """What the barge-in listener heard since its last phrase, trailing silence trimmed
        to ~100 ms, as audio() does."""
        heard = ep.listened()
        keep = len(heard) - round(ep.quiet_s * ep.sample_rate) + ep.sample_rate // 10
        return heard[: max(min(keep, len(heard)), 0)]

    @staticmethod
    def _restart_barge(ep: Endpointer) -> None:
        """A new phrase for the barge-in listener: the samples after the last full window of
        the one that ended go into it, so a cut never leaves a hole in the phrase."""
        rest = ep.leftover()
        ep.reset(timeout_s=WHILE_SPEAKING_S)
        if rest.size:
            ep.process(rest)

    async def _judge_stranded(self) -> None:
        """The reply ended while a phrase over it was still being said: what was heard of it
        (slices only checked for "basta", a phrase not closed yet) is judged now, as at its
        end. Said to CLAUDIO, it starts the next request without a beep (the person is still
        talking); anything else is left alone, as a phrase over the reply always was."""
        barge = self.barge_in
        if barge is None:
            return
        ep = barge.endpointer
        if self._takeover is not None or not (self._barge_slices or ep.started):
            self._barge_slices = []
            return
        audio = self._barge_audio(ep.partial() if ep.started else self._since_cut(ep))
        dbfs = max([d for _, d in self._barge_slices] + [ep.stats()["speech_dbfs"]])
        self._barge_slices = []
        self._restart_barge(ep)
        if not audio.size or self._before_gain(dbfs) < barge.min_dbfs:
            return
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(barge.transcribe, audio)
            if self._takeover is None and self._is_talking_to_me(text):
                # What the listener heard meanwhile is the same phrase going on.
                now = self._barge_audio(ep.partial() if ep.started else ep.listened())
                self._take_over(np.concatenate((audio, now)), text, rest=[])
                self._restart_barge(ep)
            else:
                log.info("Sentito mentre parlo, non è per me: %r", text)
        except Exception:
            log.exception("Ascolto durante la risposta non riuscito")

    def _judge_barge_phrase(self, barge: BargeIn) -> None:
        """The phrase said over the reply is over: its slices, whole, go to Whisper."""
        slices, self._barge_slices = self._barge_slices, []
        if not slices:
            return
        audio = np.concatenate([a for a, _ in slices])
        dbfs = max(d for _, d in slices)
        if self._before_gain(dbfs) < barge.min_dbfs:
            # Logged with its level: the data barge_in_min_dbfs is tuned on.
            log.info(
                "Voce troppo debole mentre parlo (%.0f dBFS, %.0f prima del guadagno): ignorata",
                dbfs,
                self._before_gain(dbfs),
            )
            return
        if self._barge_checking:
            # Still looking at the last phrase: this one waits its turn, never dropped
            # (JenJo 28/10: with the echo keeping Whisper busy, some "stop" were lost).
            prev = self._barge_pending
            if prev is not None:
                audio, dbfs = np.concatenate((prev[0], audio)), max(prev[1], dbfs)
            self._barge_pending = (audio, dbfs)
            return
        self._start_barge_check(audio, barge, dbfs)

    def _start_barge_check(
        self, audio: np.ndarray, barge: BargeIn, dbfs: float, slice_only: bool = False
    ) -> None:
        """`slice_only`: a piece of a phrase still being said, checked for "basta" alone;
        the frames heard meanwhile still belong to the phrase."""
        self._barge_checking = True
        if not slice_only:
            self._since_check.clear()
        task = asyncio.create_task(self._check_barge_in(audio, barge, dbfs, slice_only))
        self._checks.add(task)
        task.add_done_callback(self._checks.discard)

    def _next_barge_check(self, barge: BargeIn) -> None:
        """The phrase heard while Whisper was busy, if the reply is still being spoken."""
        pending, self._barge_pending = self._barge_pending, None
        if pending is not None and self.state is State.SPEAKING and self._takeover is None:
            self._start_barge_check(pending[0], barge, pending[1])

    def _sounds_like_echo(self, barge: BargeIn) -> bool:
        """The voice starting now is no louder than CLAUDIO's own echo already heard: with
        speakers the reply would keep lowering (or stopping) itself on its own voice."""
        if self._echo_dbfs is None:
            return False
        speech = self._before_gain(barge.endpointer.stats()["speech_dbfs"])
        return speech < self._echo_dbfs + ECHO_MARGIN_DB

    def _barge_loud(self, barge: BargeIn) -> bool:
        return self._before_gain(barge.endpointer.stats()["speech_dbfs"]) >= barge.min_dbfs

    async def _check_barge_in(
        self,
        audio: np.ndarray,
        barge: BargeIn,
        dbfs: float | None = None,
        slice_only: bool = False,
    ) -> None:
        """Whisper on what was heard while speaking: "basta" stops the reply, other words
        stop it and become the next request, noise or CLAUDIO's own echo change nothing.
        `slice_only`: a piece of a phrase that goes on, which only "basta" can stop."""
        text = ""
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(barge.transcribe, audio)
        except Exception:
            log.exception("Ascolto durante la risposta non riuscito")
        finally:
            self._barge_checking = False
        if self.state is not State.SPEAKING or self._takeover is not None:
            self._barge_pending = None
            return  # the reply had already ended, or someone already took over
        try:
            if not slice_only:
                self._judge_barge_in(text, audio, dbfs)
            elif text and self.responder.is_stop(text):
                # Only a slice that is just "basta": a stop word inside a longer phrase ("e
                # basta") waits for the whole phrase, which may be a request.
                self._stop_reply(text)
        finally:
            self._next_barge_check(barge)

    def _stop_reply(self, text: str, with_echo: bool = False) -> None:
        """ "basta" was heard over the reply: it stops, and the conversation with it.
        `with_echo`: "stop" said over the reply, written together with the reply's echo."""
        if with_echo:
            log.info("Interrotto a voce (insieme alla mia eco): %r", text)
        else:
            log.info("Interrotto a voce: %r", text)
        self._interrupted = True
        self._barge_slices = []
        if self._interaction is not None:
            self._interaction.info["barge_in"] = text
        self.speaker.interrupt()
        self.speaker.duck(None)

    def _judge_barge_in(self, text: str, audio: np.ndarray, dbfs: float | None) -> None:
        if self._interaction is not None and dbfs is not None:
            self._interaction.info["barge_dbfs"] = round(dbfs, 1)
        if text and self.responder.is_stop(text):
            self._stop_reply(text)
        elif self._is_talking_to_me(text):
            self._take_over(audio, text)
        elif self._says_stop(text):
            self._stop_reply(text, with_echo=True)
        else:
            if self._heard_echo and dbfs is not None:
                # How loud the speakers come back, before the gain: quieter voices no longer
                # lower the reply.
                echo = self._before_gain(dbfs)
                self._echo_dbfs = max(self._echo_dbfs or echo, echo)
            log.info("Sentito mentre parlo, non è per me: %r", text)
            if text and (saved := self._save_rejected(audio, "barge_ignored")):
                log.info("Salvato in %s (%.0f dBFS)", saved, dbfs if dbfs is not None else 0)
            self.speaker.duck(None)

    def _is_talking_to_me(self, text: str) -> bool:
        """Words said over the reply: two or more, the name or a command, and not the
        reply's own echo (with speakers instead of headphones). With needs_name only the
        name or a command."""
        self._heard_echo = False
        if not text:
            return False
        if self._is_echo(text):
            self._heard_echo = True
            log.info("La mia voce nel microfono, la ignoro: %r", text)
            return False
        named = heard_wake_word(text)
        # A memory request over the reply needs the name: the TV's "da ora in poi..." is no
        # command.
        command = self.responder.understands(text, memory=False)
        if not (content_words(text) >= 2 or named or command):
            return False
        if self.barge_in is not None and self.barge_in.needs_name and not (named or command):
            log.info("Sentito mentre parlo, senza il mio nome: lo ignoro: %r", text)
            return False
        return True

    def _says_stop(self, text: str) -> bool:
        """A stop word ("stop", "basta", "zitto"...) somewhere in `text` that the reply itself
        does not say: JenJo's voice mixed with the echo, which Whisper writes as one phrase."""
        reply = "".join(self._reply_parts)
        said = set(normalize(reply, drop_fillers=False).split())
        said |= set(normalize(" ".join(self.speaker.played), drop_fillers=False).split())
        return any(
            w not in said and self.responder.is_stop(w)
            for w in normalize(text, drop_fillers=False).split()
        )

    def _is_echo(self, text: str, reply: str | None = None) -> bool:
        """`text` is what CLAUDIO itself is saying (or `reply`, what it has just said), heard
        back."""
        recent = recent_reply(self.speaker.played, "".join(self._reply_parts) or reply or "")
        # After the reply only its end counts: a follow-up may reuse its words ("e Marte?").
        reply = "".join(self._reply_parts) if reply is None else recent
        if heard_wake_word(text) and not heard_wake_word(reply):
            return False  # the name is JenJo talking to CLAUDIO
        min_similarity = self.barge_in.echo_similarity if self.barge_in is not None else 0.0
        return is_echo(text, recent, reply, min_similarity) or echo_of(text, reply)

    def _take_over(
        self, audio: np.ndarray, text: str, rest: list[np.ndarray] | None = None
    ) -> None:
        """Stop the reply: what was said over it (`audio`, then `rest`, by default the frames
        heard while Whisper listened) is the start of the next request."""
        log.info("Interrotto da una nuova richiesta: %r", text)
        self._takeover = (audio, list(self._since_check) if rest is None else rest)
        self._since_check.clear()
        self._barge_slices = []
        self._interrupted = True
        assert self._interaction is not None
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
        try:
            again = await asyncio.to_thread(
                functools.partial(self.transcriber.transcribe, audio, language=first)
            )
        except Exception:
            log.exception("Ripetizione in %s non riuscita", first)
            return text, heard
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
        self._listen_to_held()

    def _continue(self, inter: Interaction, audio: np.ndarray, text: str) -> None:
        """The phrase was left halfway ("apri Chrome e cerca..."): keep listening up to
        continue_s for its end; then both parts are transcribed together, as one phrase."""
        log.info("Frase lasciata a metà, ascolto il resto: %r", text)
        inter.info["continued_from"] = text
        self._continued = text
        for stage in ("speech_end", "endpoint", "stt_done"):
            inter.marks.pop(stage, None)
        self._prewake = np.concatenate((self._prewake, self.endpointer.listened()))  # saved WAV
        self.endpointer.extend(audio, wait_s=self.continue_s)
        self._listen_to_held()

    def _listen_to_held(self) -> None:
        """Listening again, starting from the frames heard while transcribing."""
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
            if self.on_heard is not None:
                self.on_heard(Hearing())  # typed: nothing misheard
            stream = self._marked(self.responder.respond(text, language), inter)
            if speak:
                await self._sound(
                    self.speaker.speak(
                        stream,
                        on_first_audio=lambda: inter.mark("first_audio"),
                        language=lambda: self.responder.language,
                    )
                )
            else:
                async for _ in stream:
                    pass
        except Exception:
            outcome = "error"
            log.exception("Errore rispondendo al messaggio scritto")
        finally:
            if self.responder.needs_voice:  # level 2-3: confirmed by voice only, never armed here
                self.responder.cancel_pending()
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
            await self._sound(
                self.speaker.speak(text, on_first_audio=on_first_audio, language=lang)
            )
        finally:
            if self._takeover is not None and self.barge_in and self.barge_in.resume_s > 0:
                language_now = lang() if callable(lang) else lang
                played, now = list(self.speaker.played), self.clock()
                unsure = bool(inter.info.get("stt_unsure"))
                self._cut = CutReply(drained, played, now, language_now, unsure)
            else:
                drained.cancel()

    async def _unsaid(self, cut: CutReply | None) -> CutReply | None:
        """The reply this request cut, with its last words (Claude is usually done by now):
        `cut` with only the part that was not said, None when nothing is left or it is not
        worth going on with: the answer to a phrase Whisper was unsure of, a short reply or a
        question (30/10: "Dicevo: Non è chiaro cosa intendi, socio: chi ha fatto cosa?" after
        "Chi ce l'ha fatta?", misheard for "chiudi la chat", which had replaced it)."""
        if cut is None:
            return None
        assert self.barge_in is not None
        if cut.unsure:
            cut.stream.cancel()
            log.info("Non riprendo la risposta interrotta: rispondeva a una frase incerta")
            return None
        await cut.stream.finish(CUT_FINISH_S)
        if self.responder.awaiting:
            return None  # it asked something ("Lo uso?"): not a story to go on with
        if not cut.resumed and not worth_resuming(cut.stream.text, self.barge_in.resume_min_chars):
            # (a resumed reply was judged whole the first time: "Dicevo: <rest>" may be short)
            log.info("Non riprendo la risposta interrotta: breve o una domanda")
            return None
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
        if self._cut is not None:
            # Cut again: still the answer to the request it was judged on, not to this one.
            self._cut = replace(self._cut, unsure=cut.unsure, resumed=True)

    def _on_resume_audio(self) -> None:
        """The resumed reply can be cut again, like any other."""
        if self.barge_in is not None:
            self.barge_in.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
            self._since_check.clear()
            self._barge_slices = []
        self._set_state(State.SPEAKING)

    async def _echoed(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        async for chunk in stream:
            self._reply_parts.append(chunk)  # its echo is recognised like any reply's
            yield chunk

    async def announce(self, text: str) -> None:
        """Say something of Claudio's own (e.g. "ready" at startup) between two requests,
        never over one."""
        while self.state is not State.IDLE:
            await asyncio.sleep(0.05)
        self._set_state(State.RESPONDING)  # the wake word waits until this is said
        try:
            await self._sound(self.speaker.speak(_fixed_reply(text)))
        except Exception:
            log.exception("Avviso a voce non riuscito")
        finally:
            self.wake.reset()
            self._set_state(State.IDLE)

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
        self._barge_pending = None
        if self.barge_in is not None:
            self.barge_in.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
            self._since_check.clear()
            self._barge_slices = []
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
            self._measure_deaf(inter)
            # Not a phrase said over the reply: louder than usual, with the reply's echo in it
            # (JenJo 25/10: the gain fell to +0 dB after one and the next requests were -42 dBFS).
            if self.endpointer.started and inter.info.get("wake_word") != BARGE_IN:
                if self.level is None:
                    self._check_level(stats["speech_dbfs"])
                elif (
                    outcome in LEVEL_OUTCOMES
                    and inter.info.get("text")
                    and self._heard_on == self._microphone()
                ):
                    # Only a request Whisper heard words in: noise, a breath or the TV says
                    # nothing about the voice (26/10 19:50: an empty phrase at -50 dBFS
                    # raised the gain, the next request saturated). And only on the
                    # microphone that heard it.
                    self.level.observe(stats)
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

    def _measure_deaf(self, inter: Interaction) -> None:
        """How much voice the window's deaf start held (deaf_speech_ms): a "sì" said over the
        beep's tail used to be lost without a trace. Only for windows that opened deaf."""
        deaf = self._deaf_heard
        if not deaf.size and self._deaf:
            deaf = np.concatenate(self._deaf).astype(np.float32) / 32768.0
        self._deaf.clear()
        self._deaf_heard = np.zeros(0, dtype=np.float32)
        if not deaf.size or self.voice_meter is None:
            return
        try:
            inter.info["deaf_speech_ms"] = self.voice_meter(deaf)
        except Exception:
            log.exception("Misura della voce sorda non riuscita")

    def _microphone(self) -> str | None:
        """The microphone AutoGain is following now (None: no automatic gain)."""
        return getattr(self.level, "microphone", None)

    def _gain_db(self) -> float:
        """The microphone's digital boost now: AutoGain's, else the configured one."""
        return self.level.gain_db if self.level is not None else self.input_gain_db

    def _before_gain(self, dbfs: float) -> float:
        """A level measured on the boosted microphone, as it was before the boost."""
        return dbfs - self._gain_db()

    def _save_rejected(self, audio: np.ndarray, kind: str) -> str | None:
        """Audio CLAUDIO heard and left alone, apart from the requests (None: not saved)."""
        if self.rejected_sink is None:
            return None
        try:
            return self.rejected_sink(audio, kind)
        except Exception:
            log.exception("Impossibile salvare l'audio scartato")
            return None

    def _check_level(self, speech_dbfs: float) -> None:
        if speech_dbfs < QUIET_SPEECH_DBFS and not self._warned_quiet:
            self._warned_quiet = True
            log.warning(
                "Microfono basso: la voce arriva al massimo a %.0f dBFS (meglio tra -20 e -10). "
                "Aumenta il guadagno, per esempio CLAUDIO_AUDIO__INPUT_GAIN_DB=12 in .env",
                speech_dbfs,
            )

    async def _beep_now(self, kind: str) -> None:
        """A beep, except while off ("spegniti"): then CLAUDIO makes no sound at all."""
        if not self.system.off:
            await self._sound(self.speaker.beep(kind))

    async def _sound(self, play: Awaitable[None]) -> None:
        """Every sound of CLAUDIO's own is played through here (a reply, a beep, "Sono
        pronto"): nothing listens to it nor to its tail, the part the speakers still had
        queued when the player returned and the room's echo of it."""
        self.own_sound.started()
        try:
            await play
        finally:
            self._sound_ended()

    def _sound_ended(self) -> None:
        try:
            queued = self.speaker.queued_s()
        except Exception:  # a sound that never ends would leave CLAUDIO deaf for good
            log.exception("Durata dell'audio ancora in uscita non disponibile")
            queued = 0.0
        self.own_sound.ended(queued)

    def _beep(self, kind: str) -> None:
        """A beep played alongside: the sound starts now, before its task runs. While off
        ("spegniti") only the soft tick of a name Whisper heard is played."""
        if self.system.off and kind != "tick":
            return
        self.own_sound.started()
        task = asyncio.create_task(self.speaker.beep(kind))
        self._beeps.add(task)
        task.add_done_callback(self._beep_done)

    def _beep_done(self, task: asyncio.Task[None]) -> None:
        self._beeps.discard(task)
        self._sound_ended()

    def _hears_itself(self, own: bool) -> bool:
        """While CLAUDIO's own sound may be in the microphone the wake word does not listen
        and nothing is kept for the second chance or the lead-in: Whisper never hears CLAUDIO
        say the name (JenJo 28/10: the low beep that closes a conversation woke the "claudio"
        model at 0.6-0.8, beep after beep). Then the model starts clean."""
        if own:
            self._wake_stale = True
            return True
        if self._wake_stale:
            self._wake_stale = False
            self.wake.reset()
            self._recent.clear()
        return False

    async def wait_idle(self) -> None:
        if self._checks:
            await asyncio.gather(*self._checks, return_exceptions=True)
        # A request that listened again has started a new task: wait for that one too.
        while self._task is not None and not self._task.done():
            await asyncio.gather(self._task, return_exceptions=True)
        if self._beeps:
            await asyncio.gather(*self._beeps, return_exceptions=True)


def _distinct(text: str, candidates: Sequence[str], limit: int) -> tuple[str, ...]:
    """Up to `limit` of `candidates` that differ from `text` and from each other."""
    seen = {normalize(text, drop_fillers=False)}
    out: list[str] = []
    for candidate in candidates:
        key = normalize(candidate, drop_fillers=False)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)
    return tuple(out[:limit])


async def _fixed_reply(text: str) -> AsyncIterator[str]:
    """A fixed sentence of Claudio's own, as a reply stream."""
    yield text


# A level 2-3 action written in the chat window: it is confirmed by voice only.
TYPED_NEEDS_VOICE = {
    "it": "Questa azione va confermata a voce: chiedimela parlando.",
    "en": "This action needs a spoken confirmation: ask me out loud.",
}
