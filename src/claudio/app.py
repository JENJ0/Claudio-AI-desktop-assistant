"""Orchestrator and state machine:
IDLE -> LISTENING -> TRANSCRIBING -> RESPONDING -> SPEAKING -> IDLE."""

from __future__ import annotations

import asyncio
import enum
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
from claudio.config import NoiseSettings
from claudio.echo import is_echo, recent_reply
from claudio.intake import (
    BARGE_IN,
    CONVERSATION,
    FOLLOW_UP,
    GUESSED_DROPS,
    NEVER_GUESSED,
    PRECALL,
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
    ACTION_STOPPED,
    RESUME_PREFIX,
    DrainedStream,
    ends_the_reply,
    remainder,
    worth_resuming,
)
from claudio.router.normalizer import heard_wake_word, normalize, only_wake_words
from claudio.stt.recovery import CommandRecovery
from claudio.system import HIDDEN, OFF, PAUSE, SystemControl
from claudio.tts.sentence_splitter import joined_reply
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
CONTINUE_COMMAND = "continue_reply"  # config/commands.toml: "continua", action "resume"
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
NEARBY_CHARS = 1000  # of the PC's folder, program and window names given to the pre-call
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
    # "continua" says again the text still missing of a cut reply for this long (the time after
    # which Claude's context starts over too).
    rest_s: float = 300.0


@dataclass(frozen=True)
class CutReply:
    """A reply a new request cut short: the stream still reading it, the sentences that had
    started playing, when it was cut, whether it answered a request Whisper was
    unsure of, whether it was already a resumed one ("Dicevo: ...") and (once read) the part
    not said."""

    stream: DrainedStream
    played: list[str]
    at: float
    unsure: bool = False
    resumed: bool = False
    rest: str = ""
    handle: object | None = None  # Claude's turn that was cut: "Dicevo" completes its history
    dropped: bool = False  # its action was stopped: not resumed, the words would repeat the promise


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
        conversation_s: float = 0.0,
        barge_in: BargeIn | None = None,
        verify_wake: bool = False,
        closed_list: Callable[[np.ndarray], list[str]] | None = None,
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
        transcriber.hides_text = lambda: self.system.off  # Whisper's own logs: see _kept()
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
        # audio -> Whisper's n-best: tried on a request no command understands.
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
        self._reading = ""  # Haiku's pre-call reading of a request no command matches
        self._barge_checking = False
        # A phrase heard over the reply while Whisper was busy (audio, loudest dBFS).
        self._barge_pending: tuple[np.ndarray, float] | None = None
        # The slices of a phrase still being said over the reply (audio, loudest dBFS).
        self._barge_slices: list[tuple[np.ndarray, float]] = []
        # What the window heard while deaf, and what it was for this window (metrics).
        self._deaf: deque[np.ndarray] = deque(maxlen=DEAF_FRAMES)
        self._deaf_heard = np.zeros(0, dtype=np.float32)
        self._interrupted = False  # the reply being spoken was cut short by "basta"
        # A spoken request is being transcribed or answered and nothing has been said yet: the
        # microphone listens as it does over the reply. Typed requests and announcements never.
        self._listens_while_thinking = False
        # "basta" was heard before the first word: the request ends as stopped.
        self._stop_before_reply = False
        # The phrase the listener is hearing began while Whisper worked on the request: it is
        # not a new request, only a "basta" in it counts.
        self._phrase_stop_only = False
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
        # The text a "continua" says again: a cut reply that was not resumed by itself, or one
        # stopped with "basta" after it was fully written. None: nothing is missing.
        self._unsaid_rest: CutReply | None = None
        responder.can_resume = self._can_resume
        self.system.subscribe(self._forget_rest_when_off)  # also from the tray menu
        self.system.subscribe(self._end_questions_when_silenced)
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
            self._on_thinking_frame(frame, own)
        elif self.state == State.RESPONDING:
            self._on_thinking_frame(frame, own)
        elif self.state == State.SPEAKING and self.barge_in is not None:
            self._on_barge_frame(frame, self.barge_in)

    def _on_thinking_frame(self, frame: np.ndarray, own: bool) -> None:
        """Before the first word of a spoken request's reply the listener of the reply is on,
        except for CLAUDIO's own sound (a beep)."""
        if self._listens_while_thinking and self.barge_in is not None and not own:
            self._on_barge_frame(frame, self.barge_in)

    def _listen_while_thinking(self) -> None:
        """Arm the listener over the reply for a request being transcribed or answered."""
        self._listens_while_thinking = self.barge_in is not None
        self._stop_before_reply = False
        self._phrase_stop_only = False
        self._barge_pending = None
        self._reset_barge_listener()

    def _reset_barge_listener(self) -> None:
        if self.barge_in is not None:
            self.barge_in.endpointer.reset(timeout_s=WHILE_SPEAKING_S)
            self._since_check.clear()
            self._barge_slices = []

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
                log.debug("Wake word debole (%.2f): Whisper controlla il nome", detection[1])
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
            log.debug("Wake word su silenzio (%.2f, %d ms di voce): la ignoro", score, voice)
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
        if not self.intake.verify_wake or name not in self.always_confirm or self.system.off:
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
        off = self.system.off  # the state when it was heard, not after Whisper's pass
        text, voice, info = await self._hear_name(heard)
        # A weak detection said fast often loses the name ("apri cancolatrice"): a command
        # counts. Off, the only command understood is "accenditi".
        weak = self._weak(name, score) and bool(text) and self.responder.understands(text)
        if self.state is State.IDLE and text and not (heard_wake_word(text) or weak):
            found = await self._turns_on(heard, text)
            if found is not None:
                log.info("Da spento, lista chiusa: %r -> %r", HIDDEN, found)
                info["second_chance_from"] = HIDDEN
                text, weak = found, True
        if self.state is not State.IDLE or self.system.off is not off:
            # The model itself woke CLAUDIO meanwhile, or the tray switched it: a row for the
            # record, no WAV (it was not rejected, and it is no negative to retrain the wake
            # word on).
            self._log_near_miss(
                name, score, heard, off=off, second_chance=text, superseded=True, **info
            )
            return
        if voice is not None and voice < self.wake_min_voice_ms:
            log.debug("Seconda possibilità: nessuna voce (%d ms), niente da fare", voice)
            self._log_near_miss(name, score, heard, rejected="no_voice", **info)
            return
        if heard_wake_word(text) or weak:
            log.info(
                "Seconda possibilità: Whisper ha sentito %r (punteggio %.2f): attivo",
                self._kept(text),
                score,
            )
            self._on_wake(name, score, second_chance=self._kept(text), **info)
            if (
                not self.endpointer.started
                and not only_wake_words(text)
                and self.responder.understands(text)
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
        log.info("Seconda possibilità: Whisper ha sentito %r: niente da fare", self._kept(text))
        self._log_near_miss(name, score, heard, second_chance=text, rejected="no_name", **info)

    async def _turns_on(self, heard: np.ndarray, text: str) -> str | None:
        """Off, a second chance that heard words but not the name: Whisper's other
        transcriptions, searched for "accenditi" (30/10: "Grazie a te." was heard, "Dai,
        accenditi." was the second guess). None: not off, nothing heard, or not found."""
        if not (self.system.off and text and self.closed_list is not None):
            return None
        try:
            async with self._stt_lock:
                candidates = await asyncio.to_thread(self.closed_list, heard)
        except Exception:
            log.exception("Lista chiusa da spento non riuscita")
            return None
        understands = self.responder.understands
        return next((c for c in candidates if understands(c)), None)

    def _log_near_miss(
        self, name: str, score: float, heard: np.ndarray, off: bool | None = None, **info: Any
    ) -> None:
        """An attempt that did not wake CLAUDIO: its audio, apart from the requests when there
        is a place for it (rejected_sink), and a metrics row. A noisy model (always_confirm)
        leaves only the audio (none for no_voice, so nothing at all: only hey_claudio's
        no_voice events are counted): a row for each misfire would fill the session log.
        Heard while `off` (default: now): no audio, and none of its words in the row."""
        off = self.system.off if off is None else off
        noisy = name in self.always_confirm
        sink, kind = self.audio_sink, "near_miss"
        if off or info.get("superseded") or info.get("rejected") == "no_voice":
            # superseded: the wake's own WAV has this audio. no_voice: a click or breath, no
            # words to label (53 empty WAVs in 23 min); the row counts it, except for a noisy model.
            sink = None
        elif self.rejected_sink is not None:  # named after why: no_name, barge_ignored
            sink, kind = self.rejected_sink, str(info.get("rejected", kind))
        elif noisy:
            sink = None
        audio_file = None
        if sink is not None:
            try:
                audio_file = sink(heard, kind)
                self._note_wav(sink, audio_file, gain_db=self._gain_db())
            except Exception:
                log.exception("Impossibile salvare l'audio della wake word mancata")
        if noisy:
            return
        inter = Interaction(
            clock=self.clock,
            info={"wake_word": name, "wake_score": round(score, 3), "outcome": "near_miss", **info},
            off=off,
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
        # The state while it is heard, before the router: "spegniti" is said while on.
        self._interaction.off = self.system.off
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
            # Silence: a question, Claude's or a local command's, keeps waiting (ACTION_TTL_S) for
            # "Hey Claudio, confermo" (24/10: JenJo answered 30 s later and it had been dropped).
            self.responder.cancel_pending(silence=True)
            self._finish(outcome="timeout")
        elif event is EndpointEvent.END:
            now = self.clock()
            # The phrase really ended when the trailing silence began.
            self._interaction.marks["speech_end"] = now - self.endpointer.trailing_silence_s
            self._interaction.marks["endpoint"] = now
            self._held.clear()
            self._set_state(State.TRANSCRIBING)
            self._listen_while_thinking()
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
        # Not reset (asked_question reads it): a row takes the route only if this request
        # assigned a new one.
        route_before = self.responder.last_route
        # The reply before: the conversation window may be hearing its echo.
        self._said, self._reply_parts = "".join(self._reply_parts), []
        cut, self._cut = self._cut, None
        try:
            text, segments = await self._hear_phrase(audio, inter)
            decision = await self._decide(audio, text, inter, segments)
            outcome = decision.kind.value
            if decision.kind is Kind.RELISTEN:
                self._cut, cut = cut, None  # still the same request
                self._relisten(inter, segments)
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
            reply = await self._reply_to(decision, audio, segments, inter)
            resume = await self._unsaid(cut)  # the reply this request cut, read to its end
            if self._stop_before_reply:  # "basta" while Whisper worked: nothing is asked or said
                outcome = "stopped"
                await self._beep_now("cancel")
                return  # the reply is not started: the router and Claude never run
            cut = None
            self._set_state(State.RESPONDING)  # the listener armed at TRANSCRIBING goes on
            await self._speak_reply(
                reply, inter, on_first_audio=lambda: self._on_first_audio(inter)
            )
            outcome = self._reply_outcome(outcome)  # before the beep: it may be cancelled
            if outcome in ("paused", "stopped"):
                await self._beep_now("cancel")
            # A confirmation question ("Confermi?"): the answer comes without the wake word.
            follow_up = self.responder.awaiting and not self._interrupted and not self._silenced()
            ok = outcome == "ok" and not follow_up and self._takeover is None
            await self._go_on(resume, decision.text, inter, route_before, ok)
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
                await self._close_request(inter, outcome, follow_up, route_before)

    async def _go_on(
        self,
        resume: CutReply | None,
        request: str,
        inter: Interaction,
        route_before: dict[str, Any],
        ok: bool,
    ) -> None:
        """What comes after the answer: "continua" says the missing text, a local answer given
        over a cut reply goes back to it ("Dicevo"), anything else keeps it for "continua"."""
        barge = self.barge_in
        if self.system.off:
            self._unsaid_rest = None  # nothing is kept while off
        if barge is None:
            return  # nothing is cut without it: nothing to go on with
        route = self.responder.last_route
        mine = route is not route_before  # else this request never reached the router
        if mine and route.get("command") == CONTINUE_COMMAND:
            rest, self._unsaid_rest = self._unsaid_rest, None
            if rest is not None and ok:
                await self._resume_reply(rest, request, inter, barge.rest_s)
            return
        if mine and route.get("route") == "claude" and not self._interrupted:
            self._unsaid_rest = None  # Claude went on from the history: the old text is stale
        if resume is None:
            return
        local = mine and route.get("route") in ("local", "memory")
        if resume.dropped:
            if local and ok:
                await self._say_dropped(inter)
        elif local and ok:
            await self._resume_reply(resume, request, inter, barge.resume_s)
        elif not (self.system.off or ends_the_reply(request)):
            self._unsaid_rest = resume

    async def _hear_phrase(
        self, audio: np.ndarray, inter: Interaction
    ) -> tuple[str, list[Segment]]:
        """The phrase as Whisper heard it: its text and Whisper's confidence in it. The Whisper
        lock is free again on return (_decide's passes take it themselves)."""
        self._candidates = None
        self._reading = ""
        continued, self._continued = self._continued, None
        if not self.endpointer.extended:
            continued = None  # left over from a phrase that ended another way
        async with self._stt_lock:
            if continued is not None and not self.endpointer.spoke_again:
                # Nothing more was said: the same phrase, no second Whisper pass.
                text = continued
                segments = list(self.stt_segments()) if self.stt_segments is not None else []
            else:
                text = await asyncio.to_thread(self.transcriber.transcribe, audio)
                inter.whisper_prompt = getattr(self.transcriber, "last_prompt", "")
                segments = list(self.stt_segments()) if self.stt_segments is not None else []
                # Off, the only command understood is "accenditi": the closed list looks
                # for that one (30/10: "Dai, accenditi." was the second transcription of an
                # attempt dropped as "Grazie a te.").
                text = await self._closed_list(audio, text, inter)
        inter.mark("stt_done")
        inter.info["text"] = text
        log.info("Trascritto: %r", self._kept(text))
        if "closed_list_from" in inter.info:
            segments = []  # a known command now: the first pass's doubts no longer apply
        return text, segments

    async def _reply_to(
        self,
        decision: Decision,
        audio: np.ndarray,
        segments: Sequence[Segment],
        inter: Interaction,
    ) -> AsyncIterator[str]:
        """The reply to the phrase, not started yet: the router only runs once it is read. The
        one place where a request reaches the router."""
        if decision.kind is Kind.ASK:  # "Hai detto: ...?": the answer comes next
            if self._silenced():  # off or paused meanwhile: nothing is asked
                self.responder.cancel_pending()
                return _no_reply()
            return _fixed_reply(decision.text)
        if not self.system.off:  # off: nothing goes to Claude, no n-best pass for it
            await self._tell_hearing(audio, decision.text, segments, inter)
        return self.responder.respond(decision.text, named=decision.named)

    def _reply_outcome(self, outcome: str) -> str:
        """`outcome`, unless the route or a barge-in says the reply ended another way."""
        route = self.responder.last_route
        if route.get("route") == "off":
            return "off"  # only "accenditi" is answered, without a sound otherwise
        if route.get("route") == "paused":
            return "paused"  # only the system commands are answered
        if route.get("command") == "stop" or self._stop_before_reply:
            return "stopped"  # "basta": the conversation is over
        return "interrupted" if self._interrupted else outcome

    async def _close_request(
        self, inter: Interaction, outcome: str, follow_up: bool, route_before: dict[str, Any]
    ) -> None:
        """The request's row, then what listens next. It also runs on a cancel. Nothing may wait
        between _finish and _listen_*: for that moment CLAUDIO looks IDLE to announce,
        handle_typed and close_when_idle."""
        await self._judge_stranded()
        route = self.responder.last_route
        if route and route is not route_before:  # else it never reached the router
            inter.info.update(route)
        if self._silenced():  # off or paused: no question survives, whoever armed it and when
            self.responder.cancel_pending()
        elif not follow_up:
            # An error or silence never leaves an action armed (a phrase said over the reply does:
            # it is the answer); a real cut also drops the offer.
            cut = outcome in ("interrupted", "stopped") and self._takeover is None
            self.responder.cancel_pending(cut=cut, takeover=self._takeover is not None)
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
            deaf_voice_ms=self._deaf_voice_before_answer(info),
        )
        if segments:  # for tuning the noise filter on real logs
            info["stt_no_speech"] = round(max(s.no_speech_prob for s in segments), 3)
            info["stt_logprob"] = round(min(s.avg_logprob for s in segments), 3)
        heard = replace(heard, name_sure=self._relistened_sure(info, heard))
        decision = self.intake.decide(text, heard, self.responder)
        if decision.kind is Kind.FALSE_WAKE and await self._name_before(inter):
            heard = replace(heard, name_heard=True)
            decision = self.intake.decide(text, heard, self.responder)
        # The name counts when the wake audio held it, not only the transcript (a lone "Claudio."
        # before the request, a near miss Whisper heard, a check of the wake word).
        named = heard.name_sure or heard.name_heard or heard.second_chance
        echo = self._window_echo(text, decision, heard, info)
        if echo and not decision.dropped:
            log.info("La mia voce nel microfono dopo la risposta, la ignoro: %r", self._kept(text))
            return Decision(Kind.NOT_REQUEST, text, "echo")
        if decision.reason and not decision.dropped and decision.kind is not Kind.CONTINUE:
            info["intake"] = decision.reason  # quiet_answer, deaf_answer, unsure_yes: not counted
            log.info("Risposta non sentita (%s), la scarto: %r", decision.reason, self._kept(text))
        if not echo and self._may_recover(decision, heard):
            recovered = await self._recover_command(audio, decision.text, inter)
            if recovered is not None:
                return replace(recovered, named=named)
        if decision.dropped:
            recovery = await self._recover(decision, heard, inter) or decision
            return replace(recovery, named=named)
        if decision.kind is Kind.ANSWER:
            precalled = await self._precall(audio, decision, heard, inter)
            if precalled is not None:
                return replace(precalled, named=named)
        return replace(decision, named=named)

    def _relistened_sure(self, info: dict[str, Any], heard: Heard) -> bool:
        """The name of a lone-name phrase counts over an open question only if Whisper was sure
        of it: a hallucinated "Ehi Claudio." must not cancel the question."""
        if not heard.relistened:
            return False
        noise = self.intake.noise or NoiseSettings()
        sure = info.get("relisten_logprob", 0.0) >= noise.name_min_logprob
        return sure or not self.responder.awaiting

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

    def _may_recover(self, decision: Decision, heard: Heard) -> bool:
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
            or self.responder.understands(decision.text)
        ):
            return False
        if decision.kind is not Kind.ANSWER and (
            decision.kind not in GUESSED_DROPS or decision.reason in NEVER_GUESSED
        ):
            return False
        noise, dbfs = self.intake.noise, heard.speech_dbfs
        return dbfs is None or noise is None or dbfs >= noise.answer_min_dbfs

    async def _recover_command(
        self, audio: np.ndarray, text: str, inter: Interaction
    ) -> Decision | None:
        """Whisper's second opinion on a short phrase no command understands: the command it
        proposes, done at once (ANSWER) or asked about first ("Hai detto: ...?", ASK), None
        when there is none (stt/recovery.py, ~0.2 s, free). Any phrase can come through here,
        e.g. one said over Claudio's reply once put back together (IT-1, 2.2). It takes the
        Whisper lock itself: never call it holding it. Whether the phrase is worth it (not the
        echo, not an answer, not while paused or off) is the caller's: see _may_recover()."""
        recovery = self.command_recovery
        if recovery is None or not recovery.wants(text):
            return None
        start = self.clock()
        try:
            async with self._stt_lock:
                found = await asyncio.to_thread(recovery, audio, text)
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
            return Decision(Kind.ASK, self.responder.propose(found.text), "recovered")
        log.info("Seconda opinione: %r, Whisper propone %r: scartata", text, found.text)
        return None

    async def _recover(
        self, decision: Decision, heard: Heard, inter: Interaction
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
        doubt = self._doubt(text, decision.reason, heard, inter, self._other_hearings(text))
        return (await self._guess_decision(doubt, "guessed", inter))[0]

    async def _precall(
        self,
        audio: np.ndarray,
        decision: Decision,
        heard: Heard,
        inter: Interaction,
    ) -> Decision | None:
        """A request Whisper was unsure of and no command understands, before it goes to the
        main model: Haiku reads it (brain/guess.py, [guess] precall). A command it reads is
        decided as a guess is (done, or "Hai detto: ...?"; levels 2-3 ask by themselves), any
        other reading goes to Claude with the request in `_reading`. None: nothing changes."""
        text = decision.text
        if (
            self.guess is None
            or self.on_heard is None
            or self.unsure_logprob is None
            or not text
            or self._silenced()
            or not unsure(heard.segments, self.unsure_logprob)
            or self.responder.awaiting
            or self.responder.answer_expected  # Claude just asked: this is its answer
            or self.responder.understands(text)
        ):
            return None
        alternatives = await self._alternatives(audio, text, inter)
        names = getattr(self.transcriber, "vocabulary", None)  # Whisper's prompt names
        doubt = self._doubt(
            text,
            PRECALL,
            heard,
            inter,
            alternatives,
            nearby=(names() if names else "")[:NEARBY_CHARS],
            # shortcut: the last request's command, empty after a Claude reply; upgrade when
            # the pre-call needs the last local command whatever came between
            last_command=str(self.responder.last_route.get("command", "")),
        )
        decided, recovery = await self._guess_decision(doubt, "precall", inter)
        if decided is None and recovery is not None:
            self._reading = recovery.said
        return decided

    def _doubt(
        self,
        text: str,
        reason: str,
        heard: Heard,
        inter: Interaction,
        alternatives: tuple[str, ...],
        **more: str,
    ) -> Doubt:
        """What is known about a phrase Claude is asked to read (a drop's guess, the pre-call)."""
        noise = self.intake.noise
        return Doubt(
            text=text,
            reason=reason,
            alternatives=alternatives,
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
            **more,
        )

    async def _guess_decision(
        self, doubt: Doubt, reason: str, inter: Interaction
    ) -> tuple[Decision | None, Recovery | None]:
        """Haiku's reading of `doubt` as a Decision: the command done (ANSWER) or asked about
        first (ASK). None when it fails or the verdict is a drop; the Recovery is also returned
        (None: no call), for a reading no command matches."""
        assert self.guess is not None
        try:
            recovery = await self.guess(doubt)
        except Exception:
            log.exception("Ipotesi sulla frase incerta non riuscita")
            return None, None
        if recovery is None:
            return None, None
        inter.info.update(recovery.info)
        if recovery.verdict is Verdict.ACT:
            inter.info["guessed_from"] = doubt.text
            return Decision(Kind.ANSWER, recovery.said, reason), recovery
        if recovery.verdict is Verdict.ASK:
            inter.info["guessed_from"] = doubt.text
            return Decision(Kind.ASK, recovery.question, reason), recovery
        return None, recovery

    def _other_hearings(self, text: str) -> tuple[str, ...]:
        """Whisper's other guesses from the closed list, when it ran (no extra pass)."""
        return _distinct(text, self._candidates or (), OTHER_HEARINGS)

    def _log_drop(self, decision: Decision, inter: Interaction) -> None:
        text = self._kept(decision.text)
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

    async def _closed_list(self, audio: np.ndarray, text: str, inter: Interaction) -> str:
        """The first of Whisper's n-best transcriptions that a command understands, when the
        best one is not understood; `text` when none is (or there is nothing to fix)."""
        understands = self.responder.understands
        if self.closed_list is None or not self.intake.wants_closed_list(text, self.responder):
            return text
        start = self.clock()
        try:
            candidates = await asyncio.to_thread(self.closed_list, audio)
        except Exception:
            log.exception("Lista chiusa non riuscita")
            return text
        self._candidates = candidates
        inter.info["closed_list_ms"] = round((self.clock() - start) * 1000)
        found = next((c for c in candidates if c != text and understands(c)), None)
        if found is None:
            return text
        log.info("Lista chiusa: %r -> %r", self._kept(text), found)
        inter.info["closed_list_from"] = self._kept(text)
        return found

    async def _tell_hearing(
        self,
        audio: np.ndarray,
        text: str,
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
            and not self.responder.understands(text)
        ):
            alternatives = await self._alternatives(audio, text, inter)
            hearing = Hearing(True, alternatives, addressed, self._reading)
            inter.info["stt_unsure"] = True
            if hearing.alternatives:
                inter.info["stt_alternatives"] = list(hearing.alternatives)
            log.info("Whisper insicuro: %r, forse %s", text, list(hearing.alternatives))
        self.on_heard(hearing)

    async def _alternatives(
        self, audio: np.ndarray, text: str, inter: Interaction
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
                    candidates = await asyncio.to_thread(self.closed_list, audio)
            except Exception:
                log.exception("Trascrizioni alternative non riuscite")
                return ()
            inter.info["alternatives_ms"] = round((self.clock() - start) * 1000)
            self._candidates = candidates  # the pre-call and the note to Claude share one pass
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
            if not self._barge_slices:  # a new phrase, not the rest of one
                self._phrase_stop_only = self.state is State.TRANSCRIBING
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
        if self.state is not State.SPEAKING:
            return  # nothing is playing yet: only Whisper's words count
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
        running, waiting = set(self._checks), self._barge_pending is not None
        if running:
            # Checks of phrases heard earlier are still running: their verdict comes first, or
            # a request said then is lost.
            await asyncio.wait(running)
        if waiting and (chained := set(self._checks) - running):
            # The phrase that waited behind them was fully said before the reply ended: its
            # check too, and no more: a TV going on does not chain further.
            await asyncio.wait(chained)
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
            if (
                self._takeover is None
                and not self._phrase_stop_only  # begun in Whisper's time: not a new request
                and self._is_talking_to_me(text)
            ):
                # What the listener heard meanwhile is the same phrase going on.
                now = self._barge_audio(ep.partial() if ep.started else ep.listened())
                self._take_over(np.concatenate((audio, now)), text, rest=[])
                self._restart_barge(ep)
            else:
                log.info("Sentito mentre parlo, non è per me: %r", self._kept(text))
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
        the frames heard meanwhile still belong to the phrase. A phrase heard while Whisper
        works on the request is also checked for "basta" alone: it is not a new request."""
        slice_only = slice_only or self._phrase_stop_only
        self._barge_checking = True
        if not slice_only:
            self._since_check.clear()
        task = asyncio.create_task(self._check_barge_in(audio, barge, dbfs, slice_only))
        self._checks.add(task)
        task.add_done_callback(self._checks.discard)

    def _next_barge_check(self, barge: BargeIn) -> None:
        """The phrase heard while Whisper was busy, if the reply is still being spoken."""
        pending, self._barge_pending = self._barge_pending, None
        if pending is not None and self._hears_over_reply() and self._takeover is None:
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
        if not self._hears_over_reply() or self._takeover is not None:
            self._barge_pending = None
            return  # the reply had already ended, or someone already took over
        try:
            if not slice_only:
                self._judge_barge_in(text, audio, dbfs)
            elif text and self.responder.is_stop(text):
                # Only a slice that is just "basta": a stop word inside a longer phrase ("e
                # basta") waits for the whole phrase, which may be a request.
                self._stop_reply(text)
            elif text and self._phrase_stop_only:
                log.info(
                    "Sentito mentre elaboravo, non è un 'basta': lo ignoro: %r", self._kept(text)
                )
        finally:
            self._next_barge_check(barge)

    def _hears_over_reply(self) -> bool:
        """The listener of the reply is on: the reply is spoken, or a spoken request is still
        being transcribed or answered."""
        return self.state is State.SPEAKING or (
            self._listens_while_thinking and self.state in (State.TRANSCRIBING, State.RESPONDING)
        )

    def _stop_reply(self, text: str, with_echo: bool = False) -> None:
        """ "basta" was heard over the reply: it stops, and the conversation with it.
        `with_echo`: "stop" said over the reply, written together with the reply's echo."""
        if with_echo:
            log.info("Interrotto a voce (insieme alla mia eco): %r", self._kept(text))
        else:
            log.info("Interrotto a voce: %r", self._kept(text))
        if self.state is State.TRANSCRIBING:
            # The request is not answered yet: it ends as stopped before it reaches the router.
            self._stop_before_reply = True
            self._barge_slices = []
            return
        self._interrupted = True
        self._stop_before_reply = self.state is State.RESPONDING  # nothing has been said yet
        self._barge_slices = []
        if self._interaction is not None:
            self._interaction.info["barge_in"] = self._kept(text)
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
            log.info("Sentito mentre parlo, non è per me: %r", self._kept(text))
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
            log.info("La mia voce nel microfono, la ignoro: %r", self._kept(text))
            return False
        named = heard_wake_word(text)
        # A memory request over the reply needs the name: the TV's "da ora in poi..." is no
        # command.
        command = self.responder.understands(text, memory=False)
        if not (content_words(text) >= 2 or named or command):
            return False
        if self.barge_in is not None and self.barge_in.needs_name and not (named or command):
            log.info("Sentito mentre parlo, senza il mio nome: lo ignoro: %r", self._kept(text))
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
        log.info("Interrotto da una nuova richiesta: %r", self._kept(text))
        self._takeover = (audio, list(self._since_check) if rest is None else rest)
        self._since_check.clear()
        self._barge_slices = []
        self._interrupted = True
        assert self._interaction is not None
        self._interaction.info["barge_in"] = self._kept(text) or "voice"
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

    def _relisten(self, inter: Interaction, segments: Sequence[Segment]) -> None:
        """The phrase was only the wake word (said again, or its tail after an early detection):
        listen for the request, starting from what was heard while transcribing."""
        log.info("Solo la wake word: ascolto la richiesta")
        inter.info["relisten"] = inter.info.pop("text")
        inter.info["relisten_logprob"] = round(
            min((s.avg_logprob for s in segments), default=0.0), 3
        )
        for stage in ("speech_start", "speech_end", "endpoint", "stt_done"):
            inter.marks.pop(stage, None)
        self._prewake = np.concatenate((self._prewake, self.endpointer.listened()))  # saved WAV
        self.endpointer.reset()
        self._listen_to_held()

    def _continue(self, inter: Interaction, audio: np.ndarray, text: str) -> None:
        """The phrase was left halfway ("apri Chrome e cerca..."): keep listening up to
        continue_s for its end; then both parts are transcribed together, as one phrase."""
        log.info("Frase lasciata a metà, ascolto il resto: %r", self._kept(text))
        inter.info["continued_from"] = self._kept(text)
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

    async def handle_typed(self, text: str, speak: bool = False) -> str:
        """A message written in the chat window: the same router and Claude as a spoken
        request, between two spoken ones (never during). Silent unless `speak`: the reply is
        shown in the chat. Returns the reply."""
        while self.state is not State.IDLE:  # a spoken request goes first
            await asyncio.sleep(0.05)
        self._set_state(State.RESPONDING)  # the wake word waits until this is answered
        inter = Interaction(clock=self.clock, info={"wake_word": "typed", "text": text})
        inter.mark("stt_done")
        outcome = "ok"
        route_before = self.responder.last_route
        log.info("Scritto nella finestra: %r", text)
        cancelled = self._cancel_spoken_question()
        try:
            if self.on_heard is not None:
                self.on_heard(Hearing())  # typed: nothing misheard
            stream = self._marked(self.responder.respond(text), inter)
            if speak:
                await self._sound(
                    self.speaker.speak(
                        stream,
                        on_first_audio=lambda: self._mark_first_audio(inter),
                    )
                )
            else:
                async for _ in stream:
                    pass
        except Exception:
            outcome = "error"
            log.exception("Errore rispondendo al messaggio scritto")
        finally:
            # shortcut: the rest is shown in the chat, never spoken; upgrade when it should be said.
            if self._typed_continue(route_before):  # silent in the chat: show the rest instead
                rest, self._unsaid_rest = self._unsaid_rest, None
                if rest is not None:
                    inter.info["reply"] = rest.rest
            if self.responder.needs_voice:  # level 2-3: confirmed by voice only, never armed here
                self.responder.cancel_pending()
                note = TYPED_NEEDS_VOICE
                inter.info["reply"] = note
                outcome = "needs_voice"
            if self.responder.last_route is not route_before:  # else it never reached the router
                inter.info.update(self.responder.last_route)
            if cancelled:
                inter.info["reply"] = f"{cancelled} {inter.info.get('reply', '')}".strip()
            inter.mark("done")
            inter.info["outcome"] = outcome
            row = self.metrics.write(inter) if self.metrics else inter.summary()
            self.history.append(row)
            self.wake.reset()
            self._set_state(State.IDLE)
        return str(inter.info.get("reply", ""))

    def _cancel_spoken_question(self) -> str:
        """A question armed by voice is never answered by typing (a typed "sì" confirmed a
        level 2 action, found by Task 6's reviewer): it is cancelled and the text is a new
        request. Returns what the chat says about it ("" when nothing was waiting). The memory's
        "Quale dimentico?" is no spoken action: it can be answered by typing (`needs_voice`)."""
        if not self.responder.needs_voice:
            return ""
        log.info("Messaggio scritto con una domanda a voce in attesa: la annullo")
        note = TYPED_CANCELS_QUESTION
        self.responder.cancel_pending()
        return note

    async def _speak_reply(
        self,
        stream: AsyncIterator[str],
        inter: Interaction,
        on_first_audio: Callable[[], None],
        resumed: bool = False,
    ) -> None:
        """Speak a reply. The stream is read in a task of its own: when a new request cuts the
        reply, the rest still arrives and is kept to be resumed after it. A `resumed` reply
        is not the request's own (the session log keeps the answer)."""
        drained = DrainedStream(stream)
        text = self._echoed(drained.read()) if resumed else self._marked(drained.read(), inter)
        sounding = False

        def first_sound() -> None:
            # Own sound from its first sample, not while it waits for Claude's first words:
            # the microphone listens then (a "basta" while Claude thinks).
            nonlocal sounding
            sounding = True
            self.own_sound.started()
            on_first_audio()

        try:
            await self.speaker.speak(text, on_first_audio=first_sound)
        finally:
            if sounding:
                self._sound_ended()
            handle = None
            if self._interrupted and not resumed:  # before the stream is closed
                handle = self.responder.reply_cut(list(self.speaker.played))
                if handle is not None:
                    self._unsaid_rest = None  # a newer Claude reply: the older text is stale
            kept = self._takeover is not None and self.barge_in and self.barge_in.resume_s > 0
            if kept:
                self.responder.stop_acting()  # kept for its words: no action may follow from it
            if kept and (resumed or self.speaker.played):  # else not a word was heard: no "Dicevo"
                played, now = list(self.speaker.played), self.clock()
                unsure = bool(inter.info.get("stt_unsure"))
                self._cut = CutReply(drained, played, now, unsure, handle=handle)
            else:
                self._keep_stopped(drained, handle)
                drained.cancel()

    def _keep_stopped(self, drained: DrainedStream, handle: object | None) -> None:
        """ "basta" over Claude's reply that was already fully written: "continua" can say the
        rest for free. A stream still being written stays cancelled (no money for words nobody
        hears): then Claude continues from the history."""
        barge = self.barge_in
        if barge is None or handle is None or self._takeover is not None or not drained.ended:
            return
        if not self.speaker.played:
            return  # nothing was heard: nothing for "continua"
        # `awaiting` still matters for a question that was heard whole ("Lo uso?"); one that was
        # not heard stays armed too (marked unheard), so it also ends here.
        if (
            self.responder.awaiting
            or self.responder.action_dropped(handle)
            or not worth_resuming(drained.text, barge.resume_min_chars)
        ):
            return
        played = list(self.speaker.played)
        rest = remainder(drained.text, played)
        if rest:
            self._unsaid_rest = CutReply(drained, played, self.clock(), rest=rest, handle=handle)

    def _typed_continue(self, route_before: dict[str, Any]) -> bool:
        route = self.responder.last_route
        return route is not route_before and route.get("command") == CONTINUE_COMMAND

    def _silenced(self) -> bool:
        return self.system.off or self.system.paused

    def _end_questions_when_silenced(self, what: str) -> None:
        """Off or paused (voice or tray): a question still waiting is over, so a "sì" after
        "accenditi" or during the pause runs nothing."""
        if what in (OFF, PAUSE):
            self.responder.cancel_pending()

    def _forget_rest_when_off(self, what: str) -> None:
        if what == OFF:
            self._unsaid_rest = None

    def _can_resume(self) -> bool:
        """There is a text to say again for "continua": recent, and Claudio is on."""
        rest = self._unsaid_rest
        if rest is None or self.barge_in is None:
            return False
        if self.system.off or self.clock() - rest.at > self.barge_in.rest_s:
            self._unsaid_rest = None
            return False
        return True

    async def _say_dropped(self, inter: Interaction) -> None:
        """The answer to a local request given over a reply whose action was stopped: it was
        left, and "Dicevo" does not repeat its promise."""
        text = ACTION_STOPPED
        self._reply_parts = []
        await self._speak_reply(_fixed_reply(text), inter, self._on_resume_audio, resumed=True)

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
        if cut.handle is not None and self.responder.action_dropped(cut.handle):
            log.info("Non riprendo la risposta interrotta: la sua azione è stata fermata")
            return replace(cut, dropped=True)
        if self.responder.awaiting:
            return None  # it asked something ("Lo uso?"): not a story to go on with
        if not cut.resumed and not worth_resuming(cut.stream.text, self.barge_in.resume_min_chars):
            # (a resumed reply was judged whole the first time: "Dicevo: <rest>" may be short)
            log.info("Non riprendo la risposta interrotta: troppo breve")
            return None
        rest = remainder(cut.stream.text, cut.played)
        return replace(cut, rest=rest) if rest else None

    async def _resume_reply(
        self, cut: CutReply, request: str, inter: Interaction, max_age_s: float
    ) -> None:
        """ "Dicevo: ..." and the rest of a reply, if it was cut no longer than `max_age_s` ago."""
        if self.system.off or ends_the_reply(request) or self.clock() - cut.at > max_age_s:
            return  # off ("spegniti" cut it): nothing more is said
        rest = cut.rest
        log.info("Riprendo la risposta interrotta: %r", rest[:60])
        inter.info["resumed"] = True
        self._reply_parts = []

        async def text() -> AsyncIterator[str]:
            yield f"{RESUME_PREFIX} {rest}"

        await self._speak_reply(text(), inter, self._on_resume_audio, resumed=True)
        if cut.handle is not None:
            self.responder.reply_cut(self._heard_in_all(cut), handle=cut.handle)
        if self._cut is not None:
            # Cut again: still the answer to the request it was judged on, not to this one.
            self._cut = replace(self._cut, unsure=cut.unsure, resumed=True)

    def _heard_in_all(self, cut: CutReply) -> list[str]:
        """What JenJo heard of a resumed reply: the pieces before the one said again, then what
        started playing after "Dicevo:".
        shortcut: one resume; a reply cut again and resumed again loses the words before its
        first cut (and keeps its "Dicevo:"), upgrade when a real session shows it."""
        prefix = RESUME_PREFIX
        again = [p.removeprefix(prefix).strip() for p in self.speaker.played]
        return [*cut.played[:-1], *again]

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
                # At each chunk: the speaker may leave this stream open for good ("basta").
                inter.info["reply"] = joined_reply(parts)
                yield chunk
        finally:
            if parts:  # what was said, also when "basta" cut it short (session log, metrics)
                inter.info["reply"] = joined_reply(parts)

    def _mark_first_audio(self, inter: Interaction) -> None:
        inter.mark("first_audio")
        # Written straight in: mark() keeps the first occurrence, and so must this.
        inter.marks.setdefault("first_audible", inter.clock() + self.speaker.first_sound_delay_s)

    def _on_first_audio(self, inter: Interaction) -> None:
        self._mark_first_audio(inter)
        self._barge_pending = None
        if not self._listens_while_thinking:  # else a phrase may be under way: keep hearing it
            self._reset_barge_listener()
        self._listens_while_thinking = False  # from here the reply itself is what listens
        self._set_state(State.SPEAKING)

    def _finish(self, outcome: str) -> None:
        inter = self._interaction
        if inter is not None:
            inter.mark("done")
            inter.info["outcome"] = outcome
            before_the_name = self._prewake[:0] if inter.off else self._prewake  # said while off
            inter.off = inter.off and self.system.off  # on again: it was the turn-on, kept
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
            if self.audio_sink is not None and not inter.off:
                heard = np.concatenate((before_the_name, self.endpointer.listened()))
                try:
                    inter.info["audio_file"] = self.audio_sink(heard, outcome)
                except Exception:
                    log.exception("Impossibile salvare l'audio della richiesta")
                else:
                    self._note_request_wav(inter)
            row = self.metrics.write(inter) if self.metrics else inter.summary()
            self.history.append(row)
        self._interaction = None
        self._listens_while_thinking = False  # whichever way the request ended
        self._stop_before_reply = False
        self.wake.reset()
        self._set_state(State.IDLE)

    def _note_request_wav(self, inter: Interaction) -> None:
        """Beside the request's WAV: Whisper's prompt and the gain it was captured with (read
        from inter.info: AutoGain has already moved on, for the next request)."""
        self._note_wav(
            self.audio_sink,
            inter.info.get("audio_file"),
            whisper_prompt=inter.whisper_prompt,
            gain_db=inter.info.get("gain_db", self.input_gain_db),
        )

    def _note_wav(self, sink: object, name: str | None, **meta: Any) -> None:
        """A note beside a saved WAV, if the sink writes them. Never stops the request."""
        note = getattr(sink, "note", None)
        if note is None or name is None:
            return
        try:
            note(name, meta)
        except Exception:
            log.exception("Impossibile salvare la nota accanto al WAV %s", name)

    def _deaf_voice_ms(self) -> int | None:
        """How much voice the window's deaf start held (None: not measured). Only for windows
        that opened deaf. Does not clear what was heard: `_measure_deaf` does, at the end."""
        deaf = self._deaf_heard
        if not deaf.size and self._deaf:
            deaf = np.concatenate(self._deaf).astype(np.float32) / 32768.0
        if not deaf.size or self.voice_meter is None:
            return None
        try:
            return self.voice_meter(deaf)
        except Exception:
            log.exception("Misura della voce sorda non riuscita")
            return None

    def _deaf_voice_before_answer(self, info: dict[str, Any]) -> int:
        """The voice lost while deaf before an answer in the window right after a question: its
        head may be missing ("no, aspetta" heard from "aspetta"). A later answer, through a new
        wake word, had no deaf stretch."""
        if not self.responder.awaiting or info.get("wake_word") not in (CONVERSATION, FOLLOW_UP):
            return 0
        return self._deaf_voice_ms() or 0

    def _measure_deaf(self, inter: Interaction) -> None:
        """Writes deaf_speech_ms: a "sì" said over the beep's tail used to be lost without a
        trace."""
        voice = self._deaf_voice_ms()
        self._deaf.clear()
        self._deaf_heard = np.zeros(0, dtype=np.float32)
        if voice is not None:
            inter.info["deaf_speech_ms"] = voice

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
        """Audio CLAUDIO heard and left alone, apart from the requests (None: not saved). Off
        (a phrase over "Mi spengo"): nothing heard stays."""
        if self.rejected_sink is None or self.system.off:
            return None
        try:
            saved = self.rejected_sink(audio, kind)
            self._note_wav(self.rejected_sink, saved, gain_db=self._gain_db())
            return saved
        except Exception:
            log.exception("Impossibile salvare l'audio scartato")
            return None

    def _kept(self, text: str) -> str:
        """What may stay on disk of a phrase just heard, in a log or a field. While off only
        the turn-on: its words are a command's and nothing else, and they are known here,
        once the phrase is transcribed; all the rest is hidden. The row of a request is
        cleaned as a whole: Interaction.off."""
        if not text or not self.system.off or self.responder.understands(text):
            return text
        return HIDDEN

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


async def _no_reply() -> AsyncIterator[str]:
    return
    yield  # makes this an async generator


async def _fixed_reply(text: str) -> AsyncIterator[str]:
    """A fixed sentence of Claudio's own, as a reply stream."""
    yield text


# A typed message came while a question was waiting for a spoken answer.
TYPED_CANCELS_QUESTION = "Ho annullato la domanda a voce."

# A level 2-3 action written in the chat window: it is confirmed by voice only.
TYPED_NEEDS_VOICE = "Questa azione va confermata a voce: chiedimela parlando."
