"""Silero VAD (ONNX, no PyTorch) and the endpointer that closes a phrase."""

from __future__ import annotations

import enum
import threading
from pathlib import Path

import numpy as np

from claudio.audio.level import CLIP_LEVEL
from claudio.audio.ring_buffer import RingBuffer
from claudio.config import VadSettings
from claudio.protocols import SpeechProbability


class SileroVad:
    """Silero VAD v5/v6 ONNX model, 512-sample windows at 16 kHz."""

    window_size = 512
    _context_size = 64

    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        import onnxruntime as ort

        if not model_path.is_file():
            raise FileNotFoundError(f"Modello Silero VAD mancante: {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate((self._context, window.astype(np.float32, copy=False)))[None, :]
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[0, -self._context_size :]
        return float(out[0, 0])


LEAD_IN_GAP_S = 0.25  # a pause this long separates the phrase from what was said before
# A phrase begun before the wake word fired, with nothing said after the wake: it ends this
# long after its last word. Longer than the usual pause, because after "Claudio" alone people
# wait for the beep before the request; short enough that a request already finished when the
# wake word fired late is answered (30/10: "Claudio, spegniti." heard by the second chance,
# then 4 s of silence and a timeout).
LEAD_IN_END_S = 0.9


class EndpointEvent(enum.Enum):
    NONE = "none"
    START = "start"  # speech has started
    END = "end"  # phrase complete: audio() is ready
    TIMEOUT = "timeout"  # nothing said after the wake word


class Endpointer:
    """Turns a stream of int16 frames into one utterance, using a speech probability model."""

    def __init__(self, vad: SpeechProbability, settings: VadSettings, sample_rate: int = 16000):
        self.vad = vad
        self.settings = settings
        self.sample_rate = sample_rate
        self._win = vad.window_size
        self._neg_threshold = max(settings.threshold - 0.15, 0.01)
        self._preroll = RingBuffer(sample_rate * settings.pre_roll_ms // 1000)
        self.reset()

    def reset(self, lead_in: np.ndarray | None = None, timeout_s: float | None = None) -> None:
        """Start a new phrase. `lead_in` (float32) is what was heard just before, wake word
        included: a phrase said in one breath ("hey claudio che ore sono") began before the
        wake word fired, and one said fast may be over by then. `timeout_s` replaces
        no_speech_timeout_s for this phrase (the conversation window)."""
        self._timeout_s = self.settings.no_speech_timeout_s if timeout_s is None else timeout_s
        self.vad.reset()
        self._preroll.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self._chunks: list[np.ndarray] = []
        self._started = False
        self._finished = False
        self._speech_run = 0
        self._silence_run = 0
        self._elapsed = 0
        self._utterance_len = 0
        self._start_len = 0  # the utterance's length when the speech began (pre-roll, lead-in)
        self._listened: list[np.ndarray] = []
        self._max_prob = 0.0
        self._energy = 0.0
        self._loudest = 0.0  # mean square of the loudest window of speech
        self._speech_samples = 0
        self._clipped = 0  # samples of speech at full scale
        self._wait_ms = 0  # extend(): the pause that ends the phrase until the voice comes back
        self._extended = False
        self._spoke_again = False
        self._lead_only = False  # the phrase is all from the lead-in: nothing said since
        self._tell_start = False
        self._cut = False  # ended at max_utterance_s while the voice went on
        self._quiet_run = 0  # before the phrase starts: samples of silence in a row
        if lead_in is not None:
            self._begin_with(lead_in)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def trailing_silence_s(self) -> float:
        return self._silence_run / self.sample_rate

    @property
    def cut(self) -> bool:
        """The phrase ended at max_utterance_s, not on a pause: the voice may go on."""
        return self._cut

    @property
    def quiet_s(self) -> float:
        """Before the phrase starts: how long the audio has been silence, without a break."""
        return self._quiet_run / self.sample_rate

    def leftover(self) -> np.ndarray:
        """The samples after the last full VAD window (float32): a phrase that ended mid-frame
        leaves them, and they belong to what comes next."""
        return self._pending.copy()

    def prepend(self, audio: np.ndarray) -> None:
        """`audio` (float32) came just before the phrase began but was not heard: the window
        was deaf while CLAUDIO's own sound could be in the room. It goes at its start."""
        if not self._started or not audio.size:
            return
        audio = audio.astype(np.float32, copy=False)
        self._chunks.insert(0, audio)
        self._listened.insert(0, audio)  # the saved WAV has it too
        self._utterance_len += len(audio)
        self._start_len += len(audio)

    def process(self, frame: np.ndarray) -> EndpointEvent:
        if self._finished:
            return EndpointEvent.NONE
        samples = frame.astype(np.float32) / 32768.0 if frame.dtype == np.int16 else frame
        self._pending = np.concatenate((self._pending, samples))
        event = EndpointEvent.NONE
        if self._tell_start:  # begun in the lead-in: told with the first frame
            self._tell_start, event = False, EndpointEvent.START
        while len(self._pending) >= self._win and not self._finished:
            window, self._pending = self._pending[: self._win], self._pending[self._win :]
            result = self._on_window(window, self.vad(window))
            if result is not EndpointEvent.NONE:
                event = result
        return event

    def _on_window(self, window: np.ndarray, prob: float) -> EndpointEvent:
        s = self.settings
        self._elapsed += self._win
        self._listened.append(window)
        self._max_prob = max(self._max_prob, prob)
        energy = float(np.dot(window, window))
        self._energy += energy
        if prob >= s.threshold:  # speech only: a beep or a bump does not count as the voice level
            self._loudest = max(self._loudest, energy / len(window))
            self._speech_samples += len(window)
            self._clipped += int(np.count_nonzero(np.abs(window) >= CLIP_LEVEL))
        if not self._started:
            self._preroll.append(window)
            self._speech_run = self._speech_run + self._win if prob >= s.threshold else 0
            self._quiet_run = self._quiet_run + self._win if prob < self._neg_threshold else 0
            if self._speech_run * 1000 >= s.min_speech_ms * self.sample_rate:
                self._started = True
                self._chunks = [self._preroll.get()]
                self._utterance_len = sum(len(c) for c in self._chunks)
                self._start_len = self._utterance_len
                return EndpointEvent.START
            if self._elapsed >= self._timeout_s * self.sample_rate:
                self._finished = True
                return EndpointEvent.TIMEOUT
            return EndpointEvent.NONE

        self._chunks.append(window)
        self._utterance_len += self._win
        if self._wait_ms and prob >= s.threshold:
            self._wait_ms, self._spoke_again = 0, True  # the rest of the phrase: usual pauses
        if self._lead_only and prob >= s.threshold:
            # The phrase goes on after the wake word: usual pauses, and the longer one counts
            # from here (the lead-in and the wait for the beep are like the pre-roll).
            self._lead_only = False
            self._start_len = self._utterance_len - self._win
        self._silence_run = self._silence_run + self._win if prob < self._neg_threshold else 0
        paused = self._silence_run * 1000 >= self._end_silence_ms() * self.sample_rate
        if paused or self._utterance_len >= s.max_utterance_s * self.sample_rate:
            self._finished, self._cut = True, not paused
            return EndpointEvent.END
        return EndpointEvent.NONE

    def _end_silence_ms(self) -> int:
        """The pause that closes the phrase: short for a command ("apri Chrome"), longer once
        the phrase has gone on for long_after_s (a question: people stop to think)."""
        s = self.settings
        if self._wait_ms:
            return self._wait_ms
        if self._lead_only:
            return max(round(LEAD_IN_END_S * 1000), s.end_silence_ms)
        spoken = self._utterance_len - self._start_len - self._silence_run
        if s.long_end_silence_ms > s.end_silence_ms and spoken >= s.long_after_s * self.sample_rate:
            return s.long_end_silence_ms
        return s.end_silence_ms

    def resume(self, audio: np.ndarray, timeout_s: float | None = None) -> None:
        """Start a new phrase that has already begun: `audio` (float32) is its start, heard by
        another listener (the barge-in while CLAUDIO spoke). The next frames continue it."""
        self.reset(timeout_s=timeout_s)
        self._started = True
        self._chunks = [audio.astype(np.float32, copy=False)]
        self._listened = [self._chunks[0]]  # the saved WAV has the whole request
        self._utterance_len = len(audio)
        self._start_len = 0
        # Its level counts, so the noise filter does not take it for someone far away.
        win = self._win
        for i in range(len(audio) // win):
            window = audio[i * win : (i + 1) * win]
            self._loudest = max(self._loudest, float(np.dot(window, window)) / win)

    def extend(self, audio: np.ndarray, wait_s: float) -> None:
        """The phrase `audio` (float32) was left halfway ("apri Chrome e cerca..."): it goes
        on. It ends after `wait_s` of silence if nothing more is said, else with the usual
        pause after the new words; audio() then holds both parts, for one transcription."""
        self.resume(audio)
        self._listened = [np.zeros(0, dtype=np.float32)]  # the saved WAV already has the start
        self._wait_ms = max(int(wait_s * 1000), 1)
        self._extended = True

    @property
    def extended(self) -> bool:
        """This phrase is one left halfway, going on (extend() since the last reset())."""
        return self._extended

    @property
    def spoke_again(self) -> bool:
        """After extend(): more speech was heard (else the phrase is the same as before)."""
        return self._spoke_again

    def partial(self) -> np.ndarray:
        """The phrase heard so far (pre-roll included), float32; empty before it starts."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    def _begin_with(self, lead_in: np.ndarray) -> None:
        """The phrase began before the reset when the lead-in ends with speech, or with a
        pause shorter than LEAD_IN_END_S: its last stretch of speech (back to the pause before
        it) and the silence after it start the utterance. If nothing more is said it ends
        LEAD_IN_END_S after its last word, so a request finished before the wake word fired is
        answered; "Claudio" alone then comes back as only the wake word, and the request after
        it is listened for (the assistant's RELISTEN). The VAD keeps its state: the
        microphone's next frame follows the lead-in."""
        s, win = self.settings, self._win
        n = len(lead_in) // win
        audio = lead_in[len(lead_in) - n * win :].astype(np.float32, copy=False)
        windows = [audio[i * win : (i + 1) * win] for i in range(n)]
        probs = [self.vad(w) for w in windows]
        voiced = [i for i, p in enumerate(probs) if p >= s.threshold]
        if not voiced or (n - 1 - voiced[-1]) * win >= LEAD_IN_END_S * self.sample_rate:
            return  # no speech, or a phrase over long before the wake word: not this one
        quiet = next((i for i, p in enumerate(reversed(probs)) if p >= self._neg_threshold), n)
        gap = max(1, int(LEAD_IN_GAP_S * self.sample_rate) // win)
        start, run = 0, 0
        for i in range(voiced[-1], -1, -1):
            run = run + 1 if probs[i] < self._neg_threshold else 0
            if run >= gap:
                start = i + gap
                break
        start = max(start - 1, 0)  # one window early: soft first sounds
        voiced_samples = sum(p >= s.threshold for p in probs[start:]) * win
        if voiced_samples * 1000 < s.min_speech_ms * self.sample_rate:
            return  # a click, not a word
        self._started = self._lead_only = self._tell_start = True
        self._chunks = windows[start:]
        self._utterance_len = (n - start) * win
        self._start_len = self._utterance_len
        self._silence_run = quiet * win  # the pause after its last word has begun already
        for window, prob in zip(windows[start:], probs[start:], strict=True):
            self._max_prob = max(self._max_prob, prob)
            energy = float(np.dot(window, window))
            if prob >= s.threshold:  # its level counts: it is the request's voice
                self._loudest = max(self._loudest, energy / win)
                self._speech_samples += win
                self._clipped += int(np.count_nonzero(np.abs(window) >= CLIP_LEVEL))

    def listened(self) -> np.ndarray:
        """Everything heard since reset() (for diagnostics), float32."""
        if not self._listened:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._listened)

    def stats(self) -> dict[str, float]:
        """Since reset(): highest speech probability, average level, loudest 32 ms of speech
        (dBFS), the share of speech samples at full scale (clipping) and how much speech was
        heard (ms, pauses left out)."""
        mean_square = self._energy / self._elapsed if self._elapsed else 0.0
        clipped = self._clipped / self._speech_samples if self._speech_samples else 0.0
        return {
            "vad_max": round(self._max_prob, 3),
            "input_dbfs": dbfs(mean_square),
            "speech_dbfs": dbfs(self._loudest),
            "clipped": round(clipped, 4),
            "voice_ms": round(self._speech_samples * 1000 / self.sample_rate),
        }

    def audio(self) -> np.ndarray:
        """The utterance as float32, trailing silence trimmed to ~100 ms."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks)
        keep_tail = self.sample_rate // 10
        trim = max(self._silence_run - keep_tail, 0)
        return audio[: len(audio) - trim] if trim else audio


def dbfs(mean_square: float) -> float:
    return round(10 * float(np.log10(mean_square)), 1) if mean_square > 0 else -120.0


class VoiceMeter:
    """How much voice a stretch of audio holds, with a VAD of its own: the wake word's second
    chance asks it before Whisper, so a click, a key or the headset's gate opening never costs
    a transcription nor wakes CLAUDIO (30/10: "claudio" fired on sounds with Silero at 0.06)."""

    def __init__(self, vad: SpeechProbability, threshold: float, sample_rate: int = 16000):
        self.vad = vad
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._lock = threading.Lock()  # one VAD state: a second chance and a window may ask

    def __call__(self, audio: np.ndarray) -> int:
        """Milliseconds of `audio` (float32) in windows that sound like speech."""
        win = self.vad.window_size
        audio = audio.astype(np.float32, copy=False)
        with self._lock:
            self.vad.reset()
            voiced = sum(
                self.vad(audio[i : i + win]) >= self.threshold
                for i in range(0, len(audio) - win + 1, win)
            )
        return voiced * win * 1000 // self.sample_rate
