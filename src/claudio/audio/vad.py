"""Silero VAD (ONNX, no PyTorch) and the endpointer that closes a phrase."""

from __future__ import annotations

import enum
from pathlib import Path

import numpy as np

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


ONE_BREATH_S = 0.3  # speech starting this soon after the wake word continues the same breath
LEAD_IN_GAP_S = 0.25  # a pause this long separates the phrase from what was said before


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

    def reset(self, lead_in: np.ndarray | None = None) -> None:
        """Start a new phrase. `lead_in` (float32) is what was heard just before, wake word
        included: its last stretch of speech is kept in case the phrase was said in one breath
        ("hey claudio che ore sono") and its first words came before the wake word fired."""
        self._lead_in = np.zeros(0, dtype=np.float32)
        if lead_in is not None:
            self._lead_in = self._speech_tail(lead_in)
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
        self._listened: list[np.ndarray] = []
        self._max_prob = 0.0
        self._energy = 0.0
        self._loudest = 0.0  # mean square of the loudest window

    @property
    def started(self) -> bool:
        return self._started

    @property
    def trailing_silence_s(self) -> float:
        return self._silence_run / self.sample_rate

    def process(self, frame: np.ndarray) -> EndpointEvent:
        if self._finished:
            return EndpointEvent.NONE
        samples = frame.astype(np.float32) / 32768.0 if frame.dtype == np.int16 else frame
        self._pending = np.concatenate((self._pending, samples))
        event = EndpointEvent.NONE
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
        self._loudest = max(self._loudest, energy / len(window))
        if not self._started:
            self._preroll.append(window)
            self._speech_run = self._speech_run + self._win if prob >= s.threshold else 0
            if self._speech_run * 1000 >= s.min_speech_ms * self.sample_rate:
                self._started = True
                self._chunks = [self._preroll.get()]
                if self._elapsed <= ONE_BREATH_S * self.sample_rate and self._lead_in.size:
                    self._chunks.insert(0, self._lead_in)
                self._utterance_len = sum(len(c) for c in self._chunks)
                return EndpointEvent.START
            if self._elapsed >= s.no_speech_timeout_s * self.sample_rate:
                self._finished = True
                return EndpointEvent.TIMEOUT
            return EndpointEvent.NONE

        self._chunks.append(window)
        self._utterance_len += self._win
        self._silence_run = self._silence_run + self._win if prob < self._neg_threshold else 0
        if (
            self._silence_run * 1000 >= s.end_silence_ms * self.sample_rate
            or self._utterance_len >= s.max_utterance_s * self.sample_rate
        ):
            self._finished = True
            return EndpointEvent.END
        return EndpointEvent.NONE

    def _speech_tail(self, audio: np.ndarray) -> np.ndarray:
        """The speech at the end of `audio`, back to the last pause; empty if it ends in a pause."""
        win = self._win
        n = len(audio) // win
        audio = audio[len(audio) - n * win :]
        self.vad.reset()
        quiet = [self.vad(audio[i * win : (i + 1) * win]) < self._neg_threshold for i in range(n)]
        gap = max(1, int(LEAD_IN_GAP_S * self.sample_rate) // win)
        start, run = 0, 0
        for i in range(n - 1, -1, -1):
            run = run + 1 if quiet[i] else 0
            if run >= gap:
                start = i + gap
                break
        if start >= n:
            return audio[:0]
        return audio[max(start - 1, 0) * win :]  # one window early: soft first sounds

    def listened(self) -> np.ndarray:
        """Everything heard since reset() (for diagnostics), float32."""
        if not self._listened:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._listened)

    def stats(self) -> dict[str, float]:
        """Highest speech probability, average level and loudest 32 ms (dBFS) since reset()."""
        mean_square = self._energy / self._elapsed if self._elapsed else 0.0
        return {
            "vad_max": round(self._max_prob, 3),
            "input_dbfs": dbfs(mean_square),
            "speech_dbfs": dbfs(self._loudest),
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
