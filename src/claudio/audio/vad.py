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

    def reset(self) -> None:
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
        self._energy += float(np.dot(window, window))
        if not self._started:
            self._preroll.append(window)
            self._speech_run = self._speech_run + self._win if prob >= s.threshold else 0
            if self._speech_run * 1000 >= s.min_speech_ms * self.sample_rate:
                self._started = True
                self._chunks = [self._preroll.get()]
                self._utterance_len = len(self._chunks[0])
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

    def listened(self) -> np.ndarray:
        """Everything heard since reset() (for diagnostics), float32."""
        if not self._listened:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._listened)

    def stats(self) -> dict[str, float]:
        """Highest speech probability and input level (dBFS) since reset()."""
        rms = np.sqrt(self._energy / self._elapsed) if self._elapsed else 0.0
        level = 20 * np.log10(rms) if rms > 0 else -120.0
        return {"vad_max": round(self._max_prob, 3), "input_dbfs": round(float(level), 1)}

    def audio(self) -> np.ndarray:
        """The utterance as float32, trailing silence trimmed to ~100 ms."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks)
        keep_tail = self.sample_rate // 10
        trim = max(self._silence_run - keep_tail, 0)
        return audio[: len(audio) - trim] if trim else audio
