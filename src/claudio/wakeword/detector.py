"""openWakeWord detector (ONNX backend: tflite is not supported on Windows).

Several wake words ("hey claudio", "claudio") share one audio front-end, so each extra
model costs only its small classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from claudio.config import WakeWordSettings

log = logging.getLogger(__name__)

FEATURE_MODELS = ("melspectrogram.onnx", "embedding_model.onnx")
NEAR_MISS_FLOOR = 0.15  # default for WakeWordSettings.near_miss_floor
# The models read the features of the last 16 frames, each made from ~0.76 s of audio: 25
# frames of silence (2 s) leave nothing of what came before in what they read.
FLUSH_FRAMES = 25


def find_wakeword_model(model: str, wakeword_dir: Path) -> Path:
    """Accept a path, an exact file name, or a prefix like "hey_jarvis"."""
    candidate = Path(model)
    if candidate.is_file():
        return candidate
    name = model if model.endswith(".onnx") else f"{model}.onnx"
    if (wakeword_dir / name).is_file():
        return wakeword_dir / name
    matches = sorted(wakeword_dir.glob(f"{model}_v*.onnx"))
    if matches:
        return matches[-1]  # highest version
    raise FileNotFoundError(
        f"Modello wake word '{model}' non trovato in {wakeword_dir}. Se lo hai addestrato tu, "
        "copialo lì (scripts/wakeword/README.md); altrimenti esegui: "
        "uv run python scripts/download_models.py"
    )


class OpenWakeWordDetector:
    def __init__(self, settings: WakeWordSettings, wakeword_dir: Path, frame_size: int = 1280):
        from openwakeword.model import Model

        paths = [find_wakeword_model(m.name, wakeword_dir) for m in settings.models]
        for feature in FEATURE_MODELS:
            if not (wakeword_dir / feature).is_file():
                raise FileNotFoundError(f"Modello {feature} mancante in {wakeword_dir}")
        self._model = Model(
            wakeword_models=[str(p) for p in paths],
            inference_framework="onnx",
            melspec_model_path=str(wakeword_dir / FEATURE_MODELS[0]),
            embedding_model_path=str(wakeword_dir / FEATURE_MODELS[1]),
        )
        # openWakeWord names each model after its file stem (e.g. "hey_jarvis_v0.1").
        self.thresholds = {p.stem: m.threshold for p, m in zip(paths, settings.models, strict=True)}
        # While CLAUDIO is off ("spegniti") the assistant sets `strict`: higher thresholds.
        self._strict = False
        self.strict_thresholds = raised(self.thresholds, settings.off_threshold_boost)
        self.names = [m.name.replace("_", " ") for m in settings.models]
        self._silence = np.zeros(frame_size, dtype=np.int16)
        # Models checked by Whisper on every detection (by openWakeWord's name for them).
        self.confirmed = frozenset(
            p.stem for p, m in zip(paths, settings.models, strict=True) if m.confirm
        )
        near = {k: v for k, v in self.thresholds.items() if k not in self.confirmed}
        self._near_miss = NearMissTracker(near, settings.near_miss_floor)
        # A new Model starts from the same random-noise features as its reset()
        # (AudioFeatures.__init__): flushed the same way, the first frames read only silence.
        self.reset()
        log.info(
            "Wake word: %s", ", ".join(f"{k} (soglia {v:.2f})" for k, v in self.thresholds.items())
        )

    @property
    def strict(self) -> bool:
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        if value != self._strict:
            log.info(
                "Wake word, soglie %s: %s",
                "da spento" if value else "normali",
                ", ".join(f"{k} {v:.2f}" for k, v in self._active(value).items()),
            )
        self._strict = value

    def _active(self, strict: bool) -> dict[str, float]:
        return self.strict_thresholds if strict else self.thresholds

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        scores = self._model.predict(frame)
        detection = best_detection(scores, self._active(self._strict))
        self._near_miss.update(scores, detection is not None)
        return detection

    def pop_near_miss(self) -> tuple[str, float] | None:
        miss = self._near_miss.pop()
        if miss is not None:
            name, score = miss
            threshold = self._active(self._strict)[name]  # off: the higher one (30/10)
            log.info("Wake word sotto soglia: %s %.2f (soglia %.2f)", name, score, threshold)
        return miss

    def reset(self) -> None:
        """Forget the last ~2 s of audio so the same wake word cannot fire twice. The model's
        own reset() fills what it reads with the features of 4 s of random noise (openWakeWord
        0.6.0, AudioFeatures.reset): ~0.5 s later that noise woke CLAUDIO on silence (30/10:
        4 false wakes with prewake_ms 480, 19 of 100 quiet stretches). The silence goes in
        after it, so the models read only silence when the microphone takes over."""
        self._model.reset()
        for _ in range(FLUSH_FRAMES):
            self._model.predict(self._silence)
        self._near_miss.reset()


class NearMissTracker:
    """Peak score of each burst that rose above NEAR_MISS_FLOOR but never fired.

    These are the attempts the wake word did not catch: their scores tell how far to lower
    the threshold."""

    def __init__(self, thresholds: dict[str, float], floor: float = NEAR_MISS_FLOOR) -> None:
        self.thresholds = thresholds
        self.floor = floor
        self.reset()

    def reset(self) -> None:
        self._peak: tuple[str, float] | None = None
        self._ready: tuple[str, float] | None = None
        self._fired = False

    def update(self, scores: dict[str, float], fired: bool) -> None:
        name, score = max(
            ((n, float(scores.get(n, 0.0))) for n in self.thresholds), key=lambda item: item[1]
        )
        if score >= self.floor:
            self._fired = self._fired or fired
            if self._peak is None or score > self._peak[1]:
                self._peak = (name, score)
        elif self._peak is not None:  # burst over
            if not self._fired:
                self._ready = self._peak
            self._peak, self._fired = None, False

    def pop(self) -> tuple[str, float] | None:
        miss, self._ready = self._ready, None
        return miss


def best_detection(
    scores: dict[str, float], thresholds: dict[str, float]
) -> tuple[str, float] | None:
    """The model furthest above its own threshold, if any."""
    best: tuple[str, float] | None = None
    margin = 0.0
    for name, threshold in thresholds.items():
        score = float(scores.get(name, 0.0))
        if score >= threshold and (best is None or score - threshold > margin):
            best, margin = (name, score), score - threshold
    return best


def raised(thresholds: dict[str, float], boost: float) -> dict[str, float]:
    """Every threshold `boost` higher, never above 0.99 (a score of 1 is rare but possible)."""
    return {name: min(value + boost, 0.99) for name, value in thresholds.items()}
