"""openWakeWord detector (ONNX backend: tflite is not supported on Windows).

Several wake words ("hey claudio", "claudio") share one audio front-end, so each extra
model costs only its small classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from claudio.config import WakeWordSettings
from claudio.i18n import tr

log = logging.getLogger(__name__)

FEATURE_MODELS = ("melspectrogram.onnx", "embedding_model.onnx")
NEAR_MISS_FLOOR = 0.15  # default for WakeWordSettings.near_miss_floor


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
        tr(
            "Modello wake word '{model}' non trovato in {folder}. Se lo hai addestrato tu, "
            "copialo lì (scripts/wakeword/README.md); altrimenti esegui: "
            "uv run python scripts/download_models.py",
            model=model,
            folder=wakeword_dir,
        )
    )


class OpenWakeWordDetector:
    def __init__(self, settings: WakeWordSettings, wakeword_dir: Path, frame_size: int = 1280):
        from openwakeword.model import Model

        paths = [find_wakeword_model(m.name, wakeword_dir) for m in settings.models]
        for feature in FEATURE_MODELS:
            if not (wakeword_dir / feature).is_file():
                raise FileNotFoundError(
                    tr("Modello {name} mancante in {folder}", name=feature, folder=wakeword_dir)
                )
        self._model = Model(
            wakeword_models=[str(p) for p in paths],
            inference_framework="onnx",
            melspec_model_path=str(wakeword_dir / FEATURE_MODELS[0]),
            embedding_model_path=str(wakeword_dir / FEATURE_MODELS[1]),
        )
        # openWakeWord names each model after its file stem (e.g. "hey_jarvis_v0.1").
        self.thresholds = {p.stem: m.threshold for p, m in zip(paths, settings.models, strict=True)}
        self.names = [m.name.replace("_", " ") for m in settings.models]
        self._silence = np.zeros(frame_size, dtype=np.int16)
        # Models checked by Whisper on every detection (by openWakeWord's name for them).
        self.confirmed = frozenset(
            p.stem for p, m in zip(paths, settings.models, strict=True) if m.confirm
        )
        near = {k: v for k, v in self.thresholds.items() if k not in self.confirmed}
        self._near_miss = NearMissTracker(near, settings.near_miss_floor)
        log.info(
            "Wake word: %s", ", ".join(f"{k} (soglia {v:.2f})" for k, v in self.thresholds.items())
        )

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        scores = self._model.predict(frame)
        detection = best_detection(scores, self.thresholds)
        self._near_miss.update(scores, detection is not None)
        return detection

    def pop_near_miss(self) -> tuple[str, float] | None:
        miss = self._near_miss.pop()
        if miss is not None:
            name, score = miss
            log.info(
                "Wake word sotto soglia: %s %.2f (soglia %.2f)", name, score, self.thresholds[name]
            )
        return miss

    def reset(self) -> None:
        """Forget the last ~2 s of audio so the same wake word cannot fire twice."""
        for _ in range(25):
            self._model.predict(self._silence)
        self._model.reset()
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
