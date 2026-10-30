from collections import deque
from pathlib import Path

import numpy as np
import pytest

from claudio.config import WakeWordModel, WakeWordSettings
from claudio.wakeword.detector import (
    FEATURE_MODELS,
    NearMissTracker,
    OpenWakeWordDetector,
    best_detection,
    find_wakeword_model,
)

THRESHOLDS = {"hey_claudio": 0.5, "claudio": 0.7}


def test_no_detection_below_thresholds() -> None:
    assert best_detection({"hey_claudio": 0.4, "claudio": 0.65}, THRESHOLDS) is None


def test_single_word_needs_its_own_stricter_threshold() -> None:
    assert best_detection({"hey_claudio": 0.1, "claudio": 0.75}, THRESHOLDS) == ("claudio", 0.75)


def test_best_margin_wins() -> None:
    scores = {"hey_claudio": 0.9, "claudio": 0.8}
    assert best_detection(scores, THRESHOLDS) == ("hey_claudio", 0.9)


def test_find_model_by_prefix_and_version(tmp_path: Path) -> None:
    for name in ("hey_jarvis_v0.1.onnx", "hey_claudio.onnx", "claudio.onnx"):
        (tmp_path / name).write_bytes(b"x")
    assert find_wakeword_model("hey_jarvis", tmp_path).name == "hey_jarvis_v0.1.onnx"
    assert find_wakeword_model("claudio", tmp_path).name == "claudio.onnx"
    assert find_wakeword_model("hey_claudio", tmp_path).name == "hey_claudio.onnx"
    with pytest.raises(FileNotFoundError, match="download_models"):
        find_wakeword_model("alexa", tmp_path)


def test_settings_default_and_list() -> None:
    assert WakeWordSettings().models[0].name == "hey_jarvis"
    s = WakeWordSettings(
        models=[WakeWordModel(name="hey_claudio"), WakeWordModel(name="claudio", threshold=0.6)]
    )
    assert [m.threshold for m in s.models] == [0.5, 0.6]


def feed_scores(tracker: NearMissTracker, scores: list[float], fire_at: float = 0.5) -> list:
    misses = []
    for score in scores:
        tracker.update({"hey_claudio": score, "claudio": 0.0}, fired=score >= fire_at)
        if (miss := tracker.pop()) is not None:
            misses.append(miss)
    return misses


def test_near_miss_reports_the_peak_once_the_burst_ends() -> None:
    tracker = NearMissTracker(THRESHOLDS)
    assert feed_scores(tracker, [0.0, 0.2, 0.42, 0.3, 0.05, 0.0]) == [("hey_claudio", 0.42)]
    assert feed_scores(tracker, [0.0] * 5) == []


def test_no_near_miss_for_noise_or_a_detection() -> None:
    tracker = NearMissTracker(THRESHOLDS)
    assert feed_scores(tracker, [0.05, 0.1, 0.02, 0.0]) == []  # under the floor
    assert feed_scores(tracker, [0.2, 0.6, 0.3, 0.0]) == []  # it fired


def test_near_miss_floor_is_configurable() -> None:
    from claudio.config import WakeWordSettings

    settings = WakeWordSettings()
    assert settings.near_miss_floor == 0.15 and settings.second_chance
    tracker = NearMissTracker(THRESHOLDS, floor=0.3)
    assert feed_scores(tracker, [0.1, 0.25, 0.1, 0.0]) == []
    assert feed_scores(NearMissTracker(THRESHOLDS, floor=0.2), [0.1, 0.25, 0.1, 0.0]) != []


def test_near_miss_reset_drops_a_pending_burst() -> None:
    tracker = NearMissTracker(THRESHOLDS)
    feed_scores(tracker, [0.3])
    tracker.reset()
    assert feed_scores(tracker, [0.0]) == []


class NoisyResetModel:
    """openWakeWord's Model as its reset() behaves: what the models read is filled with the
    features of random noise, which scores high until 16 new frames have pushed it out."""

    WINDOW = 16

    def __init__(self, **_kwargs: object) -> None:
        self.window: deque[str] = deque(maxlen=self.WINDOW)

    def reset(self) -> None:
        self.window.extend(["noise"] * self.WINDOW)

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        self.window.append("audio")
        return {"claudio": 0.9 if "noise" in self.window else 0.0}


def test_reset_leaves_no_noise_for_the_microphone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """30/10: a wake ~0.5 s after every beep, on silence: the noise of the model's own reset
    was still in what it read when the microphone's frames arrived."""
    for name in ("claudio.onnx", *FEATURE_MODELS):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr("openwakeword.model.Model", NoisyResetModel)
    settings = WakeWordSettings(models=[WakeWordModel(name="claudio", threshold=0.25)])
    detector = OpenWakeWordDetector(settings, tmp_path)
    detector.reset()
    quiet = np.zeros(1280, dtype=np.int16)
    assert all(detector.process(quiet) is None for _ in range(NoisyResetModel.WINDOW + 4))
