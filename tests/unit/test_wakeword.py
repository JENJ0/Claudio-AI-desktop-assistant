from collections import deque
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from claudio.config import WakeWordModel, WakeWordSettings
from claudio.wakeword.detector import (
    FEATURE_MODELS,
    FLUSH_FRAMES,
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
    features of random noise, which scores high until 16 new frames have pushed it out. A new
    Model starts the same way (AudioFeatures.__init__ makes the same random noise)."""

    WINDOW = 16

    def __init__(self, **_kwargs: object) -> None:
        self.window: deque[str] = deque(maxlen=self.WINDOW)
        self.reset()

    def reset(self) -> None:
        self.window.extend(["noise"] * self.WINDOW)

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        self.window.append("audio")
        return {"claudio": 0.9 if "noise" in self.window else 0.0}


def fake_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: type
) -> OpenWakeWordDetector:
    for name in ("claudio.onnx", "hey_claudio.onnx", *FEATURE_MODELS):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr("openwakeword.model.Model", model)
    settings = WakeWordSettings(
        models=[
            WakeWordModel(name="hey_claudio", threshold=0.3),
            WakeWordModel(name="claudio", threshold=0.25, confirm=True),
        ]
    )
    return OpenWakeWordDetector(settings, tmp_path)


def test_reset_leaves_no_noise_for_the_microphone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """30/10: a wake ~0.5 s after every beep, on silence: the noise of the model's own reset
    was still in what it read when the microphone's frames arrived."""
    detector = fake_detector(tmp_path, monkeypatch, NoisyResetModel)
    detector.reset()
    quiet = np.zeros(1280, dtype=np.int16)
    assert all(detector.process(quiet) is None for _ in range(NoisyResetModel.WINDOW + 4))


def test_a_new_detector_leaves_no_noise_for_the_microphone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new openWakeWord Model reads the same random noise as after its reset(): the first
    frames from the microphone must find only silence there too."""
    detector = fake_detector(tmp_path, monkeypatch, NoisyResetModel)
    quiet = np.zeros(1280, dtype=np.int16)
    assert all(detector.process(quiet) is None for _ in range(NoisyResetModel.WINDOW + 4))


class ScriptedModel:
    """Scores "hey_claudio" from a list, one per frame."""

    scores: ClassVar[list[float]] = []

    def __init__(self, **_kwargs: object) -> None:
        self.frames = 0

    def reset(self) -> None:
        self.frames = -FLUSH_FRAMES  # the detector's silence after a reset is not in the script

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        self.frames += 1
        score = self.scores[self.frames - 1] if 1 <= self.frames <= len(self.scores) else 0.0
        return {"hey_claudio": score, "claudio": 0.0}


def test_the_log_says_which_thresholds_are_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """30/10: off, a 0.31 was logged "under the threshold 0.30" while the threshold in use
    was 0.40."""
    monkeypatch.setattr(ScriptedModel, "scores", [0.0, 0.31, 0.0])
    detector = fake_detector(tmp_path, monkeypatch, ScriptedModel)
    caplog.clear()  # not the thresholds it starts with
    with caplog.at_level("INFO"):
        detector.strict = True
        detector.strict = True  # once per change
        for _ in range(3):
            assert detector.process(np.zeros(1280, dtype=np.int16)) is None
        assert detector.pop_near_miss() == ("hey_claudio", 0.31)
        detector.strict = False
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [
        "Wake word, soglie da spento: hey_claudio 0.40, claudio 0.35",
        "Wake word sotto soglia: hey_claudio 0.31 (soglia 0.40)",
        "Wake word, soglie normali: hey_claudio 0.30, claudio 0.25",
    ]
