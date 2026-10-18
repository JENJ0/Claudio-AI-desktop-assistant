from pathlib import Path

import pytest

from claudio.config import WakeWordModel, WakeWordSettings
from claudio.wakeword.detector import best_detection, find_wakeword_model

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
