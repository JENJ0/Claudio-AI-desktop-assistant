"""benchmark_wakeword.py: the counting and the threshold choice, with a fake detector."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import benchmark_wakeword as bw


def test_style_comes_from_the_file_name() -> None:
    assert bw.style_of("hey_claudio_012_veloce", "hey_claudio") == "veloce"
    assert bw.style_of("hey_claudio_007_comando_cmd", "hey_claudio") == "comando_cmd"
    assert bw.style_of("claudio_003_piano", "hey_claudio") == "altro"


def test_wakes_count_like_the_runtime() -> None:
    frames_2s = 25
    burst = [0.1] * 3 + [0.9] * 4 + [0.1] * 3
    assert bw.wakes(burst, 0.5) == 1  # one burst, one detection
    assert bw.wakes(burst + [0.0] * 30 + burst, 0.5) == 2
    assert bw.wakes([0.9] * (frames_2s + 2), 0.5) == 2  # still loud after the 2 s pause
    assert bw.wakes(burst, 0.95) == 0


def test_takes_heard_are_counted_by_style() -> None:
    peaks = {"a": ("normale", 0.9), "b": ("normale", 0.2), "c": ("veloce", 0.4)}
    assert bw.heard_by_style(peaks, 0.35) == {"normale": (1, 2), "veloce": (1, 1)}
    assert bw.heard_by_style(peaks, 0.95) == {"normale": (0, 2), "veloce": (0, 1)}


def rows(*items: tuple[float, int, int, float]) -> list[dict[str, float]]:
    keys = ("threshold", "heard", "sentences", "noise_per_hour")
    return [dict(zip(keys, item, strict=True)) for item in items]


def test_suggest_wants_quiet_sentences_and_room_before_more_takes() -> None:
    table = rows((0.2, 45, 3, 9.0), (0.3, 40, 0, 4.0), (0.4, 38, 0, 0.5), (0.5, 38, 0, 0.0))
    assert bw.suggest(table, 1.0)["threshold"] == 0.5  # the highest of the equals
    assert bw.suggest(table, 5.0)["threshold"] == 0.3  # more takes, within the budget
    assert bw.suggest(rows((0.2, 45, 3, 9.0)), 1.0) is None


class Fake:
    """Scores 0.9 for audio whose level is high, the way a take with the word would."""

    def __call__(self, audio: np.ndarray) -> list[float]:
        frames = [audio[i : i + 1280] for i in range(0, len(audio) - 1279, 1280)]
        return [0.9 if np.abs(f).max() > 3000 else 0.05 for f in frames]


def write(path: Path, level: int, seconds: float = 1.0) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(int(16000 * seconds), level, dtype=np.int16), 16000, subtype="PCM_16")


def test_benchmark_prints_the_table_and_the_pick(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for i, style in enumerate(("normale", "normale", "veloce")):
        write(tmp_path / "claudio" / f"claudio_{i:03d}_{style}.wav", 9000 if i < 2 else 100)
    write(tmp_path / "negatives" / "neg_000.wav", 100)
    write(tmp_path / "noise" / "noise_000_stanza.wav", 100, seconds=30)
    bw.benchmark("claudio", Fake(), tmp_path, (0.3, 0.6), max_fph=1.0)
    out = capsys.readouterr().out
    assert "3 registrazioni, 1 frasi" in out
    assert "Soglia consigliata: 0.60 (2/3 sentite)" in out
    assert "normale      2/2" in out and "veloce       0/1" in out


def test_benchmark_without_takes_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bw.benchmark("claudio", Fake(), tmp_path, (0.3,), max_fph=1.0)
    assert "nessuna registrazione" in capsys.readouterr().out
