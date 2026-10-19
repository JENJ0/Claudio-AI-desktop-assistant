import json

import pytest

from claudio.audio.level import AutoGain


def make(tmp_path, start=12.0, mic="C270"):
    applied: list[float] = []
    gain = AutoGain(start, applied.append, tmp_path / "audio_level.json", mic)
    return gain, applied


def speech(dbfs: float, clipped: float = 0.0) -> dict[str, float]:
    return {"speech_dbfs": dbfs, "clipped": clipped, "vad_max": 0.95}


def test_quiet_voice_raises_slowly_loud_voice_lowers_quickly(tmp_path) -> None:
    gain, applied = make(tmp_path)
    assert applied == [12.0]
    gain.observe(speech(-34))  # 14 dB under the target: +3 dB at most
    assert gain.gain_db == 15.0
    gain.observe(speech(-21))  # within the tolerance: untouched
    assert gain.gain_db == 15.0
    gain.observe(speech(-6))  # 14 dB over: -6 dB at most
    assert gain.gain_db == 9.0
    gain.observe(speech(-10, clipped=0.01))  # clipping always takes 6 dB off
    assert gain.gain_db == 3.0
    assert applied == [12.0, 15.0, 9.0, 3.0]


def test_gain_stays_within_bounds_and_warns_once_when_still_quiet(tmp_path, caplog) -> None:
    gain, applied = make(tmp_path, start=29.0)
    with caplog.at_level("WARNING"):
        for _ in range(4):
            gain.observe(speech(-45))
    assert gain.gain_db == 30.0 and applied == [29.0, 30.0]
    warnings = [r for r in caplog.records if "Microfono basso" in r.getMessage()]
    assert len(warnings) == 1
    low, _ = make(tmp_path, start=1.0, mic="other")
    low.observe(speech(-3))
    assert low.gain_db == 0.0


def test_the_level_is_remembered_per_microphone(tmp_path) -> None:
    gain, _ = make(tmp_path)
    gain.observe(speech(-34))
    other, _ = make(tmp_path, start=0.0, mic="Cuffie")
    other.observe(speech(-30))
    data = json.loads((tmp_path / "audio_level.json").read_text(encoding="utf-8"))
    assert data == {"C270": 15.0, "Cuffie": 3.0}
    again, applied = make(tmp_path, start=12.0)
    assert again.gain_db == 15.0 and applied == [15.0]


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_broken_state_file_starts_from_the_config(tmp_path, content: str) -> None:
    (tmp_path / "audio_level.json").write_text(content, encoding="utf-8")
    gain, applied = make(tmp_path, start=6.0)
    assert gain.gain_db == 6.0 and applied == [6.0]
    gain.observe(speech(-34))
    assert json.loads((tmp_path / "audio_level.json").read_text(encoding="utf-8")) == {"C270": 9.0}
