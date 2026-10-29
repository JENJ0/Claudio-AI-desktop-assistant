import json

import numpy as np
import pytest

from claudio.audio.level import CLIP_LEVEL, LIMIT_KNEE, AutoGain, soft_limit


def make(tmp_path, start=12.0, mic="C270"):
    applied: list[float] = []
    gain = AutoGain(start, applied.append, tmp_path / "audio_level.json", mic)
    return gain, applied


def speech(dbfs: float, clipped: float = 0.0) -> dict[str, float]:
    return {"speech_dbfs": dbfs, "clipped": clipped, "vad_max": 0.95}


def saved(tmp_path) -> dict[str, float]:
    return json.loads((tmp_path / "audio_level.json").read_text(encoding="utf-8"))


def test_first_phrase_on_a_microphone_sets_the_gain_at_once(tmp_path) -> None:
    gain, applied = make(tmp_path)
    assert applied == [12.0]
    gain.observe(speech(-34))  # the voice is at -46 before the boost: +23 puts it at -23
    assert gain.gain_db == 23.0 and applied == [12.0, 23.0]
    assert saved(tmp_path) == {"C270": 23.0}


def test_later_phrases_move_it_gradually_quickly_down_slowly_up(tmp_path) -> None:
    gain, applied = make(tmp_path)
    gain.observe(speech(-23))  # first phrase: the voice at -35 -> +12 stays
    assert applied == [12.0]
    gain.observe(speech(-21))  # within the tolerance: untouched
    assert gain.gain_db == 12.0
    gain.observe(speech(-6))  # much louder: down, at most 6 dB a phrase
    assert gain.gain_db == 6.0
    gain.observe(speech(-10, clipped=0.01))  # clipping always takes 6 dB off
    assert gain.gain_db == 0.0
    gain.observe(speech(-36))  # quieter phrases count less: one alone moves nothing
    assert gain.gain_db == 0.0
    gain.observe(speech(-36))
    gain.observe(speech(-39))  # the same voice, boosted: then at most +3 dB a phrase
    assert applied == [12.0, 6.0, 0.0, 3.0, 6.0]


def test_one_quiet_phrase_among_normal_ones_does_not_pump_the_gain(tmp_path) -> None:
    """28/10 16:41: phrases at -25..-38 dBFS took the old gain from +15 to +29 dB in two
    minutes, then "apri la chat" saturated."""
    gain, _ = make(tmp_path, start=15.0)
    gain.observe(speech(-23))  # the voice at -38: +15 is right
    for level in (-38.5, -21.5, -25.2, -25.6, -28.6, -28.1):
        gain.observe(speech(level + (gain.gain_db - 15.0)))  # the same voice, boosted
    assert gain.gain_db <= 21.0


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


def test_each_microphone_keeps_its_own_gain(tmp_path) -> None:
    gain, applied = make(tmp_path, start=12.0, mic="")
    gain.use("Microfono (Logi C270 HD WebCam)")
    gain.observe(speech(-30))  # the webcam: voice at -42 -> +19
    gain.use("Microfono (Wireless Stereo Head")
    assert gain.gain_db == 12.0  # not the webcam's +19 (26/10 19:50: saturated at once)
    gain.observe(speech(-12))  # the headset is loud: voice at -24 -> +1
    assert saved(tmp_path) == {
        "Microfono (Logi C270 HD WebCam)": 19.0,
        "Microfono (Wireless Stereo Head": 1.0,
    }
    gain.use("Microfono (Logi C270 HD WebCam)")
    assert gain.gain_db == 19.0 and applied[-1] == 19.0


def test_a_reopened_stream_keeps_what_was_learned(tmp_path) -> None:
    gain, applied = make(tmp_path, mic="Cuffie")
    gain.observe(speech(-23))
    gain.observe(speech(-6))  # down to +6
    gain.use("Cuffie")  # the same microphone again (a USB hiccup)
    assert gain.gain_db == 6.0 and applied == [12.0, 6.0]
    again, applied_again = make(tmp_path, mic="Cuffie")  # next start: the saved value
    assert again.gain_db == 6.0 and applied_again == [6.0]
    again.observe(speech(-21))  # already learned: no first-phrase jump
    assert again.gain_db == 6.0


def test_the_old_value_of_the_setting_starts_a_new_microphone(tmp_path) -> None:
    (tmp_path / "audio_level.json").write_text('{"": 11.9}', encoding="utf-8")
    gain, applied = make(tmp_path, start=0.0, mic="")
    gain.use("Microfono (Wireless Stereo Head")
    assert applied == [11.9, 11.9]
    gain.observe(speech(-17.1))  # but it is measured afresh: voice at -29 -> +6
    assert gain.gain_db == 6.0
    assert saved(tmp_path) == {"": 11.9, "Microfono (Wireless Stereo Head": 6.0}


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_broken_state_file_starts_from_the_config(tmp_path, content: str) -> None:
    (tmp_path / "audio_level.json").write_text(content, encoding="utf-8")
    gain, applied = make(tmp_path, start=6.0)
    assert gain.gain_db == 6.0 and applied == [6.0]
    gain.observe(speech(-34))
    assert saved(tmp_path) == {"C270": 17.0}


def test_soft_limit_bends_only_the_peaks() -> None:
    x = np.linspace(-3.0, 3.0, 6001)
    y = soft_limit(x)
    quiet = np.abs(x) <= LIMIT_KNEE
    assert np.array_equal(y[quiet], x[quiet])  # untouched below -3 dBFS
    assert np.all(np.abs(y) < 1.0) and np.all(np.diff(y) > 0)  # never clips, never folds
    assert np.allclose(y, -y[::-1])  # symmetric
    just_over = soft_limit(np.array([LIMIT_KNEE + 1e-4]))[0]
    assert abs(just_over - (LIMIT_KNEE + 1e-4)) < 1e-6  # same slope at the knee
    assert 0.92 < CLIP_LEVEL < 0.94  # what full scale becomes: the clipping mark
