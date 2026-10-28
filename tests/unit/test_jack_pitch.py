"""RVC pitch cleanup in jack_worker.py (pure numpy parts: no GPU, no Applio)."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

saved = sys.stdout
try:
    from claudio.tts import jack_worker as w
finally:
    sys.stdout = saved  # the worker sends its prints to stderr when imported


def test_an_octave_error_is_folded_back() -> None:
    f0 = np.full(40, 120.0)
    f0[20] = 240.0  # one frame twice the pitch: RMVPE's typical slip
    f0[30] = 60.0  # and one at half
    out = w.smooth_f0(f0)
    assert np.allclose(out, 120.0)


def test_a_short_unvoiced_hole_in_a_vowel_is_filled_but_pauses_stay() -> None:
    f0 = np.full(60, 110.0)
    f0[10:12] = 0.0  # 20 ms hole: buzz
    f0[30:45] = 0.0  # 150 ms: a real consonant or pause
    out = w.smooth_f0(f0)
    assert np.all(out[10:12] > 100)
    assert np.all(out[30:45] == 0)


def test_a_natural_glide_is_kept() -> None:
    f0 = np.geomspace(100, 160, 50)  # rising intonation, 8 semitones in 0.5 s
    out = w.smooth_f0(f0)
    assert np.allclose(out[1:-1], f0[1:-1], rtol=0.02)


def test_nan_and_silence_stay_unvoiced() -> None:
    out = w.smooth_f0(np.array([np.nan, 0.0, -1.0, np.inf]))
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_coarse_buckets_follow_applio_offline_formula() -> None:
    f0 = np.array([0.0, 50.0, 120.0, 1100.0, 5000.0])
    mel = 1127 * np.log(1 + f0 / 700)
    expected = np.rint((mel - w.F0_MEL_MIN) * 254 / (w.F0_MEL_MAX - w.F0_MEL_MIN) + 1)
    expected = np.clip(np.where(f0 > 0, expected, 1), 1, 255)
    assert w.coarse_f0(f0).tolist() == expected.astype(int).tolist()
    assert w.coarse_f0(f0)[0] == 1 and w.coarse_f0(f0)[-1] == 255


@pytest.mark.parametrize("method", ["fcpe", "rmvpe"])
def test_realtime_pitch_scale_is_set_to_the_mel_limits(method) -> None:
    pipeline = SimpleNamespace(f0_method=method, f0_min=50.0, f0_max=1100.0)
    assert w.fix_realtime_pitch_scale(pipeline)
    assert (pipeline.f0_min, pipeline.f0_max) == (w.F0_MEL_MIN, w.F0_MEL_MAX)
    # 120 Hz now gets the training bucket instead of one ~5 higher
    mel = 1127 * np.log(1 + 120 / 700)
    bucket = round((mel - pipeline.f0_min) * 254 / (pipeline.f0_max - pipeline.f0_min) + 1)
    assert bucket == w.coarse_f0(np.array([120.0]))[0]


def test_realtime_pitch_scale_leaves_other_pipelines_alone() -> None:
    crepe = SimpleNamespace(f0_method="crepe", f0_min=50.0, f0_max=1100.0)
    assert not w.fix_realtime_pitch_scale(crepe)
    assert crepe.f0_min == 50.0
    assert not w.fix_realtime_pitch_scale(SimpleNamespace(f0_method="fcpe"))


def test_no_rvc_passes_qwen_audio_through() -> None:
    rvc = w.RvcNone()
    rvc.begin()
    (pcm,) = rvc.feed(np.array([0.0, 0.5, -2.0], dtype=np.float32))
    assert np.frombuffer(pcm, "<i2").tolist() == [0, 16383, -32767]
    assert rvc.flush() == []


def test_worker_command_adds_pitch_fix_only_when_on(tmp_path) -> None:
    from claudio.config import TtsSettings
    from claudio.tts.jack import worker_command

    assert "--pitch-fix" not in worker_command(tmp_path, TtsSettings())
    assert "--pitch-fix" in worker_command(tmp_path, TtsSettings(jack_pitch_fix=True))
    assert "--pitch-fix" in worker_command(tmp_path, TtsSettings(), jack_pitch_fix=True)


def test_qwen_alone_is_a_valid_setting() -> None:
    from claudio.config import TtsSettings

    assert TtsSettings(jack_rvc="none").jack_rvc == "none"
