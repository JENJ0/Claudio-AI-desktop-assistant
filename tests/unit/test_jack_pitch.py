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


def _peak_hz(audio: np.ndarray, rate: int = 24_000) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / rate)[np.argmax(spectrum)])


def _stretch(audio: np.ndarray, speed: float, chunk: int) -> np.ndarray:
    ts = w.TimeStretch(speed)
    parts = [ts.feed(audio[i : i + chunk]) for i in range(0, len(audio), chunk)]
    return np.concatenate([*parts, ts.flush()])


@pytest.mark.parametrize("speed", [0.9, 1.1])
def test_time_stretch_changes_the_pace_and_keeps_the_pitch(speed: float) -> None:
    rate = 24_000
    t = np.arange(2 * rate) / rate
    tone = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    out = _stretch(tone, speed, chunk=16_000)  # Qwen3 streams ~0.67 s chunks
    assert len(out) == round(len(tone) / speed)
    assert abs(_peak_hz(out) - 150) < 2  # resampling would give 150 * speed
    middle = out[rate // 4 : -rate // 4]
    assert np.sqrt(np.mean(middle**2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)  # no warble


def test_time_stretch_does_not_depend_on_chunk_size() -> None:
    rng = np.random.default_rng(0)
    voice = rng.standard_normal(30_000).astype(np.float32) * 0.1
    assert np.allclose(_stretch(voice, 0.9, 777), _stretch(voice, 0.9, 30_000), atol=1e-6)


def test_time_stretch_restarts_clean_for_each_sentence() -> None:
    tone = np.sin(np.arange(12_000) / 10).astype(np.float32)
    ts = w.TimeStretch(0.9)
    first = np.concatenate([ts.feed(tone), ts.flush()])
    ts.begin()
    again = np.concatenate([ts.feed(tone), ts.flush()])
    assert np.array_equal(first, again)


def test_no_rvc_applies_the_speed_without_rvc() -> None:
    rvc = w.RvcNone(0.9)
    rvc.begin()
    tone = (0.3 * np.sin(np.arange(24_000) / 8)).astype(np.float32)
    pcm = b"".join(rvc.feed(tone) + rvc.flush())
    assert len(pcm) // 2 == round(24_000 / 0.9)


def test_the_frame_budget_follows_the_text() -> None:
    # 12 frames a second: a short reply cannot babble for long, a long one has room
    assert w.frame_budget("Fatto, socio.") / 12 < 4
    assert w.frame_budget("x" * 220) / 12 > 220 / 13 * 1.4
    assert w.frame_budget("x" * 5000) == 2000
