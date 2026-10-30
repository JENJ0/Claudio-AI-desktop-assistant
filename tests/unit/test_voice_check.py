import logging
import time

import numpy as np
import pytest
from scipy.signal import resample_poly

from claudio.tts.voice_check import (
    MIN_VOICED,
    TakeJudge,
    VoiceMonitor,
    pitch_stats,
    pitch_track,
    syllables,
    voice_stats,
)

RATE = 24_000


def tone(hz: float, seconds: float = 1.0, amplitude: float = 8000) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    # a voice-like tone: a slow vibrato keeps the melody from being dead flat
    phase = 2 * np.pi * np.cumsum(hz * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))) / RATE
    return (amplitude * np.sin(phase)).astype(np.int16)


def loop_yin(audio: np.ndarray, rate: int, threshold: float) -> np.ndarray:
    """The frame-by-frame YIN the FFT one replaced (IT-0): the reference it must match."""
    x = resample_poly(audio.astype(np.float64) / 32768, 16_000, rate)
    lag_min, lag_max = 40, 266
    f0 = []
    for start in range(0, len(x) - 640 - lag_max, 160):
        frame = x[start : start + 640 + lag_max]
        if np.sqrt(np.mean(frame[:640] ** 2)) < 0.01:
            f0.append(0.0)
            continue
        shifted = np.lib.stride_tricks.sliding_window_view(frame, 640)[: lag_max + 1]
        diff = np.sum((shifted - frame[:640]) ** 2, axis=1)
        cmnd = np.ones_like(diff)
        cmnd[1:] = diff[1:] * np.arange(1, len(diff)) / np.maximum(np.cumsum(diff[1:]), 1e-12)
        below = np.flatnonzero(cmnd[lag_min:] < threshold)
        if not len(below):
            f0.append(0.0)
            continue
        lag = lag_min + below[0]
        while lag + 1 < len(cmnd) and cmnd[lag + 1] < cmnd[lag]:
            lag += 1
        f0.append(16_000 / lag)
    return np.array(f0)


@pytest.mark.parametrize("threshold", [0.15, 0.3])
def test_the_fft_yin_finds_what_the_frame_by_frame_one_did(threshold: float) -> None:
    rng = np.random.default_rng(1)
    voice = tone(120, 2).astype(np.float64) + rng.normal(0, 800, 2 * RATE)
    voice = np.concatenate([voice, np.zeros(RATE // 2), tone(190, 1)]).astype(np.int16)
    assert np.allclose(pitch_track(voice, RATE, threshold), loop_yin(voice, RATE, threshold))


def test_pitch_stats_finds_the_register() -> None:
    stats = pitch_stats(tone(110), RATE)
    assert 100 < stats["median_hz"] < 120
    assert stats["jumps"] == 0


def test_too_little_voice_is_unknown_not_zero_hz() -> None:
    """30/10: "f0 0 Hz" was logged for "Apro la chat.", which sounds fine."""
    assert not voice_stats(np.zeros(RATE, dtype=np.int16), RATE).known
    short = voice_stats(tone(110, 0.12), RATE)  # a few voiced frames only
    assert short.voiced < MIN_VOICED and not short.known and short.median_hz == 0.0


def test_the_highest_stretch_of_a_long_piece_is_measured() -> None:
    piece = np.concatenate([tone(110, 4), tone(260, 2)])  # a shout at the end
    stats = voice_stats(piece, RATE)
    assert stats.median_hz < 130 and stats.high_hz > 230


def test_syllables_are_counted_by_vowel_groups() -> None:
    assert syllables("Apro la chat.") == 4
    assert syllables("Mi spengo. Per riaccendermi di' Claudio, accenditi.") == 15  # about 17


@pytest.fixture
def judge() -> TakeJudge:
    return TakeJudge(target_lufs=-24.0, usual_hz=132.0)


def text_for(seconds: float) -> str:
    return "x" * round(16 * seconds)  # 16 characters a second: Jack's usual pace


def test_a_take_in_jacks_usual_register_passes(judge: TakeJudge) -> None:
    verdict = judge.judge(tone(130, 2), RATE, text_for(2), {"raw_lufs": -23.0, "raw_clipped": 0})
    assert verdict.ok and verdict.stats.known


def test_a_high_take_is_thrown_away(judge: TakeJudge) -> None:
    """+5 semitones over the usual register: 002, 004, 038, 040 of 30/10."""
    verdict = judge.judge(tone(200, 2), RATE, text_for(2), {"raw_lufs": -24.0})
    assert not verdict.ok and "registro alto" in verdict.reasons[0]
    assert judge.judge(tone(160, 2), RATE, text_for(2), {"raw_lufs": -24.0}).ok  # +3.3: lively


def test_a_shout_inside_a_long_take_is_thrown_away(judge: TakeJudge) -> None:
    piece = np.concatenate([tone(120, 5), tone(260, 2)])  # 018 of 30/10: a shout at 6-8 s
    verdict = judge.judge(piece, RATE, text_for(7), {"raw_lufs": -24.0})
    assert not verdict.ok and "registro alto" in verdict.reasons[0]


@pytest.mark.parametrize(
    ("raw", "reason"),
    [({"raw_lufs": -15.0}, "take forte"), ({"raw_lufs": -24.0, "raw_clipped": 12}, "fondo")],
)
def test_a_shouted_or_clipped_take_is_thrown_away(judge: TakeJudge, raw, reason) -> None:
    """The level guard evens out the volume, not the shout: the raw take tells."""
    verdict = judge.judge(tone(130, 2), RATE, text_for(2), raw)
    assert not verdict.ok and reason in " ".join(verdict.reasons)


def test_babbling_past_the_text_is_seen_even_before_the_take_is_whole(judge: TakeJudge) -> None:
    verdict = judge.judge(tone(130, 4), RATE, "Fatto, socio.", {"raw_lufs": -24}, complete=False)
    assert not verdict.ok and "caratteri/s" in verdict.reasons[0]


def test_too_fast_counts_only_once_the_take_is_whole(judge: TakeJudge) -> None:
    start = judge.judge(tone(130, 1), RATE, text_for(3), {"raw_lufs": -24}, complete=False)
    assert start.ok
    assert not judge.judge(tone(130, 1), RATE, text_for(3), {"raw_lufs": -24}).ok


def test_a_short_reply_with_pauses_is_not_too_slow(judge: TakeJudge) -> None:
    assert judge.judge(tone(130, 0.8), RATE, "Sì.", {"raw_lufs": -24}).ok


def test_the_monitor_flags_a_flat_melody_only_in_a_long_phrase() -> None:
    monitor = VoiceMonitor(folder=None)  # type: ignore[arg-type]  # judge() only
    flat = voice_stats(tone(130, 2), RATE)
    flat.range_st = 1.0
    flat.lufs = -24.0
    assert "piatta" not in monitor.judge(flat, "Apro la chat.")
    long_text = "Ho aperto la cartella dei documenti e non c'era niente."
    assert "piatta" in monitor.judge(flat, long_text, text_for(2))


def test_the_monitor_flags_a_level_off_the_target_after_the_guard() -> None:
    monitor = VoiceMonitor(folder=None)  # type: ignore[arg-type]
    stats = voice_stats(tone(130, 2), RATE)
    stats.lufs = -31.0
    assert "volume" in monitor.judge(stats, text_for(2))
    stats.lufs = -25.0
    assert "volume" not in monitor.judge(stats, text_for(2))


def test_monitor_saves_the_wav_and_logs_a_warning(tmp_path, caplog) -> None:
    monitor = VoiceMonitor(tmp_path / "jack", keep=2)
    caplog.set_level(logging.INFO, logger="claudio.tts.voice_check")
    quiet = 3200  # about -24 LUFS at 125 Hz: a piece the level guard left on its target
    for _ in range(3):
        monitor.check("Ciao eroe, come va.", tone(125, amplitude=quiet), RATE)
    monitor.check("Troppo alto, socio.", tone(220, amplitude=quiet), RATE)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and len(caplog.records) < 4:
        time.sleep(0.05)
    assert sorted(p.name for p in (tmp_path / "jack").glob("*.wav")) == ["003.wav", "004.wav"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "004.wav" in warnings[0] and "Troppo alto" in warnings[0]
    assert "registro alto" in warnings[0] and "LUFS" in warnings[0]


def test_the_log_says_n_d_when_the_pitch_is_unknown(tmp_path, caplog) -> None:
    monitor = VoiceMonitor(tmp_path / "jack")
    caplog.set_level(logging.INFO, logger="claudio.tts.voice_check")
    noise = (np.random.default_rng(0).normal(0, 2500, RATE)).astype(np.int16)
    monitor.check("Sì.", noise, RATE)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not caplog.records:
        time.sleep(0.05)
    assert "f0 n/d" in caplog.records[0].getMessage()
