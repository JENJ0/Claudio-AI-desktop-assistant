"""Loudness by BS.1770 and the level guard of Jack's worker (pure numpy: no GPU)."""

import json
import math

import numpy as np
import pytest

from claudio.tts.loudness import (
    LevelGuard,
    clipped_samples,
    integrated_lufs,
    k_weighting,
    peak_dbfs,
)

RATE = 24_000
CHUNK = 16_000  # one Qwen3 chunk: 8 codec frames at 12 Hz


def sine(dbfs: float, seconds: float, hz: float = 997.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return 10 ** (dbfs / 20) * np.sin(2 * np.pi * hz * t)


def voice(lufs_like: float, seconds: float) -> np.ndarray:
    """A voice-like signal: a 140 Hz buzz with harmonics, syllables of 200 ms."""
    t = np.arange(int(RATE * seconds)) / RATE
    buzz = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
    syllables = 0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t) ** 2
    x = buzz * syllables
    return x * 10 ** ((lufs_like - integrated_lufs(x, RATE)) / 20)


def guarded(audio: np.ndarray, guard: LevelGuard, chunk: int = CHUNK) -> np.ndarray:
    return np.concatenate(
        [guard.process(audio[i : i + chunk]) for i in range(0, len(audio), chunk)]
    )


def test_k_weighting_at_48k_is_the_standard_table() -> None:
    shelf, high_pass = k_weighting(48_000)
    assert shelf[:3] == pytest.approx([1.53512485958697, -2.69169618940638, 1.19839281085285])
    assert shelf[4:] == pytest.approx([-1.69065929318241, 0.73248077421585])
    assert high_pass[:3] == pytest.approx([1.0, -2.0, 1.0])
    assert high_pass[4:] == pytest.approx([-1.99004745483398, 0.99007225036621])


@pytest.mark.parametrize("rate", [24_000, 48_000])
def test_a_tone_at_minus_20_dbfs_reads_minus_23_lufs(rate: int) -> None:
    # BS.1770: a 997 Hz sine reads its RMS level (-3 dB from the peak), K-weighting adds ~0
    assert integrated_lufs(sine(-20, 5, rate=rate), rate) == pytest.approx(-23.0, abs=0.1)


def test_pauses_do_not_lower_the_loudness() -> None:
    speech = sine(-20, 3)
    with_pauses = np.concatenate([speech, np.zeros(RATE * 3), speech])
    # only the blocks across the two edges count, half full: -0.2 dB (without the gate: -1.8)
    expected = integrated_lufs(speech, RATE)
    assert integrated_lufs(with_pauses, RATE) == pytest.approx(expected, abs=0.3)


def test_silence_has_no_loudness_and_int16_reads_as_floats() -> None:
    assert integrated_lufs(np.zeros(RATE), RATE) == -math.inf
    tone = sine(-20, 2)
    pcm = (tone * 32768).astype(np.int16)
    assert integrated_lufs(pcm, RATE) == pytest.approx(integrated_lufs(tone, RATE), abs=0.01)
    assert peak_dbfs(pcm) == pytest.approx(-20, abs=0.05)


def test_clipped_samples_are_counted() -> None:
    assert clipped_samples(np.array([0, 32767, -32768, 100], dtype=np.int16)) == 2
    assert clipped_samples(np.array([0.5, 1.0, -1.2])) == 2


@pytest.mark.parametrize("level", [-36.0, -30.0, -24.0, -18.0, -12.0])
def test_every_take_comes_out_near_the_target(level: float) -> None:
    out = guarded(voice(level, 6), LevelGuard(RATE))
    expected = min(level + 6.0, -24.0)  # raised by 6 dB at most
    assert integrated_lufs(out, RATE) == pytest.approx(expected, abs=0.7)


def test_a_take_past_full_scale_is_limited_not_clipped() -> None:
    loud = voice(-6, 3) * 3  # peaks well past 1.0, as take 028 of 30/10
    guard = LevelGuard(RATE)
    out = guarded(loud, guard)
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-6
    assert guard.raw_clipped > 0 and guard.raw_peak > 1.0


def test_the_limiter_keeps_peaks_under_the_ceiling_without_hard_edges() -> None:
    guard = LevelGuard(RATE, min_gain_db=0.0, max_gain_db=0.0)  # only the limiter acts
    x = voice(-24, 2)
    x[RATE : RATE + 50] = 1.5  # a spike
    out = guarded(x, guard)
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-6
    # around the spike the gain changes smoothly: no sample jumps by more than the voice does
    assert np.max(np.abs(np.diff(out[RATE - 500 : RATE + 600]))) < 1.0


def test_the_gain_ramps_instead_of_stepping() -> None:
    """A take that starts loud and fades (30/10: 7 dB down in a long piece) is evened out
    with a gentle gain curve: no audible step between chunks."""
    x = np.concatenate([voice(-16, 3), voice(-28, 5)])
    guard = LevelGuard(RATE)
    gains = []
    for i in range(0, len(x), CHUNK):
        part = x[i : i + CHUNK]
        out = guard.process(part)
        loud = np.abs(part) > 0.05
        gains.append(20 * np.log10(np.abs(out[loud] / part[loud])))
    per_sample = np.concatenate(gains)
    assert np.max(np.abs(np.diff(per_sample))) < 0.5  # dB from one loud sample to the next
    late = guarded(x, LevelGuard(RATE))[-2 * RATE :]
    assert integrated_lufs(late, RATE) == pytest.approx(-24.0, abs=1.5)  # the fade made up


def test_a_silent_start_does_not_hold_the_gain_back() -> None:
    x = np.concatenate([np.zeros(CHUNK), voice(-32, 3)])
    out = guarded(x, LevelGuard(RATE))
    first_words = out[CHUNK : 2 * CHUNK]
    assert integrated_lufs(first_words, RATE) == pytest.approx(-26.0, abs=1.0)  # +6 at once


def test_each_piece_starts_afresh() -> None:
    guard = LevelGuard(RATE)
    guarded(voice(-12, 3), guard)  # a loud piece leaves the gain low
    guard.begin()
    out = guarded(voice(-30, 3), guard)
    assert integrated_lufs(out, RATE) == pytest.approx(-24.0, abs=0.7)


def test_the_result_hardly_depends_on_the_chunk_size() -> None:
    x = np.concatenate([voice(-20, 3), voice(-27, 3)])
    big = guarded(x, LevelGuard(RATE), CHUNK)
    small = guarded(x, LevelGuard(RATE), 6_000)  # RVC blocks of 250 ms
    assert integrated_lufs(big, RATE) == pytest.approx(integrated_lufs(small, RATE), abs=1.0)


def test_the_report_is_json_with_no_infinity() -> None:
    guard = LevelGuard(RATE)
    assert json.loads(json.dumps(guard.report())) == {
        "raw_lufs": None,
        "raw_peak_dbfs": None,
        "raw_clipped": 0,
        "gain_db": None,
    }
    guard.process(voice(-18, 1))
    report = guard.report()
    assert report["raw_lufs"] == pytest.approx(-18, abs=0.5)
    assert report["gain_db"] is not None and report["gain_db"] < 0
