import logging
import time

import numpy as np

from claudio.tts.voice_check import VoiceMonitor, pitch_stats

RATE = 24_000


def tone(hz: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    # a voice-like tone: a slow vibrato keeps the melody from being dead flat
    phase = 2 * np.pi * np.cumsum(hz * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))) / RATE
    return (8000 * np.sin(phase)).astype(np.int16)


def test_pitch_stats_finds_the_register() -> None:
    stats = pitch_stats(tone(110), RATE)
    assert 100 < stats["median_hz"] < 120
    assert stats["jumps"] == 0


def test_pitch_stats_of_silence_is_empty() -> None:
    assert pitch_stats(np.zeros(RATE, dtype=np.int16), RATE)["median_hz"] == 0.0


def test_a_piece_well_above_the_usual_register_is_flagged() -> None:
    monitor = VoiceMonitor(folder=None)  # type: ignore[arg-type]  # judge() only
    usual = {"jumps": 0, "flicker": 0, "range_st": 3.0}
    for hz in (110, 112, 108, 111):
        assert monitor.judge({**usual, "median_hz": hz}) == ""
    assert "registro alto" in monitor.judge({**usual, "median_hz": 150})  # +5 semitones
    assert monitor.judge({**usual, "median_hz": 118}) == ""  # +1 semitone: Jack excited


def test_only_octave_jumps_and_a_flat_melody_count_not_flicker() -> None:
    monitor = VoiceMonitor(folder=None)  # type: ignore[arg-type]
    assert monitor.judge({"jumps": 0, "flicker": 8, "range_st": 4.7, "median_hz": 110}) == ""
    assert "salti" in monitor.judge({"jumps": 4, "flicker": 12, "range_st": 5.5, "median_hz": 110})
    assert "piatta" in monitor.judge({"jumps": 0, "flicker": 0, "range_st": 1.0, "median_hz": 110})


def test_monitor_saves_the_wav_and_logs_a_warning(tmp_path, caplog) -> None:
    monitor = VoiceMonitor(tmp_path / "jack", keep=2)
    caplog.set_level(logging.INFO, logger="claudio.tts.voice_check")
    for _ in range(3):
        monitor.check("Ciao eroe.", tone(110), RATE)
    monitor.check("Troppo alto.", tone(200), RATE)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and len(caplog.records) < 4:
        time.sleep(0.05)
    assert sorted(p.name for p in (tmp_path / "jack").glob("*.wav")) == ["003.wav", "004.wav"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "004.wav" in warnings[0] and "Troppo alto" in warnings[0]
