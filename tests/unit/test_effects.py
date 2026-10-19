import numpy as np
import pytest

from claudio.tts.effects import PRESETS, TAIL_S, VoiceEffect, comb, pitch_shift, ring


def sentence(sr: int, seconds: float = 0.8) -> np.ndarray:
    """A vowel-like int16 'sentence': 140 Hz voice with harmonics, rising and falling."""
    t = np.arange(int(sr * seconds)) / sr
    voice = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 6))
    envelope = np.sin(np.pi * t / seconds)
    return (voice * envelope / 2.5 * 20000).astype(np.int16)


@pytest.mark.parametrize("name", list(PRESETS))
@pytest.mark.parametrize("sr", [16000, 22050, 24000])
def test_every_preset_keeps_length_and_level(name: str, sr: int) -> None:
    audio = sentence(sr)
    out = VoiceEffect(name)(audio, sr)
    assert out.dtype == np.int16
    assert len(out) == len(audio) + int(TAIL_S * sr)  # room for the echoes to fade
    assert abs(int(np.abs(out).max()) - int(0.8 * 32767)) <= 1
    assert not np.array_equal(out[: len(audio)], audio)


def test_empty_sentence_and_unknown_preset() -> None:
    assert VoiceEffect("ultron_estremo")(np.zeros(0, np.int16), 24000).size == 0
    with pytest.raises(ValueError, match="ultron_estremo"):
        VoiceEffect("dalek")


def test_comb_matches_the_textbook_filter() -> None:
    from scipy.signal import lfilter

    x = np.random.default_rng(1).normal(size=5000)
    d = int(24000 * 7 / 1000)
    a = np.zeros(d + 1)
    a[0], a[d] = 1, -0.45
    assert np.allclose(comb(x, 24000, 7, 0.45), lfilter([1.0], a, x))


def test_ring_at_full_depth_buzzes_at_its_frequency() -> None:
    sr = 16000
    x = np.ones(sr)
    y = ring(x, sr, 120, 1.0)
    spectrum = np.abs(np.fft.rfft(y))
    assert np.argmax(spectrum) == 120  # 1 Hz bins over one second


def test_pitch_shift_lowers_the_voice_and_keeps_the_duration() -> None:
    sr = 24000
    t = np.arange(sr) / sr
    y = pitch_shift(np.sin(2 * np.pi * 200 * t), sr, 0.72)
    assert len(y) == sr
    peak_hz = np.argmax(np.abs(np.fft.rfft(y[sr // 10 : sr // 10 + sr // 2]))) * 2
    # 200 Hz * 0.72 = 144 Hz; 20 ms grains move it to the nearest 50 Hz step
    assert abs(peak_hz - 144) <= 25
