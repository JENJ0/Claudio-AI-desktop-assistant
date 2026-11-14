"""The on-disk cache of fixed replies spoken by ElevenLabs."""

import logging
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from claudio.config import TtsSettings
from claudio.tts.eleven_cache import TakeCache, eleven_profile

PCM = (np.arange(2400) % 200 - 100).astype(np.int16)


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = TakeCache(tmp_path, "p1")
    assert cache.load("Apro Chrome.", 24000) is None
    cache.save("Apro Chrome.", PCM, 24000)
    loaded = cache.load(" Apro Chrome. ", 24000)
    assert loaded is not None
    assert loaded.dtype == np.int16
    assert np.array_equal(loaded, PCM)
    assert TakeCache(tmp_path, "p2").load("Apro Chrome.", 24000) is None
    assert cache.load("Apro Firefox.", 24000) is None


def test_a_broken_cache_file_is_a_miss(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cache = TakeCache(tmp_path, "p1")
    (tmp_path / f"{cache.key('Ciao.')}.wav").write_bytes(b"abc")
    with caplog.at_level(logging.WARNING):
        assert cache.load("Ciao.", 24000) is None
    assert len(caplog.records) == 1


def test_save_never_leaves_half_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TakeCache(tmp_path, "p1")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", boom)
        cache.save("Ciao.", PCM, 24000)
    assert list(tmp_path.iterdir()) == []
    cache.save("Ciao.", PCM, 24000)
    assert np.array_equal(cache.load("Ciao.", 24000), PCM)


def test_the_profile_changes_with_voice_and_settings() -> None:
    base = eleven_profile(TtsSettings())
    for change in (
        {"eleven_voice_id": "abc123"},
        {"eleven_model": "eleven_other"},
        {"eleven_stability": 0.7},
        {"jack_lufs": -20.0},
    ):
        assert eleven_profile(TtsSettings(**change)) != base


@pytest.mark.parametrize("cut", [1, 1000])
def test_a_truncated_file_is_a_miss(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, cut: int
) -> None:
    cache = TakeCache(tmp_path, "p1")
    cache.save("Ciao.", PCM, 24000)
    path = next(tmp_path.glob("*.wav"))
    path.write_bytes(path.read_bytes()[:-cut])
    with caplog.at_level(logging.WARNING):
        assert cache.load("Ciao.", 24000) is None
    assert len(caplog.records) == 1


def test_a_wrong_rate_is_a_miss(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cache = TakeCache(tmp_path, "p1")
    cache.save("Ciao.", PCM, 16000)
    with caplog.at_level(logging.WARNING):
        assert cache.load("Ciao.", 24000) is None
    assert len(caplog.records) == 1


def test_two_saves_of_one_key_leave_one_valid_file(tmp_path: Path) -> None:
    cache = TakeCache(tmp_path, "p1")
    clips = [PCM, PCM[::-1].copy()]
    threads = [threading.Thread(target=cache.save, args=("Ciao.", c, 24000)) for c in clips]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    loaded = cache.load("Ciao.", 24000)
    assert loaded is not None
    assert any(np.array_equal(loaded, c) for c in clips)
    assert len(list(tmp_path.iterdir())) == 1
