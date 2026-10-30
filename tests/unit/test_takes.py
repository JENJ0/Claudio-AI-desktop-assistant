"""The table of Jack's good takes: a seed per text, and fixed replies saved as WAV."""

import json
import logging

import numpy as np

from claudio.config import TtsSettings
from claudio.tts.takes import TakeStore, voice_profile


def store(tmp_path, profile: str = "p", **kwargs) -> TakeStore:
    return TakeStore(tmp_path / "data" / "jack_takes.json", profile, **kwargs)


def test_a_good_seed_is_kept_for_the_same_words(tmp_path) -> None:
    takes = store(tmp_path)
    key = takes.key("Chiudo la chat.", "none")
    assert takes.seed(key) is None
    takes.remember(key, 9, "Chiudo la chat.")
    again = store(tmp_path)  # after a restart
    assert again.seed(key) == 9


def test_bad_seeds_are_kept_and_a_bad_good_seed_is_forgotten(tmp_path) -> None:
    takes = store(tmp_path)
    key = takes.key("Niente magia, socio.", "none")
    takes.avoid(key, 7, "Niente magia, socio.")
    takes.avoid(key, 8, "Niente magia, socio.")
    again = store(tmp_path)
    assert again.bad_seeds(key) == {7, 8} and again.seed(key) is None
    again.remember(key, 9, "Niente magia, socio.")
    again.avoid(key, 9, "Niente magia, socio.")  # judged bad later: no longer the good one
    assert again.seed(key) is None and again.bad_seeds(key) == {7, 8, 9}
    for seed in range(20, 40):
        again.avoid(key, seed, "Niente magia, socio.")
    assert len(again.bad_seeds(key)) == 10  # the last ten only


def test_the_key_depends_on_the_words_the_voice_and_the_conversion(tmp_path) -> None:
    takes = store(tmp_path)
    key = takes.key("Apro la chat.", "none")
    assert key != takes.key("Apro la chat!", "none")
    assert key != takes.key("Apro la chat.", "blocks")
    assert key != store(tmp_path, profile="another checkpoint").key("Apro la chat.", "none")


def test_the_profile_ignores_the_seed_and_the_folder_of_the_models() -> None:
    base = voice_profile(TtsSettings())
    assert voice_profile(TtsSettings(jack_seed=42)) == base  # a remembered seed is a seed
    assert voice_profile(TtsSettings(jack_temperature=0.8)) != base
    assert voice_profile(TtsSettings(jack_lufs=-20)) != base
    assert "\\" not in base and "voices/jack/ft" in base  # relative, the same on any PC


def test_an_unreadable_table_starts_empty(tmp_path, caplog) -> None:
    path = tmp_path / "data" / "jack_takes.json"
    path.parent.mkdir()
    path.write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="claudio.tts.takes"):
        takes = TakeStore(path, "p")
    assert takes.seed(takes.key("Ciao.", "none")) is None
    assert any("illeggibile" in r.getMessage() for r in caplog.records)
    takes.remember(takes.key("Ciao.", "none"), 3, "Ciao.")  # and it is written again
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_the_least_recently_said_go_first_and_saved_takes_stay(tmp_path) -> None:
    takes = store(tmp_path, limit=3)
    saved = takes.key("Sono pronto.", "none")
    takes.save_audio(saved, 4, "Sono pronto.", np.zeros(100, dtype=np.int16), 24_000)
    keys = [takes.key(f"Frase {i}.", "none") for i in range(4)]
    for i, key in enumerate(keys):
        takes.remember(key, i, f"Frase {i}.")
        takes.seed(keys[0])  # the first one keeps being said
    kept = store(tmp_path, limit=3)
    assert kept.seed(saved) == 4 and kept.seed(keys[0]) == 0
    assert kept.seed(keys[1]) is None  # the oldest one never said again


def test_a_fixed_reply_saved_in_advance_is_played_from_its_wav(tmp_path) -> None:
    takes = store(tmp_path)
    key = takes.key("Eccomi, sono acceso.", "none")
    pcm = (np.sin(np.arange(2400) / 5) * 3000).astype(np.int16)
    path = takes.save_audio(key, 11, "Eccomi, sono acceso.", pcm, 24_000, wer=0.0)
    assert path.parent.name == "jack_cache" and path.exists()
    audio, rate = store(tmp_path).audio(key)
    assert rate == 24_000 and np.array_equal(audio, pcm)
    assert store(tmp_path).seed(key) == 11


def test_a_missing_wav_falls_back_to_generating(tmp_path, caplog) -> None:
    takes = store(tmp_path)
    key = takes.key("Fatto.", "none")
    takes.save_audio(key, 2, "Fatto.", np.zeros(10, dtype=np.int16), 24_000).unlink()
    with caplog.at_level(logging.WARNING, logger="claudio.tts.takes"):
        assert takes.audio(key) is None
    assert takes.seed(key) == 2  # its seed still helps
