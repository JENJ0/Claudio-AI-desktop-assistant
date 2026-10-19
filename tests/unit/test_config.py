from pathlib import Path

import pytest

from claudio.config import DEFAULT_CONFIG, ConfigError, load_settings


def test_project_config_loads() -> None:
    s = load_settings(DEFAULT_CONFIG)
    assert s.stt.model == "large-v3-turbo"
    assert s.stt.fallback_model == "small"
    assert s.tts.voice == "it_IT-paola-medium"
    assert s.audio.block_size == 1280
    assert s.wakeword.models[0].name == "hey_jarvis"


def test_defaults_when_sections_missing(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text('[stt]\ndevice = "cpu"\n', encoding="utf-8")
    s = load_settings(f)
    assert s.stt.device == "cpu" and s.vad.end_silence_ms == 400


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "c.toml"
    f.write_text('[stt]\ndevice = "cuda"\n', encoding="utf-8")
    monkeypatch.setenv("CLAUDIO_STT__DEVICE", "cpu")
    assert load_settings(f).stt.device == "cpu"


def test_invalid_value_has_clear_message(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text('[vad]\nthreshold = 2\n[stt]\ndevice = "tpu"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_settings(f)
    assert "vad.threshold" in str(exc.value) and "stt.device" in str(exc.value)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text("typo_key = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="typo_key"):
        load_settings(f)


def test_broken_toml(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text("[stt\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="TOML"):
        load_settings(f)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="non trovato"):
        load_settings(tmp_path / "nope.toml")


def test_resolve_relative_to_project_root() -> None:
    s = load_settings(DEFAULT_CONFIG)
    assert s.resolve(Path("models")).is_absolute()


def test_languages_voices_and_replies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = load_settings(DEFAULT_CONFIG)
    assert s.stt.languages == ["it", "en"] and s.language == "it"
    assert "Hey Jarvis" in s.stt.prompt("it") and "open Chrome" in s.stt.prompt("en")
    assert s.tts.piper_voice_for("en") == "en_US-ryan-low"
    assert s.tts.piper_voice_for("it") == s.tts.voice == s.tts.piper_voice_for("de")
    assert s.tts.kokoro_voice_for("en") == "am_michael"
    assert s.router.unknown_reply["en"] == "I can't do that yet."
    f = tmp_path / "c.toml"
    f.write_text('[router]\nunknown_reply = "Non lo so."\n', encoding="utf-8")
    replies = load_settings(f).router.unknown_reply
    assert replies["it"] == "Non lo so." and replies["en"] == "I can't do that yet."
    monkeypatch.setenv("CLAUDIO_STT__LANGUAGES", '["en"]')
    assert load_settings(f).stt.languages == ["en"]
    monkeypatch.setenv("CLAUDIO_STT__LANGUAGE", "it")  # one fixed language wins
    assert load_settings(f).stt.languages == ["it"]
