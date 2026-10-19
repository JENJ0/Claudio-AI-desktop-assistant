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


def test_plain_values_in_env_still_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A v0.1.0 .env set the replies as plain text: it must still load, and JSON works too."""
    f = tmp_path / "c.toml"
    f.write_text("", encoding="utf-8")
    monkeypatch.setenv("CLAUDIO_ROUTER__UNKNOWN_REPLY", "Non lo so fare, scusa.")
    monkeypatch.setenv("CLAUDIO_ROUTER__ERROR_REPLY", '{"it": "Ops.", "en": "Oops."}')
    monkeypatch.setenv("CLAUDIO_STT__LANGUAGES", "it, en")
    s = load_settings(f)
    assert s.router.unknown_reply == {"it": "Non lo so fare, scusa.", "en": "I can't do that yet."}
    assert s.router.error_reply == {"it": "Ops.", "en": "Oops."}
    assert s.stt.languages == ["it", "en"]
    monkeypatch.setenv("CLAUDIO_STT__LANGUAGES", "en")
    assert load_settings(f).stt.languages == ["en"]


def test_unreadable_env_value_has_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "c.toml"
    f.write_text("", encoding="utf-8")
    monkeypatch.setenv("CLAUDIO_STT__PROMPTS", "non è JSON")
    with pytest.raises(ConfigError, match="valori non validi"):
        load_settings(f)


def test_initial_prompt_from_env_replaces_the_shipped_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDIO_STT__INITIAL_PROMPT", "Hey Claudio, apri Spotify.")
    s = load_settings(DEFAULT_CONFIG)
    assert s.stt.prompt("it") == "Hey Claudio, apri Spotify."
    assert "open Chrome" in s.stt.prompt("en")


def test_configuration_errors_follow_the_language_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from claudio import i18n
    from claudio.__main__ import main

    f = tmp_path / "c.toml"
    f.write_text('language = "en"\n[stt]\nbeam_size = 0\n', encoding="utf-8")
    monkeypatch.delenv("CLAUDIO_LANGUAGE", raising=False)
    try:
        assert main(["--config", str(f), "--check"]) == 2
        assert "Configuration error" in capsys.readouterr().err
    finally:
        i18n.set_language("it")


def test_kokoro_voices_are_read_from_its_file(tmp_path: Path) -> None:
    import numpy as np

    from claudio.checks import kokoro_voices

    assert kokoro_voices(tmp_path) is None  # no file: reported by the model check
    with (tmp_path / "voices-v1.0.bin").open("wb") as fh:
        np.savez(fh, am_michael=np.zeros(3), im_nicola=np.zeros(3))
    assert kokoro_voices(tmp_path) == {"am_michael", "im_nicola"}
