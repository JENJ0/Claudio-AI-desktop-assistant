from pathlib import Path

import pytest
from pydantic import ValidationError

from claudio.config import DEFAULT_CONFIG, ConfigError, PcSettings, Settings, load_settings


@pytest.mark.parametrize("names", [[""], ["notepad", "  "]])
def test_free_typing_refuses_a_blank_name(names: list[str]) -> None:
    """An empty name is contained in every program's name: everything would need no yes."""
    with pytest.raises(ValidationError, match="free_typing"):
        PcSettings(free_typing=names)


def test_project_config_loads() -> None:
    s = load_settings(DEFAULT_CONFIG)
    assert s.stt.model == "large-v3"
    assert s.stt.fallback_model == "small"
    assert s.tts.voice == "it_IT-paola-medium"
    assert s.audio.block_size == 1280
    assert s.wakeword.models[0].name == "hey_claudio"
    assert s.wakeword.models[0].threshold == 0.30


def test_shipped_daily_budget_stays_within_80_dollars_a_month() -> None:
    assert load_settings(DEFAULT_CONFIG).cost.daily_budget * 31 <= 80


def test_type_max_chars_fits_the_retry_tokens() -> None:
    from claudio.brain.claude import CHARS_PER_TOKEN

    pc = load_settings(DEFAULT_CONFIG).pc
    assert pc.type_max_chars <= pc.retry_max_tokens * CHARS_PER_TOKEN


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


def test_replies_and_phrases_are_plain_text_and_lists(tmp_path: Path) -> None:
    s = load_settings(DEFAULT_CONFIG)
    assert s.router.unknown_reply == "Questo non lo so ancora fare."
    assert s.router.error_reply == "Non ci sono riuscito."
    assert "pensaci bene" in s.claude.smart_phrases and "a fondo" in s.claude.offer_phrases
    assert "barzelletta" in s.claude.no_offer_phrases
    f = tmp_path / "c.toml"
    f.write_text('[router]\nunknown_reply = "Non lo so."\n', encoding="utf-8")
    assert load_settings(f).router.unknown_reply == "Non lo so."


def test_plain_replies_in_env_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "c.toml"
    f.write_text("", encoding="utf-8")
    monkeypatch.setenv("CLAUDIO_ROUTER__UNKNOWN_REPLY", "Non lo so fare, scusa.")
    monkeypatch.setenv("CLAUDIO_ROUTER__ERROR_REPLY", "Ops.")
    s = load_settings(f)
    assert s.router.unknown_reply == "Non lo so fare, scusa."
    assert s.router.error_reply == "Ops."


@pytest.mark.parametrize(
    "line",
    [
        '[router]\nunknown_reply = { it = "Non lo so." }\n',
        '[claude]\nsmart_phrases = { it = ["pensaci bene"] }\n',
    ],
)
def test_the_old_per_language_forms_are_rejected(tmp_path: Path, line: str) -> None:
    f = tmp_path / "c.toml"
    f.write_text(line, encoding="utf-8")
    with pytest.raises(ConfigError, match="valori non validi"):
        load_settings(f)


def test_unreadable_env_value_has_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "c.toml"
    f.write_text("", encoding="utf-8")
    monkeypatch.setenv("CLAUDIO_STT__VOCABULARY_WORDS", "non è JSON")
    with pytest.raises(ConfigError, match="valori non validi"):
        load_settings(f)


def test_initial_prompt_from_env_replaces_the_shipped_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDIO_STT__INITIAL_PROMPT", "Hey Claudio, apri Spotify.")
    s = load_settings(DEFAULT_CONFIG)
    assert s.stt.initial_prompt == "Hey Claudio, apri Spotify."


def test_shipped_config_keeps_the_whisper_prompt() -> None:
    """The prompt moved from [stt.prompts] it to [stt] initial_prompt; an old [stt.prompts]
    left in a config would be ignored without an error, so the shipped one is pinned here."""
    assert load_settings(DEFAULT_CONFIG).stt.initial_prompt == (
        "Ehi Claudio, che ore sono? Claudio, raccontami una storia breve. "
        "Apri Chrome e la cartella Download, poi alza il volume."
    )


def test_configuration_error_is_printed_and_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from claudio.__main__ import main

    f = tmp_path / "c.toml"
    f.write_text("[stt]\nbeam_size = 0\n", encoding="utf-8")
    assert main(["--config", str(f), "--check"]) == 2  # supervisor.FINAL_EXITS relies on 2
    assert "Errore di configurazione" in capsys.readouterr().err


def test_an_old_effect_line_in_dotenv_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The voice effects are gone: a .env that still sets one must not stop Claudio."""
    env = tmp_path / ".env"
    env.write_text("CLAUDIO_TTS__EFFECT=none\n", encoding="utf-8")
    monkeypatch.setitem(Settings.model_config, "env_file", env)
    assert load_settings(DEFAULT_CONFIG).tts.engine == "jack"
