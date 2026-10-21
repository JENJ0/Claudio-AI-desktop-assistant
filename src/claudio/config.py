"""Settings loaded from config/claudio.toml, overridable by environment and .env."""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
    TomlConfigSettingsSource,
)

from claudio.i18n import tr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "claudio.toml"
DEFAULT_LANGUAGE = "it"  # commands.toml's plain `phrases` and `reply`, the first fallback


class ConfigError(Exception):
    """Configuration file missing or invalid, with a readable message."""


def _json_or(value: Any, plain: Any) -> Any:
    """A setting from .env or the environment: JSON when it looks like JSON ('{"it": ...}',
    '["it", "en"]'), else `plain(value)`. TOML values arrive already parsed."""
    if not isinstance(value, str):
        return value
    if value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return plain(value)


class AudioSettings(BaseModel):
    input_device: str | int = ""
    output_device: str | int = ""
    sample_rate: Literal[16000] = 16000
    block_ms: int = Field(80, gt=0, multiple_of=10)
    output_sample_rate: int = Field(24000, gt=0)
    beep_volume: float = Field(0.25, ge=0, le=1)
    input_gain_db: float = Field(0.0, ge=0, le=30)  # digital boost (the start, with auto_gain)
    auto_gain: bool = True  # Claudio adjusts input_gain_db itself after every request

    @property
    def block_size(self) -> int:
        return self.sample_rate * self.block_ms // 1000


class WakeWordModel(BaseModel):
    name: str
    threshold: float = Field(0.5, gt=0, lt=1)


class WakeWordSettings(BaseModel):
    # Any of these wakes CLAUDIO; each has its own threshold.
    models: list[WakeWordModel] = Field(
        default_factory=lambda: [WakeWordModel(name="hey_jarvis")], min_length=1
    )
    # A peak between this and the threshold is a missed wake word: it is logged and, with
    # second_chance, Whisper listens to the last 1.5 s and wakes CLAUDIO if it hears the name.
    near_miss_floor: float = Field(0.15, gt=0, lt=1)
    second_chance: bool = True
    # After a wake, keep the request only if Whisper heard the name at the start or it is a
    # local command: "Hey, ciao, come stai?" wakes the model but ends in silence.
    verify: bool = True


class VadSettings(BaseModel):
    threshold: float = Field(0.4, gt=0, lt=1)
    min_speech_ms: int = Field(64, ge=0)
    end_silence_ms: int = Field(400, gt=0)
    pre_roll_ms: int = Field(800, ge=0)
    no_speech_timeout_s: float = Field(4.0, gt=0)
    max_utterance_s: float = Field(8.0, gt=0)


class SttSettings(BaseModel):
    model: str = "large-v3-turbo"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "float16"
    fallback_model: str = "small"
    fallback_compute_type: str = "int8"
    # Whisper picks the language of every request among these; a single one is fixed.
    # In .env: CLAUDIO_STT__LANGUAGES=it,en (or ["it", "en"]).
    languages: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [DEFAULT_LANGUAGE], min_length=1
    )
    language: str = ""  # one fixed language (overrides `languages`)
    language_confidence: float = Field(0.6, ge=0, le=1)  # below this: the last language spoken
    # A language other than the first must be this sure (fast Italian often sounds English).
    other_confidence: float = Field(0.8, ge=0, le=1)
    # A request in another language that no command understands is transcribed again in the
    # first one (~120 ms on the GPU, only then): "cue the calculator" -> "apri la calcolatrice".
    retry_in_first_language: bool = True
    # A request no command understands: Whisper proposes 5 transcriptions and the first known
    # command wins (one more Whisper pass, only on requests not understood).
    closed_list: bool = True
    closed_list_max_words: int = Field(5, ge=0)  # longer requests skip it (0 = no limit)
    beam_size: int = Field(1, ge=1)
    initial_prompt: str = ""  # the first language's prompt, in place of prompts.<lang>
    prompts: dict[str, str] = Field(default_factory=dict)  # Whisper's prompt, per language
    no_speech_threshold: float = Field(0.6, ge=0, le=1)
    cpu_threads: int = Field(4, ge=1)

    @field_validator("languages", mode="before")
    @classmethod
    def _languages(cls, value: Any) -> Any:
        return _json_or(value, lambda text: [p for p in re.split(r"[\s,]+", text) if p])

    @model_validator(mode="after")
    def _fixed_language(self) -> SttSettings:
        if self.language:
            self.languages = [self.language]
        return self

    def prompt(self, language: str) -> str:
        """Whisper's prompt for `language`. initial_prompt (v0.1.0, often set in .env)
        replaces the first language's shipped prompt."""
        if self.initial_prompt and language == self.languages[0]:
            return self.initial_prompt
        return self.prompts.get(language, "")


class VoiceSettings(BaseModel):
    """The voices of one language; empty = the main (Italian) voice."""

    kokoro_voice: str = ""
    voice: str = ""  # Piper


class TtsSettings(BaseModel):
    engine: Literal["kokoro", "piper", "jack"] = "kokoro"
    kokoro_voice: str = "im_nicola"
    voice: str = "it_IT-paola-medium"  # Piper: engine "piper", and the fallback
    # Other languages' voices, by the language Whisper detects.
    voices: dict[str, VoiceSettings] = Field(
        default_factory=lambda: {
            "en": VoiceSettings(kokoro_voice="am_michael", voice="en_US-ryan-low")
        }
    )
    length_scale: float = Field(1.0, gt=0)
    noise_scale: float | None = Field(None, ge=0)  # None = the voice's own default
    noise_w_scale: float | None = Field(None, ge=0)
    use_cuda: bool = False
    effect: Literal["none", "estremo", "jarvis", "ultron", "ultron_estremo"] = "none"
    # Engine "jack": a worker process with its own venv; paths are relative to models_dir.
    jack_python: Path = Path("voices/jack/worker/env/Scripts/python.exe")
    jack_checkpoint: Path = Path("voices/jack/ft/0.6B/checkpoint-epoch-5")
    jack_rvc_model: Path = Path("voices/jack/rvc/Applio/logs/jack/jack_100e_2800s.pth")
    jack_applio: Path = Path("voices/jack/rvc/Applio")
    jack_startup_timeout_s: float = Field(90.0, gt=0)
    jack_sentence_timeout_s: float = Field(15.0, gt=0)

    def kokoro_voice_for(self, language: str) -> str:
        other = self.voices.get(language)
        return other.kokoro_voice if other and other.kokoro_voice else self.kokoro_voice

    def piper_voice_for(self, language: str) -> str:
        other = self.voices.get(language)
        return other.voice if other and other.voice else self.voice


UNKNOWN_REPLY = {"it": "Questo non lo so ancora fare.", "en": "I can't do that yet."}
ERROR_REPLY = {"it": "Non ci sono riuscito.", "en": "I couldn't do it."}


def in_language(texts: dict[str, str], language: str) -> str:
    """The text in `language`, else the Italian one."""
    return texts.get(language) or texts.get(DEFAULT_LANGUAGE, "")


class RouterSettings(BaseModel):
    commands_file: Path = Path("config/commands.toml")
    fuzzy_threshold: float = Field(85, ge=0, le=100)
    # Per language; a plain string (also in .env, as in v0.1.0) replaces the Italian one.
    unknown_reply: Annotated[dict[str, str], NoDecode] = Field(
        default_factory=lambda: dict(UNKNOWN_REPLY)
    )
    error_reply: Annotated[dict[str, str], NoDecode] = Field(
        default_factory=lambda: dict(ERROR_REPLY)
    )

    @field_validator("unknown_reply", mode="before")
    @classmethod
    def _unknown_reply(cls, value: Any) -> Any:
        return _json_or(value, lambda text: {**UNKNOWN_REPLY, DEFAULT_LANGUAGE: text})

    @field_validator("error_reply", mode="before")
    @classmethod
    def _error_reply(cls, value: Any) -> Any:
        return _json_or(value, lambda text: {**ERROR_REPLY, DEFAULT_LANGUAGE: text})


class ConversationSettings(BaseModel):
    # After a reply CLAUDIO keeps listening this long without the wake word (0 = off).
    window_s: float = Field(6.0, ge=0, le=30)
    # Listen while speaking: "basta" / "stop" cuts the reply short (best with headphones).
    barge_in: bool = True
    barge_in_max_s: float = Field(2.0, gt=0, le=8)  # longer speech is checked in slices this long


class CostSettings(BaseModel):
    pricing_file: Path = Path("config/pricing.toml")  # prices and cache multipliers
    db_file: Path = Path("data/spend.db")  # the ledger: one row per Claude call
    daily_budget: float = Field(1.0, gt=0)  # in the currency of the prices file (USD)
    warn_at: float = Field(0.8, gt=0, lt=1)  # share of the budget that is announced aloud


class ClaudeSettings(BaseModel):
    """Claude answers what no local command understands (Phase 2)."""

    enabled: bool = True  # false: Claudio stays local-only ("Questo non lo so ancora fare")
    model: str = "claude-haiku-5-5"  # must be in the prices file
    max_tokens: int = Field(300, ge=16, le=4096)  # a spoken reply: a few sentences
    effort: Literal["low", "medium", "high"] = "low"  # low: fastest first word
    thinking: bool = False  # Haiku thinks by default: off saves latency and tokens
    timeout_s: float = Field(10.0, gt=0, le=120)
    history_turns: int = Field(6, ge=0, le=50)  # question/answer pairs sent back as context
    history_reset_s: float = Field(300.0, gt=0)  # silence after which the context starts over
    conversation_min_words: int = Field(3, ge=1)  # without the wake word: shorter is not a request


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_audio: bool = True  # WAV of what CLAUDIO heard, in logs/audio/ (local only)
    keep_audio: int = Field(50, ge=1)
    keep_sessions: int = Field(30, ge=1)  # logs/sessions/<date_time>/: console + conversation


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAUDIO_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        extra="forbid",
    )

    # Set by load_settings() before instantiation.
    _toml_path: ClassVar[Path] = DEFAULT_CONFIG

    language: Literal["it", "en"] = "it"  # the program's own messages (console, --check)
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    data_dir: Path = Path("data")  # what Claudio learns and keeps (e.g. the microphone level)
    audio: AudioSettings = AudioSettings()
    wakeword: WakeWordSettings = WakeWordSettings()
    vad: VadSettings = VadSettings()
    stt: SttSettings = SttSettings()
    tts: TtsSettings = TtsSettings()
    router: RouterSettings = RouterSettings()
    conversation: ConversationSettings = ConversationSettings()
    cost: CostSettings = CostSettings()
    claude: ClaudeSettings = ClaudeSettings()
    # The Claude API key, only from .env or the environment (never in the TOML, never logged).
    anthropic_api_key: SecretStr | None = Field(None, validation_alias="ANTHROPIC_API_KEY")
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml = TomlConfigSettingsSource(settings_cls, toml_file=cls._toml_path)
        return (init_settings, env_settings, dotenv_settings, toml)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: Path | None = None, **overrides: Any) -> Settings:
    """Load settings; raise ConfigError with a readable message on problems."""
    path = path or Path(os.environ.get("CLAUDIO_CONFIG", DEFAULT_CONFIG))
    if not path.is_file():
        raise ConfigError(tr("File di configurazione non trovato: {path}", path=path))
    Settings._toml_path = path
    try:
        return Settings(**overrides)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(tr("{path}: TOML non valido ({error})", path=path, error=exc)) from exc
    except SettingsError as exc:  # a value in .env or the environment that cannot be read
        raise ConfigError(
            tr("{path}: valori non validi -> {problems}", path=path, problems=exc)
        ) from exc
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(
            tr("{path}: valori non validi -> {problems}", path=path, problems=problems)
        ) from exc


def configured_language(path: Path | None = None) -> str:
    """The `language` setting read without validating anything (environment, .env, TOML),
    so that even a configuration error is reported in it."""
    if value := os.environ.get("CLAUDIO_LANGUAGE"):
        return value.strip()
    try:
        for line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "CLAUDIO_LANGUAGE":
                return value.strip().strip("\"'")
    except OSError:
        pass
    path = path or Path(os.environ.get("CLAUDIO_CONFIG", DEFAULT_CONFIG))
    try:
        with path.open("rb") as fh:
            return str(tomllib.load(fh).get("language", DEFAULT_LANGUAGE))
    except (OSError, tomllib.TOMLDecodeError):
        return DEFAULT_LANGUAGE
