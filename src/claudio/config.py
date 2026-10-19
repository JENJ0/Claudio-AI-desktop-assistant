"""Settings loaded from config/claudio.toml, overridable by environment and .env."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "claudio.toml"


class ConfigError(Exception):
    """Configuration file missing or invalid, with a readable message."""


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
    language: str = "it"
    beam_size: int = Field(1, ge=1)
    initial_prompt: str = ""
    no_speech_threshold: float = Field(0.6, ge=0, le=1)
    cpu_threads: int = Field(4, ge=1)


class TtsSettings(BaseModel):
    engine: Literal["kokoro", "piper"] = "kokoro"
    kokoro_voice: str = "im_nicola"
    voice: str = "it_IT-paola-medium"  # Piper: engine "piper", and the fallback
    length_scale: float = Field(1.0, gt=0)
    noise_scale: float | None = Field(None, ge=0)  # None = the voice's own default
    noise_w_scale: float | None = Field(None, ge=0)
    use_cuda: bool = False
    effect: Literal["none", "estremo", "jarvis", "ultron", "ultron_estremo"] = "none"


class RouterSettings(BaseModel):
    commands_file: Path = Path("config/commands.toml")
    fuzzy_threshold: float = Field(85, ge=0, le=100)
    unknown_reply: str = "Questo non lo so ancora fare."


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_audio: bool = True  # WAV of what CLAUDIO heard, in logs/audio/ (local only)
    keep_audio: int = Field(50, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAUDIO_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        extra="forbid",
    )

    # Set by load_settings() before instantiation.
    _toml_path: ClassVar[Path] = DEFAULT_CONFIG

    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    data_dir: Path = Path("data")  # what Claudio learns and keeps (e.g. the microphone level)
    audio: AudioSettings = AudioSettings()
    wakeword: WakeWordSettings = WakeWordSettings()
    vad: VadSettings = VadSettings()
    stt: SttSettings = SttSettings()
    tts: TtsSettings = TtsSettings()
    router: RouterSettings = RouterSettings()
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
        raise ConfigError(f"File di configurazione non trovato: {path}")
    Settings._toml_path = path
    try:
        return Settings(**overrides)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: TOML non valido ({exc})") from exc
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{path}: valori non validi -> {problems}") from exc
