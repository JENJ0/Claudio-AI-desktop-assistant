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
    # Every detection of this model is checked by Whisper first (the name must be heard): for
    # a model that fires often on noise, like "claudio" alone. No near misses are logged.
    confirm: bool = False


class WakeWordSettings(BaseModel):
    # Any of these wakes CLAUDIO; each has its own threshold.
    models: list[WakeWordModel] = Field(
        default_factory=lambda: [WakeWordModel(name="hey_jarvis")], min_length=1
    )
    # A peak between this and the threshold is a missed wake word: it is logged and, with
    # second_chance, Whisper listens to the last 1.5 s and wakes CLAUDIO if it hears the name.
    near_miss_floor: float = Field(0.15, gt=0, lt=1)
    # A wake scoring below this is confirmed by Whisper first (the second chance, ~130 ms):
    # a weak detection on Claudio's own voice or the TV stays silent. 0 = off.
    confirm_below: float = Field(0.45, ge=0, lt=1)
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
    max_utterance_s: float = Field(15.0, gt=0)
    # A longer phrase (a question for Claude) gets a longer pause before it is closed: people
    # stop to think mid-sentence. 0 = always end_silence_ms.
    long_end_silence_ms: int = Field(800, ge=0)
    long_after_s: float = Field(1.5, ge=0)  # speech after which the longer pause applies


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
    other_confidence: float = Field(0.9, ge=0, le=1)
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
    # Claudio listens while Jack's worker is still loading: a reply in that time waits this
    # long for Jack, then the fallback voice speaks it (0 = fallback at once).
    jack_first_wait_s: float = Field(30.0, ge=0)
    # The language Qwen3 is told. "Auto" guesses it per sentence (accent and pace may drift);
    # "Italian" fixes it. "Italian" is untested on the fine-tuned model: JenJo picks by ear.
    jack_language: Literal["Auto", "Italian"] = "Auto"
    # Jack's pace: < 1 slower (0.9 = 10% slower, JenJo 23/10). The audio is stretched before
    # RVC and RVC raises the pitch back by the same amount, so the voice does not get deeper.
    jack_speed: float = Field(0.9, ge=0.7, le=1.3)
    # The first piece of a reply may end at a comma once it is this long: the voice starts
    # before the whole first sentence has arrived from Claude (0 = whole sentences only).
    first_clause_chars: int = Field(25, ge=0, le=200)

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


class StartupSettings(BaseModel):
    # Load Whisper, wake word, voice and Claude at the same time (threads) instead of one
    # after the other. Off = the old sequential start, if a model misbehaves when loaded so.
    parallel: bool = True
    # Say a short "ready" once everything, Jack's voice included, has loaded.
    announce_ready: bool = True


class ConversationSettings(BaseModel):
    # After a reply CLAUDIO keeps listening this long without the wake word (0 = off).
    window_s: float = Field(6.0, ge=0, le=30)
    # Listen while speaking: "basta" / "stop" cuts the reply short (best with headphones).
    barge_in: bool = True
    barge_in_max_s: float = Field(1.2, gt=0, le=8)  # longer speech is checked in slices this long
    # What a voice heard while speaking does: "duck" lowers the reply at once and stops it when
    # Whisper hears real words (back to full volume if it was noise); "stop" stops at once;
    # "confirm" keeps speaking until Whisper hears words. Then what was said is the next request.
    barge_in_mode: Literal["duck", "stop", "confirm"] = "duck"
    barge_in_min_dbfs: float = Field(-30.0, le=0)  # quieter voices (the TV, the echo) are ignored
    barge_in_vad: float = Field(0.6, gt=0, lt=1)  # speech probability needed (stricter than [vad])
    barge_in_min_speech_ms: int = Field(300, ge=0)  # voice this long before anything happens
    duck_volume: float = Field(0.3, ge=0, le=1)  # the reply's volume while Whisper checks
    # true: in the window only questions, "dimmi..." or the name reach Claude (the TV costs
    # nothing). false: everything is answered until "basta" or silence.
    only_addressed: bool = False


class NoiseSettings(BaseModel):
    """Noise transcribed as words (the TV, a cough, "Tremi un podcast"): a phrase no command
    understands is not sent to Claude when it is short and Whisper was unsure of all of it, or
    when it was too quiet to be someone talking to the microphone. Local commands are free and
    never judged. Every drop is logged as drop_reason=noise with the figures, to tune these."""

    enabled: bool = True
    max_words: int = Field(4, ge=0)  # only phrases this short are judged on Whisper's confidence
    no_speech_prob: float = Field(0.5, ge=0, le=1)  # a segment is unsure above this...
    avg_logprob: float = Field(-0.8, le=0)  # ...or below this
    min_speech_dbfs: float = Field(-45.0, le=0)  # loudest 32 ms below this: not a request
    # The answer to "Confermi?": quieter than this is the TV, not JenJo (treated as unclear).
    answer_min_dbfs: float = Field(-35.0, le=0)


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
    cache: bool = True  # prompt caching: from the 2nd turn the conversation is read at 10 %
    tell_time: bool = True  # the local date and time go with every request (after the cache)
    keepalive_s: float = Field(60.0, ge=0, le=600)  # keep the HTTPS connection open between turns
    # Sonnet for what needs reasoning: Haiku hands a request over with the `ask_sonnet` tool,
    # or JenJo asks for it ("pensaci bene"). Empty smart_model: Haiku only.
    smart_model: str = "claude-sonnet-5-5"  # must be in the prices file
    smart_max_tokens: int = Field(1024, ge=64, le=8192)  # its thinking counts here too
    smart_effort: Literal["low", "medium", "high"] = "low"
    escalate: bool = True  # Haiku may call Sonnet by itself (one short tool definition per call)
    # The personality (CLAUDIO.md), after the system prompt: read once at start, so it is
    # cached. Missing: copied from config/CLAUDIO.md. Longer than the limit: cut (it costs).
    personality_file: Path = Path("data/CLAUDIO.md")
    personality_max_chars: int = Field(3000, ge=0, le=20000)  # 0 = no personality
    smart_phrases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "it": ["pensaci bene", "ragionaci", "ragiona bene", "pensaci su"],
            "en": ["think hard", "think carefully", "think it through"],
        }
    )
    # A request for detail: Claudio proposes Sonnet at once ("Lo uso?"), no Haiku call.
    offer_phrases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "it": ["in dettaglio", "nel dettaglio", "approfondisci", "spiegami bene", "a fondo"],
            "en": ["in detail", "in depth", "explain thoroughly"],
        }
    )


class PcSettings(BaseModel):
    """Claude acts on the PC with tools (Phase 3). JenJo chose "Equilibrata" (23/10): harmless
    keys without a question, apps from the Start menu, files only in the user folder."""

    enabled: bool = True  # false (or not Windows): Claude answers but cannot act
    max_rounds: int = Field(4, ge=1, le=10)  # tool round trips per request
    max_tokens: int = Field(400, ge=64, le=4096)  # per call when tools are offered
    # Level 2: only "Confermi?" (JenJo, 23/10). False: the action is read out first. Level 3
    # always says what will run: an explicit yes needs to know to what.
    confirm_brief: bool = True
    roots: list[str] = Field(default_factory=lambda: ["~"])  # "*" = any local drive
    safe_keys: list[str] = Field(
        default_factory=lambda: [
            *("up", "down", "left", "right", "page_up", "page_down", "home", "end"),
            *("tab", "shift+tab", "escape", "f5", "alt+tab", "alt+left", "alt+right", "win"),
            *("ctrl+c", "ctrl+t", "ctrl+f", "ctrl+tab", "ctrl+shift+tab"),
            *("ctrl+plus", "ctrl+minus", "ctrl+0"),
        ]
    )  # level 1; every other combination asks first. [] = always ask
    type_max_chars: int = Field(500, ge=1, le=5000)
    read_chars: int = Field(1500, ge=100, le=10000)  # window text sent to Claude
    read_timeout_s: float = Field(2.0, gt=0, le=30)  # slow apps (Electron) stop here
    search_limit: int = Field(8, ge=1, le=50)
    search_timeout_s: float = Field(3.0, gt=0, le=30)  # folder walk without the Windows index
    max_windows: int = Field(25, ge=1, le=100)
    apps_refresh_s: float = Field(600.0, ge=0)
    # Windows never read (title or program): password managers.
    private_windows: list[str] = Field(
        default_factory=lambda: ["keepass", "bitwarden", "1password", "lastpass", "password"]
    )


class ChatSettings(BaseModel):
    """The chat window (`claudio --ui`): what is written there reaches the running assistant."""

    typed: bool = True  # answer the messages written in the window
    speak_typed: bool = False  # JenJo's choice (23/10): a written message gets a written reply
    poll_s: float = Field(0.3, ge=0.05, le=5)  # how often the assistant looks for new ones
    max_age_s: float = Field(120.0, gt=0)  # older ones (written while Claudio was off) are dropped


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_audio: bool = True  # WAV of what CLAUDIO heard, in logs/audio/ (local only)
    keep_audio: int = Field(50, ge=1)
    keep_sessions: int = Field(30, ge=1)  # logs/sessions/<date_time>/: console + conversation
    # Privacy: the chat (data/claudio.db), logs/sessions/ and logs/audio/ forget what is older
    # than this many days (at startup).
    retention_days: int = Field(30, ge=1)


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
    startup: StartupSettings = StartupSettings()
    noise: NoiseSettings = NoiseSettings()
    cost: CostSettings = CostSettings()
    claude: ClaudeSettings = ClaudeSettings()
    pc: PcSettings = PcSettings()
    # The Claude API key, only from .env or the environment (never in the TOML, never logged).
    anthropic_api_key: SecretStr | None = Field(None, validation_alias="ANTHROPIC_API_KEY")
    chat: ChatSettings = ChatSettings()
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
