"""Kokoro-82M (ONNX) voices: Nicola (Italian, male, the default), Sara, and Michael for English.

Phonemes come from the espeak-ng bundled with piper-tts, so nothing else to install; the
model, the voice styles and the phoneme vocabulary are fetched by scripts/download_models.py."""

from __future__ import annotations

import copy
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from claudio.config import DEFAULT_LANGUAGE, TtsSettings
from claudio.i18n import tr

log = logging.getLogger(__name__)

KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin", "config.json")
SAMPLE_RATE = 24000
MAX_TOKENS = 510
# piper's espeak-ng writes the Italian tapped r as "ɾ" and adds a palatal "ʲ"; Kokoro was
# trained on upstream espeak-ng, which writes a plain "r" and no "ʲ".
PHONEME_FIXES = str.maketrans({"ɾ": "r", "ʲ": None})
EN_PHONEME_FIXES = str.maketrans({"ʲ": None})  # "p.m.": pˌiːʲˈɛm
# Kokoro's own phonemizer ties affricates into one symbol; read as two sounds, "tʃ" drags
# words like "calcolatrice" and "dieci".
AFFRICATES = (("tʃ", "ʧ"), ("dʒ", "ʤ"), ("ts", "ʦ"), ("dz", "ʣ"))
# English keeps "ts" ("cats") and the flap "ɾ" ("better"): Kokoro's English has both.
EN_AFFRICATES = AFFRICATES[:2]
ESPEAK_VOICES = {"it": "it", "en": "en-us"}
VOICE_PREFIXES = {"it": ("if_", "im_"), "en": ("af_", "am_", "bf_", "bm_")}


def kokoro_files(kokoro_dir: Path) -> tuple[Path, Path, Path]:
    model, voices, config = (kokoro_dir / name for name in KOKORO_FILES)
    missing = [p.name for p in (model, voices, config) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            tr(
                "Kokoro: mancano {files} in {folder}. Esegui: "
                "uv run python scripts/download_models.py",
                files=", ".join(missing),
                folder=kokoro_dir,
            )
        )
    return model, voices, config


def to_tokens(phonemes: str, vocab: dict[str, int]) -> list[int]:
    """Phonemes the model knows, as ids; anything else (rare symbols) is dropped."""
    tokens = [vocab[p] for p in phonemes if p in vocab]
    if len(tokens) > MAX_TOKENS:
        log.warning("Frase troppo lunga per Kokoro (%d fonemi): la taglio", len(tokens))
        tokens = tokens[:MAX_TOKENS]
    return tokens


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 40,
    head_s: float = 0.02,
    tail_s: float = 0.12,
) -> np.ndarray:
    """Cut the silence Kokoro leaves around a sentence, keeping a short natural tail."""
    frame = sample_rate // 100
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    if rms.max() == 0:
        return audio[:0]
    loud = np.flatnonzero(rms >= rms.max() * 10 ** (-top_db / 20))
    start = max(int(loud[0]) * frame - int(head_s * sample_rate), 0)
    end = min((int(loud[-1]) + 1) * frame + int(tail_s * sample_rate), len(audio))
    return audio[start:end]


def piper_phonemizer() -> Any:
    """The espeak-ng phonemizer piper keeps for the whole process. espeak-ng must be set up
    only once: every EspeakPhonemizer() sets it up again, and on Windows the second time
    corrupts its phoneme tables (both voices speak gibberish, then an access violation)."""
    import piper.voice as piper_voice
    from piper.phonemize_espeak import EspeakPhonemizer

    with piper_voice._ESPEAK_PHONEMIZER_LOCK:
        if piper_voice._ESPEAK_PHONEMIZER is None:
            piper_voice._ESPEAK_PHONEMIZER = EspeakPhonemizer()
        return piper_voice._ESPEAK_PHONEMIZER


def kokoro_speed(length_scale: float) -> float:
    """One knob for both engines: Piper's length_scale 0.8 is Kokoro's speed 1.25."""
    return float(np.clip(1 / length_scale, 0.5, 2.0))


class KokoroSynth:
    sample_rate = SAMPLE_RATE
    language = DEFAULT_LANGUAGE

    def __init__(
        self, kokoro_dir: Path, settings: TtsSettings, language: str = DEFAULT_LANGUAGE
    ) -> None:
        import onnxruntime as ort

        model, voices, config = kokoro_files(kokoro_dir)
        self.vocab: dict[str, int] = json.loads(config.read_text(encoding="utf-8"))["vocab"]
        self._voices = np.load(voices)
        self._settings = settings
        self._use_voice(language)
        self.speed = kokoro_speed(settings.length_scale)
        self._session: Any = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self._phonemizer = piper_phonemizer()

    def _use_voice(self, language: str) -> None:
        voice = self._settings.kokoro_voice_for(language)
        if voice not in self._voices.files:
            prefixes = VOICE_PREFIXES.get(language, ("",))
            known = ", ".join(sorted(v for v in self._voices.files if v.startswith(prefixes)))
            raise ValueError(
                tr(
                    "Voce Kokoro '{voice}' sconosciuta (disponibili: {known})",
                    voice=voice,
                    known=known,
                )
            )
        self.language, self.voice = language, voice
        self._styles = self._voices[voice]  # one style per phoneme count

    def for_language(self, language: str) -> KokoroSynth:
        """The same model speaking `language` with its voice, without loading it twice."""
        other = copy.copy(self)
        other._use_voice(language)
        return other

    def phonemes(self, text: str) -> str:
        import piper.voice as piper_voice

        with piper_voice._ESPEAK_PHONEMIZER_LOCK:  # one espeak-ng, also used by Piper's thread
            sentences = self._phonemizer.phonemize(
                ESPEAK_VOICES.get(self.language, self.language), text
            )
        joined = " ".join(unicodedata.normalize("NFC", "".join(s)) for s in sentences)
        italian = self.language == "it"
        joined = joined.translate(PHONEME_FIXES if italian else EN_PHONEME_FIXES)
        for pair, tied in AFFRICATES if italian else EN_AFFRICATES:
            joined = joined.replace(pair, tied)
        return joined

    def synthesize(self, text: str) -> np.ndarray:
        tokens = to_tokens(self.phonemes(text), self.vocab)
        if not tokens:
            return np.zeros(0, dtype=np.int16)
        (audio, *_) = self._session.run(
            None,
            {
                "tokens": np.array([[0, *tokens, 0]], dtype=np.int64),
                "style": self._styles[len(tokens) - 1],
                "speed": np.array([self.speed], dtype=np.float32),
            },
        )
        audio = trim_silence(np.asarray(audio, dtype=np.float32).ravel(), SAMPLE_RATE)
        return (np.clip(audio, -1, 1) * 32767).astype(np.int16)
