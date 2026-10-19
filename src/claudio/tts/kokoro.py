"""Kokoro-82M (ONNX) Italian voices: Nicola (male, the default) and Sara.

Phonemes come from the espeak-ng bundled with piper-tts, so nothing else to install; the
model, the voice styles and the phoneme vocabulary are fetched by scripts/download_models.py."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from claudio.config import TtsSettings

log = logging.getLogger(__name__)

KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin", "config.json")
SAMPLE_RATE = 24000
MAX_TOKENS = 510
# piper's espeak-ng writes the Italian tapped r as "ɾ" and adds a palatal "ʲ"; Kokoro was
# trained on upstream espeak-ng, which writes a plain "r" and no "ʲ".
PHONEME_FIXES = str.maketrans({"ɾ": "r", "ʲ": None})
# Kokoro's own phonemizer ties affricates into one symbol; read as two sounds, "tʃ" drags
# words like "calcolatrice" and "dieci".
AFFRICATES = (("tʃ", "ʧ"), ("dʒ", "ʤ"), ("ts", "ʦ"), ("dz", "ʣ"))


def kokoro_files(kokoro_dir: Path) -> tuple[Path, Path, Path]:
    model, voices, config = (kokoro_dir / name for name in KOKORO_FILES)
    missing = [p.name for p in (model, voices, config) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Kokoro: mancano {', '.join(missing)} in {kokoro_dir}. "
            "Esegui: uv run python scripts/download_models.py"
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


def kokoro_speed(length_scale: float) -> float:
    """One knob for both engines: Piper's length_scale 0.8 is Kokoro's speed 1.25."""
    return float(np.clip(1 / length_scale, 0.5, 2.0))


class KokoroSynth:
    sample_rate = SAMPLE_RATE

    def __init__(self, kokoro_dir: Path, settings: TtsSettings) -> None:
        import onnxruntime as ort
        from piper.phonemize_espeak import EspeakPhonemizer

        model, voices, config = kokoro_files(kokoro_dir)
        self.vocab: dict[str, int] = json.loads(config.read_text(encoding="utf-8"))["vocab"]
        styles = np.load(voices)
        if settings.kokoro_voice not in styles.files:
            italian = ", ".join(sorted(v for v in styles.files if v.startswith(("if_", "im_"))))
            raise ValueError(
                f"Voce Kokoro '{settings.kokoro_voice}' sconosciuta (italiane: {italian})"
            )
        self.voice = settings.kokoro_voice
        self._styles = styles[settings.kokoro_voice]  # one style per phoneme count
        self.speed = kokoro_speed(settings.length_scale)
        self._session: Any = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self._phonemizer = EspeakPhonemizer()

    def phonemes(self, text: str) -> str:
        sentences = self._phonemizer.phonemize("it", text)
        joined = " ".join(unicodedata.normalize("NFC", "".join(s)) for s in sentences)
        joined = joined.translate(PHONEME_FIXES)
        for pair, tied in AFFRICATES:
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
