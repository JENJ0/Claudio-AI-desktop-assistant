"""faster-whisper transcription: GPU first, automatic CPU fallback. With several languages
configured, Whisper first picks which one each request is in."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from claudio.config import SttSettings
from claudio.i18n import tr
from claudio.router.normalizer import normalize, only_wake_words

log = logging.getLogger(__name__)

# Phrases Whisper invents on silence or noise (normalised form).
HALLUCINATIONS = (
    "sottotitoli creati dalla comunita amara org",
    "sottotitoli e revisione a cura di qtss",
    "sottotitoli a cura di",
    "grazie per la visione",
    "grazie per l attenzione",
    "iscriviti al canale",
    "amara org",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subtitles by",
)
# Short ones only count when they are the whole transcription.
HALLUCINATIONS_EXACT = {
    "buon appetito",
    "grazie a tutti",
    "ciao a tutti",
    "alla prossima",
    "you",
    "bye bye",
}
MIN_RMS = 1e-3  # about -60 dBFS: below this there is nothing to transcribe
MAX_NEW_TOKENS = 80  # an 8 s request is about 50 tokens; a repetition loop stops here
LOOP_RATIO = 2.4  # gzip ratio above which a text is a loop ("Ciao, ciao, ciao..."), as in Whisper


def register_cuda_dlls() -> None:
    """Make the pip-installed CUDA libraries (nvidia-cublas/cudnn-cu12) loadable on Windows."""
    if sys.platform != "win32":
        return
    try:
        import nvidia  # namespace package from nvidia-*-cu12 wheels
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def is_hallucination(text: str, prompt: str = "") -> bool:
    norm = normalize(text, drop_fillers=False)
    if not norm:
        return True
    if norm in HALLUCINATIONS_EXACT or any(h in norm for h in HALLUCINATIONS):
        return True
    # Whisper sometimes echoes the initial prompt on near-silent audio.
    return bool(prompt) and len(norm.split()) >= 6 and norm in normalize(prompt, drop_fillers=False)


def pick_language(
    probs: Mapping[str, float], languages: Sequence[str], last: str, min_confidence: float
) -> tuple[str, float]:
    """The most likely of `languages` and its share of their probability; `last` (the language
    of the previous request) when that share is below `min_confidence`. Whisper's guesses
    outside the list ("es" for Italian said fast) are ignored."""
    scores = {lang: probs.get(lang, 0.0) for lang in languages}
    total = sum(scores.values())
    best = max(scores, key=lambda lang: scores[lang])
    confidence = scores[best] / total if total > 0 else 0.0
    return (best if confidence >= min_confidence else last), confidence


class WhisperTranscriber:
    def __init__(self, settings: SttSettings, download_root: Path) -> None:
        self.settings = settings
        self.download_root = download_root
        self.device = "cpu"
        self.model_name = settings.fallback_model
        self.language = settings.languages[0]  # heard in the last audio
        self.language_confidence = 1.0
        # What an unsure detection falls back to: the language of the last real request,
        # never that of the warm-up noise, a lone wake word or a dropped hallucination.
        self.last_language = settings.languages[0]
        self._encoded: tuple[np.ndarray, Any] | None = None
        self._reused = 0  # encoder passes saved, for the tests
        self._model = self._load()
        self.language, self.language_confidence = settings.languages[0], 1.0
        self.last_language = settings.languages[0]

    def _open(self, name: str, **kwargs: Any) -> Any:
        from faster_whisper import WhisperModel

        root = str(self.download_root)
        try:  # already downloaded: start offline, without contacting huggingface.co
            return WhisperModel(name, download_root=root, local_files_only=True, **kwargs)
        except FileNotFoundError:
            return WhisperModel(name, download_root=root, **kwargs)

    def _load(self) -> Any:
        s = self.settings
        if s.device in ("auto", "cuda"):
            try:
                register_cuda_dlls()
                import ctranslate2

                if ctranslate2.get_cuda_device_count() == 0:
                    raise RuntimeError(tr("nessuna GPU CUDA trovata"))
                model = self._open(s.model, device="cuda", compute_type=s.compute_type)
                self._model, self.device, self.model_name = model, "cuda", s.model
                self._warm_up()  # CUDA kernel problems (e.g. Blackwell) show up here
                return model
            except Exception as exc:
                if s.device == "cuda":
                    raise
                log.warning(
                    "GPU non utilizzabile (%s): passo alla CPU con '%s' %s",
                    exc,
                    s.fallback_model,
                    s.fallback_compute_type,
                )
        model = self._open(
            s.fallback_model,
            device="cpu",
            compute_type=s.fallback_compute_type,
            cpu_threads=s.cpu_threads,
        )
        self._model, self.device, self.model_name = model, "cpu", s.fallback_model
        self._warm_up()
        return model

    def _warm_up(self) -> None:
        start = time.perf_counter()
        noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        self.transcribe(noise)
        log.debug("STT riscaldato in %.0f ms", (time.perf_counter() - start) * 1000)

    def detect_language(self, audio: np.ndarray) -> str:
        """Pick the request's language among the configured ones (one: no work at all)."""
        s = self.settings
        if len(s.languages) == 1:
            self.language, self.language_confidence = s.languages[0], 1.0
            return self.language
        from faster_whisper.audio import pad_or_trim

        model = self._model
        features = model.feature_extractor(audio)
        # The same first window Whisper encodes when transcribing: the output is reused.
        frames = min(model.feature_extractor.nb_max_frames, features.shape[-1] - 1)
        segment = pad_or_trim(features[:, :frames])
        encoded = model.encode(segment)
        probs = {token[2:-2]: prob for token, prob in model.model.detect_language(encoded)[0]}
        self.language, self.language_confidence = pick_language(
            probs, s.languages, self.last_language, s.language_confidence
        )
        self._encoded = (segment, encoded)
        log.debug("Lingua: %s (%.2f)", self.language, self.language_confidence)
        return self.language

    def _encode(self, features: np.ndarray) -> Any:
        """Whisper's encoder, answering from what detect_language() already encoded."""
        if self._encoded is not None:
            segment, encoded = self._encoded
            self._encoded = None
            if features.shape == segment.shape and np.array_equal(features, segment):
                self._reused += 1
                return encoded
        return type(self._model).encode(self._model, features)

    def transcribe(self, audio: np.ndarray) -> str:
        s = self.settings
        if audio.size == 0 or float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < MIN_RMS:
            return ""  # silence: Whisper would invent something ("Buon appetito!")
        audio = audio.astype(np.float32, copy=False)
        language = self.detect_language(audio)
        prompt = s.prompt(language)
        reuse = self._encoded is not None
        if reuse:
            self._model.encode = self._encode
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=s.beam_size,
                initial_prompt=prompt or None,
                vad_filter=False,  # Silero already did it upstream
                condition_on_previous_text=False,
                without_timestamps=True,
                temperature=0.0,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            parts = []
            for seg in segments:  # a generator: Whisper runs while it is consumed
                if seg.no_speech_prob > s.no_speech_threshold and seg.avg_logprob < -1.0:
                    continue
                if seg.compression_ratio > LOOP_RATIO:
                    log.info("Trascrizione scartata (ripetizione): %r", seg.text[:80])
                    continue
                parts.append(seg.text.strip())
        finally:
            if reuse:
                del self._model.encode
            self._encoded = None
        text = " ".join(p for p in parts if p).strip()
        if text and is_hallucination(text, prompt):
            log.info("Trascrizione scartata (allucinazione nota): %r", text)
            return ""
        if text and not only_wake_words(text):
            self.last_language = language
        return text
