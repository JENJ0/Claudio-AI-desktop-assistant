"""faster-whisper transcription: GPU first, automatic CPU fallback."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from claudio.config import SttSettings
from claudio.router.normalizer import normalize

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
)
# Short ones only count when they are the whole transcription.
HALLUCINATIONS_EXACT = {"buon appetito", "grazie a tutti", "ciao a tutti", "alla prossima"}
MIN_RMS = 1e-3  # about -60 dBFS: below this there is nothing to transcribe


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


class WhisperTranscriber:
    def __init__(self, settings: SttSettings, download_root: Path) -> None:
        self.settings = settings
        self.download_root = download_root
        self.device = "cpu"
        self.model_name = settings.fallback_model
        self._model = self._load()

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
                    raise RuntimeError("nessuna GPU CUDA trovata")
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

    def transcribe(self, audio: np.ndarray) -> str:
        s = self.settings
        if audio.size == 0 or float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < MIN_RMS:
            return ""  # silence: Whisper would invent something ("Buon appetito!")
        segments, _info = self._model.transcribe(
            audio.astype(np.float32, copy=False),
            language=s.language,
            beam_size=s.beam_size,
            initial_prompt=s.initial_prompt or None,
            vad_filter=False,  # Silero already did it upstream
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob > s.no_speech_threshold and seg.avg_logprob < -1.0:
                continue
            parts.append(seg.text.strip())
        text = " ".join(p for p in parts if p).strip()
        if text and is_hallucination(text, s.initial_prompt):
            log.info("Trascrizione scartata (allucinazione nota): %r", text)
            return ""
        return text
