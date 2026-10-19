"""First step of v0.1.0: does faster-whisper really run on the RTX 5070 Ti?

    uv run python scripts/check_gpu.py
Loads the configured model on CUDA (no CPU fallback), transcribes 4 s of audio 5 times
and prints the timings. Any CUDA/cuDNN/Blackwell problem shows up here with its error.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from claudio.config import load_settings
from claudio.stt.transcriber import WhisperTranscriber, register_cuda_dlls


def main() -> int:
    settings = load_settings()
    register_cuda_dlls()
    import ctranslate2

    print(f"ctranslate2 {ctranslate2.__version__}, GPU CUDA: {ctranslate2.get_cuda_device_count()}")
    if ctranslate2.get_cuda_device_count() == 0:
        print(
            "Nessuna GPU vista da CTranslate2: aggiorna il driver NVIDIA e verifica "
            "`uv sync --extra gpu`."
        )
        return 1
    print("Tipi supportati:", ", ".join(sorted(ctranslate2.get_supported_compute_types("cuda"))))
    stt_settings = settings.stt.model_copy(update={"device": "cuda"})
    t0 = time.perf_counter()
    try:
        stt = WhisperTranscriber(stt_settings, settings.resolve(settings.models_dir) / "whisper")
    except Exception as exc:
        print(f"ERRORE caricando {stt_settings.model} su CUDA: {exc!r}")
        return 1
    print(
        f"{stt.model_name} {stt_settings.compute_type} caricato e riscaldato in "
        f"{time.perf_counter() - t0:.1f} s"
    )
    audio = (np.random.default_rng(1).standard_normal(16000 * 4) * 0.05).astype(np.float32)
    times = []
    for _ in range(5):
        t1 = time.perf_counter()
        stt.transcribe(audio)
        times.append((time.perf_counter() - t1) * 1000)
    print("Trascrizione di 4 s:", ", ".join(f"{t:.0f}" for t in times), "ms")
    print("GPU OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
