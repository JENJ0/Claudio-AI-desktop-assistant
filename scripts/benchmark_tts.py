"""Time to the first sentence for Kokoro and every Piper voice in models/.

uv run python scripts/benchmark_tts.py
uv run python scripts/benchmark_tts.py --play   # also listen to each voice
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

from claudio.config import load_settings
from claudio.tts.speaker import PiperSynth

SENTENCES = [
    "Apro la cartella Download.",
    "Sono le 14 e 35.",
    "Oggi è sabato 18 ottobre 2026.",
    "Questo non lo so ancora fare, ma presto imparerò.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    models = settings.resolve(settings.models_dir)
    engines: list[tuple[str, Callable[[], Any]]] = []
    if (models / "kokoro").is_dir():
        from claudio.tts.kokoro import KokoroSynth

        for voice in ("im_nicola", "if_sara"):
            tts = settings.tts.model_copy(update={"kokoro_voice": voice})
            engines.append((f"kokoro {voice}", lambda tts=tts: KokoroSynth(models / "kokoro", tts)))
    for path in sorted((models / "piper").glob("*.onnx")):
        engines.append((path.stem, lambda path=path: PiperSynth(path, settings.tts)))
    if not engines:
        raise SystemExit("Nessuna voce: uv run python scripts/download_models.py")
    print(f"length_scale {settings.tts.length_scale}")
    for name, make in engines:
        t0 = time.perf_counter()
        synth = make()
        synth.synthesize("Ciao.")
        load_s = time.perf_counter() - t0
        times, rtf = [], []
        for text in SENTENCES:
            t1 = time.perf_counter()
            audio = synth.synthesize(text)
            elapsed = time.perf_counter() - t1
            times.append(elapsed * 1000)
            rtf.append(elapsed / (len(audio) / synth.sample_rate))
            if args.play:
                import sounddevice as sd

                sd.play(audio, synth.sample_rate)
                sd.wait()
        print(
            f"{name:<28} caricata in {load_s:.1f} s | prima frase: mediana "
            f"{statistics.median(times):.0f} ms, max {max(times):.0f} ms | RTF "
            f"{statistics.median(rtf):.3f}"
        )


if __name__ == "__main__":
    main()
