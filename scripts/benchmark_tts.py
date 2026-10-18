"""Time to the first sentence for every Piper voice in models/piper.

uv run python scripts/benchmark_tts.py
uv run python scripts/benchmark_tts.py --play   # also listen to each voice
"""

from __future__ import annotations

import argparse
import statistics
import time

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
    voices = sorted((settings.resolve(settings.models_dir) / "piper").glob("*.onnx"))
    if not voices:
        raise SystemExit("Nessuna voce: uv run python scripts/download_models.py")
    for path in voices:
        t0 = time.perf_counter()
        synth = PiperSynth(path, settings.tts)
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
            f"{path.stem:<28} caricata in {load_s:.1f} s | prima frase: mediana "
            f"{statistics.median(times):.0f} ms, max {max(times):.0f} ms | RTF "
            f"{statistics.median(rtf):.3f}"
        )


if __name__ == "__main__":
    main()
