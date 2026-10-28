"""Manual test of Jack's worker: speaks a few sentences, prints timings, saves WAVs.

  models\\voices\\jack\\worker\\env\\Scripts\\python.exe scripts\\try_jack.py [--play]

Runs with the worker venv (numpy and soundfile are there); Claudio's venv is not needed.
Personal use only: the WAVs go to logs/jack-try/, which is not committed."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from claudio.tts.jack import JackSynth  # noqa: E402

SENTENCES = [
    "Ciao, sono Claudio. Ho aperto Chrome: non serve ringraziarmi.",
    "Volume al massimo. Adesso sì che si ragiona!",
    "Ti ho trovato tre file nella cartella Download. Prego, eroe.",
    "Non ci sono riuscito. Riprova tra un attimo.",
]


class Silent:
    sample_rate = 22050

    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(2205, dtype=np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--checkpoint", default="voices/jack/ft/1.7B/checkpoint-epoch-7")
    parser.add_argument("--play", action="store_true", help="also play through the default output")
    args = parser.parse_args()
    jack = args.models / "voices" / "jack"
    command = [
        str(jack / "worker" / "env" / "Scripts" / "python.exe"),
        str(ROOT / "src" / "claudio" / "tts" / "jack_worker.py"),
        "--checkpoint",
        str(args.models / args.checkpoint),
        "--rvc-model",
        str(jack / "rvc" / "Applio" / "logs" / "jack" / "jack_100e_2800s.pth"),
        "--applio",
        str(jack / "rvc" / "Applio"),
    ]
    out = ROOT / "logs" / "jack-try"
    out.mkdir(parents=True, exist_ok=True)
    synth = JackSynth(command, Silent())
    t0 = time.perf_counter()
    synth.start()
    print(
        f"start: {time.perf_counter() - t0:.1f} s (worker load {synth.load_s} s, "
        f"GPU in use {synth.vram_mb} MB)"
    )
    for n, text in enumerate(SENTENCES * 2, 1):  # twice: the second round shows the warm timing
        t = time.perf_counter()
        pieces, first = [], None
        for piece in synth.stream(text):
            first = first or 1000 * (time.perf_counter() - t)
            pieces.append(piece)
        wall = 1000 * (time.perf_counter() - t)
        audio = np.concatenate(pieces)
        seconds = len(audio) / synth.sample_rate
        print(
            f"{n}: first chunk after {first:.0f} ms (worker {synth.last_first_ms} ms), "
            f"sentence done in {wall:.0f} ms, {len(pieces)} chunks, {seconds:.2f} s of speech, "
            f"peak {np.abs(audio).max()}"
        )
        sf.write(out / f"{n}.wav", audio, synth.sample_rate)
        if args.play:
            import sounddevice as sd

            sd.play(audio, synth.sample_rate, blocking=True)
    synth.close()
    print(f"WAV in {out}")


if __name__ == "__main__":
    main()
