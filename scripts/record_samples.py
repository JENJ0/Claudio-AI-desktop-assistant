"""Record WAV samples with the real microphone.

Wake words (≈50 takes per phrase, for validation and the verifier):
    uv run python scripts/record_samples.py wakeword --phrase "Hey Claudio" --count 50
    uv run python scripts/record_samples.py wakeword --phrase "Claudio" --count 50
Commands (the first phrase of every command, x2, for benchmark_latency.py):
    uv run python scripts/record_samples.py commands --repeat 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from claudio.config import PROJECT_ROOT, load_settings
from claudio.router.matcher import load_commands

SR = 16000


def record(seconds: float, device: int | str | None) -> np.ndarray:
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16", device=device)
    sd.wait()
    return audio[:, 0]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["wakeword", "commands"])
    parser.add_argument("--phrase", default="Hey Claudio")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "samples")
    args = parser.parse_args()
    device = settings.audio.input_device or None

    if args.mode == "wakeword":
        prompts = [(args.phrase, f"{slug(args.phrase)}_{i:03d}") for i in range(args.count)]
        out, seconds = args.out / slug(args.phrase), args.seconds or 2.0
    else:
        commands = load_commands(settings.resolve(settings.router.commands_file))
        prompts = [
            (c.phrases[0], f"{c.id}__{slug(c.phrases[0])}__{r}")
            for c in commands
            for r in range(args.repeat)
        ]
        out, seconds = args.out / "commands", args.seconds or 3.5
    out.mkdir(parents=True, exist_ok=True)

    print(f"Registro {len(prompts)} clip da {seconds:.1f} s in {out}")
    for i, (text, name) in enumerate(prompts, 1):
        input(f'[{i}/{len(prompts)}] Premi Invio e di\': "{text}" ')
        audio = record(seconds, device)
        peak = np.abs(audio).max() / 32768
        sf.write(out / f"{name}.wav", audio, SR)
        print(f"   salvato (picco {peak:.0%}){'  <- troppo basso, ripeti?' if peak < 0.05 else ''}")


if __name__ == "__main__":
    main()
