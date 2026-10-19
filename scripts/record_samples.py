"""Record WAV samples with the real microphone, for training and measuring CLAUDIO.

Wake words (50 takes each; the script asks for a different way of speaking every few takes):
    uv run python scripts/record_samples.py wakeword --phrase "Hey Claudio"
    uv run python scripts/record_samples.py wakeword --phrase "Claudio"
Ordinary sentences without the wake word (about 3 minutes), and the room's own noise:
    uv run python scripts/record_samples.py negatives
    uv run python scripts/record_samples.py noise
The zip to upload to Colab (recordings + training scripts):
    uv run python scripts/record_samples.py pack
Commands (the first phrase of every command, x2, for benchmark_latency.py), in Italian or English:
    uv run python scripts/record_samples.py commands --repeat 2
    uv run python scripts/record_samples.py commands --repeat 2 --language en

Running a mode again adds new takes after the ones already there: nothing is overwritten.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "wakeword"))

import phrases  # scripts/wakeword/phrases.py, shared with the trainer

SR = 16000
WAKEWORD_SECONDS = {"lento": 3.0, "lontano": 3.0, "distratto": 3.0, "comando_cmd": 4.0}
FAR_DELAY_S = 4  # time to walk 2 m away after pressing Enter
NOISE_TAKES = [
    ("stanza", "la stanza com'è di solito, senza parlare"),
    ("tv", "la TV o la musica accesa al volume di sempre, senza parlare tu"),
]
PACK_DIRS = ("hey_claudio", "claudio", "negatives", "noise")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def next_index(directory: Path, prefix: str) -> int:
    """First free take number for `prefix_NNN...wav` files: new takes never overwrite."""
    numbers = [
        int(m.group(1))
        for p in directory.glob(f"{prefix}_*.wav")
        if (m := re.match(rf"{re.escape(prefix)}_(\d+)", p.stem))
    ]
    return max(numbers, default=-1) + 1


def wakeword_prompts(phrase: str, count: int, start: int) -> list[tuple[str, str, str, float]]:
    """(header, text, file stem, seconds) per take. A known wake word gets the styles of
    phrases.RECORDING_STYLES; any other phrase is recorded plainly."""
    model = slug(phrase)
    if model in phrases.RECORDING_STYLES:
        plan = phrases.recording_plan(model, count)
    else:
        plan = [("normale", "normale", phrase)] * count
    return [
        (
            how,
            f'di\': "{say}"',
            f"{model}_{start + i:03d}_{style}",
            WAKEWORD_SECONDS.get(style, 2.5),
        )
        for i, (style, how, say) in enumerate(plan)
    ]


def negative_prompts(start: int) -> list[tuple[str, str, str, float]]:
    return [
        ("frasi normali, senza dire Claudio", f'di\': "{text}"', f"neg_{start + i:03d}", 4.0)
        for i, text in enumerate(phrases.NEGATIVE_PROMPTS)
    ]


def noise_prompts(start: int, seconds: float) -> list[tuple[str, str, str, float]]:
    return [
        (
            instruction,
            f"resta in silenzio per {seconds:.0f} s",
            f"noise_{start + i:03d}_{name}",
            seconds,
        )
        for i, (name, instruction) in enumerate(NOISE_TAKES)
    ]


def pack(samples: Path, scripts: Path, out: Path) -> dict[str, int]:
    """Zip the recordings and the training scripts, laid out the way the notebook expects:
    samples/<kind>/*.wav and wakeword/*.py."""
    counts = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for kind in PACK_DIRS:
            files = sorted((samples / kind).glob("*.wav"))
            counts[kind] = len(files)
            for f in files:
                zf.write(f, f"samples/{kind}/{f.name}")
        for f in sorted(scripts.glob("*.py")):
            zf.write(f, f"wakeword/{f.name}")
        notebook = scripts / "train_claudio.ipynb"
        if notebook.is_file():
            zf.write(notebook, f"wakeword/{notebook.name}")
    return counts


def record(seconds: float, device: int | None) -> np.ndarray:
    """The "speak" cue comes only once the microphone is running, so no word is lost."""
    import sounddevice as sd

    from claudio.audio.devices import stream_settings

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="int16",
        device=device,
        extra_settings=stream_settings(device),
    ) as stream:
        print("   ● parla", flush=True)
        audio, _ = stream.read(int(seconds * SR))
    return np.asarray(audio[:, 0])


def check(audio: np.ndarray, min_peak: float) -> str:
    peak = float(np.abs(audio).max()) / 32768
    if peak >= 0.99:
        return f"picco {peak:.0%}  <- distorto, allontanati un po' e rifai (r)"
    if peak < min_peak:
        return f"picco {peak:.0%}  <- troppo basso, rifai (r) più vicino o più forte"
    return f"picco {peak:.0%}"


def min_peak_for(stem: str) -> float:
    """Below this peak a take is probably too quiet (room noise has no minimum)."""
    if stem.startswith("noise_"):
        return 0.0
    return 0.015 if stem.endswith(("_piano", "_lontano")) else 0.05


def session(prompts: list[tuple[str, str, str, float]], out: Path, device: int | None) -> int:
    """Records the prompts in order; returns how many different clips were saved."""
    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    print(f"Registro {len(prompts)} clip in {out}")
    print("Invio = registra   r + Invio = rifai la precedente   q + Invio = basta così\n")
    i, header, saved = 0, "", set()
    while True:
        if i == len(prompts):  # after the last take, one more chance to redo it
            again = input("r + Invio = rifai l'ultima   Invio = fine ") if prompts else ""
            if again.strip().lower() != "r":
                break
            i -= 1
            print(f"   rifai: {prompts[i][1]}")
        else:
            title, text, _, _ = prompts[i]
            if title != header:
                header = title
                print(f"--- Ora: {title}")
            answer = input(f"[{i + 1}/{len(prompts)}] Invio, poi {text} ").strip().lower()
            if answer == "q":
                break
            if answer == "r":
                if i == 0:
                    print("   niente da rifare")
                    continue
                i -= 1
                print(f"   rifai: {prompts[i][1]}")
        _, _, stem, seconds = prompts[i]
        if stem.endswith("_lontano"):
            for s in range(FAR_DELAY_S, 0, -1):
                print(f"   {s}...", end=" ", flush=True)
                time.sleep(1)
            print()
        audio = record(seconds, device)
        sf.write(out / f"{stem}.wav", audio, SR, subtype="PCM_16")
        saved.add(stem)
        print(f"   salvato, {check(audio, min_peak_for(stem))}")
        i += 1
    return len(saved)


def command_prompts(
    commands: list[Any], language: str, repeat: int, seconds: float, out: Path
) -> list[tuple[str, str, str, float]]:
    """The first phrase of every command, `repeat` times, numbered after the takes already
    in `out` (benchmark_latency.py reads the command id before the first "__")."""
    prompts = []
    for c in commands:
        if not c.phrases_in(language):
            continue
        phrase = c.phrases_in(language)[0]
        prefix = f"{c.id}__{slug(phrase)}"
        start = next_index(out, f"{prefix}_")
        for r in range(repeat):
            prompts.append(("comandi", f'di\': "{phrase}"', f"{prefix}__{start + r}", seconds))
    return prompts


def main() -> None:
    from claudio.audio.devices import resolve_device
    from claudio.config import PROJECT_ROOT, load_settings
    from claudio.router.matcher import load_commands

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["wakeword", "negatives", "noise", "commands", "pack"])
    parser.add_argument("--phrase", default="Hey Claudio")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "samples")
    parser.add_argument("--language", default="it", help="commands: lingua delle frasi (it, en)")
    args = parser.parse_args()

    if args.mode == "pack":
        zip_path = args.out / "claudio_training.zip"
        counts = pack(args.out, PROJECT_ROOT / "scripts" / "wakeword", zip_path)
        print(", ".join(f"{k}: {v}" for k, v in counts.items()))
        print(f"Creato {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB): caricalo su Colab")
        if not counts["hey_claudio"] or not counts["claudio"]:
            print("Attenzione: mancano registrazioni della wake word, il modello sarà meno preciso")
        return

    settings = load_settings()
    device = resolve_device(settings.audio.input_device, "input")  # as CLAUDIO opens it
    if args.mode == "wakeword":
        out = args.out / slug(args.phrase)
        prompts = wakeword_prompts(args.phrase, args.count, next_index(out, slug(args.phrase)))
    elif args.mode == "negatives":
        out = args.out / "negatives"
        prompts = negative_prompts(next_index(out, "neg"))
    elif args.mode == "noise":
        out = args.out / "noise"
        prompts = noise_prompts(next_index(out, "noise"), args.seconds or 60.0)
    else:
        out = args.out / "commands"
        commands = load_commands(settings.resolve(settings.router.commands_file))
        prompts = command_prompts(commands, args.language, args.repeat, args.seconds or 3.5, out)
    if args.seconds and args.mode in ("wakeword", "negatives"):
        prompts = [(h, t, s, args.seconds) for h, t, s, _ in prompts]
    saved = session(prompts, out, device)
    print(f"\nFatto: {saved} clip salvate in {out}")


if __name__ == "__main__":
    main()
