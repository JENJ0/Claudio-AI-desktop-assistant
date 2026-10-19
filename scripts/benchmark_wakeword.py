"""How well do the trained wake word models hear JenJo's own recordings, and how often do they
wake on things that are not the wake word? Runs the real streaming detector, as CLAUDIO does.

    uv run python scripts/benchmark_wakeword.py
    uv run python scripts/benchmark_wakeword.py --models hey_claudio --thresholds 0.3,0.4,0.5

Reads samples/<model>/ (made by record_samples.py wakeword), samples/negatives/ (ordinary
sentences) and samples/noise/ (the quiet room and the TV). Per threshold it prints the takes
heard (all of them, then by way of speaking), the sentences that wake the model and the false
wakes per hour in the room noise, then the threshold it would pick.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

REFRACTORY_S = 2.0  # the runtime ignores 2 s after a detection (detector.reset())
FRAME = 1280  # 80 ms at 16 kHz
DEFAULT_THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7)
Scorer = Callable[[np.ndarray], list[float]]  # int16 audio -> the model's score per 80 ms frame


def style_of(stem: str, model: str) -> str:
    """ "hey_claudio_012_veloce" -> "veloce"; takes that do not follow the pattern -> "altro"."""
    m = re.match(rf"^{re.escape(model)}_\d+_(.+)$", stem)
    return m.group(1) if m else "altro"


def wakes(scores: list[float], threshold: float) -> int:
    """Detections the runtime would make: after each one, 2 s are ignored."""
    skip, count, quiet = int(REFRACTORY_S * 16000 / FRAME), 0, 0
    for score in scores:
        if quiet > 0:
            quiet -= 1
        elif score >= threshold:
            count += 1
            quiet = skip
    return count


def heard_by_style(
    peaks: dict[str, tuple[str, float]], threshold: float
) -> dict[str, tuple[int, int]]:
    """{style: (takes heard, takes)} from {take: (style, peak score)}."""
    table: dict[str, list[int]] = {}
    for style, peak in peaks.values():
        row = table.setdefault(style, [0, 0])
        row[0] += peak >= threshold
        row[1] += 1
    return {style: (heard, total) for style, (heard, total) in sorted(table.items())}


def suggest(rows: list[dict[str, float]], max_fph: float) -> dict[str, float] | None:
    """The threshold hearing the most takes with no ordinary sentence waking it and at most
    `max_fph` false wakes per hour in the room noise (the highest of equals: fewer surprises)."""
    ok = [r for r in rows if r["sentences"] == 0 and r["noise_per_hour"] <= max_fph]
    return max(ok, key=lambda r: (r["heard"], r["threshold"])) if ok else None


def load_scorer(model_path: Path, features: Path) -> Scorer:
    from openwakeword.model import Model

    detector = Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        melspec_model_path=str(features / "melspectrogram.onnx"),
        embedding_model_path=str(features / "embedding_model.onnx"),
    )
    silence = np.zeros(FRAME, dtype=np.int16)

    def score(audio: np.ndarray) -> list[float]:
        detector.reset()
        for _ in range(25):  # forget the previous clip, as CLAUDIO does after a request
            detector.predict(silence)
        padded = np.concatenate((np.zeros(16000, dtype=np.int16), audio, np.zeros(16000, np.int16)))
        return [
            float(next(iter(detector.predict(padded[i : i + FRAME]).values())))
            for i in range(0, len(padded) - FRAME + 1, FRAME)
        ]

    return score


def read(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        raise SystemExit(f"{path.name}: {sr} Hz, servono 16000 (le registra record_samples.py)")
    return np.asarray(audio)


def benchmark(
    model: str,
    scorer: Scorer,
    samples: Path,
    thresholds: tuple[float, ...],
    max_fph: float,
) -> None:
    takes = sorted((samples / model).glob("*.wav"))
    sentences = sorted((samples / "negatives").glob("*.wav"))
    noise = sorted((samples / "noise").glob("*.wav"))
    if not takes:
        print(f"{model}: nessuna registrazione in {samples / model}")
        return
    audio = {p: read(p) for p in [*takes, *sentences, *noise]}
    series = {p: scorer(a) for p, a in audio.items()}
    peaks = {p.name: (style_of(p.stem, model), max(series[p])) for p in takes}
    hours = sum(len(audio[p]) for p in noise) / 16000 / 3600
    print(
        f"\n== {model}: {len(takes)} registrazioni, {len(sentences)} frasi, "
        f"{hours * 60:.1f} min di rumore"
    )
    print(" soglia | sentite | frasi che scattano | falsi/ora nel rumore")
    rows = []
    for t in thresholds:
        heard = sum(peak >= t for _, peak in peaks.values())
        woken = sum(wakes(series[p], t) > 0 for p in sentences)
        fph = sum(wakes(series[p], t) for p in noise) / hours if hours else 0.0
        rows.append({"threshold": t, "heard": heard, "sentences": woken, "noise_per_hour": fph})
        print(
            f"  {t:.2f}  | {heard:3d}/{len(takes):<3d} | "
            f"{woken:3d}/{len(sentences):<3d}            | {fph:6.1f}"
        )
    pick = suggest(rows, max_fph)
    if pick is None:
        print(
            f" Nessuna soglia sentita senza far scattare le frasi o il rumore (max {max_fph}/ora)."
        )
        pick = max(rows, key=lambda r: (-r["sentences"], r["heard"]))
        print(f" La meno peggio per le frasi: {pick['threshold']:.2f}")
    else:
        heard_of = f"{int(pick['heard'])}/{len(takes)}"
        print(f" Soglia consigliata: {pick['threshold']:.2f} ({heard_of} sentite)")
    print(f" Per stile alla soglia {pick['threshold']:.2f}:")
    for style, (heard, total) in heard_by_style(peaks, pick["threshold"]).items():
        print(f"   {style:12s} {heard}/{total}")
    loud = sorted(((max(series[p]), p.name) for p in sentences), reverse=True)[:3]
    print(" Frasi normali più vicine a scattare: " + ", ".join(f"{n} ({s:.2f})" for s, n in loud))


def main(argv: list[str] | None = None) -> None:
    from claudio.config import PROJECT_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=PROJECT_ROOT / "samples")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "wakeword")
    parser.add_argument("--models", default="hey_claudio,claudio", help="separati da virgola")
    parser.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    parser.add_argument("--max-fph", type=float, default=1.0, help="falsi allarmi/ora nel rumore")
    args = parser.parse_args(argv)
    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        path = args.models_dir / f"{model}.onnx"
        if not path.is_file():
            print(f"{model}: manca {path}", file=sys.stderr)
            continue
        benchmark(model, load_scorer(path, args.models_dir), args.samples, thresholds, args.max_fph)


if __name__ == "__main__":
    main()
