"""Which speech-to-text understands your own requests best, and how fast?

Runs every request saved in logs/audio (with its row in logs/metrics.jsonl) through several
engines and counts how many become a known local command:

    uv run python scripts/compare_stt.py
    uv run python scripts/compare_stt.py --engines turbo lista --details
    uv run python scripts/compare_stt.py --engines parakeet     # needs: uv pip install onnx-asr

Engines:
    turbo     Whisper large-v3-turbo, as CLAUDIO uses it today
    large-v3  Whisper large-v3: slower, sometimes more accurate on fast speech
    lista     the "closed list": turbo proposes 5 transcriptions, the first known command wins
    parakeet  NVIDIA Parakeet TDT 0.6B v3 (ONNX int8, via onnx-asr): fast, guesses the language

The audio is what Whisper got live: the request plus the 1.5 s before the wake word. There is
no "right answer" in the log, so the score is how many requests become a command; --details
shows every transcription side by side to judge the rest by ear. The last column is the wake
verifier: "scarta" marks a request it would treat as a false wake.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from claudio.config import Settings, load_settings
from claudio.router.matcher import CommandMatcher, load_commands
from claudio.wakeword.verifier import wake_confirmed

ENGINES = ("turbo", "large-v3", "lista", "parakeet")
LEAD_IN_S = 1.5  # app.LEAD_IN_S: the audio before the detection that reaches Whisper


@dataclass
class Request:
    name: str
    audio: np.ndarray  # float32, 16 kHz mono
    live_text: str


@dataclass
class Result:
    engine: str
    texts: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    ms: list[float] = field(default_factory=list)


def load_requests(audio_dir: Path, log_path: Path, sample_rate: int = 16000) -> list[Request]:
    """The saved requests that still have their metrics row, cut as they reached Whisper."""
    if not log_path.exists():
        raise SystemExit(f"Manca {log_path}: avvia CLAUDIO e fai qualche richiesta")
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    live = {r["audio_file"]: r for r in rows if r.get("audio_file") and "prewake_ms" in r}
    requests = []
    for wav in sorted(audio_dir.glob("*.wav")):
        row = live.get(wav.name)
        if row is None or row.get("outcome") == "near_miss":
            continue
        audio, sr = sf.read(wav, dtype="float32")
        if sr != sample_rate:
            continue  # every WAV CLAUDIO saves is 16 kHz: anything else is not one of them
        start = max(0, round((row["prewake_ms"] / 1000 - LEAD_IN_S) * sr))
        requests.append(Request(wav.stem, audio[start:], row.get("text", "")))
    if not requests:
        raise SystemExit(f"Nessuna richiesta salvata in {audio_dir} con le sue metriche")
    return requests


def whisper_engine(
    settings: Settings, model: str, closed_list: bool, matcher: CommandMatcher
) -> Callable[[np.ndarray], str]:
    from claudio.stt.transcriber import WhisperTranscriber, first_known

    stt_settings = settings.stt.model_copy(update={"model": model})
    stt = WhisperTranscriber(stt_settings, settings.resolve(settings.models_dir) / "whisper")
    print(f"  {model} su {stt.device}{' (lista chiusa, 5 proposte)' if closed_list else ''}")
    if not closed_list:
        return stt.transcribe
    return lambda audio: first_known(stt.candidates(audio), lambda t: matcher.match(t) is not None)


def parakeet_engine() -> Callable[[np.ndarray], str]:
    try:
        import onnx_asr
    except ImportError:
        raise SystemExit("Parakeet richiede onnx-asr: uv pip install onnx-asr") from None
    model = onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3",
        quantization="int8",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print("  Parakeet TDT 0.6B v3 int8")
    return lambda audio: str(model.recognize(audio, sample_rate=16000)).strip()


def build_engine(
    name: str, settings: Settings, matcher: CommandMatcher
) -> Callable[[np.ndarray], str]:
    if name == "parakeet":
        return parakeet_engine()
    model = "large-v3" if name == "large-v3" else "large-v3-turbo"
    return whisper_engine(settings, model, name == "lista", matcher)


def run(
    name: str,
    transcribe: Callable[[np.ndarray], str],
    requests: list[Request],
    matcher: CommandMatcher,
) -> Result:
    result = Result(name)
    transcribe(requests[0].audio)  # warm-up: the first pass on the GPU is slow
    for request in requests:
        start = time.perf_counter()
        text = transcribe(request.audio)
        result.ms.append((time.perf_counter() - start) * 1000)
        match = matcher.match(text) if text else None
        result.texts.append(text)
        result.commands.append(match.command.id if match else "-")
    return result


def report(requests: list[Request], results: list[Result], details: bool) -> None:
    n = len(requests)
    print(f"\n== {n} richieste")
    print(f"  {'motore':<10} {'comandi':>9} {'mediana':>9} {'p90':>8}")
    for r in results:
        known = sum(c != "-" for c in r.commands)
        print(
            f"  {r.engine:<10} {known:>4}/{n:<4} {np.median(r.ms):>6.0f} ms "
            f"{np.percentile(r.ms, 90):>5.0f} ms"
        )
    best = max(results, key=lambda r: (sum(c != "-" for c in r.commands), -np.median(r.ms)))
    print(f"Capisce di più: {best.engine}")
    first = results[0]
    rejected = [
        i
        for i, (text, cmd) in enumerate(zip(first.texts, first.commands, strict=True))
        if not wake_confirmed(text, cmd != "-")
    ]
    print(f"Verificatore wake word ({first.engine}): scarterebbe {len(rejected)}/{n} richieste")
    if not details:
        return
    print("\n== dettagli: dal vivo, poi ogni motore (-> comando)")
    for i, request in enumerate(requests):
        print(f"{request.name}  dal vivo {request.live_text!r}")
        for r in results:
            mark = "  scarta" if r is first and i in rejected else ""
            print(f"    {r.engine:<10} {r.texts[i]!r} -> {r.commands[i]}{mark}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", type=Path, default=Path("logs/audio"))
    parser.add_argument("--log", type=Path, default=Path("logs/metrics.jsonl"))
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=["turbo", "lista"])
    parser.add_argument("--details", action="store_true", help="ogni trascrizione")
    args = parser.parse_args(argv)

    settings = load_settings()
    matcher = CommandMatcher(
        load_commands(settings.resolve(settings.router.commands_file)),
        settings.router.fuzzy_threshold,
    )
    requests = load_requests(args.audio, args.log)
    print(f"{len(requests)} richieste da {args.audio}")
    results = []
    for name in args.engines:
        transcribe = build_engine(name, settings, matcher)
        results.append(run(name, transcribe, requests, matcher))
        del transcribe
        gc.collect()  # one big model at a time on the GPU
    report(requests, results, args.details)


if __name__ == "__main__":
    sys.exit(main())
