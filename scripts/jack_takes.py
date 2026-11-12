"""Jack's fixed replies chosen in advance: each one is generated with several seeds, every
take is judged (register, level, pace and what Whisper hears) and the best one is kept as a
WAV in data/jack_cache/, its seed in data/jack_takes.json. Claudio then plays it with no TTS
at all (0 ms) and always the same good take (30/10: "Mi spengo... accenditi" came out as
"Accendete" every time, "Chiudo la chat." always high).

  uv run python scripts/jack_takes.py                 # commands.toml + the usual fixed lines
  uv run python scripts/jack_takes.py --seeds 12      # more takes per reply (default 8)
  uv run python scripts/jack_takes.py --from-logs 2   # + replies said at least twice in logs
  uv run python scripts/jack_takes.py --only "Apro la chat."

Every take is also saved in logs/jack-takes/ (with report.md) to listen to. A new
checkpoint or new [tts] settings need a new run: the old takes no longer match them. The
worker needs the GPU alone: close Claudio first (~2 s of GPU per take)."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claudio.config import load_settings  # noqa: E402
from claudio.router.matcher import load_commands  # noqa: E402
from claudio.safety.confirmation import (  # noqa: E402
    ASK_LEVEL_2,
    ASK_LEVEL_3,
    CANCELLED,
    NOT_UNDERSTOOD,
)
from claudio.tts.jack import JackSynth, worker_command  # noqa: E402
from claudio.tts.takes import TakeStore, voice_profile  # noqa: E402
from claudio.tts.voice_check import TakeJudge, semitones  # noqa: E402
from jack_variants import Silent, wer  # noqa: E402

STARTUP = "Sono pronto."  # said once everything has loaded (__main__.announce_ready)


@dataclass
class Candidate:
    seed: int
    audio: np.ndarray
    raw: dict[str, float]
    reasons: list[str]
    median_hz: float
    heard: str = ""
    wer: float = 1.0


def fixed_replies(settings: Any, from_logs: int) -> list[str]:
    """What Jack says word for word again and again: the Italian replies and questions of
    commands.toml, the confirmation lines, "Sono pronto." and, with `from_logs`, the replies
    said at least that many times in logs/sessions/ (one piece each: short enough)."""
    commands = load_commands(settings.resolve(settings.router.commands_file))
    lines = [STARTUP, ASK_LEVEL_2, ASK_LEVEL_3, CANCELLED, NOT_UNDERSTOOD]
    for command in commands:
        lines += [command.reply, command.confirm]
    if from_logs:
        said: Counter[str] = Counter()
        for log in (ROOT / "logs" / "sessions").glob("*/conversazione.md"):
            for line in log.read_text(encoding="utf-8").splitlines():
                if line.startswith("**Claudio:**"):
                    said[line.split("**Claudio:**", 1)[1].strip()] += 1
        lines += [text for text, n in said.most_common() if n >= from_logs]
    limit = settings.tts.jack_group_chars or 220
    return [t for t in dict.fromkeys(lines) if t and "{" not in t and len(t) <= limit]


def transcribe(settings: Any, candidates: dict[str, list[Candidate]]) -> None:
    from scipy.signal import resample_poly

    from claudio.stt.transcriber import WhisperTranscriber

    stt = WhisperTranscriber(settings.stt, settings.resolve(settings.models_dir) / "whisper")
    for text, takes in candidates.items():
        for take in takes:
            audio16 = resample_poly(take.audio.astype(np.float32) / 32768, 2, 3)
            segments, _ = stt._model.transcribe(audio16, language="it", beam_size=5)
            take.heard = " ".join(s.text.strip() for s in segments)
            take.wer = wer(text, take.heard)


def best(takes: list[Candidate], usual_hz: float, target: float) -> Candidate:
    """Heard right first, then no reason to throw it away, then closest to Jack's usual
    register, then to the level target."""
    return min(
        takes,
        key=lambda t: (
            round(t.wer, 3),
            len(t.reasons),
            abs(semitones(t.median_hz, usual_hz)) if t.median_hz else 6.0,
            abs(t.raw.get("raw_lufs", target) - target),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=8, help="takes per reply")
    parser.add_argument("--from-logs", type=int, default=0, help="also replies said N+ times")
    parser.add_argument("--only", nargs="+", help="just these replies")
    args = parser.parse_args()
    settings = load_settings()
    tts = settings.tts
    texts = args.only or fixed_replies(settings, args.from_logs)
    target = tts.jack_lufs or -24.0
    judge = TakeJudge(target_lufs=target, usual_hz=tts.jack_usual_hz)
    takes = TakeStore(settings.resolve(settings.data_dir) / "jack_takes.json", voice_profile(tts))
    out = ROOT / "logs" / "jack-takes"
    out.mkdir(parents=True, exist_ok=True)
    models = settings.resolve(settings.models_dir)
    synth = JackSynth(
        worker_command(models, tts),
        Silent(),
        tts.jack_startup_timeout_s,
        tts.jack_sentence_timeout_s,
        rvc=tts.jack_rvc,
        words=tts.jack_words,
    )
    print(f"{len(texts)} risposte fisse x {args.seeds} take - avvio del worker...")
    synth.start()
    candidates: dict[str, list[Candidate]] = {}
    try:
        for n, text in enumerate(texts, 1):
            spoken = synth._spoken(text)
            for seed in range(max(tts.jack_seed, 0), max(tts.jack_seed, 0) + args.seeds):
                audio, raw = synth.generate_take(text, seed)
                verdict = judge.judge(audio, synth.sample_rate, spoken, raw)
                candidates.setdefault(text, []).append(
                    Candidate(seed, audio, raw, verdict.reasons, verdict.stats.median_hz)
                )
                sf.write(out / f"{n:02d}_seme{seed}.wav", audio, synth.sample_rate)
            print(f"  {n:02d}: {text}")
    finally:
        synth.close()
    print("Whisper ascolta le take...")
    transcribe(settings, candidates)
    rvc = "blocks" if tts.jack_rvc == "auto" else tts.jack_rvc
    lines = ["# Risposte fisse di Jack", "", "| # | Risposta | Seme | WER | f0 | Scartate |"]
    lines.append("|---|---|---|---|---|---|")
    registers = []
    for n, (text, found) in enumerate(candidates.items(), 1):
        pick = best(found, judge.usual_hz, target)
        key = takes.key(synth._spoken(text), rvc)
        takes.save_audio(
            key,
            pick.seed,
            text,
            pick.audio,
            synth.sample_rate,
            wer=round(pick.wer, 3),
            median_hz=round(pick.median_hz, 1),
        )
        rejected = sum(bool(t.reasons) for t in found)
        registers += [t.median_hz for t in found if t.median_hz and not t.reasons]
        lines.append(
            f"| {n:02d} | {text} | {pick.seed} | {pick.wer:.2f} | {pick.median_hz:.0f} Hz | "
            f"{rejected}/{len(found)} |"
        )
        if pick.wer > 0:
            lines.append(f"|  | Whisper sente: {pick.heard} |  |  |  |  |")
    if registers:
        lines += [
            "",
            f"Registro mediano delle take buone: {statistics.median(registers):.0f} Hz "
            f"(jack_usual_hz = {judge.usual_hz:.0f}).",
        ]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Take scelte in {takes.cache}, tutte le take e il report in {out}")


if __name__ == "__main__":
    main()
