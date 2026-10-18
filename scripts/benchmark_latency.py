"""Latency benchmark.

From recorded commands (see record_samples.py), offline, per STT model:
    uv run python scripts/benchmark_latency.py --wav-dir samples/commands \
        --models large-v3-turbo small base
From the live log written by `uv run claudio`:
    uv run python scripts/benchmark_latency.py --log logs/metrics.jsonl

Offline M1 = VAD silence wait + STT + router + Piper first sentence (playback start ≈ +30-60 ms).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from claudio.config import load_settings
from claudio.metrics import summarize


def print_summary(title: str, rows: list[dict], keys: tuple[str, ...]) -> None:
    print(f"\n== {title}")
    for key, stats in summarize(rows, keys).items():
        print(
            f"  {key:<14} n={stats['n']:<3} mediana={stats['median']:>7.1f} ms   "
            f"p90={stats['p90']:>7.1f} ms"
        )


def from_log(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    misses = [r for r in rows if r.get("outcome") == "near_miss"]
    rows = [r for r in rows if r.get("outcome") != "near_miss"]
    ok = [r for r in rows if r.get("outcome") == "ok"]
    print(f"{len(rows)} interazioni, {len(ok)} completate")
    if misses:
        peaks = ", ".join(f"{r['wake_score']:.2f}" for r in misses[-20:])
        print(f"{len(misses)} wake word sotto soglia (ultimi picchi: {peaks})")
    print_summary("dal vivo", ok, ("m1_ms", "m2_ms", "vad_wait_ms", "stt_ms", "tts_first_ms"))
    print("\n== ultime richieste: esito, wake word, volume microfono, VAD, testo, audio")
    for r in rows[-20:]:
        level = r.get("input_dbfs")
        print(
            f"  {r.get('outcome', '?'):<8} wake={r.get('wake_score', 0):.2f} "
            f"vol={'-' if level is None else f'{level:.0f}':>4} dBFS "
            f"vad={r.get('vad_max', '-')!s:<5} {r.get('text', '')!r} {r.get('audio_file', '')}"
        )


def offline(wav_dir: Path, model_names: list[str], device: str) -> None:
    from claudio.__main__ import build_responder
    from claudio.audio.vad import Endpointer, EndpointEvent, SileroVad
    from claudio.stt.transcriber import WhisperTranscriber
    from claudio.tts.speaker import PiperSynth, voice_path

    settings = load_settings()
    models = settings.resolve(settings.models_dir)
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"Nessun WAV in {wav_dir}: registra con scripts/record_samples.py")
    responder = build_responder(settings)
    vad = SileroVad(models / "vad" / "silero_vad.onnx")
    endpointer = Endpointer(vad, settings.vad)
    try:
        synth = PiperSynth(voice_path(models / "piper", settings.tts.voice), settings.tts)
        synth.synthesize("Ciao.")
    except FileNotFoundError as exc:
        print(f"TTS escluso: {exc}")
        synth = None
    block = settings.audio.block_size

    clips = []
    for wav in wavs:
        audio, sr = sf.read(wav, dtype="int16")
        assert sr == 16000, f"{wav.name}: serve 16 kHz"
        padded = np.concatenate((audio, np.zeros(16000 * 2, dtype=np.int16)))
        endpointer.reset()
        event = EndpointEvent.NONE
        for i in range(0, len(padded) - block + 1, block):
            event = endpointer.process(padded[i : i + block])
            if event in (EndpointEvent.END, EndpointEvent.TIMEOUT):
                break
        utterance = endpointer.audio() if event is EndpointEvent.END else audio / 32768
        clips.append((wav, utterance.astype(np.float32), settings.vad.end_silence_ms))

    for name in model_names:
        stt_settings = settings.stt.model_copy(
            update={"model": name, "fallback_model": name, "device": device}
        )
        t0 = time.perf_counter()
        stt = WhisperTranscriber(stt_settings, models / "whisper")
        print(
            f"\n### {name} su {stt.device.upper()} (caricato in {time.perf_counter() - t0:.1f} s)"
        )
        rows, correct = [], 0
        for wav, utterance, vad_wait in clips:
            t1 = time.perf_counter()
            text = stt.transcribe(utterance)
            t2 = time.perf_counter()

            # Route only (actions are not executed in the benchmark).
            match = responder.matcher.match(text)
            t3 = time.perf_counter()
            reply = match.command.reply if match else settings.router.unknown_reply
            tts_ms = 0.0
            if synth and reply:
                t4 = time.perf_counter()
                synth.synthesize(reply.split(".")[0] + ".")
                tts_ms = (time.perf_counter() - t4) * 1000
            expected = wav.stem.split("__")[0]
            got = match.command.id if match else "-"
            correct += got == expected
            row = {"stt_ms": (t2 - t1) * 1000, "route_ms": (t3 - t2) * 1000, "tts_first_ms": tts_ms}
            row["m1_ms"] = vad_wait + row["stt_ms"] + row["route_ms"] + tts_ms
            rows.append(row)
            flag = "ok " if got == expected else "ERR"
            print(f"  {flag} {wav.stem[:40]:<40} {row['stt_ms']:6.0f} ms  {text!r} -> {got}")
        print(f"  riconosciuti {correct}/{len(clips)}")
        print_summary(name, rows, ("m1_ms", "stt_ms", "route_ms", "tts_first_ms"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", type=Path, nargs="?", const=Path("logs/metrics.jsonl"))
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--models", nargs="+", default=["large-v3-turbo", "small", "base"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    if args.log:
        from_log(args.log)
    elif args.wav_dir:
        offline(args.wav_dir, args.models, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
