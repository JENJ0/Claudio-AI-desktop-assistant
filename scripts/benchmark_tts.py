"""Time to the first sentence for Kokoro and every Piper voice in models/.

uv run python scripts/benchmark_tts.py
uv run python scripts/benchmark_tts.py --play   # also listen to each voice
uv run python scripts/benchmark_tts.py --elevenlabs --reps 12
    # online voice vs Jack and Kokoro: first chunk, total time, credits, WER.
    # It SPENDS ElevenLabs credits. Close Claudio first (Jack and Whisper need the GPU).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from claudio.config import PROJECT_ROOT, load_settings
from claudio.tts.speaker import PiperSynth

SENTENCES = [
    "Apro la cartella Download.",
    "Sono le 14 e 35.",
    "Oggi è sabato 18 ottobre 2026.",
    "Questo non lo so ancora fare, ma presto imparerò.",
]

IDLE_S = 6.0  # httpx2 closes idle connections after 5 s: the first sentence starts cold
RATE = 24_000
OUT = PROJECT_ROOT / "logs" / "tts-bench"
CREDITS_PER_CHARS = (16, 78)  # v4 Turbo: 16 credits every 78 characters
ONLINE_ENDPOINTS = 2  # tts and dialogue


@dataclass
class Row:
    voice: str
    index: int  # 0 is the first sentence of a reply (cold), the others are warm
    rep: int
    first_ms: float | None = None
    total_ms: float | None = None
    cost: int | None = None
    audio: np.ndarray | None = None
    rate: int = RATE
    error: str = ""
    wer: float | None = None


def credit_estimate(characters: int) -> int:
    credits, chars = CREDITS_PER_CHARS
    return math.ceil(characters * credits / chars)


def summarize(values: list[float]) -> tuple[float, float, float]:
    """Median, p95 and max; with a single value all three are that value."""
    if len(values) < 2:
        return values[0], values[0], values[0]
    return (
        statistics.median(values),
        statistics.quantiles(values, n=20, method="inclusive")[-1],
        max(values),
    )


def split_cold_warm(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    return [r for r in rows if r.index == 0], [r for r in rows if r.index > 0]


def timed_stream(
    stream: Callable[[str], Iterator[np.ndarray]], text: str
) -> tuple[float, float, np.ndarray]:
    """Milliseconds to the first chunk and to the end, and the whole audio."""
    start = time.perf_counter()
    first_ms: float | None = None
    chunks: list[np.ndarray] = []
    for chunk in stream(text):
        if first_ms is None:
            first_ms = (time.perf_counter() - start) * 1000
        chunks.append(chunk)
    total_ms = (time.perf_counter() - start) * 1000
    if first_ms is None:
        raise RuntimeError("nessun audio")
    return first_ms, total_ms, np.concatenate(chunks)


def run_voice(
    name: str, once: Callable[[str, int], Row], reps: int, folder: Path | None = None
) -> list[Row]:
    """`once(text, index)` speaks one sentence. Before the first sentence of each repetition the
    connection rests IDLE_S, as it does between two replies."""
    rows: list[Row] = []
    for rep in range(1, reps + 1):
        for index, text in enumerate(SENTENCES):
            if index == 0:
                time.sleep(IDLE_S)
            row = once(text, index)
            row.voice, row.index, row.rep = name, index, rep
            if folder is not None and row.audio is not None:
                import soundfile as sf

                sf.write(folder / f"{name}_{index + 1}_{rep}.wav", row.audio, row.rate, "PCM_16")
            rows.append(row)
            if folder is not None:
                _append_csv(folder / "rows.csv", row)
            print(f"  {name} giro {rep} frase {index + 1}: {_brief(row)}", flush=True)
    return rows


def _append_csv(path: Path, row: Row) -> None:
    """One line per row, written at once: a crash later in the run keeps the paid numbers."""
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(["voice", "sentence", "rep", "first_ms", "total_ms", "cost", "error"])
        writer.writerow(
            [row.voice, row.index + 1, row.rep, row.first_ms, row.total_ms, row.cost, row.error]
        )


def _brief(row: Row) -> str:
    if row.error:
        return f"ERRORE {row.error[:120]}"
    return f"primo chunk {row.first_ms:.0f} ms, totale {row.total_ms:.0f} ms, costo {row.cost}"


def stream_once(
    stream: Callable[[str], Iterator[np.ndarray]],
    rate_of: Callable[[], int],
    cost_of: Callable[[], int | None] = lambda: None,
) -> Callable[[str, int], Row]:
    def once(text: str, index: int) -> Row:
        row = Row("", index, 0)
        try:
            row.first_ms, row.total_ms, row.audio = timed_stream(stream, text)
        except Exception as exc:  # a failing voice must show in the table, not stop the run
            row.error = f"{type(exc).__name__}: {exc}"
            return row
        row.rate, row.cost = rate_of(), cost_of()
        return row

    return once


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def catch_warnings(
    once: Callable[[str, int], Row], logger_name: str = "claudio.tts.elevenlabs"
) -> Callable[[str, int], Row]:
    """The online voice ends a stream cut after audio with a WARNING, not an exception: make
    that, and the real cause of a failure before audio, an error of the row."""

    def wrapped(text: str, index: int) -> Row:
        logger = logging.getLogger(logger_name)
        handler = _Collect()
        logger.addHandler(handler)
        try:
            row = once(text, index)
        finally:
            logger.removeHandler(handler)
        if handler.messages:
            row.error = " | ".join([row.error, *handler.messages]).strip(" |")
        return row

    return wrapped


def make_bench_jack(settings: Any) -> Any:
    """Jack as production builds it, without the saved takes: they would be read from disk
    instead of spoken, and the run would write seeds into data/jack_takes.json."""
    from claudio.__main__ import make_jack

    jack = make_jack(settings)
    jack.takes = None
    return jack


class NoFallback:
    """The online voice's fallback: it raises, so a failure shows instead of hiding behind Jack."""

    sample_rate = RATE

    def synthesize(self, text: str) -> np.ndarray:
        raise RuntimeError("ElevenLabs non ha risposto")


def dialogue_once(client: Any, key: str, voice_id: str, model: str) -> Callable[[str, int], Row]:
    import try_elevenlabs as smoke

    def once(text: str, index: int) -> Row:
        attempt = smoke.try_once(client, key, "dialogue", voice_id, model, text)
        row = Row("", index, 0, attempt.first_byte_ms, attempt.total_ms)
        if attempt.status != 200 or not attempt.pcm:
            row.error = f"HTTP {attempt.status} {attempt.error or 'nessun audio'}"
            return row
        row.audio = np.frombuffer(attempt.pcm[: len(attempt.pcm) // 2 * 2], dtype="<i2")
        cost = next((v for k, v in attempt.headers.items() if k.lower() == "character-cost"), "")
        row.cost = int(cost) if cost.isdigit() else None
        return row

    return once


def zero_retention_probe(client: Any, key: str, voice_id: str, model: str) -> None:
    """One request with enable_logging=false (enterprise only): status, masked error, and
    whether a history item was made (no history-item-id header = zero retention applied)."""
    import try_elevenlabs as smoke

    attempt = smoke.try_once(
        client, key, "tts", voice_id, model, SENTENCES[0], params={"enable_logging": "false"}
    )
    has_history = any(k.lower() == "history-item-id" for k in attempt.headers)
    print(
        f"Zero retention (enable_logging=false): HTTP {attempt.status} {attempt.error} "
        f"cronologia: {'sì' if has_history else 'no'} (no = zero retention applicata)"
    )


def needs_wer(row: Row) -> bool:
    """A row with an error (e.g. a stream cut short) keeps its WAV but stays out of the WER."""
    return row.audio is not None and not row.error


def transcribe_rows(settings: Any, rows: list[Row]) -> None:
    """Whisper writes down every clip; Jack must be closed by now (both need the GPU)."""
    from scipy.signal import resample_poly

    from claudio.stt.transcriber import WhisperTranscriber
    from jack_variants import wer

    stt = WhisperTranscriber(settings.stt, settings.resolve(settings.models_dir) / "whisper")
    for row in rows:
        if not needs_wer(row):
            continue
        assert row.audio is not None
        audio = row.audio.astype(np.float32) / 32768
        audio16 = resample_poly(audio, 2, 3) if row.rate == RATE else audio
        segments, _ = stt._model.transcribe(audio16, language="it", beam_size=5)
        heard = " ".join(s.text.strip() for s in segments)
        row.wer = wer(SENTENCES[row.index], heard)


def _stats_cells(group: list[Row]) -> list[str]:
    ok = [r for r in group if not r.error]
    if not ok:
        return ["-"] * 4
    first = summarize([r.first_ms for r in ok if r.first_ms is not None])
    total = statistics.median(r.total_ms for r in ok if r.total_ms is not None)
    return [f"{first[0]:.0f}", f"{first[1]:.0f}", f"{first[2]:.0f}", f"{total:.0f}"]


def table_lines(rows: list[Row]) -> list[str]:
    head = ["voce", "quando", "n", "mediana", "p95", "max", "totale", "crediti", "WER", "errori"]
    lines = [head]
    for voice in dict.fromkeys(r.voice for r in rows):
        cold, warm = split_cold_warm([r for r in rows if r.voice == voice])
        for label, group in (("a freddo", cold), ("a caldo", warm)):
            costs = [r.cost for r in group if r.cost is not None]
            wers = [r.wer for r in group if r.wer is not None and not r.error]
            lines.append(
                [
                    voice,
                    label,
                    str(len(group)),
                    *_stats_cells(group),
                    str(sum(costs)) if costs else "-",
                    f"{statistics.mean(wers):.3f}" if wers else "-",
                    str(sum(1 for r in group if r.error)),
                ]
            )
    widths = [max(len(line[i]) for line in lines) for i in range(len(head))]
    body = [
        "  ".join(
            c.ljust(w) if i < 2 else c.rjust(w)
            for i, (c, w) in enumerate(zip(line, widths, strict=True))
        )
        for line in lines
    ]
    return ["Primo chunk in ms (mediana, p95, max); totale = mediana in ms.", *body]


def run_elevenlabs(reps: int) -> None:
    import httpx2

    import try_elevenlabs as smoke
    from claudio.__main__ import make_elevenlabs
    from claudio.tts.jack import WorkerError
    from claudio.tts.kokoro import KokoroSynth

    settings = load_settings()
    if settings.elevenlabs_api_key is None:
        raise SystemExit(
            "ELEVENLABS_API_KEY manca in .env: segui la guida V0 in piani/IT-2-passo-0b-prove.md"
        )
    key = settings.elevenlabs_api_key.get_secret_value()
    tts = settings.tts
    chars = sum(len(s) for s in SENTENCES) * reps * ONLINE_ENDPOINTS
    print(
        f"{chars} caratteri online: stima {credit_estimate(chars)} crediti, "
        "più la prova a zero retention"
    )
    eleven = make_elevenlabs(settings, NoFallback())
    if eleven is None:
        raise SystemExit("ELEVENLABS_API_KEY non valida in .env")
    eleven.cache = None  # the benchmark measures the network and the credits, never the disk
    if problem := eleven.probe():  # free: stops a doomed run before the first paid request
        raise SystemExit(
            f"ElevenLabs non risponde ({problem}): benchmark fermato, nessun credito speso"
        )
    folder = OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []

    speak = stream_once(eleven.stream, lambda: eleven.sample_rate, lambda: eleven.last_cost)

    def eleven_once(text: str, index: int) -> Row:
        if index == 0:
            eleven.begin_reply()  # no previous_text from the last repetition
        return speak(text, index)

    eleven_once = catch_warnings(eleven_once)
    print("ElevenLabs v4 Turbo, text-to-speech")
    rows += run_voice("eleven", eleven_once, reps, folder)

    timeout = httpx2.Timeout(20.0, connect=5.0)
    with httpx2.Client(base_url=smoke.API, headers={"xi-api-key": key}, timeout=timeout) as client:
        print("ElevenLabs v4 Turbo, dialogue")
        once = dialogue_once(client, key, tts.eleven_voice_id, tts.eleven_model)
        rows += run_voice("dialogue", once, reps, folder)
        zero_retention_probe(client, key, tts.eleven_voice_id, tts.eleven_model)

    kokoro = KokoroSynth(settings.resolve(settings.models_dir) / "kokoro", tts)
    kokoro.synthesize("Ciao.")
    print("Kokoro")
    kokoro_once = stream_once(lambda t: iter([kokoro.synthesize(t)]), lambda: kokoro.sample_rate)
    rows += run_voice("kokoro", kokoro_once, reps, folder)

    jack = make_bench_jack(settings)
    try:
        jack.start()
    except WorkerError as exc:
        print(f"Jack non parte: {exc}")
    else:
        print("Jack")
        try:
            rows += run_voice(
                "jack", stream_once(jack.stream, lambda: jack.sample_rate), reps, folder
            )
        finally:
            jack.close()
    try:
        transcribe_rows(settings, rows)
    except Exception as exc:  # the paid rows are in rows.csv: the table still prints
        print(f"WER non calcolato ({type(exc).__name__}: {exc})")
    print()
    print("\n".join(table_lines(rows)))
    print(f"WAV in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--elevenlabs", action="store_true", help="online voice vs Jack and Kokoro")
    parser.add_argument("--reps", type=int, default=12, help="repetitions with --elevenlabs")
    args = parser.parse_args()
    if args.elevenlabs:
        if args.reps < 1:
            parser.error("--reps deve essere almeno 1")
        run_elevenlabs(args.reps)
        return
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
