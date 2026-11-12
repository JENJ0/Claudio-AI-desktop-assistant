"""How well does CLAUDIO understand the test set of scripts/stt_testset.py? For every microphone
and every variant of the speech-to-text settings: the word error rate (WER, lower is better),
how many commands come out right and the time per sentence.

    uv run python scripts/stt_bench.py
    uv run python scripts/stt_bench.py --details
    uv run python scripts/stt_bench.py --variants live live+beam1 parakeet

A variant is a "+"-joined list of changes from how CLAUDIO transcribes today ("live"):
    live      about the gain CLAUDIO learns on that microphone (its sentences' typical loudest
              32 ms at -20 dBFS, soft limiter), the names on this PC and the sentences of
              [stt] initial_prompt as Whisper's prompt, Silero's cut of the sentence ([vad]
              settings).
              What comes after the transcription live (the closed list, waiting for the rest
              of a phrase left halfway) is not in it: compare_stt.py measures the closed list
    recupero  then, as live, a phrase no command understands goes through the closed list and
              Whisper's second opinion ([recovery]): the command proposed counts, asked or done
    grezzo    the recording as the microphone gives it, no gain
    frasi     prompt: only the sentences of [stt] initial_prompt, no names
    vuoto     no prompt at all
    beam1     beam size 1 instead of [stt] beam_size: faster
    large     Whisper large-v3 instead of [stt] model
    turbo     Whisper large-v3-turbo instead of [stt] model
    novad     the whole recording instead of Silero's cut
    parakeet  NVIDIA Parakeet TDT 0.6B v3 instead of Whisper (uv pip install onnx-asr)

"falsi": sentences that are no command, heard as one (the closed list and the second opinion
must keep them at 0). "tagli": sentences Silero ended before the voice did (a pause
mid-sentence). With --details an empty transcription says why (Silero heard no voice, Whisper
took it for silence or for a known hallucination). Every transcription goes to
data/stt_testset/risultati.json.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from claudio import intake
from claudio.audio.capture import apply_gain
from claudio.audio.level import TARGET_DBFS
from claudio.audio.vad import Endpointer, EndpointEvent
from claudio.config import Settings, load_settings
from claudio.router.matcher import CommandMatcher, load_commands

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stt_testset import BLOCK, MUTE_DBFS, OUT, SAMPLE_RATE, peak_dbfs, read_sentences

PARTS = (
    "live",
    "recupero",
    "grezzo",
    "frasi",
    "vuoto",
    "beam1",
    "large",
    "turbo",
    "novad",
    "parakeet",
)
DEFAULT_VARIANTS = ("live", "live+grezzo", "live+frasi", "live+vuoto", "live+novad", "live+beam1")
MAX_GAIN_DB = 30.0  # [audio] auto_gain's ceiling
SAME_WORD = {"hey": "ehi", "ei": "ehi", "eh": "ehi"}  # ways of writing the same word


@dataclass(frozen=True)
class Variant:
    name: str
    gain: bool = True
    prompt: str = "nomi"  # nomi | frasi | vuoto
    beam: int | None = None  # None: [stt] beam_size
    model: str = ""  # "": [stt] model
    vad: bool = True
    engine: str = "whisper"  # whisper | parakeet
    recover: bool = False  # the closed list and Whisper's second opinion, as live


def parse_variant(text: str) -> Variant:
    parts = text.split("+")
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise SystemExit(f"Variante {text!r}: non conosco {', '.join(unknown)} (vedi --help)")
    prompt = "vuoto" if "vuoto" in parts else "frasi" if "frasi" in parts else "nomi"
    return Variant(
        name=text,
        gain="grezzo" not in parts,
        prompt=prompt,
        beam=1 if "beam1" in parts else None,
        model="large-v3" if "large" in parts else "large-v3-turbo" if "turbo" in parts else "",
        vad="novad" not in parts,
        engine="parakeet" if "parakeet" in parts else "whisper",
        recover="recupero" in parts,
    )


def words(text: str) -> list[str]:
    """Lower case, no punctuation, an apostrophe ends its word ("c'è" -> "c'", "è")."""
    text = re.sub(r"[^\w'\s]", " ", text.lower().replace("’", "'").replace("'", "' "))
    return [SAME_WORD.get(w, w) for w in text.split() if w != "'"]


def word_errors(reference: list[str], heard: list[str]) -> int:
    """Words substituted, deleted or inserted (Levenshtein distance on words)."""
    row = list(range(len(heard) + 1))
    for i, ref in enumerate(reference, 1):
        diagonal, row[0] = row[0], i
        for j, word in enumerate(heard, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (ref != word))
    return row[-1]


def wer(references: list[str], texts: list[str]) -> float:
    errors = sum(word_errors(words(r), words(t)) for r, t in zip(references, texts, strict=True))
    total = sum(len(words(r)) for r in references)
    return errors / total if total else 0.0


def microphone_gain_db(recordings: list[np.ndarray]) -> float:
    """About the gain AutoGain settles on for this microphone (live it follows the louder
    phrases a little more): its typical sentence (the median of the loudest 32 ms) at
    TARGET_DBFS, within 0..MAX_GAIN_DB."""
    typical = statistics.median(peak_dbfs(audio) for audio in recordings)
    return round(min(max(TARGET_DBFS - typical, 0.0), MAX_GAIN_DB), 1)


def silero_cut(audio: np.ndarray, endpointer: Endpointer) -> tuple[np.ndarray, bool]:
    """What the Endpointer hands Whisper (float32) and whether it ended before the voice did.
    The recording is followed by a second of silence, as a pause would be live."""
    endpointer.reset()
    padded = np.concatenate((audio, np.zeros(SAMPLE_RATE, dtype=np.float32)))
    for start in range(0, len(padded) - BLOCK + 1, BLOCK):
        event = endpointer.process(padded[start : start + BLOCK])
        if event is EndpointEvent.END:
            rest = audio[start + BLOCK :]
            vad, threshold = endpointer.vad, endpointer.settings.threshold
            more = any(
                vad(rest[i : i + BLOCK]) >= threshold
                for i in range(0, len(rest) - BLOCK + 1, BLOCK)
            )
            return endpointer.audio(), more
        if event is EndpointEvent.TIMEOUT:
            break
    return (endpointer.audio() if endpointer.started else np.zeros(0, np.float32)), False


@dataclass
class Microphone:
    name: str
    folder: Path
    numbers: list[int]  # the sentences recorded (1-based lines of frasi.txt)
    recordings: list[np.ndarray]  # int16
    gain_db: float = 0.0


@dataclass
class Score:
    variant: str
    texts: list[str] = field(default_factory=list)
    ms: list[float] = field(default_factory=list)
    cuts: int = 0
    why: dict[int, str] = field(default_factory=dict)  # sentence index -> why it came out empty


class Notes(logging.Handler):
    """What the transcriber logs while it works: why a sentence came out empty."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def load_microphones(out: Path) -> tuple[list[str], list[Microphone]]:
    import soundfile as sf

    if not (out / "frasi.txt").is_file():
        raise SystemExit(f"Nessuna registrazione in {out}: prima scripts/stt_testset.py")
    sentences = read_sentences(out / "frasi.txt")
    microphones = []
    for folder in sorted(p for p in out.iterdir() if p.is_dir()):
        wavs = sorted(folder.glob("[0-9][0-9].wav"))
        if not wavs:
            continue
        recordings = [sf.read(w, dtype="int16")[0] for w in wavs]
        name_file = folder / "microfono.txt"
        name = name_file.read_text(encoding="utf-8") if name_file.is_file() else folder.name
        if max(peak_dbfs(a) for a in recordings) < MUTE_DBFS:
            print(f"  {name}: muto, salto")
            continue
        mic = Microphone(name, folder, [int(w.stem) for w in wavs], recordings)
        mic.gain_db = microphone_gain_db(recordings)
        microphones.append(mic)
    if not microphones:
        raise SystemExit(f"Nessuna registrazione in {out}: prima scripts/stt_testset.py")
    return sentences, microphones


def whisper_engine(
    settings: Settings, model: str, cache: dict[str, object]
) -> Callable[[np.ndarray, Variant], str]:
    from claudio.stt.transcriber import WhisperTranscriber

    if cache.get("model") != model:
        cache.clear()
        gc.collect()  # one big model at a time on the GPU
        stt_settings = settings.stt.model_copy(update={"model": model})
        root = settings.resolve(settings.models_dir) / "whisper"
        cache.update(model=model, stt=WhisperTranscriber(stt_settings, root))
        loaded = cache["stt"]
        assert isinstance(loaded, WhisperTranscriber)
        print(f"  Whisper {loaded.model_name} su {loaded.device}", flush=True)  # CPU: fell back
    stt = cache["stt"]
    assert isinstance(stt, WhisperTranscriber)
    if "names" not in cache:
        from claudio.__main__ import build_vocabulary

        vocabulary = build_vocabulary(settings)
        vocabulary.read_now()
        cache["names"] = vocabulary.text
    base = settings.stt
    matcher = CommandMatcher(
        load_commands(settings.resolve(settings.router.commands_file)),
        settings.router.fuzzy_threshold,
    )
    from claudio.__main__ import build_recovery

    recovery = build_recovery(settings, stt, matcher)
    if recovery is None:  # off, on the CPU or another model: "recupero" is then the closed list
        print("  Seconda opinione spenta per questo modello: recupero = solo lista chiusa")

    def transcribe(audio: np.ndarray, variant: Variant) -> str:
        stt.settings = base.model_copy(update={"beam_size": variant.beam or base.beam_size})
        stt.vocabulary = cache["names"] if variant.prompt == "nomi" else None  # type: ignore[assignment]
        if variant.prompt == "vuoto":
            stt.prompt = lambda language: ""  # type: ignore[method-assign]
        else:
            stt.__dict__.pop("prompt", None)
        text = stt.transcribe(audio)
        if not variant.recover:
            return text
        return as_live(audio, text, stt, matcher, recovery, base.closed_list_max_words)

    return transcribe


def as_live(
    audio: np.ndarray,
    text: str,
    stt: Any,
    matcher: CommandMatcher,
    recovery: Callable[[np.ndarray, str, str], Any] | None,
    closed_list_max_words: int,
) -> str:
    """What CLAUDIO goes on with after the transcription: a command understood at once, else
    the first command of the closed list (short phrases, no "?"), else the command Whisper's
    second opinion proposes, asked or done (a yes would run it)."""
    if not text or matcher.match(text) is not None:
        return text
    if intake.words(text) <= closed_list_max_words and not text.rstrip().endswith("?"):
        listed = stt.candidates(audio)
        found = next((c for c in listed if c != text and matcher.match(c) is not None), None)
        if found:
            return str(found)
    second = recovery(audio, text, stt.language) if recovery is not None else None
    if second is None or second.verdict is intake.Verdict.DROP:
        return text
    return str(second.text)


def parakeet(cache: dict[str, object]) -> Callable[[np.ndarray, Variant], str]:
    from compare_stt import parakeet_engine

    if "parakeet" not in cache:
        cache.clear()
        gc.collect()
        cache["parakeet"] = parakeet_engine()
    engine = cache["parakeet"]
    assert callable(engine)
    return lambda audio, _variant: str(engine(audio))


def run(
    variant: Variant,
    microphone: Microphone,
    transcribe: Callable[[np.ndarray, Variant], str],
    endpointer: Endpointer,
) -> Score:
    from claudio.stt.transcriber import MIN_RMS

    score = Score(variant.name)
    gain = 10 ** ((microphone.gain_db if variant.gain else 0.0) / 20)
    engine = variant.engine.capitalize()
    notes, logger = Notes(), logging.getLogger("claudio.stt.transcriber")
    level = logger.level
    logger.addHandler(notes)
    logger.setLevel(logging.INFO)
    warm = False
    try:
        for i, recording in enumerate(microphone.recordings):
            audio = apply_gain(recording, gain).astype(np.float32) / 32768.0
            if variant.vad:
                audio, cut = silero_cut(audio, endpointer)
                score.cuts += cut
            if not audio.size:
                score.texts.append("")
                score.why[i] = "Silero non ha sentito voce"
                continue
            if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < MIN_RMS:
                score.texts.append("")  # CLAUDIO does not transcribe it either
                score.why[i] = "sotto -60 dBFS: non trascritta"
                continue
            if not warm:
                transcribe(audio, variant)  # warm-up: the first pass on the GPU is slow
                warm = True
            notes.lines.clear()
            start = time.perf_counter()
            text = transcribe(audio, variant)
            score.ms.append((time.perf_counter() - start) * 1000)  # only what was transcribed
            score.texts.append(text)
            if not text:
                score.why[i] = "; ".join(notes.lines) or f"{engine} non ha scritto niente"
    finally:
        logger.removeHandler(notes)
        logger.setLevel(level)
    return score


def command_of(matcher: CommandMatcher, text: str) -> str | None:
    match = matcher.match(text) if text else None
    return match.command.id if match else None


def report(
    microphone: Microphone,
    references: list[str],
    scores: list[Score],
    matcher: CommandMatcher,
    details: bool,
) -> None:
    print(f"\n== {microphone.name}: {len(references)} frasi, guadagno +{microphone.gain_db:.0f} dB")
    expected = [command_of(matcher, r) for r in references]
    commands = sum(1 for c in expected if c)
    head = f"  {'variante':<22} {'WER':>6}  {'comandi':>8}  {'falsi':>5}  {'ms/frase':>8}"
    print(f"{head}  {'tagli':>5}")
    for s in scores:
        heard = [command_of(matcher, t) for t in s.texts]
        right = sum(1 for c, h in zip(expected, heard, strict=True) if c and h == c)
        false = sum(1 for c, h in zip(expected, heard, strict=True) if h and not c)
        print(
            f"  {s.variant:<22} {wer(references, s.texts):>6.1%}  {right:>3}/{commands:<4}"
            f"  {false:>5}  {statistics.mean(s.ms) if s.ms else 0:>8.0f}  {s.cuts:>5}"
        )
    if details:
        for i, reference in enumerate(references):
            wrong = [s for s in scores if word_errors(words(reference), words(s.texts[i]))]
            if wrong:
                print(f"  {microphone.numbers[i]:02d} «{reference}»")
                for s in wrong:
                    why = f"  ({s.why[i]})" if i in s.why else ""
                    print(f"       {s.variant:<20} {s.texts[i]!r}{why}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--details", action="store_true", help="ogni frase sbagliata")
    parser.add_argument("--set", type=Path, default=OUT, help="la cartella del set di prova")
    args = parser.parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):  # accents and «» also in a redirected file
        sys.stdout.reconfigure(encoding="utf-8")
    variants = [parse_variant(v) for v in args.variants]

    settings = load_settings()
    matcher = CommandMatcher(
        load_commands(settings.resolve(settings.router.commands_file)),
        settings.router.fuzzy_threshold,
    )
    from claudio.audio.vad import SileroVad

    vad = SileroVad(settings.resolve(settings.models_dir) / "vad" / "silero_vad.onnx", SAMPLE_RATE)
    endpointer = Endpointer(vad, settings.vad, SAMPLE_RATE)
    sentences, microphones = load_microphones(args.set)
    results: dict[str, dict[str, list[str]]] = {m.name: {} for m in microphones}
    scores: dict[str, list[Score]] = {m.name: [] for m in microphones}
    cache: dict[str, object] = {}
    # Grouped by engine and model: each big model is loaded once.
    model_of = {v.name: v.model or settings.stt.model for v in variants}
    for variant in sorted(variants, key=lambda v: (v.engine, model_of[v.name])):
        transcribe = None  # it holds the last model: let it go before the next one loads
        if variant.engine == "parakeet":
            transcribe = parakeet(cache)
        else:
            transcribe = whisper_engine(settings, model_of[variant.name], cache)
        for mic in microphones:
            print(f"  {variant.name} su {mic.name}...", flush=True)
            score = run(variant, mic, transcribe, endpointer)
            scores[mic.name].append(score)
            results[mic.name][variant.name] = score.texts
    order = {v.name: i for i, v in enumerate(variants)}
    for mic in microphones:
        references = [sentences[n - 1] for n in mic.numbers]
        ranked = sorted(scores[mic.name], key=lambda s: order[s.variant])
        report(mic, references, ranked, matcher, args.details)
    out = args.set / "risultati.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nTutte le trascrizioni: {out}")


if __name__ == "__main__":
    main()
