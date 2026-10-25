"""Compare Jack's voice settings by ear and by numbers: every variant speaks the same replies,
split into sentences as Claudio does, then Whisper writes down what it hears (WER).

  uv run python scripts/jack_variants.py                 # all variants (~5 min of GPU)
  uv run python scripts/jack_variants.py --only A C      # some of them
  uv run python scripts/jack_variants.py --words         # spellings of hard words (~1 min)
  uv run python scripts/jack_variants.py --from-log latest   # Claudio's real replies, C/S/V

Output in logs/jack-variants/ (not committed, personal use only): <variant>/<n>.wav and
report.md (WER, first audio, gaps). The worker needs the GPU alone: close Claudio first."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claudio.config import load_settings  # noqa: E402
from claudio.tts.jack import JackSynth  # noqa: E402
from claudio.tts.number_words import digits_to_words_it  # noqa: E402
from claudio.tts.sentence_splitter import SentenceSplitter  # noqa: E402
from claudio.tts.spoken_text import to_spoken_it  # noqa: E402

# Replies like Haiku's with the Jack personality: commas, a question, numbers, an acronym.
REPLIES = [
    "Ah, eccoti, eroe. Ho aperto Chrome, e non serve che mi ringrazi.",
    "Sono le 22 e 51, e domani alle 8 hai la riunione con il capo.",
    "Il PC è al 35% di batteria: attaccalo, se non vuoi perdere tutto.",
    "Ho trovato tre file nella cartella Download, vuoi che apra il più recente?",
    "Non ci sono riuscito. Qualcuno ha rotto qualcosa, e scommetto che non sono stato io.",
    "Fuori ci sono 18 gradi e piove: perfetto per restare a casa a lavorare per me.",
]
GAP_S = 0.12  # silence between the sentences of a reply in the WAV (like the player's)

# name -> (what it is, [tts] overrides, first_clause_chars or None for the configured one)
VARIANTS: dict[str, tuple[str, dict[str, Any], int | None]] = {
    "A": (
        "prima: Auto, temperatura 0.9, RVC a blocchi",
        {"language": "Auto", "jack_temperature": 0.9, "jack_rvc": "blocks"},
        None,
    ),
    "B": (
        "Italian (ora fisso), temperatura 0.7, RVC a blocchi",
        {"jack_temperature": 0.7, "jack_rvc": "blocks"},
        None,
    ),
    "C": (
        "B + RVC sulla frase intera (prima frase a blocchi)",
        {"jack_temperature": 0.7, "jack_rvc": "auto"},
        None,
    ),
    "D": (
        "C con il checkpoint dell'epoca 3",
        {
            "jack_temperature": 0.7,
            "jack_rvc": "auto",
            "jack_checkpoint": Path("voices/jack/ft/0.6B/checkpoint-epoch-3"),
        },
        None,
    ),
    "S": (
        "C con RVC sulla frase intera anche per la prima frase",
        {"jack_temperature": 0.7, "jack_rvc": "sentence"},
        None,
    ),
    "V": (
        "C a velocità normale (senza rallentare del 10%)",
        {"jack_temperature": 0.7, "jack_rvc": "auto", "jack_speed": 1.0},
        None,
    ),
    "E": (
        "C senza taglio alla prima virgola",
        {"jack_temperature": 0.7, "jack_rvc": "auto"},
        0,
    ),
}


class Silent:
    """Fallback that makes a failed worker obvious: silence, and a counter."""

    sample_rate = 24_000
    used = 0

    def synthesize(self, text: str) -> np.ndarray:
        Silent.used += 1
        return np.zeros(2400, dtype=np.int16)


_CLOCK = re.compile(r"\b(\d{1,2})\.(\d{2})\b")  # Whisper writes "22 e 51" as "22.51"


def words(text: str) -> list[str]:
    text = digits_to_words_it(to_spoken_it(_CLOCK.sub(r"\1:\2", text))).lower()
    return re.findall(r"[a-zàèéìòóù]+", text)


def wer(reference: str, heard: str) -> float:
    ref, hyp = words(reference), words(heard)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / max(len(ref), 1)


def speak_reply(synth: JackSynth, reply: str, first_clause: int) -> tuple[np.ndarray, dict]:
    """The reply as Claudio speaks it: sentence by sentence, audio played as it arrives.
    `gap_ms` is the longest wait for audio after playback ran dry (a pause heard)."""
    splitter = SentenceSplitter(first_clause_chars=first_clause)
    sentences = splitter.feed(reply) + splitter.flush()
    pieces: list[np.ndarray] = []
    t0 = time.perf_counter()
    first_ms = None
    played_until = 0.0  # when the audio received so far ends, in seconds from t0
    gap = 0.0
    for n, sentence in enumerate(sentences):
        for chunk in synth.stream(sentence):
            now = time.perf_counter() - t0
            if first_ms is None:
                first_ms = 1000 * now
                played_until = now
            elif now > played_until:
                gap = max(gap, now - played_until)
                played_until = now
            played_until += len(chunk) / synth.sample_rate
            pieces.append(chunk)
        if n < len(sentences) - 1:
            pieces.append(np.zeros(int(GAP_S * synth.sample_rate), dtype=np.int16))
            played_until += GAP_S
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    return audio, {"first_ms": first_ms or 0.0, "gap_ms": 1000 * gap, "pieces": sentences}


def run_variant(
    settings: Any, name: str, out: Path, replies: list[str] | None = None
) -> list[dict]:
    label, overrides, first_clause = VARIANTS[name]
    from claudio.tts.jack import worker_command

    tts = settings.tts
    models = settings.resolve(settings.models_dir)
    checkpoint = models / overrides.get("jack_checkpoint", tts.jack_checkpoint)
    if not checkpoint.exists():
        print(f"{name}: salto, manca {checkpoint}")
        return []
    synth = JackSynth(
        worker_command(models, tts, **overrides),
        Silent(),
        tts.jack_startup_timeout_s,
        tts.jack_sentence_timeout_s,
        language=overrides.get("language", "Italian"),
        rvc=overrides.get("jack_rvc", tts.jack_rvc),
        words=tts.jack_words,
    )
    print(f"{name}: {label} - avvio del worker...")
    synth.start()
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        synth.synthesize("Pronto.")  # warm path, not measured
        clause = tts.first_clause_chars if first_clause is None else first_clause
        for n, reply in enumerate(replies or REPLIES, 1):
            time.sleep(1.5)  # a new reply each time, as after a question
            audio, info = speak_reply(synth, reply, clause)
            sf.write(folder / f"{n}.wav", audio, synth.sample_rate)
            rows.append({"n": n, "reply": reply, "audio": audio, "rate": synth.sample_rate, **info})
            first, gap = info["first_ms"], info["gap_ms"]
            print(f"  {n}: prima audio {first:.0f} ms, pausa max {gap:.0f} ms")
    finally:
        synth.close()
    return rows


def transcribe(settings: Any, results: dict[str, list[dict]]) -> None:
    from scipy.signal import resample_poly

    from claudio.stt.transcriber import WhisperTranscriber

    stt = WhisperTranscriber(settings.stt, settings.resolve(settings.models_dir) / "whisper")
    for rows in results.values():
        for row in rows:
            audio = row["audio"].astype(np.float32) / 32768
            audio16 = resample_poly(audio, 2, 3) if row["rate"] == 24_000 else audio
            segments, _ = stt._model.transcribe(audio16, language="it", beam_size=5)
            row["heard"] = " ".join(s.text.strip() for s in segments)
            row["wer"] = wer(row["reply"], row["heard"])


def report(results: dict[str, list[dict]], out: Path) -> str:
    lines = [
        "# Varianti della voce di Jack",
        "",
        "| Variante | Cosa cambia | WER medio | Prima audio (ms) | Pausa max (ms) |",
        "|---|---|---|---|---|",
    ]
    for name, rows in results.items():
        if not rows:
            continue
        lines.append(
            f"| {name} | {VARIANTS[name][0]} | {statistics.mean(r['wer'] for r in rows):.3f} | "
            f"{statistics.median(r['first_ms'] for r in rows):.0f} | "
            f"{max(r['gap_ms'] for r in rows):.0f} |"
        )
    lines += ["", "## Cosa ha sentito Whisper", ""]
    for name, rows in results.items():
        for row in rows:
            lines.append(f"- {name}/{row['n']}.wav (WER {row['wer']:.2f}): {row['heard']}")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


# Ways to write a word Jack says wrong; each goes in a sentence, with the configured settings.
# The winner goes in [tts.jack_words] of claudio.toml.
SPELLINGS = {
    "PC": ["pi ci", "pi-ci", "pì cì", "picì"],
    "file": ["fàil", "fail", "fàiil", "fàyl"],
    "download": ["dàunlod", "daunlòd", "dàunlòud"],
}
SPELLING_SENTENCE = {
    "PC": "Il {} è al trentacinque per cento di batteria.",
    "file": "Ho trovato tre {} nella cartella documenti.",
    "download": "Apro la cartella {}, eroe.",
}


def log_replies(path: Path, limit: int = 15) -> list[str]:
    """Claudio's replies in a conversazione.md (a session folder or the file itself;
    "latest" = the newest session in logs/sessions/)."""
    if str(path) == "latest":
        sessions = sorted((ROOT / "logs" / "sessions").glob("*/conversazione.md"))
        if not sessions:
            sys.exit("nessuna sessione in logs/sessions/")
        path = sessions[-1]
    elif path.is_dir():
        path = path / "conversazione.md"
    replies = [
        line.split("**Claudio:**", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("**Claudio:**")
    ]
    replies = [r for r in dict.fromkeys(replies) if len(words(r)) >= 3]  # once each
    print(f"{len(replies)} risposte da {path} (uso le ultime {limit})")
    return replies[-limit:]


def misheard_words(results: dict[str, list[dict]]) -> list[tuple[str, int, list[str]]]:
    """Words of the replies Whisper did not hear as written, most frequent first, with what
    it heard instead: the candidates for [tts.jack_words]."""
    from collections import Counter, defaultdict
    from difflib import SequenceMatcher

    counts: Counter[str] = Counter()
    heard_as: dict[str, list[str]] = defaultdict(list)
    for rows in results.values():
        for row in rows:
            ref, hyp = words(row["reply"]), words(row["heard"])
            for op, i1, i2, j1, j2 in SequenceMatcher(a=ref, b=hyp).get_opcodes():
                if op in ("replace", "delete"):
                    got = " ".join(hyp[j1:j2]) or "(niente)"
                    for word in ref[i1:i2]:
                        counts[word] += 1
                        heard_as[word].append(got)
    return [(w, n, heard_as[w][:3]) for w, n in counts.most_common()]


def run_spellings(settings: Any, out: Path) -> None:
    from claudio.tts.jack import worker_command

    tts = settings.tts
    models = settings.resolve(settings.models_dir)
    synth = JackSynth(
        worker_command(models, tts),
        Silent(),
        tts.jack_startup_timeout_s,
        tts.jack_sentence_timeout_s,
        rvc="sentence",  # the whole sentence: how most sentences of a reply are spoken
    )  # no respellings: the candidates are sent as written
    print("Parole difficili - avvio del worker...")
    synth.start()
    folder = out / "parole"
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["# Parole difficili", ""]
    try:
        synth.synthesize("Pronto.")
        for word, candidates in SPELLINGS.items():
            for n, spelling in enumerate(candidates, 1):
                sentence = SPELLING_SENTENCE[word].format(spelling)
                audio = synth.synthesize(sentence)
                name = f"{word.lower()}_{n}.wav"
                sf.write(folder / name, audio, synth.sample_rate)
                lines.append(f"- {name}: {sentence}")
                print(f"  {name}: {sentence}")
    finally:
        synth.close()
    (folder / "elenco.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--words", action="store_true", help="only the hard-word spellings")
    parser.add_argument(
        "--from-log",
        type=Path,
        help='speak Claudio\'s replies of a session (folder, conversazione.md or "latest")',
    )
    args = parser.parse_args()
    settings = load_settings()
    out = ROOT / "logs" / "jack-variants"
    out.mkdir(parents=True, exist_ok=True)
    if args.words:
        run_spellings(settings, out)
        print(f"WAV in {out / 'parole'}")
        return
    replies = log_replies(args.from_log) if args.from_log else None
    names = args.only if (args.only != sorted(VARIANTS) or not replies) else ["C", "S", "V"]
    results = {name: run_variant(settings, name, out, replies) for name in names}
    if Silent.used:
        print(f"ATTENZIONE: {Silent.used} frasi non dette da Jack (worker caduto)")
    print("Whisper ascolta le frasi...")
    transcribe(settings, results)
    text = report(results, out)
    hard = misheard_words(results)
    if hard:
        lines = ["", "## Parole sentite male (parola: volte, sentita come)", ""]
        lines += [f"- {w}: {n}, {' / '.join(got)}" for w, n, got in hard[:40]]
        text += "\n".join(lines) + "\n"
        (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"WAV e report in {out}")


if __name__ == "__main__":
    main()
