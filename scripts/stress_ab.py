"""A/B of the accents Jack writes (claudio.tts.stress_it): each word of a fixed list is said in a
short sentence as written and with its accent, with two seeds, then Whisper writes down what it
heard. A word is a lexicon error only if it fails with both seeds as written and is right with
the accent; Whisper does not hear a stress that leaves the vowels the same, so the ear decides.

  uv run python scripts/stress_ab.py                      # ~90 words, seeds 7 and 42 (~8 min)
  uv run python scripts/stress_ab.py --seeds 7 --only rare
  uv run python scripts/stress_ab.py --models C:\\path\\to\\models    # from a git worktree

Output in logs/stress-ab/ (not committed): <seed>/<group>_<word>_{scritta,accentata}.wav and
report.md. The worker needs the GPU alone: close Claudio and games first."""

from __future__ import annotations

import argparse
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import claudio.tts.jack as jack_module  # noqa: E402
from claudio.config import load_settings  # noqa: E402
from claudio.tts.spoken_text import to_spoken_it  # noqa: E402
from claudio.tts.stress_it import StressLexicon, plain  # noqa: E402

SENTENCE = "Socio, ho detto: {}. Chiaro?"  # the word after a colon: where a clause starts
GROUPS: dict[str, list[str]] = {
    # Verb + pronouns, the case of "accenditi" (heard "accendete" 4/4 on 30/10).
    "pronomi": [
        "accenditi",
        "riaccendermi",
        "accomodati",
        "spegniti",
        "chiuditi",
        "spegnerti",
        "chiudermi",
        "aprilo",
        "guardalo",
        "dimmelo",
        "dammelo",
        "vattene",
        "mandamelo",
        "scrivimelo",
        "ricordamelo",
        "cercalo",
        "prendilo",
        "mettilo",
        "chiamami",
        "ascoltami",
        "parlami",
        "eccolo",
        "accendendolo",
        "accendetelo",
        "ricordati",
        "sbrigati",
        "calmati",
        "siediti",
        "svegliati",
        "fermati",
    ],
    # Rarer sdrucciole and bisdrucciole: the voice may stress them on the second-to-last.
    "rare": [
        "abitano",
        "telefonano",
        "sedano",
        "polizza",
        "aquila",
        "statua",
        "Brindisi",
        "Taranto",
        "regolano",
        "decidere",
        "rispondere",
        "scrivere",
        "chiudere",
        "spegnere",
        "accendere",
        "fulmine",
        "perdita",
        "simbolo",
        "capsula",
        "indice",
        "ordinano",
        "cliccano",
    ],
    # Common sdrucciole: the voice should know them; the accent must not make them worse.
    "comuni": [
        "numero",
        "musica",
        "visita",
        "pagina",
        "camera",
        "facile",
        "ultimo",
        "rapido",
        "medico",
        "tavolo",
        "piccolo",
        "macchina",
        "telefono",
        "utile",
        "dodici",
        "tredici",
    ],
    # Regular words, left as written by mark_stress: Whisper's baseline on this pipeline.
    "piane": ["socio", "cartella", "accendi", "finestra", "chiave", "ragazzo", "subito", "ancora"],
}
_WORDS = re.compile(r"[a-zàèéìíòóùú]+")


class Silent:
    """Fallback that makes a failed worker obvious: silence, and a counter."""

    sample_rate = 24_000
    used = 0

    def synthesize(self, text: str) -> np.ndarray:
        Silent.used += 1
        return np.zeros(2400, dtype=np.int16)


def texts(word: str, table: dict[str, str]) -> tuple[str, str]:
    """(as written, with the accent) for `word` in its sentence, as Jack's worker gets them."""
    sentence = SENTENCE.format(word)
    return (
        to_spoken_it(sentence, table, stress=StressLexicon()),
        to_spoken_it(sentence, table),
    )


def tokens(text: str) -> list[str]:
    return _WORDS.findall(plain(text.lower()))


def heard_as(word: str, sentence: str, heard: str) -> str:
    """What Whisper wrote where `word` was in `sentence` ("(niente)" if it dropped it)."""
    ref, hyp = tokens(sentence), tokens(heard)
    target = plain(word.lower())
    for op, i1, i2, j1, j2 in SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if target in ref[i1:i2]:
            return target if op == "equal" else " ".join(hyp[j1:j2]) or "(niente)"
    return "(niente)"


def speak_all(settings: Any, models: Path, seed: int, jobs: list[dict], out: Path) -> None:
    """Jack says every job's text with `seed`; the audio goes in job["audio"][seed]."""
    tts = settings.tts
    synth = jack_module.JackSynth(
        jack_module.worker_command(models, tts, jack_seed=seed),
        Silent(),
        tts.jack_startup_timeout_s,
        tts.jack_sentence_timeout_s,
        rvc=tts.jack_rvc,
    )
    print(f"Seme {seed}: avvio del worker...")
    synth.start()
    folder = out / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        synth.synthesize("Pronto.")  # warm path, not measured
        for job in jobs:
            start = time.perf_counter()
            audio = synth.synthesize(job["text"])
            job.setdefault("audio", {})[seed] = audio
            sf.write(folder / f"{job['name']}.wav", audio, synth.sample_rate)
            print(f"  {job['name']}: {time.perf_counter() - start:.1f} s")
    finally:
        synth.close()


def transcribe_all(settings: Any, models: Path, jobs: list[dict]) -> None:
    from scipy.signal import resample_poly

    from claudio.stt.transcriber import WhisperTranscriber

    stt = WhisperTranscriber(settings.stt, models / "whisper")
    for job in jobs:
        job["heard"] = {}
        for seed, audio in job.get("audio", {}).items():
            audio16 = resample_poly(audio.astype(np.float32) / 32768, 2, 3)
            segments, _ = stt._model.transcribe(audio16, language="it", beam_size=5)
            heard = " ".join(s.text.strip() for s in segments)
            job["heard"][seed] = (heard, heard_as(job["word"], job["text"], heard))


def report(jobs: list[dict], seeds: list[int], out: Path) -> str:
    def ok(job: dict, seed: int) -> bool:
        return job["heard"][seed][1] == plain(job["word"].lower())

    lines = ["# A/B degli accenti di Jack", "", f"Frase: «{SENTENCE}», semi {seeds}.", ""]
    lines += ["| Gruppo | Scritta giusta | Accentata giusta |", "|---|---|---|"]
    for group in GROUPS:
        rows = [j for j in jobs if j["group"] == group]
        cells = []
        for arm in ("scritta", "accentata"):
            said = [j for j in rows if j["arm"] in (arm, "uguale")]  # "uguale": same text
            cells.append(f"{sum(ok(j, s) for j in said for s in seeds)}/{len(said) * len(seeds)}")
        if rows:
            lines.append(f"| {group} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Parola per parola (cosa ha scritto Whisper, seme per seme)", ""]
    by_word: dict[str, dict[str, dict]] = {}
    for job in jobs:
        by_word.setdefault(job["word"], {})[job["arm"]] = job
    for word, arms in by_word.items():
        cells = []
        for arm, job in arms.items():
            got = ", ".join(job["heard"][s][1] for s in seeds)
            cells.append(f"{arm} «{job['text']}» → {got}")
        lines.append(f"- **{word}** ({next(iter(arms.values()))['group']}): " + "; ".join(cells))
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def make_jobs(groups: list[str], table: dict[str, str]) -> list[dict]:
    jobs = []
    for group in groups:
        for word in GROUPS[group]:
            written, accented = texts(word, table)
            if written == accented:  # nothing to compare: said once
                jobs.append({"group": group, "word": word, "arm": "uguale", "text": written})
                continue
            for arm, text in (("scritta", written), ("accentata", accented)):
                jobs.append({"group": group, "word": word, "arm": arm, "text": text})
    for job in jobs:
        job["name"] = f"{job['group']}_{plain(job['word'].lower())}_{job['arm']}"
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42])
    parser.add_argument("--only", nargs="+", choices=sorted(GROUPS), default=list(GROUPS))
    parser.add_argument("--models", type=Path, help="models folder (default: the configured one)")
    args = parser.parse_args()
    settings = load_settings()
    models = args.models or settings.resolve(settings.models_dir)
    jobs = make_jobs(args.only, settings.tts.jack_words)
    # The texts are final: the worker must not convert them again (with or without accents).
    jack_module.to_spoken_it = lambda text, words=None: text  # type: ignore[assignment]
    out = ROOT / "logs" / "stress-ab"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} frasi x {len(args.seeds)} semi")
    for seed in args.seeds:
        speak_all(settings, models, seed, jobs, out)
    if Silent.used:
        print(f"ATTENZIONE: {Silent.used} frasi non dette da Jack (worker caduto)")
    print("Whisper ascolta le frasi...")
    transcribe_all(settings, models, jobs)
    print(report(jobs, args.seeds, out))
    print(f"WAV e report in {out}")


if __name__ == "__main__":
    main()
