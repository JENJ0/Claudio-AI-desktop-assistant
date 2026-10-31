"""A/B of the accents Jack writes (claudio.tts.stress_it): each text is said as written and with
its accents, with several seeds, then Whisper writes down what it heard in place of each word.
A word is a lexicon error only if it fails as written and is right with the accent; Whisper does
not hear a stress that leaves the vowels the same, so the ear decides last.

  uv run python scripts/stress_ab.py --frasi       # 26 real sentences, seeds 7 42 99 (~10 min)
  uv run python scripts/stress_ab.py               # ~90 words alone in "Socio, ho detto: ..."
  uv run python scripts/stress_ab.py --seeds 7 --only rare
  uv run python scripts/stress_ab.py --models C:\\path\\to\\models    # from a git worktree

Output in logs/stress-ab/ (not committed): <seed>/<name>_{scritta,accentata}.wav and report.md.
The worker needs the GPU alone: close Claudio and games first."""

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

SENTENCE = "Socio, ho detto: {}. Chiaro?"  # a word alone, after a colon: where a clause starts
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
# Sentences as Claudio says them, with the words to check. The first two are pieces Jack said on
# 30/10 (logs/sessions/2026-10-30_14-00): "accenditi" was heard "accendete" there, not alone.
REAL: list[tuple[str, str]] = [
    ("Per riaccendermi di' Claudio, accenditi.", "riaccendermi accenditi"),
    (
        "Accomodati, socio, che il narratore più bello dell'universo sta per cominciare.",
        "accomodati",
    ),
    ("Per spegnermi di' Claudio, spegniti.", "spegnermi spegniti"),
    ("Se vuoi chiudermi, di' Claudio, chiuditi.", "chiudermi chiuditi"),
    ("Se il PC non risponde, spegnilo e riaccendilo.", "spegnilo riaccendilo"),
    ("Ho trovato il file sul desktop: aprilo quando vuoi.", "aprilo"),
    ("Il video è nella cartella Download, guardalo con calma.", "guardalo"),
    ("Se ti serve il numero, dimmelo e lo cerco.", "numero dimmelo"),
    ("Quando hai finito il documento, mandamelo e lo salvo.", "mandamelo"),
    ("Prendi il caricatore e mettilo nella borsa, poi chiamami.", "mettilo chiamami"),
    ("Ti ho mandato il link: cliccalo e scaricalo.", "cliccalo scaricalo"),
    ("Cercavi la cartella della tesi? Eccola, socio.", "eccola"),
    ("Calmati, socio, ci penso io.", "calmati"),
    ("Ricordati di salvare il file prima di chiudere.", "ricordati"),
    ("Sbrigati, eroe, la riunione comincia tra cinque minuti.", "sbrigati"),
    ("Siediti comodo, che ti spiego tutto.", "siediti"),
    ("Le persone che abitano qui pagano poco di affitto.", "abitano"),
    ("I tuoi amici ti telefonano sempre all'ora di cena.", "telefonano"),
    ("Il treno per Brindisi passa da Taranto alle nove.", "brindisi taranto"),
    ("Metti un po' di musica, così è più facile lavorare.", "musica facile"),
    ("È l'ultimo file utile della cartella, socio.", "ultimo utile"),
    ("Ho prenotato la visita dal medico per domani.", "visita medico"),
    ("La macchina è in garage e il telefono è sul tavolo.", "macchina telefono tavolo"),
    ("Apri la prima pagina, c'è un piccolo errore.", "pagina piccolo"),
    ("È l'unico modo, socio: un'isola senza ospiti.", "unico isola ospiti"),
    ("Metti in ordine gli alberi del giardino.", "ordine alberi"),
]
_WORDS = re.compile(r"[a-zàèéìíòóùú]+")


class Silent:
    """Fallback that makes a failed worker obvious: silence, and a counter."""

    sample_rate = 24_000
    used = 0

    def synthesize(self, text: str) -> np.ndarray:
        Silent.used += 1
        return np.zeros(2400, dtype=np.int16)


def texts(sentence: str, table: dict[str, str]) -> tuple[str, str]:
    """(as written, with the accents) of `sentence`, as Jack's worker gets them."""
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


def make_jobs(groups: list[str], table: dict[str, str], real: bool = False) -> list[dict]:
    """One job per text and arm ("scritta", "accentata", or "uguale" when the accents change
    nothing). `targets`: the words whose hearing is checked."""
    items = (
        [("frasi", sentence, words.split()) for sentence, words in REAL]
        if real
        else [(g, SENTENCE.format(word), [word]) for g in groups for word in GROUPS[g]]
    )
    jobs = []
    for n, (group, sentence, targets) in enumerate(items, 1):
        written, accented = texts(sentence, table)
        name = f"{group}_{n:02d}_{plain(targets[0].lower())}"
        if written == accented:
            arms = [("uguale", written)]
        else:
            arms = [("scritta", written), ("accentata", accented)]
        for arm, text in arms:
            jobs.append(
                {"group": group, "targets": targets, "arm": arm, "text": text, "name": name}
            )
    for job in jobs:
        job["name"] += f"_{job['arm']}"
    return jobs


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
    print(f"Seme {seed}: avvio del worker...", flush=True)
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
            print(f"  {job['name']}: {time.perf_counter() - start:.1f} s", flush=True)
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
            job["heard"][seed] = {t: heard_as(t, job["text"], heard) for t in job["targets"]}
            job["heard"][seed][""] = heard


def score(jobs: list[dict], seeds: list[int], arm: str) -> tuple[int, int]:
    """(right, said) over every target word of the jobs said in `arm` (or the same text)."""
    said = [(j, t) for j in jobs if j["arm"] in (arm, "uguale") for t in j["targets"]]
    right = sum(j["heard"][s][t] == plain(t.lower()) for j, t in said for s in seeds)
    return right, len(said) * len(seeds)


def report(jobs: list[dict], seeds: list[int], out: Path) -> str:
    lines = ["# A/B degli accenti di Jack", "", f"Semi {seeds}.", ""]
    lines += ["| Gruppo | Scritta giusta | Accentata giusta |", "|---|---|---|"]
    for group in dict.fromkeys(j["group"] for j in jobs):
        mine = [j for j in jobs if j["group"] == group]
        cells = ["{}/{}".format(*score(mine, seeds, arm)) for arm in ("scritta", "accentata")]
        lines.append(f"| {group} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Parola per parola (cosa ha scritto Whisper, seme per seme)", ""]
    for name in dict.fromkeys(j["name"].rsplit("_", 1)[0] for j in jobs):
        arms = [j for j in jobs if j["name"].rsplit("_", 1)[0] == name]
        for target in arms[0]["targets"]:
            cells = []
            for job in arms:
                got = ", ".join(job["heard"][s][target] for s in seeds)
                cells.append(f"{job['arm']} → {got}")
            lines.append(f"- **{target}** ({name}): " + "; ".join(cells))
        lines.append(f"  - testo accentato: «{arms[-1]['text']}»")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frasi", action="store_true", help="real sentences instead of words")
    parser.add_argument("--seeds", nargs="+", type=int, help="default 7 42 (words), 7 42 99")
    parser.add_argument("--only", nargs="+", choices=sorted(GROUPS), default=list(GROUPS))
    parser.add_argument("--models", type=Path, help="models folder (default: the configured one)")
    args = parser.parse_args()
    seeds = args.seeds or ([7, 42, 99] if args.frasi else [7, 42])
    settings = load_settings()
    models = args.models or settings.resolve(settings.models_dir)
    jobs = make_jobs(args.only, settings.tts.jack_words, real=args.frasi)
    # The texts are final: the worker must not convert them again (with or without accents).
    jack_module.to_spoken_it = lambda text, words=None: text  # type: ignore[assignment]
    out = ROOT / "logs" / ("stress-ab-frasi" if args.frasi else "stress-ab")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} testi x {len(seeds)} semi", flush=True)
    for seed in seeds:
        speak_all(settings, models, seed, jobs, out)
    if Silent.used:
        print(f"ATTENZIONE: {Silent.used} frasi non dette da Jack (worker caduto)")
    print("Whisper ascolta le frasi...", flush=True)
    transcribe_all(settings, models, jobs)
    print(report(jobs, seeds, out))
    print(f"WAV e report in {out}")


if __name__ == "__main__":
    main()
