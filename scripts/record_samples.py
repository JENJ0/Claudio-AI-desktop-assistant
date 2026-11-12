"""Record WAV samples with the real microphone, for training and measuring CLAUDIO.

Wake words (50 takes each; the script asks for a different way of speaking every few takes):
    uv run python scripts/record_samples.py wakeword --phrase "Hey Claudio"
    uv run python scripts/record_samples.py wakeword --phrase "Claudio"
Ordinary sentences without the wake word (about 3 minutes), and the room's own noise:
    uv run python scripts/record_samples.py negatives
    uv run python scripts/record_samples.py noise
The zip to upload to Colab (recordings + training scripts):
    uv run python scripts/record_samples.py pack
Commands (the first phrase of every command, x2, for benchmark_latency.py):
    uv run python scripts/record_samples.py commands --repeat 2
Safety phrases (a yes and near-yes phrases, x2, for tests/integration/test_safety_audio.py):
    uv run python scripts/record_samples.py safety --repeat 2
The 145-take safety set for the final test (one stream open for the whole session, about 25 min):
    uv run python scripts/record_samples.py safety-v2

Only some ways of speaking (a few takes of each, in a few minutes):
    uv run python scripts/record_samples.py wakeword --phrase "Claudio" --styles veloce,distratto

Running a mode again adds new takes after the ones already there: nothing is overwritten.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zipfile
from dataclasses import dataclass
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
SAFETY_V2_SEED = 2026  # the order of the takes is shuffled, the same way on every restart
QUESTION_NOTE = "  (come una vera domanda, con il tono che sale)"
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


def wakeword_prompts(
    phrase: str, count: int, start: int, only: tuple[str, ...] = ()
) -> list[tuple[str, str, str, float]]:
    """(header, text, file stem, seconds) per take. A known wake word gets the styles of
    phrases.RECORDING_STYLES (`only`: just those); any other phrase is recorded plainly."""
    model = slug(phrase)
    if model in phrases.RECORDING_STYLES:
        plan = phrases.recording_plan(model, count, only)
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


def negative_labels(start: int) -> dict[str, str]:
    """The sentence of every take negative_prompts(start) asks for, by file stem."""
    return {f"neg_{start + i:03d}": text for i, text in enumerate(phrases.NEGATIVE_PROMPTS)}


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
    samples/<kind>/*.wav and wakeword/*.py. The ordinary sentences with "Claudia" stay out
    (and the zip says which sentence each take is, for the training report)."""
    counts = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for kind in PACK_DIRS:
            if kind == "negatives":
                files, left_out = phrases.usable_negatives(samples / kind)
                texts = phrases.negative_texts(samples / kind)
                zf.writestr(
                    f"samples/{kind}/{phrases.PROMPTS_FILE}",
                    json.dumps({f.stem: texts.get(f.stem, "") for f in files}, ensure_ascii=False),
                )
                if left_out:
                    print(
                        "Lasciate fuori le frasi con Claudia: "
                        + ", ".join(f'{f.name} ("{texts[f.stem]}")' for f in left_out)
                    )
            else:
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


def peak_problem(audio: np.ndarray, min_peak: float) -> str | None:
    peak = float(np.abs(audio).max()) / 32768
    if peak >= 0.99:
        return f"picco {peak:.0%}  <- distorto, allontanati un po' e rifai (r)"
    if peak < min_peak:
        return f"picco {peak:.0%}  <- troppo basso, rifai (r) più vicino o più forte"
    return None


def check(audio: np.ndarray, min_peak: float) -> str:
    return peak_problem(audio, min_peak) or f"picco {float(np.abs(audio).max()) / 32768:.0%}"


class OpenMic:
    """One input stream for the whole session, takes are slices of it: a stream opened per take
    made the headset's voice processing adapt at each start, and a burst followed each start."""

    def __init__(self, device: int | None) -> None:
        self.device = device
        self.stream: Any = None

    def __enter__(self) -> OpenMic:
        import sounddevice as sd

        from claudio.audio.devices import stream_settings

        self.stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype="int16",
            device=self.device,
            extra_settings=stream_settings(self.device),
        )
        self.stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stream.stop()
        self.stream.close()

    def take(self, seconds: float) -> np.ndarray:
        """Drops what the stream heard since the last take, then reads the next `seconds`."""
        if stale := self.stream.read_available:
            self.stream.read(stale)
        print("   ● parla", flush=True)
        audio, _ = self.stream.read(int(seconds * SR))
        return np.asarray(audio[:, 0])


def min_peak_for(stem: str) -> float:
    """Below this peak a take is probably too quiet (room noise has no minimum)."""
    if stem.startswith("noise_"):
        return 0.0
    return 0.015 if stem.endswith(("_piano", "_lontano")) else 0.05


def session(
    prompts: list[tuple[str, str, str, float]],
    out: Path,
    device: int | None,
    labels: dict[str, str] | None = None,
    mic: OpenMic | None = None,
) -> int:
    """Records the prompts in order; returns how many different clips were saved. `labels`
    (file stem -> sentence) is written next to the takes, so a take is never taken for
    another sentence when the list of prompts changes. With `mic` (one stream for the whole
    session) a take under the peak check or clipped is not saved: the prompt comes back."""
    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    print(f"Registro {len(prompts)} clip in {out}")
    print("Invio = registra   r + Invio = rifai la precedente   q + Invio = basta così\n")
    i, header, saved = 0, "", set()
    held: np.ndarray | None = None  # a quiet noise take, kept if JenJo answers "a"
    while True:
        kept, held, accept = held, None, False
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
            accept = answer == "a" and kept is not None
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
        audio = kept if accept and kept is not None else None
        if audio is None:
            audio = mic.take(seconds) if mic else record(seconds, device)
        if mic and not accept and (problem := peak_problem(audio, min_peak_for(stem))):
            print(f"   scartata, {problem}")
            if stem.startswith("noise/") and float(np.abs(audio).max()) / 32768 < 0.99:
                # a whispered "psst" may never reach the peak; a clipped take is always redone
                held = audio
                print("   (a + Invio = tienila lo stesso)")
            continue
        (out / f"{stem}.wav").parent.mkdir(parents=True, exist_ok=True)  # safety: yes/, no/
        sf.write(out / f"{stem}.wav", audio, SR, subtype="PCM_16")
        if labels and stem in labels:
            phrases.remember_text(out, stem, labels[stem])
        saved.add(stem)
        print(f"   salvato, {check(audio, min_peak_for(stem))}")
        i += 1
    return len(saved)


def command_prompts(
    commands: list[Any], repeat: int, seconds: float, out: Path
) -> list[tuple[str, str, str, float]]:
    """The first phrase of every command, `repeat` times, numbered after the takes already
    in `out` (benchmark_latency.py reads the command id before the first "__")."""
    prompts = []
    for c in commands:
        phrase = c.phrases[0]
        prefix = f"{c.id}__{slug(phrase)}"
        start = next_index(out, f"{prefix}_")
        for r in range(repeat):
            prompts.append(("comandi", f'di\': "{phrase}"', f"{prefix}__{start + r}", seconds))
    return prompts


def safety_lines(path: Path, columns: int) -> list[tuple[int, list[str]]]:
    """(line number, fields) per data line of the TSV; blank lines and # comments are skipped.
    A line with too few columns, an unknown answer or an empty phrase stops with its number."""
    lines = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        where = f"{path.name}, riga {number}"
        if len(fields) < columns:
            raise ValueError(
                f"{where}: servono {columns} colonne separate da tab, ce ne sono {len(fields)}"
            )
        if fields[0] not in ("yes", "no", "unclear"):
            raise ValueError(f"{where}: risposta {fields[0]!r}, attesa una tra yes, no, unclear")
        if not fields[1].strip():
            raise ValueError(f"{where}: frase vuota")
        lines.append((number, fields))
    return lines


def load_safety_phrases(path: Path) -> list[tuple[str, str]]:
    """(expected answer, phrase) per line of the TSV; the columns after the phrase are for
    safety-v2."""
    return [(f[0], f[1].strip()) for _, f in safety_lines(path, 2)]


@dataclass(frozen=True)
class SafetyRow:
    expected: str
    phrase: str
    group: str  # folder of the takes in samples/safety_v2/
    takes: int


def load_safety_v2(path: Path) -> list[SafetyRow]:
    """Rows of the TSV with their group and number of takes (the columns after the phrase)."""
    rows = []
    for number, (expected, phrase, group, takes, *_) in safety_lines(path, 4):
        if not takes.isdigit():
            raise ValueError(f"{path.name}, riga {number}: prese {takes!r}, atteso un numero")
        rows.append(SafetyRow(expected, phrase.strip(), group, int(takes)))
    return rows


def safety_v2_prompts(
    phrases_file: Path, seconds: float, out: Path
) -> list[tuple[str, str, str, float]]:
    """The takes still missing in out/<group>/, two rounds, each shuffled (the same way on every
    run), so a restart goes on from where the session stopped."""
    rounds: dict[int, list[tuple[str, str, str, float]]] = {}
    for row in load_safety_v2(phrases_file):
        prefix = slug(row.phrase)
        text = f'di\': "{row.phrase}"' + (QUESTION_NOTE if row.phrase.endswith("?") else "")
        for take in range(row.takes):
            if (out / row.group / f"{prefix}__{take}.wav").exists():
                continue  # the takes are not recorded in order: ask for each one that is missing
            stem = f"{row.group}/{prefix}__{take}"
            # a phrase with many takes (the 8 "sì?") is spread over the first two rounds
            rounds.setdefault(take % 2, []).append(("frasi di sicurezza", text, stem, seconds))
    rng = random.Random(SAFETY_V2_SEED)
    prompts = []
    for take in sorted(rounds):
        rng.shuffle(rounds[take])
        prompts += rounds[take]
    return prompts


def missing_takes(phrases_file: Path, safety_dir: Path) -> tuple[list[str], list[str]]:
    """(expected WAVs not in safety_dir, WAVs there that nothing expects), as `group/stem.wav`."""
    expected = {
        f"{r.group}/{slug(r.phrase)}__{n}.wav"
        for r in load_safety_v2(phrases_file)
        for n in range(r.takes)
    }
    found = {p.relative_to(safety_dir).as_posix() for p in safety_dir.glob("*/*.wav")}
    return sorted(expected - found), sorted(found - expected)


def run_safety_v2(prompts: list[tuple[str, str, str, float]], out: Path, device: int | None) -> int:
    with OpenMic(device) as mic:
        return session(prompts, out, device, mic=mic)


def safety_prompts(
    phrases_file: Path, repeat: int, seconds: float, out: Path
) -> list[tuple[str, str, str, float]]:
    """Every safety phrase up to `repeat` takes, the missing ones only; the take lands in
    out/<expected>/, so the folder says what the answer must be."""
    prompts = []
    for expected, phrase in load_safety_phrases(phrases_file):
        prefix = slug(phrase)
        start = next_index(out / expected, f"{prefix}_")
        for take in range(start, repeat):  # a rerun records only what is missing
            stem = f"{expected}/{prefix}__{take}"
            prompts.append(("frasi di sicurezza", f'di\': "{phrase}"', stem, seconds))
    return prompts


def main() -> None:
    from claudio.audio.devices import resolve_device
    from claudio.config import PROJECT_ROOT, load_settings
    from claudio.router.matcher import load_commands

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        choices=["wakeword", "negatives", "noise", "commands", "safety", "safety-v2", "pack"],
    )
    parser.add_argument("--phrase", default="Hey Claudio")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--styles",
        default="",
        help="wakeword: solo questi modi di parlare, separati da virgola (es. veloce,distratto)",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "samples")
    args = parser.parse_args()

    if args.mode == "pack":
        zip_path = args.out / "claudio_training.zip"
        counts = pack(args.out, PROJECT_ROOT / "scripts" / "wakeword", zip_path)
        print(", ".join(f"{k}: {v}" for k, v in counts.items()))
        print(f"Creato {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB): caricalo su Colab")
        if not counts["hey_claudio"] or not counts["claudio"]:
            print("Attenzione: mancano registrazioni della wake word, il modello sarà meno preciso")
        return

    labels: dict[str, str] | None = None
    settings = load_settings()
    device = resolve_device(settings.audio.input_device, "input")  # as CLAUDIO opens it
    if args.mode == "wakeword":
        out = args.out / slug(args.phrase)
        only = tuple(s.strip() for s in args.styles.split(",") if s.strip())
        known = [s[0] for s in phrases.RECORDING_STYLES.get(slug(args.phrase), [])]
        if unknown := [s for s in only if s not in known]:
            parser.error(f"stili sconosciuti: {', '.join(unknown)} (ci sono {', '.join(known)})")
        start = next_index(out, slug(args.phrase))
        prompts = wakeword_prompts(args.phrase, args.count, start, only)
    elif args.mode == "negatives":
        out = args.out / "negatives"
        start = next_index(out, "neg")
        prompts = negative_prompts(start)
        labels = negative_labels(start)
    elif args.mode == "noise":
        out = args.out / "noise"
        prompts = noise_prompts(next_index(out, "noise"), args.seconds or 60.0)
    elif args.mode == "safety":
        out = args.out / "safety"
        phrases_file = PROJECT_ROOT / "scripts" / "safety_phrases.tsv"
        prompts = safety_prompts(phrases_file, args.repeat, args.seconds or 3.0, out)
    elif args.mode == "safety-v2":
        out = args.out / "safety_v2"
        phrases_file = PROJECT_ROOT / "scripts" / "safety_phrases.tsv"
        prompts = safety_v2_prompts(phrases_file, args.seconds or 3.0, out)
        print('Le 8 "sì?" vanno dette come vere domande, con il tono che sale.')
        print(f"{len(prompts)} riprese da fare. Lo stream resta aperto fino alla fine.\n")
        saved = run_safety_v2(prompts, out, device)
        print(f"\nFatto: {saved} clip salvate in {out}")
        return
    else:
        out = args.out / "commands"
        commands = load_commands(settings.resolve(settings.router.commands_file))
        prompts = command_prompts(commands, args.repeat, args.seconds or 3.5, out)
    if args.seconds and args.mode in ("wakeword", "negatives"):
        prompts = [(h, t, s, args.seconds) for h, t, s, _ in prompts]
    saved = session(prompts, out, device, labels)
    print(f"\nFatto: {saved} clip salvate in {out}")


if __name__ == "__main__":
    main()
