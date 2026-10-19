"""Synthetic voices saying the wake words (and what is not the wake word), as 16 kHz WAVs.

Two engines, both through onnxruntime on the CPU:
- Piper: the Italian voices (Paola, Riccardo) and en_US-libritts_r, 904 English speakers.
  The English voices also speak Italian (espeak-ng "it" phonemes): hundreds of accents.
- Kokoro: 54 voice styles in 9 languages, blended in pairs, all speaking Italian phonemes.
Speed varies from slow to very fast, because fast speech is what the old model missed."""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from augment import SR, resample, trim, write_wav

# Same fixes as src/claudio/tts/kokoro.py: piper's espeak-ng writes the Italian r and the
# affricates differently from the espeak-ng Kokoro was trained on.
PHONEME_FIXES = str.maketrans({"ɾ": "r", "ʲ": None})
AFFRICATES = (("tʃ", "ʧ"), ("dʒ", "ʤ"), ("ts", "ʦ"), ("dz", "ʣ"))
KOKORO_MAX_TOKENS = 510


@dataclass(frozen=True)
class Job:
    text: str
    lang: str  # espeak-ng voice: "it" or "en-us"
    engine: str  # "piper:<voice name>" or "kokoro"
    seed: int
    out: str


class PiperEngine:
    def __init__(self, model: Path) -> None:
        from piper import PiperVoice

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        self.voice = PiperVoice.load(model)
        self.native = self.voice.config.espeak_voice
        self.speakers = self.voice.config.num_speakers

    def __call__(self, text: str, lang: str, rng: np.random.Generator) -> np.ndarray:
        from piper import SynthesisConfig

        # A voice reads with its own language unless told otherwise: an English speaker
        # reading Italian phonemes sounds like an English speaker saying "Claudio".
        self.voice.config.espeak_voice = lang if lang else self.native
        config = SynthesisConfig(
            speaker_id=int(rng.integers(self.speakers)) if self.speakers > 1 else None,
            length_scale=float(rng.uniform(0.55, 1.35)),
            noise_scale=float(rng.uniform(0.3, 1.0)),
            noise_w_scale=float(rng.uniform(0.4, 1.2)),
        )
        chunks = [c.audio_float_array for c in self.voice.synthesize(text, config)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return resample(audio, self.voice.config.sample_rate, SR)


class KokoroEngine:
    def __init__(self, kokoro_dir: Path) -> None:
        import onnxruntime as ort
        from piper.phonemize_espeak import EspeakPhonemizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        model = next(
            p
            for p in (kokoro_dir / "kokoro-v1.0.int8.onnx", kokoro_dir / "kokoro-v1.0.onnx")
            if p.is_file()
        )
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.vocab: dict[str, int] = json.loads((kokoro_dir / "config.json").read_text())["vocab"]
        styles = np.load(kokoro_dir / "voices-v1.0.bin")
        self.styles = {name: styles[name] for name in styles.files}
        self.names = sorted(self.styles)
        self.phonemizer = EspeakPhonemizer()

    def phonemes(self, text: str, lang: str) -> str:
        sentences = self.phonemizer.phonemize(lang, text)
        joined = " ".join(unicodedata.normalize("NFC", "".join(s)) for s in sentences)
        if lang == "it":
            joined = joined.translate(PHONEME_FIXES)
            for pair, tied in AFFRICATES:
                joined = joined.replace(pair, tied)
        return joined

    def __call__(self, text: str, lang: str, rng: np.random.Generator) -> np.ndarray:
        tokens = [self.vocab[p] for p in self.phonemes(text, lang) if p in self.vocab]
        tokens = tokens[:KOKORO_MAX_TOKENS]
        if not tokens:
            return np.zeros(0, dtype=np.float32)
        a, b = rng.choice(self.names, size=2, replace=False)
        w = float(rng.uniform(0.0, 1.0)) if rng.random() < 0.7 else 1.0
        style = w * self.styles[a][len(tokens) - 1] + (1 - w) * self.styles[b][len(tokens) - 1]
        (audio, *_) = self.session.run(
            None,
            {
                "tokens": np.array([[0, *tokens, 0]], dtype=np.int64),
                "style": style.astype(np.float32),
                "speed": np.array([rng.uniform(0.8, 1.6)], dtype=np.float32),
            },
        )
        return resample(np.asarray(audio, dtype=np.float32).ravel(), 24000, SR)


def plan(
    phrases: dict[str, list[str]],
    n: int,
    engines: dict[str, float],
    out_dir: Path,
    seed: int,
) -> list[Job]:
    """n jobs: each picks a language (weighted by its number of phrases), a phrase in it and
    an engine (by the engines' shares). Every engine reads every language: an Italian voice
    reading English sounds like JenJo speaking English."""
    rng = np.random.default_rng(seed)
    langs = [lang for lang, texts in phrases.items() if texts]
    sizes = np.array([len(phrases[lang]) for lang in langs], dtype=float)
    names = list(engines)
    shares = np.array([engines[e] for e in names], dtype=float)
    jobs = []
    for i in range(n):
        lang = langs[int(rng.choice(len(langs), p=sizes / sizes.sum()))]
        text = phrases[lang][int(rng.integers(len(phrases[lang])))]
        engine = names[int(rng.choice(len(names), p=shares / shares.sum()))]
        name = f"{uuid.uuid4().hex}.wav"
        jobs.append(Job(text, lang, engine, seed * 1_000_003 + i, str(out_dir / name)))
    return jobs


_ENGINES: dict[str, Callable[[str, str, np.random.Generator], np.ndarray]] = {}
_PATHS: dict[str, Path] = {}


def _init(paths: dict[str, Path]) -> None:
    _PATHS.update(paths)


def _engine(name: str) -> Callable[[str, str, np.random.Generator], np.ndarray]:
    if name not in _ENGINES:
        if name == "kokoro":
            _ENGINES[name] = KokoroEngine(_PATHS["kokoro"])
        else:
            _ENGINES[name] = PiperEngine(_PATHS[name])
    return _ENGINES[name]


def _run(job: Job) -> bool:
    rng = np.random.default_rng(job.seed)
    try:
        audio = trim(_engine(job.engine)(job.text, job.lang, rng))
    except Exception as exc:  # one bad phrase or voice must not stop thousands of clips
        print(f"  saltata: {job.engine} {job.text!r}: {exc}")
        return False
    if len(audio) < SR // 10:
        return False
    write_wav(Path(job.out), audio)
    return True


def run(jobs: list[Job], paths: dict[str, Path], workers: int) -> int:
    """Synthesize the jobs; returns how many clips were written."""
    if not jobs:
        return 0
    Path(jobs[0].out).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    ctx = get_context("spawn")  # onnxruntime sessions do not survive a fork
    with ctx.Pool(workers, initializer=_init, initargs=(paths,)) as pool:
        for i, ok in enumerate(pool.imap_unordered(_run, jobs, chunksize=8), 1):
            done += ok
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} clip")
    return done


def engine_paths(data: Path) -> dict[str, Path]:
    """The engines whose files are in data/ (piper/*.onnx and kokoro/)."""
    paths = {f"piper:{p.stem}": p for p in sorted((data / "piper").glob("*.onnx"))}
    if (data / "kokoro" / "voices-v1.0.bin").is_file():
        paths["kokoro"] = data / "kokoro"
    return paths


def iter_clips(directory: Path) -> Iterator[Path]:
    yield from sorted(directory.glob("*.wav"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--text", nargs="+", required=True)
    parser.add_argument("--lang", default="it")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    paths = engine_paths(args.data)
    engines: dict[str, Any] = {name: 1.0 for name in paths}
    jobs = plan({args.lang: args.text}, args.n, engines, args.out, seed=0)
    print(f"{run(jobs, paths, args.workers)} clip in {args.out}")


if __name__ == "__main__":
    main()
