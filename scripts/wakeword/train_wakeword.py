"""Train CLAUDIO's wake word models ("hey_claudio", "claudio") on Colab or any Linux PC.

    python wakeword/train_wakeword.py download --data data
    python wakeword/train_wakeword.py train --model hey_claudio --data data --samples samples

Each stage keeps its output (in --work, the models in --out) and is skipped when already
there and made from the same recordings, scripts and options, so running the same command
again after a disconnect carries on where it stopped (a variant already trained included).
What the model learns:
- yes: synthetic voices saying the wake word (generate_clips.py) and JenJo's recordings;
- no: similar words and ordinary speech (synthetic and JenJo's), ~2000 hours of varied
  audio (openWakeWord's ACAV100M features), JenJo's room noise.
The report at the end gives, per threshold, how many of JenJo's held-out recordings wake
Claudio and how many false wakes per hour the 11 hours of validation audio cause."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from collections.abc import Iterator
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import augment
import generate_clips
import numpy as np
import phrases

TOTAL = 32000  # 2 s windows: 16 embedding frames, the input of every openWakeWord model
FRAMES = 16
FRAME_S = 0.08  # one embedding frame every 80 ms
REFRACTORY = 25  # frames (2 s) the runtime ignores after a detection (detector.reset())
THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.96, 0.05)]
# How much a checkpoint's synthetic and its held-out real windows count in its recall.
RECALL_WEIGHTS = (0.3, 0.7)
# The version of the clips, features and training; with a fingerprint of the scripts that make
# them (recipe_id), so older ones in a kept work folder are made again by themselves.
RECIPE = 2
SCRIPTS = Path(__file__).resolve().parent
DATA_SCRIPTS = ("phrases.py", "augment.py", "generate_clips.py")  # what the clips come from
# The first model trained with the "base" settings heard JenJo's own voice badly, so each run
# also tries these and keeps the one that hears most of his held-out takes. A variant is
# openWakeWord's two rounds with another network width and another weight on the negatives.
VARIANTS: dict[str, dict[str, Any]] = {
    "base": {},
    "wide": {"layer": 64},
    "gentle": {"layer": 64, "max_weights": (400.0, 800.0)},
}

HF = "https://huggingface.co/"
OWW_FEATURES = HF + "datasets/davidscripka/openwakeword_features/resolve/main/"
PIPER = HF + "rhasspy/piper-voices/resolve/main/"
PIPER_VOICES = (
    "it/it_IT/paola/medium/it_IT-paola-medium",
    "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low",
    "en/en_US/libritts_r/medium/en_US-libritts_r-medium",
)
KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_VOCAB = (
    "https://raw.githubusercontent.com/thewh1teagle/kokoro-onnx/"
    "3596b26764286a7de9d90c363e988d50578918e5/src/kokoro_onnx/config.json"
)
OWW_MODELS = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
BACKGROUNDS = {  # name: mirrors, tried in order
    "audioset": (
        HF + "datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar",
        HF + "datasets/agkphysics/AudioSet/resolve/main/bal_train09.tar",
    ),
    "esc50": ("https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",),
}
# Kokoro sounds the most natural but makes about 1 clip/s per CPU core, Piper about ten times
# more: a quarter of the clips from Kokoro keeps its 54 voices without doubling Colab's time.
ENGINE_SHARES = {
    "kokoro": 0.25,
    "en_US-libritts_r": 0.45,
    "it_IT-paola": 0.15,
    "it_IT-riccardo": 0.15,
}


# --- downloads -------------------------------------------------------------------------


def fetch(url: str, dest: Path) -> bool:
    """Download once (resuming a partial file with wget when there is one)."""
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  {dest.name} ...", flush=True)
    try:
        if shutil.which("wget"):
            subprocess.run(
                ["wget", "-q", "--show-progress", "-c", "-O", str(part), url], check=True
            )
        else:
            with urllib.request.urlopen(url, timeout=120) as resp, part.open("wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
    except Exception as exc:
        print(f"  NON scaricato {url}: {exc}")
        return False
    part.replace(dest)
    return True


def convert_backgrounds(files: Iterator[Path], out_dir: Path, limit: int) -> int:
    """Any audio soundfile reads -> 16 kHz mono WAV, at most `limit` files."""
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in files:
        if n >= limit:
            break
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            continue
        augment.write_wav(
            out_dir / f"{path.stem}.wav", augment.resample(audio.mean(axis=1), sr, augment.SR)
        )
        n += 1
    return n


def download(data: Path) -> dict[str, bool]:
    """Everything training needs, in data/. Background audio is a bonus: if a source is
    down, coloured noise and speech stand in for it."""
    ok: dict[str, bool] = {}
    print("openWakeWord: 2000 ore di audio vario (17 GB) e 11 ore di validazione")
    ok["acav"] = fetch(
        OWW_FEATURES + "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        data / "features" / "acav.npy",
    )
    ok["validation"] = fetch(
        OWW_FEATURES + "validation_set_features.npy", data / "features" / "validation.npy"
    )
    print("openWakeWord: modelli delle feature")
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        ok[name] = fetch(OWW_MODELS + name, data / "oww" / name)
    print("Voci Piper")
    for voice in PIPER_VOICES:
        stem = voice.rsplit("/", 1)[-1]
        ok[stem] = all(
            fetch(f"{PIPER}{voice}{suffix}", data / "piper" / f"{stem}{suffix}")
            for suffix in (".onnx", ".onnx.json")
        )
    print("Voci Kokoro")
    ok["kokoro"] = all(
        (
            fetch(KOKORO + "kokoro-v1.0.int8.onnx", data / "kokoro" / "kokoro-v1.0.int8.onnx"),
            fetch(KOKORO + "voices-v1.0.bin", data / "kokoro" / "voices-v1.0.bin"),
            fetch(KOKORO_VOCAB, data / "kokoro" / "config.json"),
        )
    )
    print("Rumori e musica di sottofondo")
    bg = data / "backgrounds"
    for name, urls in BACKGROUNDS.items():
        if any((bg / name).glob("*.wav")):
            ok[name] = True
            continue
        archive = data / "downloads" / urls[0].rsplit("/", 1)[-1]
        ok[name] = any(fetch(url, archive) for url in urls)
        if not ok[name]:
            continue
        unpacked = data / "downloads" / name
        try:
            if archive.suffix == ".tar":
                with tarfile.open(archive) as tar:
                    tar.extractall(unpacked, filter="data")
            else:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(unpacked)
        except Exception as exc:
            print(f"  archivio {archive.name} illeggibile: {exc}")
            ok[name] = False
            continue
        files = (p for p in sorted(unpacked.rglob("*")) if p.suffix in (".flac", ".wav", ".mp3"))
        n = convert_backgrounds(files, bg / name, limit=3000)
        print(f"  {name}: {n} file")
        ok[name] = n > 0
        shutil.rmtree(unpacked, ignore_errors=True)
        archive.unlink(missing_ok=True)
    return ok


# --- data ------------------------------------------------------------------------------


def is_eval(path: Path) -> bool:
    return phrases.is_held_out(path.stem)


def split_recordings(directory: Path) -> tuple[list[Path], list[Path]]:
    takes = sorted(directory.glob("*.wav")) if directory.is_dir() else []
    return [p for p in takes if not is_eval(p)], [p for p in takes if is_eval(p)]


def fingerprint(files: list[Path]) -> str:
    """Names and contents of JenJo's recordings (a redone take has the same name and size):
    when they change, the features and the model made from the old ones are made again."""
    digest = hashlib.sha1()
    for f in sorted(files):
        digest.update(f"{f.parent.name}/{f.name}\n".encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()[:16]


def stamp_of(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def code_id(names: tuple[str, ...]) -> str:
    """Fingerprint of scripts (line endings aside: the zip is made on Windows)."""
    digest = hashlib.sha1()
    for name in names:
        digest.update((SCRIPTS / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:8]


def recipe_id() -> str:
    """What the clips and features depend on: RECIPE and the scripts that make them. Changed
    by itself when phrases, augmentation or the voices change."""
    return f"{RECIPE}-{code_id(DATA_SCRIPTS)}"


def model_stamp(
    args: argparse.Namespace, recordings: str, names: list[str], variant: str | None = None
) -> dict[str, Any]:
    """Everything a finished model (or one variant of it) was made from: when any of it
    differs, it is trained again."""
    stamp: dict[str, Any] = {
        "recordings": recordings,
        "recipe": recipe_id(),
        "code": code_id((*DATA_SCRIPTS, "train_wakeword.py")),
    }
    for key in (
        "steps",
        "layer",
        "lr",
        "seed",
        "real_fraction",
        "real_copies",
        "target_fph",
        "n_positive",
        "n_negative",
    ):
        stamp[key] = getattr(args, key, None)
    if variant is None:
        stamp["variants"] = names
    else:
        stamp["variant"] = variant
        stamp["settings"] = {k: str(v) for k, v in VARIANTS[variant].items()}
    return json.loads(json.dumps(stamp))  # as read back from a file


def synth_phrases(model: str, positive: bool) -> dict[str, list[str]]:
    if positive:
        return {lang: phrases.weighted(items) for lang, items in phrases.POSITIVES[model].items()}
    return {
        lang: phrases.ADVERSARIAL[lang] * 3 + phrases.SENTENCES[lang]
        for lang in phrases.ADVERSARIAL
    }


def engine_shares(paths: dict[str, Path]) -> dict[str, float]:
    shares = {}
    for name in paths:
        for key, share in ENGINE_SHARES.items():
            if key in name:
                shares[name] = share
    return shares


def synthesize(
    out: Path, texts: dict[str, list[str]], n: int, paths: dict[str, Path], seed: int, workers: int
) -> None:
    have = len(list(out.glob("*.wav"))) if out.is_dir() else 0
    if have >= 0.95 * n:
        print(f"  {out.name}: {have} clip già pronte")
        return
    jobs = generate_clips.plan(texts, n - have, engine_shares(paths), out, seed + have)
    t0 = time.time()
    done = generate_clips.run(jobs, paths, workers)
    print(f"  {out.name}: {done} clip in {time.time() - t0:.0f} s")


# --- features --------------------------------------------------------------------------

_BG: augment.Backgrounds | None = None
_TAILS: list[np.ndarray] = []


def _init_worker(bg_files: list[str], tail_files: list[str]) -> None:
    global _BG
    _BG = augment.Backgrounds([Path(f) for f in bg_files])
    _TAILS.extend(augment.read_wav(Path(f)) for f in tail_files)


def _augment_one(task: tuple[str, bool, int]) -> np.ndarray:
    path, positive, seed = task
    assert _BG is not None
    rng = np.random.default_rng(seed)
    return augment.augment(augment.read_wav(Path(path)), TOTAL, rng, _BG, positive, _TAILS)


def embedder(oww_dir: Path) -> Any:
    from openwakeword.utils import AudioFeatures

    return AudioFeatures(
        melspec_model_path=str(oww_dir / "melspectrogram.onnx"),
        embedding_model_path=str(oww_dir / "embedding_model.onnx"),
        inference_framework="onnx",
    )


def featurize(
    out: Path,
    clips: list[tuple[Path, int]],
    positive: bool,
    oww_dir: Path,
    bg_files: list[Path],
    tail_files: list[Path],
    workers: int,
    seed: int,
    made_from: str,
) -> None:
    """Augmented windows of the clips (each repeated `copies` times) -> (N, 16, 96) features.
    `made_from` stands for everything they depend on (the clips and takes, the copies, the
    background sounds): saved next to them, they are made again when it changes."""
    stamp = out.with_suffix(".stamp")
    if out.is_file():
        if stamp.is_file() and stamp.read_text() == made_from:
            print(f"  {out.name}: già pronte")
            return
        print(f"  {out.name}: sono cambiate le registrazioni, i clip o le opzioni: le rifaccio")
    tasks = [
        (str(p), positive, seed * 7919 + i)
        for i, (p, _) in enumerate(c for c in clips for _ in range(c[1]))
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(tasks)  # type: ignore[arg-type]
    features = embedder(oww_dir)
    result = np.zeros((len(tasks), FRAMES, 96), dtype=np.float32)
    t0 = time.time()
    batch = 256
    ctx = get_context("spawn")
    with ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=([str(f) for f in bg_files], [str(f) for f in tail_files]),
    ) as pool:
        windows = pool.imap(_augment_one, tasks, chunksize=32)
        for start in range(0, len(tasks), batch):
            chunk = np.stack([next(windows) for _ in range(min(batch, len(tasks) - start))])
            result[start : start + len(chunk)] = features.embed_clips(chunk, batch_size=64)
            if (start // batch) % 10 == 0:
                print(f"  {out.name}: {start + len(chunk)}/{len(tasks)}", flush=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, result)
    tmp.replace(out)
    stamp.write_text(made_from)
    print(f"  {out.name}: {len(tasks)} finestre in {time.time() - t0:.0f} s")


# --- training --------------------------------------------------------------------------


def build_net(layer: int, blocks: int = 1) -> Any:
    """openWakeWord's DNN: the same shape the runtime loads (input 1x16x96, output 1x1)."""
    from torch import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc, self.norm, self.relu = nn.Linear(layer, layer), nn.LayerNorm(layer), nn.ReLU()

        def forward(self, x: Any) -> Any:
            return self.relu(self.norm(self.fc(x)))

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(FRAMES * 96, layer),
        nn.LayerNorm(layer),
        nn.ReLU(),
        *[Block() for _ in range(blocks)],
        nn.Linear(layer, 1),
        nn.Sigmoid(),
    )


def windows(stream: np.ndarray) -> np.ndarray:
    """Every 16-frame window of a (frames, 96) stream, one per 80 ms (a view, no copy)."""
    return np.lib.stride_tricks.sliding_window_view(stream, (FRAMES, 96))[:, 0]


def scores(net: Any, x: np.ndarray, device: str, batch: int = 16384) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(x[i : i + batch], dtype=np.float32))
            out.append(net(chunk.to(device)).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def false_wakes(stream_scores: np.ndarray, threshold: float) -> int:
    """Detections the runtime would make: after each one, REFRACTORY frames are ignored."""
    count, i = 0, 0
    above = np.flatnonzero(stream_scores >= threshold)
    while i < len(above):
        count += 1
        i = int(np.searchsorted(above, above[i] + REFRACTORY))
    return count


def acav_batches(acav: np.ndarray, n: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Consecutive slices from a random start: the 17 GB file is read in order, which is
    what keeps Colab's disk fast enough."""
    rows = len(acav)
    pos = int(rng.integers(rows))
    while True:
        if pos + n > rows:
            pos = 0
        chunk = np.asarray(acav[pos : pos + n], dtype=np.float32)
        pos += n
        if chunk.shape[1] != FRAMES:  # other window lengths: re-cut into 16-frame windows
            flat = chunk.reshape(-1, 96)
            chunk = windows(flat)[::FRAMES]
        yield chunk


def reset_stale(folder: Path) -> None:
    """Scratch made by an older recipe is deleted (only `folder`, never anything above it)."""
    marker = folder / "recipe.txt"
    if folder.is_dir() and (not marker.is_file() or marker.read_text() != recipe_id()):
        print(f"  {folder.parent.name}/{folder.name}: fatto con una versione vecchia, lo rifaccio")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker.write_text(recipe_id())


def positive_batch(
    synthetic: np.ndarray,
    real: np.ndarray,
    rng: np.random.Generator,
    size: int,
    real_fraction: float,
) -> np.ndarray:
    """`size` positive windows, `real_fraction` of them from JenJo's own takes. Drawn evenly
    from everything, the few real windows were lost among thousands of synthetic ones and
    the model learned "synthetic voice" instead of "Claudio"."""
    n_real = round(size * real_fraction) if len(real) else 0
    parts = [synthetic[rng.integers(len(synthetic), size=size - n_real)]]
    if n_real:
        parts.append(real[rng.integers(len(real), size=n_real)])
    return np.concatenate(parts)


def checkpoint_value(
    val: np.ndarray, val_hours: float, synth: np.ndarray, real: np.ndarray, target_fph: float
) -> dict[str, Any]:
    """How good a checkpoint is, judged like pick_threshold judges the final model: at the
    best threshold, not at a fixed 0.5. Within the false wake target the best threshold hears
    the most (synthetic windows and his held-out ones, `RECALL_WEIGHTS`); a checkpoint with no
    threshold within the target is worth its recall cut down by how far over it is, so one that
    hears nothing never wins by being quiet. `val`, `synth`, `real`: window scores."""
    best: dict[str, Any] = {}
    for t in THRESHOLDS:
        fph = false_wakes(val, t) / val_hours
        synth_recall = float(np.mean(synth >= t)) if len(synth) else 0.0
        real_recall = float(np.mean(real >= t)) if len(real) else None
        recall = (
            synth_recall
            if real_recall is None
            else RECALL_WEIGHTS[0] * synth_recall + RECALL_WEIGHTS[1] * real_recall
        )
        within = fph <= target_fph
        ok = within and recall > 0  # a checkpoint that hears nothing is no good, however quiet
        value = recall if within else recall * target_fph / fph
        if not best or (ok, value) > (best["ok"], best["value"]):  # a tie keeps the lowest t
            best = {
                "ok": ok,
                "within": within,
                "value": value,
                "threshold": t,
                "fph": fph,
                "synth_recall": synth_recall,
                "real_recall": real_recall,
            }
    return best


def load_windows(path: Path) -> np.ndarray:
    return np.load(path) if path.is_file() else np.zeros((0, FRAMES, 96), dtype=np.float32)


def train(
    work: Path,
    data: Path,
    model: str,
    steps: int,
    layer: int,
    target_fph: float,
    seed: int,
    lr: float = 1e-4,
    real_fraction: float = 0.4,
    max_weights: tuple[float, float] = (1500.0, 3000.0),
) -> dict[str, Any]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feats = work / model / "features"
    pos_synth = np.load(feats / "positive_synth_train.npy")
    pos_real = load_windows(feats / "positive_real_train.npy")  # JenJo's takes
    pos_test = np.load(feats / "positive_test.npy")
    real_test = load_windows(feats / "positive_real_test.npy")
    neg_train = np.load(work / "shared" / "negative_train.npy")
    neg_test = np.load(work / "shared" / "negative_test.npy")
    acav = np.load(data / "features" / "acav.npy", mmap_mode="r")
    val = np.load(data / "features" / "validation.npy", mmap_mode="r")
    val_stream = np.asarray(val, dtype=np.float32).reshape(-1, 96)
    val_windows = windows(val_stream)
    val_hours = len(val_stream) * FRAME_S / 3600  # 11.3 h for openWakeWord's file
    print(
        f"  positivi {len(pos_synth)} sintetici + {len(pos_real)} tuoi, "
        f"negativi {len(neg_train)} + {len(acav)} ACAV, validazione {val_hours:.1f} h, su {device}"
    )

    net = build_net(layer).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    acav_iter = acav_batches(acav, 1024, rng)
    best: dict[str, Any] = {}

    def evaluate(step: int) -> None:
        net.eval()
        # The checkpoint kept must hear JenJo, not only the synthetic voices: his held-out
        # takes weigh most when there are any.
        now = checkpoint_value(
            scores(net, val_windows, device),
            val_hours,
            scores(net, pos_test, device),
            scores(net, real_test, device) if len(real_test) else np.zeros(0),
            target_fph,
        )
        neg_fp = float(np.mean(scores(net, neg_test, device) >= now["threshold"]))
        net.train()
        better = not best or (now["ok"], now["value"]) > (best["ok"], best["value"])
        heard = (
            "" if now["real_recall"] is None else f", tue registrazioni {now['real_recall']:.3f}"
        )
        print(
            f"  passo {step}: soglia {now['threshold']:.2f}, richiamo sintetico "
            f"{now['synth_recall']:.3f}{heard}, falsi/ora {now['fph']:.2f}"
            f"{'' if now['within'] else ' (sopra il limite)'}, negativi simili {neg_fp:.3f}"
            f"{'  <- migliore' if better else ''}",
            flush=True,
        )
        if better:
            best.update(
                now,
                step=step,
                state={k: v.detach().clone() for k, v in net.state_dict().items()},
            )

    # Two rounds like openWakeWord's auto_train: a long one, then a short fine-tune at a
    # tenth of the rate. Negatives weigh more and more, which is what cuts false wakes.
    rounds = [(steps, lr, max_weights[0]), (max(steps // 10, 1), lr / 10, max_weights[1])]
    t0 = time.time()
    for r, (n_steps, top_lr, max_weight) in enumerate(rounds):
        checks = set(
            np.linspace(n_steps * (0.75 if r == 0 else 0.1), n_steps - 1, 12 if r == 0 else 6)
            .astype(int)
            .tolist()
        )
        warmup, hold = n_steps // 5, n_steps // 3
        for step in range(n_steps):
            if step < warmup:
                rate = top_lr * step / max(warmup, 1)
            elif step < warmup + hold:
                rate = top_lr
            else:
                rate = (
                    0.5
                    * top_lr
                    * (1 + np.cos(np.pi * (step - warmup - hold) / max(n_steps - warmup - hold, 1)))
                )
            for group in optim.param_groups:
                group["lr"] = rate
            x = np.concatenate(
                (
                    positive_batch(pos_synth, pos_real, rng, 64, real_fraction),
                    neg_train[rng.integers(len(neg_train), size=64)],
                    next(acav_iter),
                )
            )
            y = np.zeros(len(x), dtype=np.float32)
            y[:64] = 1
            xt = torch.from_numpy(x).to(device)
            yt = torch.from_numpy(y).to(device)
            p = net(xt).squeeze(-1)
            # Only what the model still gets wrong teaches it something.
            keep = ((yt == 0) & (p >= 0.001)) | ((yt == 1) & (p < 0.999))
            if keep.sum() > 0:
                weight = torch.where(yt == 1, 1.0, 1 + (max_weight - 1) * step / n_steps)
                loss = torch.nn.functional.binary_cross_entropy(p[keep], yt[keep], weight[keep])
                optim.zero_grad()
                loss.backward()
                optim.step()
            if step in checks:
                evaluate(step if r == 0 else steps + step)
        print(f"  giro {r + 1} finito in {time.time() - t0:.0f} s")
    net.load_state_dict(best["state"])
    net.eval()
    return {
        "net": net,
        "device": device,
        "val_windows": val_windows,
        "val_hours": val_hours,
        "best": best,
    }


def export(net: Any, path: Path) -> None:
    import onnxruntime as ort
    import torch

    net = copy.deepcopy(net).to("cpu").eval()  # the caller's copy stays on its device
    dummy = torch.rand(1, FRAMES, 96)
    kwargs: dict[str, Any] = {"opset_version": 13, "input_names": ["x"], "output_names": ["score"]}
    try:
        torch.onnx.export(net, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # torch without the dynamo switch
        torch.onnx.export(net, dummy, str(path), **kwargs)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})[0]
    want = net(dummy).detach().numpy()
    if got.shape != (1, 1) or not np.allclose(got, want, atol=1e-4):
        raise RuntimeError(f"ONNX diverso da PyTorch: {got} contro {want}")


def stream_scores(onnx_path: Path, oww_dir: Path, clips: list[Path]) -> list[float]:
    """Peak score of each clip through the real streaming detector, as Claudio runs it."""
    from openwakeword.model import Model

    detector = Model(
        wakeword_models=[str(onnx_path)],
        inference_framework="onnx",
        melspec_model_path=str(oww_dir / "melspectrogram.onnx"),
        embedding_model_path=str(oww_dir / "embedding_model.onnx"),
    )
    peaks = []
    silence = np.zeros(augment.SR, dtype=np.int16)
    for clip in clips:
        audio = np.concatenate((silence, augment.to_int16(augment.read_wav(clip)), silence))
        detector.reset()
        for _ in range(25):
            detector.predict(np.zeros(1280, dtype=np.int16))
        peak = 0.0
        for i in range(0, len(audio) - 1279, 1280):
            peak = max(peak, float(next(iter(detector.predict(audio[i : i + 1280]).values()))))
        peaks.append(peak)
    return peaks


def report(
    model: str,
    onnx_path: Path,
    trained: dict[str, Any],
    oww_dir: Path,
    takes: dict[str, list[Path]],
    synth_test: list[Path],
    target_fph: float,
    said: dict[str, str] | None = None,
) -> dict[str, Any]:
    """`takes` holds JenJo's recordings by role: "eval" (kept out of training, what the table
    counts), "train" (what the model learned from) and "negatives" (his ordinary sentences,
    `said`: the sentence of each by file stem). The held-out takes also choose the checkpoint,
    the variant and the threshold, so the table is a little rosier than a new take will be."""
    val = scores(trained["net"], trained["val_windows"], trained["device"])
    clips = [*takes["eval"], *takes["train"], *takes["negatives"], *synth_test]
    peaks = stream_scores(onnx_path, oww_dir, clips)
    cuts = np.cumsum([len(takes["eval"]), len(takes["train"]), len(takes["negatives"])])
    real, learned, spoken, synth = np.split(np.array(peaks), cuts)
    rows = []
    for t in THRESHOLDS:
        rows.append(
            {
                "threshold": t,
                "jenjo": int(np.sum(real >= t)),
                "synthetic": round(float(np.mean(synth >= t)) if len(synth) else 0.0, 3),
                "false_per_hour": round(false_wakes(val, t) / trained["val_hours"], 2),
            }
        )
    pick = pick_threshold(rows, target_fph)
    cut = pick["threshold"]
    names = {role: [p.name for p in paths] for role, paths in takes.items()}
    missed = [
        (name, round(float(s), 3)) for name, s in zip(names["eval"], real, strict=True) if s < cut
    ]
    said = said or {}
    loud = sorted(zip(takes["negatives"], spoken, strict=True), key=lambda x: -x[1])[:5]
    problems = []
    if len(real) and pick["jenjo"] == 0:
        problems.append("non sente nessuna delle tue registrazioni di prova")
    if len(synth) and pick["synthetic"] < 0.5:
        problems.append("sente meno della metà delle voci sintetiche di prova")
    learned_heard = int(np.sum(learned >= cut))
    if len(learned) and learned_heard < len(learned) / 2:
        problems.append("non riconosce nemmeno la metà delle registrazioni da cui ha imparato")
    result = {
        "model": model,
        "recommended_threshold": cut,
        "meets_target": pick["false_per_hour"] <= target_fph,
        "healthy": not problems,
        "problems": problems,
        "jenjo_takes": len(takes["eval"]),
        "best_step": trained["best"]["step"],
        "table": rows,
        "missed_at_recommended": missed,
        "peaks": {n: round(float(s), 3) for n, s in zip(names["eval"], real, strict=True)},
        # Told apart on purpose: takes it learned from but does not hear mean the training
        # failed; takes it hears only after learning them mean it does not generalise.
        "learned_takes": len(takes["train"]),
        "learned_heard_at_recommended": learned_heard,
        "learned_peaks": {
            n: round(float(s), 3) for n, s in zip(names["train"], learned, strict=True)
        },
        "sentences_waking_at_recommended": int(np.sum(spoken >= cut)),
        "loudest_sentences": [
            {"file": p.name, "said": said.get(p.stem, ""), "peak": round(float(s), 3)}
            for p, s in loud
        ],
    }
    print(f"\n{model}: soglia consigliata {cut} (falsi allarmi/ora al massimo {target_fph})")
    print(" soglia | tue registrazioni di prova | sintetiche | falsi/ora")
    for r in rows:
        mark = "  <-" if r is pick else ""
        print(
            f"  {r['threshold']:.2f} | {r['jenjo']:3d}/{len(takes['eval']):<3d}                  | "
            f"{r['synthetic']:.3f}      | {r['false_per_hour']:.2f}{mark}"
        )
    print(
        f" registrazioni usate per imparare e sentite alla soglia: "
        f"{result['learned_heard_at_recommended']}/{len(takes['train'])}"
    )
    if len(takes["negatives"]):
        print(
            f" frasi normali che scattano alla soglia: {result['sentences_waking_at_recommended']}"
            f"/{len(takes['negatives'])}; le più alte: "
            + ", ".join(
                f'"{x["said"] or x["file"]}" ({x["peak"]})' for x in result["loudest_sentences"][:3]
            )
        )
    if missed:
        print(" mancate alla soglia consigliata: " + ", ".join(f"{n} ({s})" for n, s in missed))
    if not result["meets_target"]:
        print(
            f" ATTENZIONE: nessuna soglia arriva a {target_fph} falsi allarmi/ora: questa è la"
            " meno rumorosa che sente ancora la maggior parte delle registrazioni. Mandami il"
            " report: serve un altro giro di addestramento."
        )
    for problem in problems:
        print(f" ATTENZIONE: il modello {problem}: l'addestramento non è riuscito.")
    return result


def recommended_row(result: dict[str, Any]) -> dict[str, Any]:
    return next(r for r in result["table"] if r["threshold"] == result["recommended_threshold"])


def variant_rank(result: dict[str, Any]) -> tuple[bool, bool, float, float, float]:
    """Which variant is better: one that works first, then one within the false wake target
    (the most of JenJo's held-out takes heard at its own recommended threshold, then the
    synthetic ones); when none is within it, the fewest false wakes, then the takes heard."""
    row = recommended_row(result)
    if result["meets_target"]:
        quality = (row["jenjo"], row["synthetic"], -row["false_per_hour"])
    else:
        quality = (-row["false_per_hour"], row["jenjo"], row["synthetic"])
    return (result["healthy"], result["meets_target"], *quality)


def pick_threshold(rows: list[dict[str, Any]], target_fph: float) -> dict[str, Any]:
    """Among the thresholds within the false wake target: the one that hears the most of
    JenJo's takes (then synthetic ones), lowest first. If none is within the target: the
    fewest false wakes that still hears at least 80% of the takes, else the strictest."""
    ok = [r for r in rows if r["false_per_hour"] <= target_fph]
    if ok:
        return max(ok, key=lambda r: (r["jenjo"], r["synthetic"], -r["threshold"]))
    most = max(r["jenjo"] for r in rows)
    usable = [r for r in rows if r["jenjo"] >= 0.8 * most and (most or r["synthetic"] >= 0.8)]
    if usable:
        return min(usable, key=lambda r: (r["false_per_hour"], r["threshold"]))
    return rows[-1]


# --- the whole run -----------------------------------------------------------------------


REQUIRED = (
    "features/acav.npy",
    "features/validation.npy",
    "oww/melspectrogram.onnx",
    "oww/embedding_model.onnx",
)


def check_options(args: argparse.Namespace) -> list[str]:
    """The variants to train, from --variants; any option that cannot work stops the run here,
    before hours of synthesis."""
    names = list(dict.fromkeys(v.strip() for v in args.variants.split(",") if v.strip()))
    if not names:
        raise SystemExit(f"--variants è vuoto (ci sono {', '.join(VARIANTS)})")
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(
            f"Varianti sconosciute: {', '.join(unknown)} (ci sono {', '.join(VARIANTS)})"
        )
    if not 0 <= args.real_fraction <= 1:
        raise SystemExit("--real-fraction va da 0 a 1")
    if args.real_copies < 1 or args.steps < 10 or args.target_fph < 0:
        raise SystemExit("--real-copies, --steps e --target-fph non possono essere così")
    return names


def run_training(args: argparse.Namespace) -> None:
    data, work, samples, model = args.data, args.work, args.samples, args.model
    missing = [name for name in REQUIRED if not (data / name).is_file()]
    if missing:
        raise SystemExit(f"Mancano {', '.join(missing)} in {data}: esegui prima 'download'")
    paths = generate_clips.engine_paths(data)
    if not paths:
        raise SystemExit("Nessuna voce in data/piper o data/kokoro: esegui prima 'download'")
    names = check_options(args)
    real_train, real_eval = split_recordings(samples / model)
    negatives, _ = phrases.usable_negatives(samples / "negatives")  # not the "Claudia" ones
    said = phrases.negative_texts(samples / "negatives")
    noise = sorted((samples / "noise").glob("*.wav"))
    recordings = fingerprint(real_train + real_eval + negatives + noise)
    stamp = model_stamp(args, recordings, names)
    report_path = args.out / f"{model}_report.json"
    done = json.loads(report_path.read_text()) if report_path.is_file() else {}
    finished = (args.out / f"{model}.onnx").is_file() and done.get("stamp") == stamp
    if not args.force and finished and not done.get("failed_variants"):
        print(f"== {model}: modello già pronto in {args.out} (--force per rifarlo)")
        return
    oww_dir = data / "oww"
    # Before any synthesis: the clips and features an older recipe made chose phrases and takes
    # differently.
    reset_stale(work / "shared")
    reset_stale(work / model / "features")
    reset_stale(work / model / "synthetic")
    print(f"== {model}: voci sintetiche ({', '.join(paths)})")
    synth = work / model / "synthetic"
    shared = work / "shared" / "synthetic"
    synthesize(
        synth / "positive_train",
        synth_phrases(model, True),
        args.n_positive,
        paths,
        1,
        args.workers,
    )
    synthesize(
        synth / "positive_test",
        synth_phrases(model, True),
        args.n_positive // 10,
        paths,
        2,
        args.workers,
    )
    synthesize(
        shared / "negative_train",
        synth_phrases(model, False),
        args.n_negative,
        paths,
        3,
        args.workers,
    )
    synthesize(
        shared / "negative_test",
        synth_phrases(model, False),
        args.n_negative // 10,
        paths,
        4,
        args.workers,
    )

    print(
        f"== tue registrazioni: {len(real_train)} per imparare, {len(real_eval)} per misurare, "
        f"{len(negatives)} frasi normali, {len(noise)} di rumore"
    )
    bg_files = sorted((data / "backgrounds").rglob("*.wav")) + noise
    tails = sorted((shared / "negative_test").glob("*.wav"))[:200]
    print(f"== finestre di 2 s con rumore ed eco ({len(bg_files)} file di sottofondo)")
    feats = work / model / "features"
    clips = lambda d, copies: [(p, copies) for p in sorted(d.glob("*.wav"))]  # noqa: E731
    # A take with a command glued after the wake word measures the streaming detector only:
    # its last 2 s do not end with the wake word, so it is no window to score.
    real_windows = [p for p in real_eval if not phrases.is_eval_only(p.stem)]
    for name, wanted in (("positive_real_train", real_train), ("positive_real_test", real_windows)):
        if not wanted:
            (feats / f"{name}.npy").unlink(missing_ok=True)
    # What every file of windows depends on besides its own clips: the background sounds
    # (the downloaded ones and his noise recordings), the command tails, the scripts.
    sounds = stamp_of(recipe_id(), len(bg_files), fingerprint(noise))
    with_tails = stamp_of(sounds, len(tails))
    pos_train, pos_test = clips(synth / "positive_train", 1), clips(synth / "positive_test", 1)
    neg_train, neg_test = clips(shared / "negative_train", 1), clips(shared / "negative_test", 1)
    featurize(
        feats / "positive_synth_train.npy",
        pos_train,
        True,
        oww_dir,
        bg_files,
        tails,
        args.workers,
        11,
        stamp_of(with_tails, len(pos_train)),
    )
    if real_train:
        featurize(
            feats / "positive_real_train.npy",
            [(p, args.real_copies) for p in real_train],
            True,
            oww_dir,
            bg_files,
            tails,
            args.workers,
            16,
            stamp_of(with_tails, fingerprint(real_train), args.real_copies),
        )
    featurize(
        feats / "positive_test.npy",
        pos_test,
        True,
        oww_dir,
        bg_files,
        tails,
        args.workers,
        12,
        stamp_of(with_tails, len(pos_test)),
    )
    if real_windows:
        featurize(
            feats / "positive_real_test.npy",
            [(p, 8) for p in real_windows],
            True,
            oww_dir,
            bg_files,
            tails,
            args.workers,
            15,
            stamp_of(with_tails, fingerprint(real_windows), 8),
        )
    featurize(
        work / "shared" / "negative_train.npy",
        neg_train + [(p, 10) for p in negatives] + [(p, 100) for p in noise],
        False,
        oww_dir,
        bg_files,
        [],
        args.workers,
        13,
        stamp_of(sounds, len(neg_train), fingerprint(negatives + noise)),
    )
    featurize(
        work / "shared" / "negative_test.npy",
        neg_test,
        False,
        oww_dir,
        bg_files,
        [],
        args.workers,
        14,
        stamp_of(sounds, len(neg_test)),
    )

    synth_test = sorted((synth / "positive_test").glob("*.wav"))[:300]
    takes = {"eval": real_eval, "train": real_train, "negatives": negatives}
    candidates = work / model / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}
    for name in names:
        settings = {"layer": args.layer, "max_weights": (1500.0, 3000.0)} | VARIANTS[name]
        onnx_path, saved = candidates / f"{model}.{name}.onnx", candidates / f"{model}.{name}.json"
        made_from = model_stamp(args, recordings, names, name)
        if onnx_path.is_file() and saved.is_file():  # a disconnect cost no finished variant
            kept = json.loads(saved.read_text())
            if kept.get("stamp") == made_from:
                print(f'== "{name}": già addestrata, la riprendo')
                results[name] = kept["result"]
                continue
        print(f'== addestramento "{name}" ({args.steps} passi, rete {settings["layer"]})')
        try:
            trained = train(
                work,
                data,
                model,
                args.steps,
                settings["layer"],
                args.target_fph,
                args.seed,
                args.lr,
                args.real_fraction,
                settings["max_weights"],
            )
            export(trained["net"], onnx_path)
            result = report(
                model, onnx_path, trained, oww_dir, takes, synth_test, args.target_fph, said
            )
        except Exception as error:  # the other variants may still give a model
            failed[name] = f"{type(error).__name__}: {error}"
            print(f'== "{name}" non è riuscita: {failed[name]}')
            continue
        result["settings"] = {k: str(v) for k, v in settings.items()}
        results[name] = result
        saved.write_text(json.dumps({"stamp": made_from, "result": result}, indent=1))
    if not results:
        raise SystemExit("Nessuna variante è arrivata in fondo: " + "; ".join(failed.values()))
    best = max(results, key=lambda n: variant_rank(results[n]))
    if len(results) > 1:
        print(f'\n== scelta: "{best}" (la migliore sulle tue registrazioni di prova)')
    out = args.out / f"{model}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidates / f"{model}.{best}.onnx", out)
    print(f"== modello salvato: {out}")
    result = results[best]
    result["variant"] = best
    result["variants"] = {
        n: {
            "recommended_threshold": r["recommended_threshold"],
            "meets_target": r["meets_target"],
            "healthy": r["healthy"],
            "jenjo_heard": recommended_row(r)["jenjo"],
            "learned_heard": r["learned_heard_at_recommended"],
            "settings": r["settings"],
        }
        for n, r in results.items()
    }
    result["failed_variants"] = failed
    result["args"] = {k: str(v) for k, v in vars(args).items()}
    result["recordings"] = recordings
    result["recipe"] = recipe_id()
    result["stamp"] = stamp
    report_path.write_text(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="scarica dati, voci e modelli")
    dl.add_argument("--data", type=Path, default=Path("data"))
    sub.add_parser("version", help="la versione di questi script (cambia se cambiano)")
    tr = sub.add_parser("train", help="addestra un modello")
    tr.add_argument("--model", choices=sorted(phrases.POSITIVES), required=True)
    tr.add_argument("--data", type=Path, default=Path("data"))
    tr.add_argument("--samples", type=Path, default=Path("samples"))
    tr.add_argument("--work", type=Path, default=Path("work"))
    tr.add_argument("--out", type=Path, default=Path("output"))
    tr.add_argument("--n-positive", type=int, default=6000)
    tr.add_argument("--n-negative", type=int, default=8000)
    tr.add_argument("--real-copies", type=int, default=60, help="augmented windows per take")
    tr.add_argument(
        "--real-fraction",
        type=float,
        default=0.4,
        help="share of each batch's positives that come from your own takes",
    )
    tr.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"settings to try, the best is kept ({', '.join(VARIANTS)})",
    )
    tr.add_argument("--steps", type=int, default=30000)
    tr.add_argument("--layer", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument(
        "--target-fph",
        type=float,
        default=1.0,
        help="falsi allarmi all'ora accettati sulle ore di validazione (1.0 = uno all'ora)",
    )
    tr.add_argument("--workers", type=int, default=2)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--force", action="store_true", help="rifà anche un modello già pronto")
    args = parser.parse_args(argv)
    if args.command == "version":
        every = (*DATA_SCRIPTS, "train_wakeword.py")
        print(f"Script di addestramento: ricetta {recipe_id()}, codice {code_id(every)}")
    elif args.command == "download":
        ok = download(args.data)
        print("Scaricati: " + ", ".join(f"{k} {'OK' if v else 'NO'}" for k, v in ok.items()))
        needed = ("acav", "validation", "melspectrogram.onnx", "embedding_model.onnx")
        if not all(ok[name] for name in needed):
            raise SystemExit("Mancano i dati di openWakeWord (righe NO): così non si addestra")
    else:
        run_training(args)


if __name__ == "__main__":
    main()
