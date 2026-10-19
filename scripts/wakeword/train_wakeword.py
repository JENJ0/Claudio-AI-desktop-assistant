"""Train CLAUDIO's wake word models ("hey_claudio", "claudio") on Colab or any Linux PC.

    python wakeword/train_wakeword.py download --data data
    python wakeword/train_wakeword.py train --model hey_claudio --data data --samples samples

Each stage keeps its output in --work and is skipped when already there, so running the
same command again after a disconnect carries on where it stopped. What the model learns:
- yes: synthetic voices saying the wake word (generate_clips.py) and JenJo's recordings;
- no: similar words and ordinary speech (synthetic and JenJo's), ~2000 hours of varied
  audio (openWakeWord's ACAV100M features), JenJo's room noise.
The report at the end gives, per threshold, how many of JenJo's held-out recordings wake
Claudio and how many false wakes per hour the 11 hours of validation audio cause."""

from __future__ import annotations

import argparse
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
    """JenJo's takes kept out of training to measure the model: every fifth (by name, so
    the split never changes) and every take with a command after the wake word."""
    digest = int(hashlib.md5(path.stem.encode()).hexdigest(), 16)
    return phrases.is_eval_only(path.stem) or digest % 5 == 0


def split_recordings(directory: Path) -> tuple[list[Path], list[Path]]:
    takes = sorted(directory.glob("*.wav")) if directory.is_dir() else []
    return [p for p in takes if not is_eval(p)], [p for p in takes if is_eval(p)]


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
) -> None:
    """Augmented windows of the clips (each repeated `copies` times) -> (N, 16, 96) features."""
    if out.is_file():
        print(f"  {out.name}: già pronte")
        return
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


def train(
    work: Path,
    data: Path,
    model: str,
    steps: int,
    layer: int,
    target_fph: float,
    seed: int,
    lr: float = 1e-4,
) -> dict[str, Any]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feats = work / model / "features"
    pos_train = np.load(feats / "positive_train.npy")
    pos_test = np.load(feats / "positive_test.npy")
    neg_train = np.load(work / "shared" / "negative_train.npy")
    neg_test = np.load(work / "shared" / "negative_test.npy")
    acav = np.load(data / "features" / "acav.npy", mmap_mode="r")
    val = np.load(data / "features" / "validation.npy", mmap_mode="r")
    val_stream = np.asarray(val, dtype=np.float32).reshape(-1, 96)
    val_windows = windows(val_stream)
    val_hours = len(val_stream) * FRAME_S / 3600  # 11.3 h for openWakeWord's file
    print(
        f"  positivi {len(pos_train)}, negativi {len(neg_train)} + {len(acav)} ACAV, "
        f"validazione {val_hours:.1f} h, su {device}"
    )

    net = build_net(layer).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    acav_iter = acav_batches(acav, 1024, rng)
    best: dict[str, Any] = {}

    def evaluate(step: int) -> None:
        net.eval()
        recall = float(np.mean(scores(net, pos_test, device) >= 0.5))
        neg_fp = float(np.mean(scores(net, neg_test, device) >= 0.5))
        fph = false_wakes(scores(net, val_windows, device), 0.5) / val_hours
        net.train()
        ok = fph <= target_fph
        better = (
            not best
            or (ok and (not best["ok"] or recall > best["recall"]))
            or (not ok and not best["ok"] and fph < best["fph"])
        )
        print(
            f"  passo {step}: richiamo {recall:.3f}, falsi/ora {fph:.2f}, "
            f"negativi simili {neg_fp:.3f}{'  <- migliore' if better else ''}",
            flush=True,
        )
        if better:
            best.update(
                step=step,
                recall=recall,
                fph=fph,
                ok=ok,
                state={k: v.detach().clone() for k, v in net.state_dict().items()},
            )

    # Two rounds like openWakeWord's auto_train: a long one, then a short fine-tune at a
    # tenth of the rate. Negatives weigh more and more, which is what cuts false wakes.
    rounds = [(steps, lr, 1500.0), (max(steps // 10, 1), lr / 10, 3000.0)]
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
                    pos_train[rng.integers(len(pos_train), size=64)],
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

    net = net.to("cpu").eval()
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
    eval_takes: list[Path],
    synth_test: list[Path],
    target_fph: float,
) -> dict[str, Any]:
    val = scores(trained["net"], trained["val_windows"], trained["device"])
    real = stream_scores(onnx_path, oww_dir, eval_takes)
    synth = stream_scores(onnx_path, oww_dir, synth_test)
    rows = []
    for t in THRESHOLDS:
        rows.append(
            {
                "threshold": t,
                "jenjo": int(sum(s >= t for s in real)),
                "synthetic": round(float(np.mean([s >= t for s in synth])) if synth else 0.0, 3),
                "false_per_hour": round(false_wakes(val, t) / trained["val_hours"], 2),
            }
        )
    pick = pick_threshold(rows, target_fph)
    missed = [
        (p.name, round(s, 3))
        for p, s in zip(eval_takes, real, strict=True)
        if s < pick["threshold"]
    ]
    result = {
        "model": model,
        "recommended_threshold": pick["threshold"],
        "meets_target": pick["false_per_hour"] <= target_fph,
        "jenjo_takes": len(eval_takes),
        "best_step": trained["best"]["step"],
        "table": rows,
        "missed_at_recommended": missed,
        "peaks": {p.name: round(s, 3) for p, s in zip(eval_takes, real, strict=True)},
    }
    print(
        f"\n{model}: soglia consigliata {pick['threshold']} "
        f"(falsi allarmi/ora al massimo {target_fph})"
    )
    print(" soglia | tue registrazioni | sintetiche | falsi/ora")
    for r in rows:
        mark = "  <-" if r is pick else ""
        print(
            f"  {r['threshold']:.2f} | {r['jenjo']:3d}/{len(eval_takes):<3d}           | "
            f"{r['synthetic']:.3f}      | {r['false_per_hour']:.2f}{mark}"
        )
    if missed:
        print(" mancate alla soglia consigliata: " + ", ".join(f"{n} ({s})" for n, s in missed))
    if not result["meets_target"]:
        print(
            f" ATTENZIONE: nessuna soglia arriva a {target_fph} falsi allarmi/ora: questa è la"
            " meno rumorosa che sente ancora la maggior parte delle registrazioni. Mandami il"
            " report: serve un altro giro di addestramento."
        )
    return result


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


def run_training(args: argparse.Namespace) -> None:
    data, work, samples, model = args.data, args.work, args.samples, args.model
    paths = generate_clips.engine_paths(data)
    if not paths:
        raise SystemExit("Nessuna voce in data/piper o data/kokoro: esegui prima 'download'")
    oww_dir = data / "oww"
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

    real_train, real_eval = split_recordings(samples / model)
    negatives = sorted((samples / "negatives").glob("*.wav"))
    noise = sorted((samples / "noise").glob("*.wav"))
    print(
        f"== tue registrazioni: {len(real_train)} per imparare, {len(real_eval)} per misurare, "
        f"{len(negatives)} frasi normali, {len(noise)} di rumore"
    )
    bg_files = sorted((data / "backgrounds").rglob("*.wav")) + noise
    tails = sorted((shared / "negative_test").glob("*.wav"))[:200]
    print(f"== finestre di 2 s con rumore ed eco ({len(bg_files)} file di sottofondo)")
    feats = work / model / "features"
    feats.mkdir(parents=True, exist_ok=True)
    (work / "shared").mkdir(parents=True, exist_ok=True)
    clips = lambda d, copies: [(p, copies) for p in sorted(d.glob("*.wav"))]  # noqa: E731
    featurize(
        feats / "positive_train.npy",
        clips(synth / "positive_train", 1) + [(p, args.real_copies) for p in real_train],
        True,
        oww_dir,
        bg_files,
        tails,
        args.workers,
        11,
    )
    featurize(
        feats / "positive_test.npy",
        clips(synth / "positive_test", 1),
        True,
        oww_dir,
        bg_files,
        tails,
        args.workers,
        12,
    )
    featurize(
        work / "shared" / "negative_train.npy",
        clips(shared / "negative_train", 1)
        + [(p, 10) for p in negatives]
        + [(p, 100) for p in noise],
        False,
        oww_dir,
        bg_files,
        [],
        args.workers,
        13,
    )
    featurize(
        work / "shared" / "negative_test.npy",
        clips(shared / "negative_test", 1),
        False,
        oww_dir,
        bg_files,
        [],
        args.workers,
        14,
    )

    print(f"== addestramento ({args.steps} passi)")
    trained = train(work, data, model, args.steps, args.layer, args.target_fph, args.seed, args.lr)
    out = args.out / f"{model}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    export(trained["net"], out)
    print(f"== modello salvato: {out}")
    synth_test = sorted((synth / "positive_test").glob("*.wav"))[:300]
    result = report(model, out, trained, oww_dir, real_eval, synth_test, args.target_fph)
    result["settings"] = {k: str(v) for k, v in vars(args).items()}
    (args.out / f"{model}_report.json").write_text(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="scarica dati, voci e modelli")
    dl.add_argument("--data", type=Path, default=Path("data"))
    tr = sub.add_parser("train", help="addestra un modello")
    tr.add_argument("--model", choices=sorted(phrases.POSITIVES), required=True)
    tr.add_argument("--data", type=Path, default=Path("data"))
    tr.add_argument("--samples", type=Path, default=Path("samples"))
    tr.add_argument("--work", type=Path, default=Path("work"))
    tr.add_argument("--out", type=Path, default=Path("output"))
    tr.add_argument("--n-positive", type=int, default=6000)
    tr.add_argument("--n-negative", type=int, default=8000)
    tr.add_argument("--real-copies", type=int, default=25)
    tr.add_argument("--steps", type=int, default=30000)
    tr.add_argument("--layer", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument(
        "--target-fph",
        type=float,
        default=0.5,
        help="falsi allarmi all'ora accettati (0.5 = uno ogni 2 ore)",
    )
    tr.add_argument("--workers", type=int, default=2)
    tr.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "download":
        ok = download(args.data)
        print("Scaricati: " + ", ".join(f"{k} {'OK' if v else 'NO'}" for k, v in ok.items()))
        if not (ok["acav"] and ok["validation"]):
            raise SystemExit("Mancano le feature di openWakeWord: senza non si può addestrare")
    else:
        run_training(args)


if __name__ == "__main__":
    main()
