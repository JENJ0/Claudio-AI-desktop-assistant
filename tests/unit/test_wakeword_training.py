"""The wake word training kit (scripts/wakeword) and the recording script: everything that
runs without torch, voices or downloads. The network itself is tested only where torch is
installed (Colab, a Linux training PC)."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "wakeword"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import augment
import generate_clips
import phrases
import train_wakeword

import record_samples

SR = augment.SR


def tone(seconds: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- phrases ---------------------------------------------------------------------------


def test_every_model_has_positives_and_recording_styles() -> None:
    assert set(phrases.POSITIVES) == set(phrases.RECORDING_STYLES) == {"hey_claudio", "claudio"}
    for styles in phrases.RECORDING_STYLES.values():
        slugs = [slug for slug, _, _ in styles]
        assert len(set(slugs)) == len(slugs)
        assert sum(phrases.is_eval_only(f"x_000_{s}") for s in slugs) == 1  # one command style


def test_negatives_never_contain_the_wake_word() -> None:
    texts = [*phrases.NEGATIVE_PROMPTS]
    for lang in phrases.ADVERSARIAL:
        texts += phrases.ADVERSARIAL[lang] + phrases.SENTENCES[lang]
    assert not [t for t in texts if "claudio" in t.lower()]


def test_weighted_repeats_by_weight() -> None:
    assert phrases.weighted([("a", 2), ("b", 1)]) == ["a", "a", "b"]


def test_recording_plan_groups_styles_in_order() -> None:
    plan = phrases.recording_plan("hey_claudio", 50)
    slugs = [slug for slug, _, _ in plan]
    assert len(plan) == 50
    assert slugs[:5] == ["normale"] * 5 and slugs[5] == "veloce"
    assert slugs[-1] == "comando_cmd"
    short = [slug for slug, _, _ in phrases.recording_plan("claudio", 12)]
    assert short[:3] == ["normale", "normale", "veloce"] and len(set(short)) == 10


# --- augment ---------------------------------------------------------------------------


def test_trim_drops_silence_but_keeps_a_margin() -> None:
    clip = np.concatenate([np.zeros(SR), tone(0.5), np.zeros(SR)]).astype(np.float32)
    trimmed = augment.trim(clip)
    assert 0.5 * SR <= len(trimmed) <= 0.5 * SR + 2 * 0.08 * SR + 2 * SR // 100
    assert len(augment.trim(np.zeros(SR, dtype=np.float32))) == 0


def test_change_speed_and_resample_lengths() -> None:
    clip = tone(1.0)
    assert len(augment.change_speed(clip, 1.25)) == pytest.approx(0.8 * SR, rel=0.02)
    assert len(augment.change_speed(clip, 0.8)) == pytest.approx(1.25 * SR, rel=0.02)
    assert len(augment.resample(tone(1.0), 24000, SR)) == pytest.approx(SR * 2 / 3, abs=1)


def test_set_peak_and_int16_conversion() -> None:
    out = augment.set_peak(tone(0.1), -6.0)
    assert float(np.abs(out).max()) == pytest.approx(10 ** (-6 / 20), rel=1e-4)
    assert augment.to_int16(np.array([2.0, -2.0])).tolist() == [32767, -32767]


def test_mix_hits_the_requested_snr() -> None:
    rng = np.random.default_rng(0)
    fg = tone(1.0)
    noise = augment.coloured_noise(SR, rng)
    mixed = augment.mix(fg, noise, 10.0)
    snr = 10 * np.log10(np.mean(fg**2) / np.mean((mixed - fg) ** 2))
    assert snr == pytest.approx(10.0, abs=0.3)


def test_place_puts_the_wake_word_at_the_end_of_the_window() -> None:
    rng = np.random.default_rng(1)
    clip = tone(0.6)
    for _ in range(20):
        out = augment.place(clip, 2 * SR, rng, at_end=True)
        loud = np.flatnonzero(np.abs(out) > 0.01)
        assert 2 * SR - loud[-1] <= 0.2 * SR + 1
    tail = tone(1.0, freq=200)
    out = augment.place(clip, 2 * SR, rng, at_end=True, end_jitter_s=0.35, tail=tail)
    assert np.abs(out[-100:]).max() > 0  # the command after the wake word fills the window


def test_augment_returns_a_full_int16_window() -> None:
    rng = np.random.default_rng(2)
    backgrounds = augment.Backgrounds([])
    for positive in (True, False):
        out = augment.augment(tone(0.7), 2 * SR, rng, backgrounds, positive, [tone(1.0)])
        assert out.dtype == np.int16 and len(out) == 2 * SR
    silent = augment.augment(np.zeros(SR, dtype=np.float32), 2 * SR, rng, backgrounds, True)
    assert len(silent) == 2 * SR


def test_backgrounds_read_wav_segments(tmp_path: Path) -> None:
    path = tmp_path / "bg.wav"
    augment.write_wav(path, tone(3.0, amp=0.1))
    segment = augment.Backgrounds([path]).segment(SR, np.random.default_rng(3))
    assert len(segment) == SR and np.std(segment) == pytest.approx(1.0, rel=0.01)
    assert len(augment.read_wav(path)) == 3 * SR


# --- generate_clips --------------------------------------------------------------------


def test_plan_follows_languages_and_engine_shares(tmp_path: Path) -> None:
    texts = {"it": ["ciao", "ehi"], "en-us": ["hello"], "fr": []}
    jobs = generate_clips.plan(texts, 300, {"a": 0.75, "b": 0.25, "c": 0.0}, tmp_path, seed=7)
    assert len(jobs) == 300 and len({j.out for j in jobs}) == 300
    assert {j.lang for j in jobs} == {"it", "en-us"}
    assert {j.engine for j in jobs} == {"a", "b"}
    share_a = sum(j.engine == "a" for j in jobs) / len(jobs)
    assert 0.65 < share_a < 0.85
    again = generate_clips.plan(texts, 300, {"a": 0.75, "b": 0.25, "c": 0.0}, tmp_path, seed=7)
    assert [(j.text, j.engine, j.seed) for j in jobs] == [(j.text, j.engine, j.seed) for j in again]


def test_engine_paths_finds_downloaded_voices(tmp_path: Path) -> None:
    (tmp_path / "piper").mkdir()
    (tmp_path / "piper" / "it_IT-paola-medium.onnx").write_bytes(b"x")
    assert generate_clips.engine_paths(tmp_path) == {
        "piper:it_IT-paola-medium": tmp_path / "piper" / "it_IT-paola-medium.onnx"
    }
    (tmp_path / "kokoro").mkdir()
    (tmp_path / "kokoro" / "voices-v1.0.bin").write_bytes(b"x")
    assert "kokoro" in generate_clips.engine_paths(tmp_path)


# --- train_wakeword --------------------------------------------------------------------


def test_eval_split_keeps_command_takes_and_a_fifth_out(tmp_path: Path) -> None:
    takes = [f"hey_claudio_{i:03d}_normale" for i in range(200)]
    held_out = [t for t in takes if train_wakeword.is_eval(Path(t))]
    assert 0.1 < len(held_out) / len(takes) < 0.3
    assert train_wakeword.is_eval(Path("hey_claudio_007_comando_cmd.wav"))
    for name in ("a_001_normale.wav", "a_002_comando_cmd.wav"):
        (tmp_path / name).write_bytes(b"")
    _, evaluate = train_wakeword.split_recordings(tmp_path)
    assert Path(tmp_path / "a_002_comando_cmd.wav") in evaluate
    assert train_wakeword.split_recordings(tmp_path / "missing") == ([], [])


def test_false_wakes_counts_like_the_runtime() -> None:
    scores = np.zeros(200, dtype=np.float32)
    scores[[10, 11, 12, 30, 40, 100]] = 0.9  # 30 and 40 fall in the 25-frame refractory of 10
    assert train_wakeword.false_wakes(scores, 0.5) == 3
    assert train_wakeword.false_wakes(scores, 0.95) == 0


def test_threshold_pick_prefers_recall_within_the_false_wake_target() -> None:
    def row(t, jenjo, synthetic, fph):
        return {"threshold": t, "jenjo": jenjo, "synthetic": synthetic, "false_per_hour": fph}

    rows = [row(0.3, 10, 0.9, 2.0), row(0.5, 10, 0.8, 0.4), row(0.7, 8, 0.7, 0.1)]
    assert train_wakeword.pick_threshold(rows, 0.5)["threshold"] == 0.5
    noisy = [row(0.3, 10, 0.9, 9.0), row(0.5, 9, 0.8, 4.0), row(0.7, 5, 0.5, 1.0)]
    assert train_wakeword.pick_threshold(noisy, 0.5)["threshold"] == 0.5  # 9/10 still heard
    deaf = [row(0.3, 0, 0.2, 9.0), row(0.9, 0, 0.1, 4.0)]
    assert train_wakeword.pick_threshold(deaf, 0.5)["threshold"] == 0.9


def test_windows_and_acav_batches() -> None:
    stream = np.arange(40 * 96, dtype=np.float32).reshape(40, 96)
    w = train_wakeword.windows(stream)
    assert w.shape == (25, 16, 96) and w[3, 0, 0] == stream[3, 0]
    rng = np.random.default_rng(0)
    acav = np.zeros((10, 16, 96), dtype=np.float16)
    batches = train_wakeword.acav_batches(acav, 4, rng)
    assert all(next(batches).shape == (4, 16, 96) for _ in range(5))  # wraps around
    long = np.zeros((10, 32, 96), dtype=np.float16)  # other window lengths are re-cut
    assert next(train_wakeword.acav_batches(long, 2, rng)).shape[1:] == (16, 96)


def test_synthetic_phrases_and_engine_shares() -> None:
    positive = train_wakeword.synth_phrases("claudio", True)
    assert positive["it"].count("claudio") == 8
    negative = train_wakeword.synth_phrases("claudio", False)
    assert "claudia" in negative["it"] and "apri chrome" in negative["it"]
    shares = train_wakeword.engine_shares(
        {"kokoro": Path("k"), "piper:it_IT-paola-medium": Path("p"), "piper:other": Path("o")}
    )
    assert shares == {"kokoro": 0.25, "piper:it_IT-paola-medium": 0.15}


def test_network_exports_to_the_runtime_shape(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    net = train_wakeword.build_net(32)
    assert net(torch.zeros(3, 16, 96)).shape == (3, 1)
    train_wakeword.export(net, tmp_path / "m.onnx")  # checks onnxruntime agrees, shape (1, 1)
    assert (tmp_path / "m.onnx").stat().st_size > 0


# --- record_samples --------------------------------------------------------------------


def test_new_takes_never_overwrite(tmp_path: Path) -> None:
    assert record_samples.next_index(tmp_path, "claudio") == 0
    for name in ("claudio_000.wav", "claudio_007_veloce.wav", "hey_claudio_020_lento.wav"):
        (tmp_path / name).write_bytes(b"")
    assert record_samples.next_index(tmp_path, "claudio") == 8
    prompts = record_samples.wakeword_prompts("Claudio", 10, start=8)
    assert prompts[0][2] == "claudio_008_normale" and prompts[-1][2] == "claudio_017_comando_cmd"
    assert prompts[-1][3] > prompts[0][3]  # a command needs more time


def test_prompts_for_every_mode() -> None:
    plain = record_samples.wakeword_prompts("Ciao Mondo", 3, start=0)
    assert [p[2] for p in plain] == [f"ciao_mondo_{i:03d}_normale" for i in range(3)]
    negatives = record_samples.negative_prompts(5)
    assert len(negatives) == len(phrases.NEGATIVE_PROMPTS) and negatives[0][2] == "neg_005"
    noise = record_samples.noise_prompts(0, 60.0)
    assert [p[2] for p in noise] == ["noise_000_stanza", "noise_001_tv"]
    assert record_samples.min_peak_for("noise_000_tv") == 0.0
    assert record_samples.min_peak_for("claudio_003_piano") < record_samples.min_peak_for("x")


def test_pack_lays_out_what_the_notebook_expects(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    for kind in ("hey_claudio", "negatives"):
        (samples / kind).mkdir(parents=True)
        augment.write_wav(samples / kind / f"{kind}_000.wav", tone(0.1))
    zip_path = samples / "claudio_training.zip"
    counts = record_samples.pack(samples, PROJECT_ROOT / "scripts" / "wakeword", zip_path)
    assert counts == {"hey_claudio": 1, "claudio": 0, "negatives": 1, "noise": 0}
    names = zipfile.ZipFile(zip_path).namelist()
    assert "samples/hey_claudio/hey_claudio_000.wav" in names
    assert {"wakeword/train_wakeword.py", "wakeword/phrases.py"} <= set(names)
    assert "wakeword/train_claudio.ipynb" in names
