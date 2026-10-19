"""The wake word training kit (scripts/wakeword) and the recording script: everything that
runs without torch, voices or downloads. The network itself is tested only where torch is
installed (Colab, a Linux training PC)."""

from __future__ import annotations

import argparse
import json
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


def test_no_positive_is_the_start_of_a_negative() -> None:
    """'claudi' was a positive and 'claudia' (and the English 'cloudy') negatives: the same
    sounds with both labels teach the model nothing but confusion."""

    def letters(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalpha() or ch == " ").strip()

    negatives = [letters(t) for lang in phrases.ADVERSARIAL for t in phrases.ADVERSARIAL[lang]]
    for model in phrases.POSITIVES:
        for items in phrases.POSITIVES[model].values():
            for text, _ in items:
                assert not [n for n in negatives if n.startswith(letters(text))], text


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


def test_trim_finds_a_quiet_take_above_the_room_noise() -> None:
    """A quiet or far take: the room noise is within 35 dB of the voice, so a threshold
    relative to the peak alone kept the whole 3 s and the wake word was cut later."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(3 * SR).astype(np.float32) * 10 ** (-60 / 20)
    take = noise.copy()
    take[int(0.4 * SR) : int(1.3 * SR)] += tone(0.9, amp=10 ** (-30 / 20))
    trimmed = augment.trim(take)
    assert 0.9 * SR <= len(trimmed) < 1.2 * SR


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


def test_reverb_ranges_from_desk_to_across_the_room() -> None:
    rng = np.random.default_rng(4)
    drr = []
    for _ in range(200):
        rir = augment.synthetic_rir(rng)
        drr.append(10 * np.log10(rir[0] ** 2 / float(np.sum(rir[1:] ** 2))))
    assert min(drr) > -10.1 and 5 < max(drr) < 12.1  # some as dry as a desk mic


def test_write_wav_leaves_no_partial_file(tmp_path: Path) -> None:
    augment.write_wav(tmp_path / "a.wav", tone(0.1))
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


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


def test_engine_paths_finds_only_complete_voices(tmp_path: Path) -> None:
    piper = tmp_path / "piper"
    piper.mkdir()
    (piper / "it_IT-paola-medium.onnx").write_bytes(b"x")
    assert generate_clips.engine_paths(tmp_path) == {}  # its .onnx.json is missing
    (piper / "it_IT-paola-medium.onnx.json").write_text("{}")
    assert generate_clips.engine_paths(tmp_path) == {
        "piper:it_IT-paola-medium": piper / "it_IT-paola-medium.onnx"
    }
    kokoro = tmp_path / "kokoro"
    kokoro.mkdir()
    for name in ("voices-v1.0.bin", "kokoro-v1.0.int8.onnx"):
        (kokoro / name).write_bytes(b"x")
    assert "kokoro" not in generate_clips.engine_paths(tmp_path)  # no config.json yet
    (kokoro / "config.json").write_text("{}")
    assert "kokoro" in generate_clips.engine_paths(tmp_path)


def test_a_voice_that_fails_to_load_is_not_loaded_again(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = []

    def broken(path: Path) -> None:
        loads.append(path)
        raise OSError("file mancante")

    monkeypatch.setattr(generate_clips, "PiperEngine", broken)
    monkeypatch.setattr(generate_clips, "_PATHS", {"piper:x": Path("x.onnx")})
    monkeypatch.setattr(generate_clips, "_BROKEN", {})
    for _ in range(3):
        with pytest.raises(Exception, match="file mancante"):
            generate_clips._engine("piper:x")
    assert len(loads) == 1


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


def training_args(tmp_path: Path, **extra: object) -> argparse.Namespace:
    values = {
        "data": tmp_path / "data",
        "work": tmp_path / "work",
        "samples": tmp_path / "samples",
        "out": tmp_path / "output",
        "model": "claudio",
        "force": False,
        "n_positive": 10,
        "n_negative": 10,
        "workers": 1,
        "variants": "base",
        "steps": 100,
        "layer": 32,
        "lr": 1e-4,
        "seed": 0,
        "real_fraction": 0.4,
        "real_copies": 60,
        "target_fph": 1.0,
    }
    values.update(extra)
    return argparse.Namespace(**values)


def test_training_stops_before_synthesis_when_data_is_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"oww/embedding_model\.onnx"):
        train_wakeword.run_training(training_args(tmp_path))


def test_an_unknown_variant_is_refused_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="gentel"):
        train_wakeword.run_training(with_training_data(tmp_path, variants="base,gentel"))


@pytest.mark.parametrize(
    "options",
    [{"variants": " , "}, {"real_fraction": 1.5}, {"real_copies": 0}, {"steps": 3}],
)
def test_options_that_cannot_work_are_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict[str, object]
) -> None:
    def synthesize(*_: object) -> None:
        raise AssertionError("troppo tardi")

    monkeypatch.setattr(train_wakeword, "synthesize", synthesize)
    with pytest.raises(SystemExit):
        train_wakeword.run_training(with_training_data(tmp_path, **options))


def test_the_same_variant_twice_is_trained_once() -> None:
    args = training_args(Path("."), variants="base, wide,base")
    assert train_wakeword.check_options(args) == ["base", "wide"]


def test_download_fails_without_the_feature_models(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = {"acav": True, "validation": True, "melspectrogram.onnx": True}
    monkeypatch.setattr(
        train_wakeword, "download", lambda data: ok | {"embedding_model.onnx": False}
    )
    with pytest.raises(SystemExit):
        train_wakeword.main(["download"])


def with_training_data(tmp_path: Path, **extra: object) -> argparse.Namespace:
    """Arguments for run_training with the downloaded files faked (empty but present)."""
    args = training_args(tmp_path, **extra)
    for name in train_wakeword.REQUIRED:
        (args.data / name).parent.mkdir(parents=True, exist_ok=True)
        (args.data / name).write_bytes(b"x")
    (args.data / "piper").mkdir(exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        (args.data / "piper" / f"it_IT-paola-medium{suffix}").write_bytes(b"x")
    return args


def test_a_finished_model_is_skipped_until_anything_it_was_made_from_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = with_training_data(tmp_path)
    takes = args.samples / "claudio"
    takes.mkdir(parents=True)
    augment.write_wav(takes / "claudio_000_normale.wav", tone(0.5))
    recordings = train_wakeword.fingerprint([takes / "claudio_000_normale.wav"])
    args.out.mkdir()
    (args.out / "claudio.onnx").write_bytes(b"x")
    report = args.out / "claudio_report.json"
    stamp = train_wakeword.model_stamp(args, recordings, ["base"])
    report.write_text(json.dumps({"stamp": stamp}))

    def synthesize(*_: object) -> None:
        raise AssertionError("ricomincia da capo")

    monkeypatch.setattr(train_wakeword, "synthesize", synthesize)
    train_wakeword.run_training(args)  # nothing to do
    for change in (
        {"steps": 200},  # an option
        {"target_fph": 0.5},
        {"variants": "base,wide"},
    ):
        with pytest.raises(AssertionError, match="ricomincia"):
            train_wakeword.run_training(with_training_data(tmp_path, **change))
    augment.write_wav(takes / "claudio_001_veloce.wav", tone(0.4))
    with pytest.raises(AssertionError, match="ricomincia"):
        train_wakeword.run_training(args)  # a new take: train again
    (takes / "claudio_001_veloce.wav").unlink()
    with pytest.raises(AssertionError, match="ricomincia"):
        train_wakeword.run_training(training_args(tmp_path, force=True))
    monkeypatch.setattr(train_wakeword, "code_id", lambda names: "other-scripts")
    with pytest.raises(AssertionError, match="ricomincia"):
        train_wakeword.run_training(args)  # other scripts: train again
    monkeypatch.undo()
    monkeypatch.setattr(train_wakeword, "synthesize", synthesize)
    report.write_text(json.dumps({"stamp": stamp, "failed_variants": {"wide": "boom"}}))
    with pytest.raises(AssertionError, match="ricomincia"):
        train_wakeword.run_training(args)  # a variant failed: try it again
    report.write_text(json.dumps({"recordings": recordings}))
    with pytest.raises(AssertionError, match="ricomincia"):
        train_wakeword.run_training(args)  # made before stamps existed: train again


def stub_the_heavy_stages(monkeypatch: pytest.MonkeyPatch, failing: set[int]) -> list[int]:
    """run_training with no synthesis, features or torch: train() just records the network
    width it was asked for (and raises for the widths in `failing`)."""
    asked: list[int] = []

    def train(work: Path, data: Path, model: str, steps: int, layer: int, *_: object) -> dict:
        asked.append(layer)
        if layer in failing:
            raise RuntimeError("boom")
        return {"net": None, "best": {"step": 1}}

    def report(*_: object) -> dict[str, object]:
        row = {"threshold": 0.5, "jenjo": 3, "synthetic": 0.9, "false_per_hour": 0.1}
        return {
            "recommended_threshold": 0.5,
            "meets_target": True,
            "healthy": True,
            "table": [row],
            "learned_heard_at_recommended": 0,
        }

    monkeypatch.setattr(train_wakeword, "synthesize", lambda *_: None)
    monkeypatch.setattr(train_wakeword, "featurize", lambda *_: None)
    monkeypatch.setattr(train_wakeword, "train", train)
    monkeypatch.setattr(train_wakeword, "export", lambda _net, path: path.write_bytes(b"onnx"))
    monkeypatch.setattr(train_wakeword, "report", report)
    return asked


def test_a_variant_already_trained_is_kept_and_a_failing_one_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = with_training_data(tmp_path, variants="base,wide")
    asked = stub_the_heavy_stages(monkeypatch, failing={64})
    train_wakeword.run_training(args)
    assert asked == [32, 64]  # "wide" failed, "base" gave the model
    saved = json.loads((args.out / "claudio_report.json").read_text())
    assert saved["variant"] == "base" and list(saved["failed_variants"]) == ["wide"]
    assert saved["variants"]["base"]["settings"]["layer"] == "32"
    assert (args.out / "claudio.onnx").is_file()
    # a disconnect or a failure costs only what was not finished
    asked = stub_the_heavy_stages(monkeypatch, failing=set())  # nothing fails now
    train_wakeword.run_training(args)
    assert asked == [64]
    saved = json.loads((args.out / "claudio_report.json").read_text())
    assert saved["failed_variants"] == {} and set(saved["variants"]) == {"base", "wide"}
    assert saved["variants"]["wide"]["settings"]["layer"] == "64"  # the winner's is not copied
    asked.clear()
    train_wakeword.run_training(args)  # finished, nothing failed: skipped
    assert asked == []
    train_wakeword.run_training(with_training_data(tmp_path, variants="base,wide", steps=300))
    assert asked == [32, 64]  # other options: every variant again


def test_training_stops_when_every_variant_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_the_heavy_stages(monkeypatch, failing={32})
    with pytest.raises(SystemExit, match="boom"):
        train_wakeword.run_training(with_training_data(tmp_path, variants="base"))
    assert not (tmp_path / "output" / "claudio.onnx").exists()


def test_old_scratch_is_cleared_before_the_first_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clips and features of an older recipe (no recipe marker) must be gone before anything
    new is made: kept, the "Claudia" negatives would be trained on again."""
    args = with_training_data(tmp_path)
    old = [
        args.work / "shared" / "synthetic" / "negative_train" / "old.wav",
        args.work / "claudio" / "features" / "positive_train.npy",
        args.work / "claudio" / "synthetic" / "positive_train" / "old.wav",
    ]
    for path in old:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"old")
    seen: list[bool] = []

    def synthesize(*_: object) -> None:
        seen.append(any(path.exists() for path in old))
        raise AssertionError("stop here")

    monkeypatch.setattr(train_wakeword, "synthesize", synthesize)
    with pytest.raises(AssertionError, match="stop here"):
        train_wakeword.run_training(args)
    assert seen == [False]


def test_features_are_made_again_when_what_they_come_from_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "positive_train.npy"
    np.save(out, np.zeros((1, 16, 96), dtype=np.float32))
    out.with_suffix(".stamp").write_text("abc")

    def embedder(_: Path) -> None:
        raise AssertionError("rifatte")

    monkeypatch.setattr(train_wakeword, "embedder", embedder)
    featurize = train_wakeword.featurize
    featurize(out, [], True, tmp_path, [], [], 1, 0, "abc")  # same takes, sounds, copies: kept
    with pytest.raises(AssertionError, match="rifatte"):
        featurize(out, [], True, tmp_path, [], [], 1, 0, "def")
    out.with_suffix(".stamp").unlink()  # made before stamps existed
    with pytest.raises(AssertionError, match="rifatte"):
        featurize(out, [], True, tmp_path, [], [], 1, 0, "abc")


def test_a_new_noise_recording_or_copy_count_changes_the_stamps() -> None:
    assert train_wakeword.stamp_of("a", 1) == train_wakeword.stamp_of("a", 1)
    assert train_wakeword.stamp_of("a", 60) != train_wakeword.stamp_of("a", 3)


def test_fingerprint_follows_names_and_contents(tmp_path: Path) -> None:
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    augment.write_wav(a, tone(0.1))
    augment.write_wav(b, tone(0.1))
    first = train_wakeword.fingerprint([a, b])
    assert train_wakeword.fingerprint([b, a]) == first
    augment.write_wav(b, tone(0.1, freq=300))  # a redone take: same name and size
    assert train_wakeword.fingerprint([a, b]) != first
    assert train_wakeword.fingerprint([a]) != first


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


def test_claudia_may_wake_claudio_and_no_synthetic_negative_says_it() -> None:
    """JenJo does not mind "Claudia" waking CLAUDIO, and refusing it taught the single-word
    model to refuse "Claudio" in his own voice too."""
    texts = [*phrases.NEGATIVE_PROMPTS]
    for lang in phrases.ADVERSARIAL:
        texts += phrases.ADVERSARIAL[lang] + phrases.SENTENCES[lang]
    assert not [t for t in texts if phrases.mentions_claudia(t)]


def test_the_sentences_to_record_are_the_first_forty_without_the_two_with_claudia() -> None:
    legacy = phrases.LEGACY_NEGATIVE_PROMPTS
    assert len(legacy) == 40 and len(phrases.NEGATIVE_PROMPTS) == 38
    assert legacy[7] == "ho parlato con claudia ieri sera"  # row 8 of 40
    assert legacy[30] == "Claudia, mi apri la porta?"  # row 31 of 40
    assert [t for t in legacy if not phrases.mentions_claudia(t)] == phrases.NEGATIVE_PROMPTS
    assert phrases.NEGATIVE_PROMPTS[7] == "il mio amico paolo arriva domani"  # neg_008 on disk
    assert not phrases.mentions_claudia("audio") and phrases.mentions_claudia("CLAUDIA, vieni")


def test_a_take_of_a_sentence_with_claudia_is_left_out_however_it_was_recorded(
    tmp_path: Path,
) -> None:
    for i in (6, 7, 8, 30, 39):  # the first version's folder: numbers follow its list
        augment.write_wav(tmp_path / f"neg_{i:03d}.wav", tone(0.1))
    kept, left_out = phrases.usable_negatives(tmp_path)
    assert [p.stem for p in left_out] == ["neg_007", "neg_030"]
    assert [p.stem for p in kept] == ["neg_006", "neg_008", "neg_039"]
    # the script writes what each take says: that decides, not the number
    phrases.remember_text(tmp_path, "neg_040", "ciao Claudia")
    phrases.remember_text(tmp_path, "neg_041", "Apri Chrome")
    for i in (40, 41):
        augment.write_wav(tmp_path / f"neg_{i:03d}.wav", tone(0.1))
    texts = phrases.negative_texts(tmp_path)
    assert texts["neg_040"] == "ciao Claudia" and texts["neg_006"] == "fa freddo qui dentro"
    _, left_out = phrases.usable_negatives(tmp_path)
    assert [p.stem for p in left_out] == ["neg_007", "neg_030", "neg_040"]
    assert phrases.usable_negatives(tmp_path / "missing") == ([], [])


def test_positive_batch_takes_a_share_from_his_own_takes() -> None:
    synthetic = np.zeros((500, 16, 96), dtype=np.float32)
    real = np.ones((20, 16, 96), dtype=np.float32)
    rng = np.random.default_rng(0)
    batch = train_wakeword.positive_batch(synthetic, real, rng, 64, 0.4)
    assert batch.shape == (64, 16, 96) and int(batch[:, 0, 0].sum()) == 26  # round(64 * 0.4)
    no_real = train_wakeword.load_windows(Path("missing.npy"))
    assert no_real.shape == (0, 16, 96)
    assert train_wakeword.positive_batch(synthetic, no_real, rng, 64, 0.4).shape[0] == 64
    assert train_wakeword.positive_batch(synthetic, real, rng, 64, 0.0)[:, 0, 0].sum() == 0


def test_stale_scratch_is_rebuilt_only_when_made_by_another_recipe(tmp_path: Path) -> None:
    folder = tmp_path / "work" / "shared"
    folder.mkdir(parents=True)
    (folder / "negative_train.npy").write_bytes(b"old")
    (tmp_path / "work" / "keep.txt").write_text("outside the folder")
    train_wakeword.reset_stale(folder)  # no marker: from before the recipe existed
    assert not (folder / "negative_train.npy").exists()
    assert (tmp_path / "work" / "keep.txt").exists()
    (folder / "negative_train.npy").write_bytes(b"new")
    train_wakeword.reset_stale(folder)  # same recipe: kept
    assert (folder / "negative_train.npy").read_bytes() == b"new"
    (folder / "recipe.txt").write_text("1-abcdef01")  # made by another recipe or other scripts
    train_wakeword.reset_stale(folder)
    assert not (folder / "negative_train.npy").exists()
    assert (folder / "recipe.txt").read_text() == train_wakeword.recipe_id()


def test_the_recipe_follows_the_scripts_the_clips_come_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in train_wakeword.DATA_SCRIPTS:
        (tmp_path / name).write_bytes(b"a = 1\r\n")
    monkeypatch.setattr(train_wakeword, "SCRIPTS", tmp_path)
    first = train_wakeword.recipe_id()
    (tmp_path / "phrases.py").write_bytes(b"a = 1\n")  # Windows line endings: the same code
    assert train_wakeword.recipe_id() == first
    (tmp_path / "phrases.py").write_bytes(b"a = 2\n")
    assert train_wakeword.recipe_id() != first
    assert first.startswith(f"{train_wakeword.RECIPE}-")


def variant(
    meets: bool, heard: int, synthetic: float, fph: float, healthy: bool = True
) -> dict[str, object]:
    row = {"threshold": 0.5, "jenjo": heard, "synthetic": synthetic, "false_per_hour": fph}
    return {
        "recommended_threshold": 0.5,
        "meets_target": meets,
        "healthy": healthy,
        "table": [row],
    }


def test_variant_rank_prefers_the_target_then_the_takes_he_is_heard_on() -> None:
    rank = train_wakeword.variant_rank
    assert rank(variant(True, 3, 0.5, 0.9)) > rank(variant(False, 10, 0.9, 0.1))
    assert rank(variant(True, 8, 0.5, 0.9)) > rank(variant(True, 3, 0.9, 0.1))
    assert rank(variant(True, 8, 0.9, 0.9)) > rank(variant(True, 8, 0.5, 0.1))
    assert rank(variant(True, 8, 0.9, 0.1)) > rank(variant(True, 8, 0.9, 0.9))


def test_variant_rank_with_none_within_the_target_prefers_the_quietest() -> None:
    rank = train_wakeword.variant_rank
    assert rank(variant(False, 7, 0.9, 0.4)) > rank(variant(False, 12, 0.9, 6.0))
    assert rank(variant(False, 12, 0.9, 0.4)) > rank(variant(False, 7, 0.9, 0.4))


def test_a_model_that_works_beats_a_silent_one_however_quiet() -> None:
    rank = train_wakeword.variant_rank
    assert rank(variant(False, 5, 0.9, 20.0)) > rank(variant(True, 0, 0.0, 0.0, healthy=False))


def noisy_validation() -> np.ndarray:
    """An hour of stream: four bursts at 0.6, far enough apart to be four false wakes."""
    val = np.zeros(200, dtype=np.float32)
    val[[10, 60, 110, 160]] = 0.6
    return val


def test_checkpoint_is_judged_at_its_best_threshold_not_at_a_fixed_one() -> None:
    value = train_wakeword.checkpoint_value
    hears = np.full(10, 0.9, dtype=np.float32)
    got = value(noisy_validation(), 1.0, hears, hears, 1.0)
    # 4 false wakes an hour up to 0.6, none above: it hears everything from 0.65
    assert got["ok"] and got["threshold"] == 0.65 and got["value"] == pytest.approx(1.0)
    deaf = np.zeros(10, dtype=np.float32)
    assert (got["ok"], got["value"]) > (value(np.zeros(200), 1.0, deaf, deaf, 1.0)["ok"], 0.0)


def test_a_checkpoint_that_hears_nothing_loses_to_one_that_hears_but_is_noisy() -> None:
    value = train_wakeword.checkpoint_value
    loud = np.full(200, 0.99, dtype=np.float32)
    loud[::30] = 0.0  # a false wake every other stretch, at every threshold
    hears = np.full(10, 0.99, dtype=np.float32)
    noisy = value(loud, 1.0, hears, hears, 1.0)
    quiet = np.zeros(10, dtype=np.float32)
    silent = value(np.zeros(200, dtype=np.float32), 1.0, quiet, quiet, 1.0)
    assert not noisy["within"] and not silent["ok"]
    assert (noisy["ok"], noisy["value"]) > (silent["ok"], silent["value"])


def test_a_checkpoint_hearing_him_beats_one_hearing_only_synthetic_voices() -> None:
    value = train_wakeword.checkpoint_value
    val = np.zeros(200, dtype=np.float32)
    synth = np.full(10, 0.9, dtype=np.float32)
    mine = np.full(10, 0.9, dtype=np.float32)
    deaf_to_him = np.zeros(10, dtype=np.float32)
    both = value(val, 1.0, synth, mine, 1.0)
    only_synth = value(val, 1.0, synth, deaf_to_him, 1.0)
    assert both["value"] > only_synth["value"] and only_synth["value"] == pytest.approx(0.3)
    no_real = value(val, 1.0, synth, np.zeros(0, dtype=np.float32), 1.0)
    assert no_real["real_recall"] is None and no_real["value"] == pytest.approx(1.0)


def fake_trained(tmp_path: Path) -> dict[str, object]:
    return {
        "net": None,
        "device": "cpu",
        "val_hours": 1.0,
        "val_windows": np.zeros((100, 16, 96)),
        "best": {"step": 7},
    }


def run_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peaks: dict[str, float]
) -> dict[str, object]:
    takes = {
        "eval": [tmp_path / "c_e1.wav", tmp_path / "c_e2.wav"],
        "train": [tmp_path / "c_t1.wav", tmp_path / "c_t2.wav", tmp_path / "c_t3.wav"],
        "negatives": [tmp_path / "n1.wav", tmp_path / "n2.wav"],
    }
    synth = [tmp_path / "s1.wav", tmp_path / "s2.wav"]
    monkeypatch.setattr(
        train_wakeword,
        "stream_scores",
        lambda _o, _w, clips: [peaks.get(c.stem, 0.05) for c in clips],
    )
    val = np.zeros(100, dtype=np.float32)
    val[[10, 50, 90]] = 0.6  # three false wakes in the hour, up to threshold 0.6
    monkeypatch.setattr(train_wakeword, "scores", lambda *_: val)
    said = {"n1": "una frase normale"}
    return train_wakeword.report(
        "claudio", tmp_path / "m.onnx", fake_trained(tmp_path), tmp_path, takes, synth, 1.0, said
    )


def test_report_tells_learned_takes_from_held_out_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    peaks = {"c_e1": 0.9, "c_e2": 0.3, "c_t1": 0.9, "c_t2": 0.05, "c_t3": 0.8, "n1": 0.7}
    peaks |= {"s1": 0.9, "s2": 0.7}
    result = run_report(tmp_path, monkeypatch, peaks)
    assert result["learned_takes"] == 3 and result["learned_heard_at_recommended"] == 2
    assert result["sentences_waking_at_recommended"] == 1
    assert result["loudest_sentences"][0] == {
        "file": "n1.wav",
        "said": "una frase normale",
        "peak": 0.7,
    }
    assert result["peaks"] == {"c_e1.wav": 0.9, "c_e2.wav": 0.3}
    # three false wakes an hour at 0.6 and below, over the limit of one: the first threshold
    # within it is 0.65, which still hears one of his two held-out takes
    assert result["recommended_threshold"] == 0.65 and result["meets_target"]
    assert result["healthy"] and result["problems"] == []
    json.dumps(result)  # it is saved as JSON


def test_report_says_when_the_model_hears_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = run_report(tmp_path, monkeypatch, {})  # every clip scores 0.05
    assert not result["healthy"] and len(result["problems"]) == 3
    assert "l'addestramento non è riuscito" in capsys.readouterr().out
    quiet = train_wakeword.variant_rank(result)
    assert quiet[0] is False  # a working variant always beats it


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
    assert "apri chrome" in negative["it"] and "audio" in negative["it"]
    shares = train_wakeword.engine_shares(
        {"kokoro": Path("k"), "piper:it_IT-paola-medium": Path("p"), "piper:other": Path("o")}
    )
    assert shares == {"kokoro": 0.25, "piper:it_IT-paola-medium": 0.15}


def test_network_exports_to_the_runtime_shape(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    net = train_wakeword.build_net(32)
    assert net(torch.zeros(3, 16, 96)).shape == (3, 1)
    before = next(net.parameters())
    train_wakeword.export(net, tmp_path / "m.onnx")  # checks onnxruntime agrees, shape (1, 1)
    assert (tmp_path / "m.onnx").stat().st_size > 0
    assert next(net.parameters()) is before  # exported from a copy: the net stays put


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


def test_pack_leaves_out_the_takes_of_sentences_with_claudia_and_says_what_each_says(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    samples = tmp_path / "samples"
    (samples / "negatives").mkdir(parents=True)
    for i in (6, 7, 8, 30):  # the first version's numbering: 7 and 30 say "Claudia"
        augment.write_wav(samples / "negatives" / f"neg_{i:03d}.wav", tone(0.1))
    zip_path = samples / "claudio_training.zip"
    counts = record_samples.pack(samples, PROJECT_ROOT / "scripts" / "wakeword", zip_path)
    assert counts["negatives"] == 2
    zf = zipfile.ZipFile(zip_path)
    wavs = [n for n in zf.namelist() if n.startswith("samples/negatives/") and n.endswith(".wav")]
    assert sorted(wavs) == [f"samples/negatives/neg_{i:03d}.wav" for i in (6, 8)]
    said = json.loads(zf.read("samples/negatives/prompts.json"))
    assert said == {
        "neg_006": "fa freddo qui dentro",
        "neg_008": "il mio amico paolo arriva domani",
    }
    out = capsys.readouterr().out
    assert "neg_007.wav" in out and "neg_030.wav" in out and "Claudia" in out


def test_each_negative_take_remembers_its_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = iter(["", ""])
    monkeypatch.setattr("builtins.input", lambda _="": next(replies))
    monkeypatch.setattr(
        record_samples, "record", lambda seconds, device: np.full(SR // 10, 8000, np.int16)
    )
    prompts = record_samples.negative_prompts(40)[:2]
    labels = record_samples.negative_labels(40)
    record_samples.session(prompts[:1], tmp_path, None, labels)
    assert phrases.negative_texts(tmp_path)["neg_040"] == phrases.NEGATIVE_PROMPTS[0]
    assert (tmp_path / phrases.PROMPTS_FILE).is_file()


def test_recording_only_some_styles() -> None:
    plan = phrases.recording_plan("hey_claudio", 8, ("veloce", "ehi"))
    assert [slug for slug, _, _ in plan] == ["veloce"] * 4 + ["ehi"] * 4
    prompts = record_samples.wakeword_prompts("Claudio", 6, 10, ("distratto",))
    assert [p[2] for p in prompts] == [f"claudio_{10 + i:03d}_distratto" for i in range(6)]


def run_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answers: list[str], count: int = 2
) -> tuple[int, list[str]]:
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _="": next(replies))
    monkeypatch.setattr(
        record_samples, "record", lambda seconds, device: np.full(SR // 10, 8000, np.int16)
    )
    prompts = [("prova", f"frase {i}", f"neg_{i:03d}", 0.1) for i in range(count)]
    saved = record_samples.session(prompts, tmp_path, None)
    return saved, sorted(p.stem for p in tmp_path.glob("*.wav"))


def test_session_can_redo_the_last_take(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # r on the first prompt (nothing to redo), two takes, redo the last one, done.
    saved, files = run_session(monkeypatch, tmp_path, ["r", "", "", "r", ""])
    assert saved == 2 and files == ["neg_000", "neg_001"]  # a redo is not a new clip


def test_session_stops_on_q(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved, files = run_session(monkeypatch, tmp_path, ["", "q"], count=3)
    assert saved == 1 and files == ["neg_000"]


def test_command_takes_never_overwrite(tmp_path: Path) -> None:
    class Command:
        def __init__(self, id_: str, phrase: str) -> None:
            self.id, self.phrase = id_, phrase

        def phrases_in(self, language: str) -> list[str]:
            return [self.phrase] if language == "it" else []

    commands = [Command("open_chrome", "apri chrome"), Command("mute", "muto")]
    (tmp_path / "open_chrome__apri_chrome__1.wav").write_bytes(b"")
    prompts = record_samples.command_prompts(commands, "it", 2, 3.5, tmp_path)
    assert [p[2] for p in prompts] == [
        "open_chrome__apri_chrome__2",
        "open_chrome__apri_chrome__3",
        "mute__muto__0",
        "mute__muto__1",
    ]
    assert record_samples.command_prompts(commands, "en", 2, 3.5, tmp_path) == []
