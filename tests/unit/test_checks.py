"""--check on a machine without the models: it reports every missing piece, never crashes."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from claudio.checks import kokoro_voices, run_checks
from claudio.config import load_settings


def test_missing_models_fail_the_check(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    settings = load_settings().model_copy(update={"models_dir": tmp_path})
    with caplog.at_level(logging.INFO):
        assert run_checks(settings) is False
    text = caplog.text
    assert "commands.toml" in text or "comandi" in text.lower()  # the router was checked
    assert len(caplog.records) >= 5  # one line per check


def test_kokoro_voices_without_the_folder(tmp_path: Path) -> None:
    assert kokoro_voices(tmp_path / "missing") is None


# --- IT-1: the Whisper model CLAUDIO will load must be downloaded --------------------------


def fake_snapshot(root: Path, repo: str, files: tuple[str, ...]) -> Path:
    """A Hugging Face cache as faster-whisper leaves it in models/whisper (no symlinks)."""
    cache = root / f"models--{repo.replace('/', '--')}"
    revision = "0" * 40
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text(revision, encoding="utf-8")
    folder = cache / "snapshots" / revision
    folder.mkdir(parents=True)
    for name in files:
        (folder / name).write_bytes(b"x" * 10)
    return folder


COMPLETE = ("model.bin", "config.json", "tokenizer.json", "vocabulary.json")


def test_whisper_model_found_without_downloading(tmp_path: Path) -> None:
    from claudio.checks import find_whisper_model

    folder = fake_snapshot(tmp_path, "Systran/faster-whisper-large-v3", COMPLETE)
    assert find_whisper_model("large-v3", tmp_path) == folder


def test_whisper_model_missing_or_half_downloaded(tmp_path: Path) -> None:
    from claudio.checks import find_whisper_model

    with pytest.raises(FileNotFoundError, match="large-v3 non scaricato"):
        find_whisper_model("large-v3", tmp_path)
    fake_snapshot(tmp_path, "Systran/faster-whisper-small", ("config.json", "vocabulary.txt"))
    with pytest.raises(FileNotFoundError, match=r"incompleto.*model\.bin, tokenizer\.json"):
        find_whisper_model("small", tmp_path)
    with pytest.raises(ValueError):  # a typo in [stt] model
        find_whisper_model("large-v9", tmp_path)


def test_whisper_model_in_a_local_folder(tmp_path: Path) -> None:
    from claudio.checks import find_whisper_model

    folder = tmp_path / "my-model"
    folder.mkdir()
    for name in COMPLETE:
        (folder / name).write_bytes(b"x")
    assert find_whisper_model(str(folder), tmp_path / "unused") == folder


@pytest.mark.parametrize(
    ("device", "gpus", "expected"),
    [
        ("auto", 1, ("large-v3", "small")),  # the GPU model, the CPU one in case
        ("auto", 0, ("small", "large-v3")),  # no GPU: Whisper falls back to the CPU model
        ("cuda", 1, ("large-v3", None)),  # never falls back
        ("cpu", 1, ("small", None)),
    ],
)
def test_the_whisper_model_that_will_load(device: str, gpus: int, expected: tuple) -> None:
    from claudio.checks import whisper_models

    stt = load_settings().stt.model_copy(
        update={"device": device, "model": "large-v3", "fallback_model": "small"}
    )
    assert whisper_models(stt, gpus) == expected


def test_check_reports_the_whisper_model(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from claudio.checks import check_whisper

    base = load_settings()
    stt = base.stt.model_copy(
        update={"device": "auto", "model": "large-v3", "fallback_model": "small"}
    )
    settings = base.model_copy(update={"models_dir": tmp_path, "stt": stt})
    results: list[tuple[bool, str]] = []
    with caplog.at_level(logging.WARNING):
        check_whisper(settings, 1, lambda good, what: results.append((good, what)))
    assert results[0][0] is False
    assert "large-v3" in results[0][1] and "download_models.py --whisper large-v3" in results[0][1]
    assert "small" in caplog.text and "senza GPU" in caplog.text  # the fallback: a warning

    fake_snapshot(tmp_path / "whisper", "Systran/faster-whisper-large-v3", COMPLETE)
    results.clear()
    check_whisper(settings, 1, lambda good, what: results.append((good, what)))
    assert results == [(True, "modello Whisper large-v3 (0.0 GB)")]


def test_cuda_without_a_gpu_is_an_error(tmp_path: Path) -> None:
    """device = "cuda" never falls back to the CPU: without a GPU CLAUDIO cannot start."""
    from claudio.checks import check_whisper

    fake_snapshot(tmp_path / "whisper", "Systran/faster-whisper-large-v3", COMPLETE)
    base = load_settings()
    stt = base.stt.model_copy(update={"device": "cuda", "model": "large-v3"})
    settings = base.model_copy(update={"models_dir": tmp_path, "stt": stt})
    results: list[tuple[bool, str]] = []
    check_whisper(settings, 0, lambda good, what: results.append((good, what)))
    assert results[0] == (False, '[stt] device = "cuda" senza GPU CUDA: Whisper non partirà')
    results.clear()
    check_whisper(settings, 1, lambda good, what: results.append((good, what)))
    assert [good for good, _ in results] == [True]


def test_a_broken_faster_whisper_does_not_stop_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from claudio import checks

    def broken(name: str, root: Path) -> Path:
        raise ImportError("DLL load failed while importing _ext")

    monkeypatch.setattr(checks, "find_whisper_model", broken)
    settings = load_settings().model_copy(update={"models_dir": tmp_path})
    results: list[tuple[bool, str]] = []
    checks.check_whisper(settings, 1, lambda good, what: results.append((good, what)))
    assert results[0][0] is False
    assert "DLL load failed" in results[0][1]
