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
