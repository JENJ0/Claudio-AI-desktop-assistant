"""The supervisor behind the start at login: CLAUDIO starts again after a crash, never after
a clean close, and not forever when it crashes at every start."""

from __future__ import annotations

import logging
import sys
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from claudio import supervisor
from claudio.single_instance import EXIT_ALREADY_RUNNING
from claudio.supervisor import child_command, run_child, supervise

ACCESS_VIOLATION = 0xC0000005  # how Windows reports a crash in C code (a CUDA driver...)


class Runs:
    """Plays back exit codes; each run takes `took` seconds of a fake clock."""

    def __init__(self, codes: Sequence[int], took: float = 1.0) -> None:
        self.codes = list(codes)
        self.took = took
        self.now = 0.0
        self.calls = 0
        self.slept: list[float] = []

    def run(self, command: Sequence[str]) -> int:
        self.calls += 1
        self.now += self.took
        return self.codes.pop(0)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def go(runs: Runs) -> int:
    return supervise(["claudio"], run=runs.run, sleep=runs.sleep, clock=runs.clock)


@pytest.mark.parametrize("code", [0, 2, EXIT_ALREADY_RUNNING])
def test_a_clean_close_ends_the_supervisor(code: int) -> None:
    """0: "chiuditi" or the tray menu; 2: configuration error; 3: another CLAUDIO."""
    runs = Runs([code])
    assert go(runs) == code
    assert runs.calls == 1
    assert runs.slept == []


@pytest.mark.parametrize("crash", [1, ACCESS_VIOLATION])
def test_a_crash_is_restarted(crash: int, caplog: pytest.LogCaptureFixture) -> None:
    runs = Runs([crash, 0])
    with caplog.at_level(logging.WARNING):
        assert go(runs) == 0
    assert runs.calls == 2
    assert runs.slept == [5.0]
    assert "lo riavvio tra 5 s" in caplog.text


def test_a_crash_at_every_start_gives_up(caplog: pytest.LogCaptureFixture) -> None:
    runs = Runs([1] * 10)
    with caplog.at_level(logging.ERROR):
        assert go(runs) == 1
    assert runs.calls == 4  # the first start and 3 restarts
    assert runs.slept == [5.0, 30.0, 60.0]  # waits longer each time
    assert "non lo riavvio più" in caplog.text


def test_crashes_far_apart_are_always_restarted() -> None:
    """One crash a day for a week: each is the first in its 10 minutes."""
    runs = Runs([1] * 7 + [0], took=86_400)
    assert go(runs) == 0
    assert runs.calls == 8
    assert runs.slept == [5.0] * 7


def test_claudio_runs_with_the_same_python_and_arguments() -> None:
    assert child_command(["--config", "x.toml"]) == [
        sys.executable,
        "-m",
        "claudio",
        "--config",
        "x.toml",
    ]


def test_real_processes_crash_then_close(tmp_path: Path) -> None:
    """Two real Python processes: the first exits with 1 (a crash), the second with 0."""
    marker = tmp_path / "runs.txt"
    script = tmp_path / "child.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import pathlib, sys
            marker = pathlib.Path({str(marker)!r})
            runs = int(marker.read_text()) + 1 if marker.exists() else 1
            marker.write_text(str(runs))
            sys.exit(1 if runs == 1 else 0)
            """
        ),
        encoding="utf-8",
    )
    code = supervise([sys.executable, str(script)], run=run_child, delays=(0.0,))
    assert code == 0
    assert marker.read_text() == "2"


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in saved:
        root.addHandler(handler)
    root.setLevel(level)


def test_main_passes_the_arguments_and_logs_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_logging: None
) -> None:
    config = tmp_path / "c.toml"
    logs = tmp_path / "logs"
    config.write_text(f'logs_dir = "{logs.as_posix()}"\n', encoding="utf-8")
    seen: list[Sequence[str]] = []
    monkeypatch.setattr(supervisor, "supervise", lambda command: seen.append(command) or 0)
    assert supervisor.main(["--config", str(config)]) == 0
    assert seen == [[sys.executable, "-m", "claudio", "--config", str(config)]]
    assert (logs / supervisor.LOG_FILE).exists()  # not claudio.log.jsonl: CLAUDIO rotates it


def test_main_with_a_broken_configuration_still_runs_claudio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_logging: None
) -> None:
    """CLAUDIO itself reports the error (exit code 2), and the supervisor stops on it."""
    config = tmp_path / "broken.toml"
    config.write_text("[stt]\nbeam_size = 0\n", encoding="utf-8")
    monkeypatch.setattr(supervisor, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "supervise", lambda command: 2)
    assert supervisor.main(["--config", str(config)]) == 2
    assert (tmp_path / "logs" / supervisor.LOG_FILE).exists()
