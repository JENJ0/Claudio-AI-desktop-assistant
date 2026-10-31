"""One CLAUDIO at a time: the named mutex, in this process and across processes. Every test
uses a name of its own, so a CLAUDIO running on the PC is never disturbed."""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from claudio import single_instance
from claudio.single_instance import SingleInstance

SRC = str(Path(single_instance.__file__).resolve().parents[1])
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows mutex")


@pytest.fixture
def name() -> str:
    return f"Local\\CLAUDIO-test-{uuid.uuid4().hex}"


def try_in_another_process(name: str, then: str = "") -> subprocess.CompletedProcess[str]:
    """A second Python asks for the name: prints 1 if it got it, 0 if not; `then` runs after."""
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {SRC!r})
        from claudio.single_instance import SingleInstance
        lock = SingleInstance({name!r})
        print(int(lock.acquire()), flush=True)
        {then}
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False
    )


@windows_only
def test_the_second_one_is_refused_until_the_first_ends(name: str) -> None:
    first, second = SingleInstance(name), SingleInstance(name)
    assert first.acquire() is True
    assert first.held
    assert second.acquire() is False
    assert not second.held
    first.release()
    assert second.acquire() is True
    second.release()


@windows_only
def test_acquire_twice_and_release_twice_are_harmless(name: str) -> None:
    lock = SingleInstance(name)
    assert lock.acquire() and lock.acquire()
    lock.release()
    lock.release()
    with SingleInstance(name) as alone:
        assert alone is True


@windows_only
def test_another_process_is_refused(name: str) -> None:
    with SingleInstance(name) as alone:
        assert alone
        assert try_in_another_process(name).stdout.strip() == "0"
    assert try_in_another_process(name).stdout.strip() == "1"


@windows_only
def test_a_crash_never_leaves_the_name_taken(name: str) -> None:
    """os._exit skips every cleanup, like a crash: Windows closes the handle anyway."""
    crashed = try_in_another_process(name, then="os._exit(1)")
    assert crashed.stdout.strip() == "1" and crashed.returncode == 1
    with SingleInstance(name) as alone:
        assert alone is True


@windows_only
def test_held_by_an_administrator_is_taken_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CLAUDIO started as administrator: CreateMutexW cannot open its mutex and says
    ERROR_ACCESS_DENIED. That is "already running" (code 3), not a crash (code 1)."""
    monkeypatch.setattr(single_instance, "_create_mutex", lambda name: (0, 5))
    assert SingleInstance("x").acquire() is False
    monkeypatch.setattr(single_instance, "_create_mutex", lambda name: (0, 87))
    with pytest.raises(OSError):  # any other failure is a real error
        SingleInstance("x").acquire()


def test_outside_windows_there_is_no_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(single_instance.sys, "platform", "linux")
    lock = SingleInstance("whatever")
    assert lock.acquire() is True
    lock.release()


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


def test_a_second_claudio_closes_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_logging: None
) -> None:
    """Before any model loads, and without a session folder of its own."""
    from claudio.__main__ import main
    from claudio.single_instance import EXIT_ALREADY_RUNNING

    logs = tmp_path / "logs"
    config = tmp_path / "c.toml"
    config.write_text(f'logs_dir = "{logs.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setattr(SingleInstance, "acquire", lambda self: False)
    assert main(["--config", str(config)]) == EXIT_ALREADY_RUNNING
    assert not (logs / "sessions").exists()
    assert "già acceso" in (logs / "claudio.log.jsonl").read_text(encoding="utf-8")


def test_an_unexpected_error_reaches_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_logging: None
) -> None:
    """Under pythonw there is no console: the traceback must be in the log file, and the
    exit code 1 tells the supervisor it was a crash."""
    from claudio import __main__ as entry

    async def broken(*args: object) -> None:
        raise RuntimeError("CUDA failed to initialize")

    logs = tmp_path / "logs"
    config = tmp_path / "c.toml"
    config.write_text(f'logs_dir = "{logs.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setattr(SingleInstance, "acquire", lambda self: True)
    monkeypatch.setattr(entry, "run_assistant", broken)
    assert entry.main(["--config", str(config)]) == 1
    text = (logs / "claudio.log.jsonl").read_text(encoding="utf-8")
    assert "si è fermato per un errore" in text
    assert "CUDA failed to initialize" in text  # the traceback is in the record
