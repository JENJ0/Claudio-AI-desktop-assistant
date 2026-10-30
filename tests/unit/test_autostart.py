"""The login task: its definition and commands, never registered by the tests."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from claudio import autostart
from claudio.autostart import TaskSpec, task_xml

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


@pytest.fixture
def spec(tmp_path: Path) -> TaskSpec:
    python = tmp_path / ".venv" / "Scripts" / "pythonw.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    (tmp_path / "src" / "claudio").mkdir(parents=True)
    return TaskSpec("PC\\JenJo & Co", python, tmp_path)


def test_xml_runs_pythonw_at_login_for_the_user(spec: TaskSpec) -> None:
    root = ET.fromstring(task_xml(spec).split("\n", 1)[1])  # without the UTF-16 header
    exec_ = root.find("t:Actions/t:Exec", NS)
    assert exec_ is not None
    assert exec_.findtext("t:Command", namespaces=NS) == str(spec.python)
    assert exec_.findtext("t:Arguments", namespaces=NS) == "-m claudio"
    assert exec_.findtext("t:WorkingDirectory", namespaces=NS) == str(spec.project)
    trigger = root.find("t:Triggers/t:LogonTrigger", NS)
    assert trigger is not None
    assert trigger.findtext("t:UserId", namespaces=NS) == "PC\\JenJo & Co"  # & escaped
    assert trigger.findtext("t:Delay", namespaces=NS) == "PT30S"
    principal = root.find("t:Principals/t:Principal", NS)
    assert principal is not None
    assert principal.findtext("t:RunLevel", namespaces=NS) == "LeastPrivilege"  # no admin
    settings = root.find("t:Settings", NS)
    assert settings is not None
    assert settings.findtext("t:MultipleInstancesPolicy", namespaces=NS) == "IgnoreNew"
    assert settings.findtext("t:ExecutionTimeLimit", namespaces=NS) == "PT0S"  # runs for weeks


def test_install_replaces_and_uninstall_is_forced() -> None:
    assert autostart.install_command(Path("t.xml"))[-1] == "/F"  # twice = same task
    assert autostart.uninstall_command() == ["schtasks", "/Delete", "/TN", "Claudio", "/F"]


def test_validate_refuses_a_missing_python(spec: TaskSpec) -> None:
    spec.python.unlink()
    with pytest.raises(FileNotFoundError, match="uv sync"):
        spec.validate()


def test_validate_refuses_another_folder(spec: TaskSpec, tmp_path: Path) -> None:
    other = TaskSpec(spec.user, spec.python, tmp_path / "elsewhere")
    with pytest.raises(FileNotFoundError, match="not the CLAUDIO project"):
        other.validate()


def test_dry_run_changes_nothing(spec: TaskSpec, monkeypatch, capsys) -> None:
    def no_run(*args, **kwargs):
        raise AssertionError("a dry run must not call schtasks")

    monkeypatch.setattr(autostart.subprocess, "run", no_run)
    assert autostart.install(spec, dry_run=True) == 0
    assert autostart.uninstall(dry_run=True) == 0
    assert "-m claudio" in capsys.readouterr().out


def test_uninstall_when_not_installed_is_fine(monkeypatch) -> None:
    calls: list[list[str]] = []

    class Done:
        returncode = 1  # schtasks /Query: no such task

    def run(cmd, **kwargs):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr(autostart.subprocess, "run", run)
    assert autostart.uninstall() == 0
    assert calls == [autostart.status_command()]  # never /Delete


def test_install_writes_utf16_xml_and_calls_schtasks(spec: TaskSpec, monkeypatch) -> None:
    seen: dict[str, str] = {}

    class Done:
        returncode = 0

    def run(cmd, **kwargs):
        seen["xml"] = Path(cmd[cmd.index("/XML") + 1]).read_text(encoding="utf-16")
        return Done()

    monkeypatch.setattr(autostart.subprocess, "run", run)
    assert autostart.install(spec) == 0
    assert "<Arguments>-m claudio</Arguments>" in seen["xml"]


def test_logging_without_a_console(tmp_path: Path, monkeypatch) -> None:
    """pythonw.exe (the start at login) has sys.stderr = None: only the log file is used."""
    import logging

    from claudio.logging_setup import setup_logging

    monkeypatch.setattr("sys.stderr", None)
    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        setup_logging(tmp_path, "INFO", "test.log.jsonl")
        assert all(
            not isinstance(h, logging.StreamHandler) or isinstance(h, logging.FileHandler)
            for h in root.handlers
        )
        logging.getLogger("claudio.test").info("ciao")
        assert (tmp_path / "test.log.jsonl").read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved:
            root.addHandler(handler)
