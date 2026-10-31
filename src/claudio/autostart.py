"""Start CLAUDIO at Windows login: a Task Scheduler task, installed and removed on request.

The task runs `pythonw.exe -m claudio.supervisor` from the project's virtual environment
(pythonw: no console window), 30 s after login so the microphone and the network are ready,
for the logged-in user only and without admin rights. The supervisor starts CLAUDIO and
starts it again after a crash: Task Scheduler's own RestartOnFailure (kept below) only
covers a program that cannot be started at all. Installing twice replaces the task (/F), so
the commands are idempotent. scripts/install_autostart.ps1 does the same in PowerShell."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from claudio.config import PROJECT_ROOT

TASK_NAME = "Claudio"
LOGON_DELAY = "PT30S"  # ISO 8601: the audio device is often not ready at the first second
# Task Scheduler retries only a start that failed (pythonw.exe missing for a moment): a
# crash of CLAUDIO is handled by claudio.supervisor.
RESTARTS = 3
RESTART_EVERY = "PT1M"
ARGUMENTS = "-m claudio.supervisor"


@dataclass(frozen=True)
class TaskSpec:
    user: str  # DOMAIN\user, the account that logs in
    python: Path  # pythonw.exe of the virtual environment
    project: Path  # working directory: config/, models/ and .env are found from here

    def validate(self) -> None:
        """Fail before touching Task Scheduler: a task pointing nowhere starts and dies."""
        if not self.user.strip():
            raise ValueError("Unknown Windows user (USERNAME is empty)")
        if not self.python.is_file():
            raise FileNotFoundError(
                f"{self.python} not found: run `uv sync` in {self.project} first"
            )
        if not (self.project / "src" / "claudio").is_dir():
            raise FileNotFoundError(f"{self.project} is not the CLAUDIO project folder")


def default_spec(project: Path = PROJECT_ROOT) -> TaskSpec:
    domain = os.environ.get("USERDOMAIN", "")
    name = os.environ.get("USERNAME", "")
    user = f"{domain}\\{name}" if domain and name else name
    return TaskSpec(user, project / ".venv" / "Scripts" / "pythonw.exe", project)


def task_xml(spec: TaskSpec) -> str:
    """The task definition for `schtasks /Create /XML` (schema 1.2, Windows 7 and later)."""
    user, python, project = escape(spec.user), escape(str(spec.python)), escape(str(spec.project))
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>CLAUDIO voice assistant, started at login.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{user}</UserId>
      <Delay>{LOGON_DELAY}</Delay>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>{RESTART_EVERY}</Interval>
      <Count>{RESTARTS}</Count>
    </RestartOnFailure>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{python}</Command>
      <Arguments>{ARGUMENTS}</Arguments>
      <WorkingDirectory>{project}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def install_command(xml_path: Path) -> list[str]:
    return ["schtasks", "/Create", "/TN", TASK_NAME, "/XML", str(xml_path), "/F"]


def uninstall_command() -> list[str]:
    return ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"]


def status_command() -> list[str]:
    return ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"]


def is_installed() -> bool:
    return subprocess.run(status_command(), capture_output=True, check=False).returncode == 0


def install(spec: TaskSpec, dry_run: bool = False) -> int:
    spec.validate()
    if dry_run:
        print(task_xml(spec))
        print(" ".join(install_command(Path("<task.xml>"))))
        return 0
    # schtasks reads the XML from a file, in UTF-16 as its header says.
    with tempfile.TemporaryDirectory() as folder:
        xml_path = Path(folder) / "claudio-task.xml"
        xml_path.write_text(task_xml(spec), encoding="utf-16")
        return subprocess.run(install_command(xml_path), check=False).returncode


def uninstall(dry_run: bool = False) -> int:
    if dry_run:
        print(" ".join(uninstall_command()))
        return 0
    if not is_installed():
        print(f"Task '{TASK_NAME}' is not installed: nothing to remove.")
        return 0  # idempotent: removing twice is not an error
    return subprocess.run(uninstall_command(), check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start CLAUDIO at Windows login (or stop it).")
    parser.add_argument("action", choices=["install", "uninstall", "status"])
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would run, change nothing"
    )
    args = parser.parse_args(argv)
    if sys.platform != "win32" and not args.dry_run:
        print("Task Scheduler exists only on Windows.", file=sys.stderr)
        return 2
    try:
        if args.action == "install":
            return install(default_spec(), args.dry_run)
        if args.action == "uninstall":
            return uninstall(args.dry_run)
        if args.dry_run:
            print(" ".join(status_command()))
            return 0
        return subprocess.run(status_command(), check=False).returncode
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
