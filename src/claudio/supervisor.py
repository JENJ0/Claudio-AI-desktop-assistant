"""Keep CLAUDIO running: `python -m claudio.supervisor` starts it and starts it again after a
crash. The task at login runs this, not CLAUDIO itself: Task Scheduler's "restart on
failure" fires only when the program cannot be started, never when it ends with an error
(tried on JenJo's PC on 30/10: a task that exited with code 1, allowed 3 restarts a minute
apart, was not run again in 3 minutes).

What CLAUDIO's exit code means here:
- 0: closed on purpose ("chiuditi", the tray menu, Ctrl+C): the supervisor ends too;
- 2: configuration error: it would fail the same way until the file is fixed;
- 3: another CLAUDIO is already running (single_instance.py);
- anything else is a crash: CLAUDIO starts again after RESTART_DELAYS_S, at most that many
  times within WINDOW_S, so a crash at every start (no microphone) does not loop forever.

The arguments are passed to CLAUDIO as they are (`--config other.toml`)."""

from __future__ import annotations

import argparse
import contextlib
import logging
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

from claudio.config import PROJECT_ROOT, ConfigError, configured_language, load_settings
from claudio.i18n import set_language
from claudio.logging_setup import setup_logging
from claudio.single_instance import EXIT_ALREADY_RUNNING

log = logging.getLogger("claudio.supervisor")

EXIT_CONFIG_ERROR = 2  # __main__.main() on a ConfigError
FINAL_EXITS = frozenset({0, EXIT_CONFIG_ERROR, EXIT_ALREADY_RUNNING})
RESTART_DELAYS_S = (5.0, 30.0, 60.0)  # before the 1st, 2nd and 3rd restart in WINDOW_S
WINDOW_S = 600.0  # a crash older than this is forgotten: one a day is always restarted
LOG_FILE = "supervisor.log.jsonl"  # its own file: two processes rotating one fail on Windows
WAIT_SLICE_S = 0.5  # how often the wait for CLAUDIO lets a Ctrl+C through
CTRL_C_GRACE_S = 10.0  # after Ctrl+C, CLAUDIO gets this long to close before it is killed


def child_command(args: Sequence[str]) -> list[str]:
    """CLAUDIO with the same Python: pythonw.exe at login, so no console window either."""
    return [sys.executable, "-m", "claudio", *args]


def run_child(command: Sequence[str]) -> int:
    process = subprocess.Popen(list(command))
    try:
        while True:  # in slices: on Windows a wait without a timeout ignores Ctrl+C
            try:
                return process.wait(timeout=WAIT_SLICE_S)
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        # Ctrl+C in a terminal reaches CLAUDIO too: it says goodbye and closes by itself.
        try:
            process.wait(timeout=CTRL_C_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
        # The supervisor itself was asked to stop: whatever CLAUDIO's code (a Ctrl+C before
        # CLAUDIO handles it ends it with STATUS_CONTROL_C_EXIT), never start it again.
        return 0


def supervise(
    command: Sequence[str],
    run: Callable[[Sequence[str]], int] = run_child,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    delays: Sequence[float] = RESTART_DELAYS_S,
    window_s: float = WINDOW_S,
) -> int:
    """Run `command` until it ends with one of FINAL_EXITS or crashes too often; returns
    its last exit code."""
    crashes: deque[float] = deque()
    while True:
        started = clock()
        code = run(command)
        if code in FINAL_EXITS:
            log.info("CLAUDIO chiuso (codice %d): il supervisore si ferma", code)
            return code
        now = clock()
        while crashes and now - crashes[0] > window_s:
            crashes.popleft()
        crashes.append(now)
        if len(crashes) > len(delays):
            log.error(
                "CLAUDIO si è chiuso per errore %d volte in %.0f minuti: non lo riavvio più. "
                "Il motivo è in logs/claudio.log.jsonl",
                len(crashes),
                window_s / 60,
            )
            return code
        delay = delays[len(crashes) - 1]
        log.warning(
            "CLAUDIO si è chiuso per errore (codice %d dopo %.0f s): lo riavvio tra %.0f s",
            code,
            now - started,
            delay,
        )
        sleep(delay)


def logs_dir(config: Path | None) -> Path:
    """The configured log folder; the default one when the configuration does not load
    (CLAUDIO reports that error itself, with exit code 2)."""
    try:
        settings = load_settings(config)
    except ConfigError:
        return PROJECT_ROOT / "logs"
    set_language(settings.language)
    return settings.resolve(settings.logs_dir)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    config = parser.parse_known_args(args)[0].config
    with contextlib.suppress(ValueError):
        set_language(configured_language(config))
    setup_logging(logs_dir(config), file_name=LOG_FILE)
    command = child_command(args)
    log.info("Supervisore: avvio %s", " ".join(command))
    try:
        return supervise(command)
    except KeyboardInterrupt:  # Ctrl+C while waiting to start CLAUDIO again
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
