"""Open and close the chat window from the running assistant ("apri la chat").

The window is `claudio --ui` in a child process: pywebview wants the main thread, which the
assistant's event loop already holds, and a crash in WebView2 must not take the voice down.
Both processes share data/claudio.db, so the window shows the conversation live."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def window_command(config: Path | None = None) -> list[str]:
    """The same Python and the same configuration as the assistant."""
    command = [sys.executable, "-m", "claudio", "--ui"]
    if config is not None:
        command += ["--config", str(config.resolve())]
    return command


def _popen(command: Sequence[str]) -> Any:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console box behind the window
    return subprocess.Popen(
        list(command),
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


class ChatLauncher:
    def __init__(
        self, command: Sequence[str], popen: Callable[[Sequence[str]], Any] = _popen
    ) -> None:
        self.command = list(command)
        self._popen = popen
        self._process: Any = None

    @property
    def is_open(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def open(self) -> bool:
        """Open the window; False when it is already open (the user closed it: reopened)."""
        if self.is_open:
            return False
        self._process = self._popen(self.command)
        log.info("Chat aperta", extra={"pid": getattr(self._process, "pid", None)})
        return True

    def close(self) -> bool:
        """Close the window; False when it was not open."""
        if not self.is_open:
            return False
        self._process.terminate()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
        log.info("Chat chiusa")
        return True
