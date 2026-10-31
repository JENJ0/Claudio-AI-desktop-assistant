"""One CLAUDIO at a time: a named Windows mutex, held while the assistant runs.

With the start at login, a second `claudio` (typed in a terminal, or the task started again)
would share the microphone with the first: two wake words, two replies to every request.
The name is per Windows session ("Local\\"): another user logged in on the same PC has
their own CLAUDIO. Windows closes the handle when the process ends, a crash included, so a
restart never finds a stale lock, as it could with a lock file."""

from __future__ import annotations

import ctypes
import functools
import sys
from types import TracebackType
from typing import Any

MUTEX_NAME = "Local\\CLAUDIO-voice-assistant"
ERROR_ACCESS_DENIED = 5
ERROR_ALREADY_EXISTS = 183
# The exit code of a CLAUDIO that found another one running: not a crash, never restarted.
EXIT_ALREADY_RUNNING = 3


@functools.cache
def _kernel32() -> Any:
    from ctypes import wintypes

    # A private WinDLL: argtypes set here do not leak into other users of ctypes.windll.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _create_mutex(name: str) -> tuple[int, int]:
    """(handle or 0, GetLastError) of CreateMutexW: the existing mutex or a new one.
    Not owned (bInitialOwner False): the name existing is the lock. The handle is not
    inherited (no security attributes), so Jack's worker and the chat window never keep it
    alive after CLAUDIO ends."""
    handle = _kernel32().CreateMutexW(None, False, name)
    return handle or 0, ctypes.get_last_error()


class SingleInstance:
    """`with SingleInstance() as alone:` alone is False when another CLAUDIO holds the name."""

    def __init__(self, name: str = MUTEX_NAME) -> None:
        self.name = name
        self._handle: int | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """True when this is the only CLAUDIO (the name is now held); False when another
        process holds it. Outside Windows there is no mutex: always True."""
        if self._handle is not None:
            return True
        if sys.platform != "win32":
            return True
        handle, error = _create_mutex(self.name)
        if not handle:
            # A CLAUDIO started as administrator holds the name and its mutex cannot be
            # opened by this one (CreateMutexW documents ERROR_ACCESS_DENIED): it is taken.
            if error == ERROR_ACCESS_DENIED:
                return False
            raise ctypes.WinError(error)
        if error == ERROR_ALREADY_EXISTS:
            _kernel32().CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is not None:
            _kernel32().CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
