"""How full the graphics card is, read with nvidia-smi (it comes with the NVIDIA driver: no
package to install). Jack's worker needs a few GB of it: with a game holding the card it
loaded in 21 s, or never (30/10 morning: no answer in 90 s).

Only the first card is read: CLAUDIO's PC has one. With several, the order of nvidia-smi
(PCI bus) may not be the one CUDA uses."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.free,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]
TIMEOUT_S = 3.0  # it answers in ~0.1 s; a hung driver must not hold the start
CREATE_NO_WINDOW = 0x08000000  # Windows: no console window flashing at startup


@dataclass(frozen=True)
class GpuMemory:
    free_mb: int  # as the driver counts it: total minus used minus its own reserve
    total_mb: int
    busy_pct: int | None = None  # compute load over the last second; None: not reported


def read_gpu(popen: Callable[..., Any] = subprocess.Popen) -> GpuMemory | None:
    """The first card's free memory and load; None without an NVIDIA card or driver."""
    flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        proc = popen(
            QUERY,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=flags,
        )
    except OSError as exc:
        log.debug("nvidia-smi non disponibile: %s", exc)
        return None
    try:
        out, _ = proc.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # subprocess.run would wait again for a process that cannot die: give up on it.
        proc.kill()
        log.debug("nvidia-smi non disponibile: nessuna risposta in %.0f s", TIMEOUT_S)
        return None
    if proc.returncode != 0 or not out.strip():
        return None
    fields = [f.strip() for f in out.splitlines()[0].split(",")]
    try:
        free, total = int(fields[0]), int(fields[1])
    except (ValueError, IndexError):  # "[N/A]" or a format we do not know
        log.debug("nvidia-smi: risposta non letta: %r", out[:80])
        return None
    busy = int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else None
    return GpuMemory(free, total, busy)


def has_room(need_mb: int, read: Callable[[], GpuMemory | None] = read_gpu) -> bool:
    """At least `need_mb` free, or nothing to tell (no NVIDIA card: no reason to wait)."""
    gpu = read()
    return gpu is None or gpu.free_mb >= need_mb


class GpuSampler:
    """The last reading of the card for the metrics rows. read_gpu() starts a process (~0.1 s,
    up to TIMEOUT_S when the driver hangs): it never runs on the reply path. latest() returns
    what is there at once, and when that is older than `max_age_s` starts one refresh in a
    daemon thread (closing the program does not wait for it). The first row may get None."""

    def __init__(
        self,
        read: Callable[[], GpuMemory | None] = read_gpu,
        max_age_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._read, self._max_age_s, self._clock = read, max_age_s, clock
        self._lock = threading.Lock()
        self._reading: GpuMemory | None = None
        self._tried_at: float | None = None  # when the last refresh started, failed or not
        self._in_flight = False
        self.thread: threading.Thread | None = None  # the last refresh started

    def latest(self) -> GpuMemory | None:
        with self._lock:
            now = self._clock()
            stale = self._tried_at is None or now - self._tried_at >= self._max_age_s
            if stale and not self._in_flight:
                self._start_refresh(now)
            return self._reading

    def _start_refresh(self, now: float) -> None:
        """Called with the lock held."""
        self._in_flight, self._tried_at = True, now
        try:
            self.thread = threading.Thread(target=self._refresh, name="gpu-sampler", daemon=True)
            self.thread.start()
        except RuntimeError:  # no thread to be had: try again in max_age_s
            self._in_flight = False
            log.warning("Lettura della GPU non avviata", exc_info=True)

    def _refresh(self) -> None:
        try:
            reading = self._read()
            with self._lock:
                self._reading = reading  # None: no NVIDIA card or driver
        except Exception:
            log.warning("Lettura della GPU non riuscita", exc_info=True)
        finally:
            with self._lock:
                self._in_flight = False
