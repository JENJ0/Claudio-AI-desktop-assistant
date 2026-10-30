"""How full the graphics card is, read with nvidia-smi (it comes with the NVIDIA driver: no
package to install). Jack's worker needs a few GB of it: with a game holding the card it
loaded in 21 s, or never (30/10 morning: no answer in 90 s).

Only the first card is read: CLAUDIO's PC has one. With several, the order of nvidia-smi
(PCI bus) may not be the one CUDA uses."""

from __future__ import annotations

import logging
import subprocess
import sys
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
