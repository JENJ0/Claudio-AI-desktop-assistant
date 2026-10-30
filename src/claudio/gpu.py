"""How full the graphics card is, read with nvidia-smi (it comes with the NVIDIA driver: no
package to install). Jack's worker needs a few GB of it: with a game holding the card it
loaded in 21 s, or never (30/10 morning: no answer in 90 s)."""

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
    "--query-gpu=memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]
TIMEOUT_S = 3.0  # it answers in ~0.1 s; a hung driver must not hold the start
CREATE_NO_WINDOW = 0x08000000  # Windows: no console window flashing at startup


@dataclass(frozen=True)
class GpuMemory:
    used_mb: int
    total_mb: int
    busy_pct: int  # how much of the last second the card was computing

    @property
    def free_mb(self) -> int:
        return max(self.total_mb - self.used_mb, 0)


def read_gpu(run: Callable[..., Any] = subprocess.run) -> GpuMemory | None:
    """The first card's memory and load; None without an NVIDIA card or driver."""
    flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        done = run(QUERY, capture_output=True, text=True, timeout=TIMEOUT_S, creationflags=flags)
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("nvidia-smi non disponibile: %s", exc)
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    try:
        used, total, busy = (int(v) for v in done.stdout.splitlines()[0].split(","))
    except ValueError:  # "[N/A]" or a format we do not know
        log.debug("nvidia-smi: risposta non letta: %r", done.stdout[:80])
        return None
    return GpuMemory(used, total, busy)
