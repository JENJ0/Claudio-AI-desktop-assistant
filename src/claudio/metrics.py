"""Per-interaction timestamps and the latency metrics.

audible_ms: end of speech -> first audible sample (what we optimise), per path.
M1: end of speech -> first audio handed to the player (kept for continuity).
M2: wake word -> first sound of the reply (kept for continuity).
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claudio.gpu import GpuSampler

log = logging.getLogger(__name__)

# The fields of a row that hold what was said, or may quote it (a reply, a tool's arguments). The
# figures around them say nothing of it.
SAID_FIELDS = frozenset(
    {"text", "second_chance", "second_chance_from", "wake_unclear", "wake_check", "relisten"}
    | {"continued_from", "closed_list_from", "retried_from", "recovered_from", "guessed_from"}
    | {"barge_in", "recovery", "guess", "stt_alternatives", "reply", "tools", "shown"}
)


@dataclass
class Interaction:
    clock: Any = time.perf_counter
    marks: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    off: bool = False  # heard while CLAUDIO was off: what was said is not in the row
    # Whisper's prompt for the pass whose text was kept: never in a row (folder names).
    whisper_prompt: str = ""

    def mark(self, stage: str) -> None:
        # Only the first occurrence counts (e.g. first audio chunk).
        if stage not in self.marks:
            self.marks[stage] = self.clock()

    def ms(self, start: str, end: str) -> float | None:
        if start in self.marks and end in self.marks:
            return round((self.marks[end] - self.marks[start]) * 1000, 1)
        return None

    def summary(self) -> dict[str, Any]:
        """The row: every file and the history get this. `info` stays as it is, the decisions
        read it."""
        info = self.info
        if self.off:
            info = {key: value for key, value in info.items() if key not in SAID_FIELDS}
        return {
            "m1_ms": self.ms("speech_end", "first_audio"),
            "m2_ms": self.ms("wake", "first_audio"),
            "audible_ms": self.ms("speech_end", "first_audible"),
            "vad_wait_ms": self.ms("speech_end", "endpoint"),
            "stt_ms": self.ms("endpoint", "stt_done"),
            "route_ms": self.ms("stt_done", "first_text"),
            "tts_first_ms": self.ms("first_text", "first_audio"),
            "speech_ms": self.ms("speech_start", "speech_end"),
            **info,
        }


class MetricsWriter:
    """Appends one JSON line per interaction to logs/metrics.jsonl."""

    def __init__(
        self,
        path: Path,
        conversation: Any = None,
        chat: Any = None,
        gpu: GpuSampler | None = None,
        max_bytes: int = 5_000_000,
        backups: int = 3,
    ) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self.conversation = conversation  # session_log.ConversationLog: the readable copy
        self.chat = chat  # chat.store.ChatStore: what the chat window shows
        self.gpu = gpu  # the card's load and free memory in every row, as last read
        path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate(self) -> None:
        """metrics.jsonl -> .1 -> .2 -> .3 once it passes `max_bytes`: always-on runs last weeks.
        On Windows a file open elsewhere cannot be renamed: keep appending, retry next time."""
        try:
            if self.path.stat().st_size < self.max_bytes:
                return
            for i in range(self.backups, 0, -1):
                src = self.path if i == 1 else self.path.with_name(f"{self.path.name}.{i - 1}")
                if src.exists():
                    src.replace(self.path.with_name(f"{self.path.name}.{i}"))
        except OSError:
            log.warning("Rotazione di %s non riuscita", self.path.name, exc_info=True)

    def write(self, interaction: Interaction) -> dict[str, Any]:
        row = {"ts": time.time(), **interaction.summary()}
        if self.gpu is not None:  # never waits: the last reading, None before the first
            reading = self.gpu.latest()
            row["gpu_free_mb"] = reading.free_mb if reading else None
            row["gpu_busy_pct"] = reading.busy_pct if reading else None
        self._rotate()
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Held by another program, or a value JSON cannot encode: the row still goes to the caller.
        except (OSError, TypeError, ValueError):
            log.exception("Metriche non salvate")
        if self.conversation is not None:
            self.conversation.add(row)
        if self.chat is not None:
            self.chat.add(row)
        if row["m2_ms"] is None:  # timeout, near miss...: nothing was said back
            log.debug("Metriche: %s", row.get("outcome"), extra={"metrics": row})
            return row
        log.info(
            "Latenza udibile=%s ms M1=%s ms M2=%s ms (STT %s ms, TTS prima frase %s ms)",
            row["audible_ms"],
            row["m1_ms"],
            row["m2_ms"],
            row["stt_ms"],
            row["tts_first_ms"],
            extra={"metrics": row},
        )
        return row


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(
    rows: list[dict[str, Any]], keys: tuple[str, ...] = ("m1_ms", "m2_ms", "audible_ms", "stt_ms")
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if vals:
            out[key] = {
                "n": len(vals),
                "median": round(statistics.median(vals), 1),
                "p90": round(percentile(vals, 90), 1),
            }
    return out
