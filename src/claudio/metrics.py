"""Per-interaction timestamps and the two latency metrics.

M1: end of speech -> first sound of the reply (what we optimise).
M2: wake word -> first sound of the reply (the CLAUDE.md metric).
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STAGES = (
    "wake",
    "speech_start",
    "speech_end",
    "endpoint",
    "stt_done",
    "first_text",
    "first_audio",
    "done",
)


@dataclass
class Interaction:
    clock: Any = time.perf_counter
    marks: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        # Only the first occurrence counts (e.g. first audio chunk).
        if stage not in self.marks:
            self.marks[stage] = self.clock()

    def ms(self, start: str, end: str) -> float | None:
        if start in self.marks and end in self.marks:
            return round((self.marks[end] - self.marks[start]) * 1000, 1)
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "m1_ms": self.ms("speech_end", "first_audio"),
            "m2_ms": self.ms("wake", "first_audio"),
            "vad_wait_ms": self.ms("speech_end", "endpoint"),
            "stt_ms": self.ms("endpoint", "stt_done"),
            "route_ms": self.ms("stt_done", "first_text"),
            "tts_first_ms": self.ms("first_text", "first_audio"),
            "speech_ms": self.ms("speech_start", "speech_end"),
            **self.info,
        }


class MetricsWriter:
    """Appends one JSON line per interaction to logs/metrics.jsonl."""

    def __init__(self, path: Path, conversation: Any = None) -> None:
        self.path = path
        self.conversation = conversation  # session_log.ConversationLog: the readable copy
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, interaction: Interaction) -> dict[str, Any]:
        row = {"ts": time.time(), **interaction.summary()}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.conversation is not None:
            self.conversation.add(row)
        if row["m2_ms"] is None:  # timeout, near miss...: nothing was said back
            log.debug("Metriche: %s", row.get("outcome"), extra={"metrics": row})
            return row
        log.info(
            "Latenza M1=%s ms M2=%s ms (STT %s ms, TTS prima frase %s ms)",
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
    rows: list[dict[str, Any]], keys: tuple[str, ...] = ("m1_ms", "m2_ms", "stt_ms")
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
