"""One folder per run in logs/sessions/, named by its start (2026-10-20_16-32): the console log
as JenJo sees it, plus the conversation, one block per request. A test on the PC can then be
read as it is (Claude reads the folder) instead of being copied into the chat."""

from __future__ import annotations

import logging
import logging.handlers
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from claudio.i18n import TranslateFilter

log = logging.getLogger(__name__)

CONSOLE_FILE = "console.log"
CONVERSATION_FILE = "conversazione.md"
LOG_MAX_BYTES = 5_000_000
LOG_BACKUPS = 3
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def start_session(logs_dir: Path, keep: int, now: datetime | None = None) -> Path:
    """Create this run's folder and mirror the console into it. Only the newest `keep`
    sessions stay (a few KB each; the audio has its own limit)."""
    now = now or datetime.now()
    root = logs_dir / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    folder = root / now.strftime("%Y-%m-%d_%H-%M")
    n = 2
    while folder.exists():  # two runs in the same minute
        folder = root / f"{now.strftime('%Y-%m-%d_%H-%M')}_{n}"
        n += 1
    folder.mkdir()
    handler = logging.handlers.RotatingFileHandler(  # a session can last days
        folder / CONSOLE_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    handler.addFilter(TranslateFilter())
    logging.getLogger().addHandler(handler)
    (folder / CONVERSATION_FILE).write_text(
        f"# CLAUDIO, sessione del {now:%d/%m/%Y %H:%M}\n", encoding="utf-8"
    )
    for old in sorted(p for p in root.iterdir() if p.is_dir())[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return folder


def purge_old_logs(
    logs_dir: Path, days: int, current: Path | None = None, now: float | None = None
) -> int:
    """Privacy (logging.retention_days): delete what Claudio heard more than `days` ago, by
    last change. Session folders (console and conversation), the WAVs in logs/audio/ (near
    misses too) and the rotated metrics files; the live metrics.jsonl and `current` stay.
    Returns how many were deleted."""
    cutoff = (now if now is not None else time.time()) - days * 86400
    old: list[Path] = []
    sessions = logs_dir / "sessions"
    if sessions.is_dir():
        old += [p for p in sessions.iterdir() if p.is_dir() and p != current]
    audio = logs_dir / "audio"
    if audio.is_dir():
        old += list(audio.glob("*.wav"))
    old += list(logs_dir.glob("metrics.jsonl.*"))
    removed = 0
    for path in old:
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except OSError:
            log.warning("Impossibile eliminare il vecchio log %s", path)
    return removed


class ConversationLog:
    """Appends each finished request (a metrics row) to conversazione.md."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / CONVERSATION_FILE

    def add(self, row: dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(format_row(row))
        except OSError:
            log.exception("Conversazione non salvata")


def format_row(row: dict[str, Any]) -> str:
    when = datetime.fromtimestamp(row.get("ts") or 0).strftime("%H:%M:%S")
    wake = row.get("wake_word", "?")
    score = row.get("wake_score")
    head = f"\n## {when} · {wake}" + (f" {score:.2f}" if score else "")
    head += f" · {row.get('outcome', '?')}"
    route = row.get("route")
    if route:
        head += f" · {route}"
        if row.get("command"):
            head += f" ({row['command']})"
    lines = [head, ""]
    if row.get("text"):
        lines.append(f"**JenJo:** {row['text']}")
    if row.get("reply"):
        lines.append(f"**Claudio:** {row['reply']}")
    if row.get("tools"):
        lines.append("**Azioni:** " + "; ".join(row["tools"]))
    details = []
    for key, label in (("m1_ms", "M1"), ("m2_ms", "M2"), ("stt_ms", "STT")):
        if row.get(key) is not None:
            details.append(f"{label} {row[key]:.0f} ms")
    if row.get("claude_first_ms") is not None:
        details.append(f"Claude prima parola {row['claude_first_ms']} ms")
    if row.get("cost") is not None:
        details.append(
            f"{row.get('model')} {row.get('input_tokens')} in / {row.get('output_tokens')} out"
            f" = {row['cost']:.5f} $"
        )
    for key in ("claude_error", "barge_in", "wake_check", "closed_list_from", "retried_from"):
        if row.get(key):
            details.append(f"{key}: {row[key]}")
    if row.get("audio_file"):
        details.append(f"audio: {row['audio_file']}")
    if details:
        lines.append("")
        lines.append("_" + " · ".join(details) + "_")
    return "\n".join(lines) + "\n"
