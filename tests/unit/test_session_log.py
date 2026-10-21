"""logs/sessions/<date_time>/: the console and the conversation of one run, ready to read."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytest

from claudio.metrics import Interaction, MetricsWriter
from claudio.session_log import (
    CONSOLE_FILE,
    CONVERSATION_FILE,
    ConversationLog,
    format_row,
    start_session,
)

NOW = datetime(2026, 10, 20, 16, 32, 44)


@pytest.fixture
def session(tmp_path: Path):
    folder = start_session(tmp_path, keep=30, now=NOW)
    yield folder
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
        if Path(handler.baseFilename).parent == folder:
            root.removeHandler(handler)
            handler.close()


def test_folder_is_named_by_date_and_minute(tmp_path: Path, session: Path) -> None:
    assert session == tmp_path / "sessions" / "2026-10-20_16-32"
    assert (
        (session / CONVERSATION_FILE)
        .read_text(encoding="utf-8")
        .startswith("# CLAUDIO, sessione del 20/10/2026 16:32")
    )


def test_console_is_mirrored(session: Path) -> None:
    logging.getLogger("claudio.test").warning("Prova %d", 1)
    text = (session / CONSOLE_FILE).read_text(encoding="utf-8")
    assert "WARNING claudio.test: Prova 1" in text


def test_same_minute_gets_a_second_folder(tmp_path: Path, session: Path) -> None:
    again = start_session(tmp_path, keep=30, now=NOW)
    try:
        assert again.name == "2026-10-20_16-32_2"
    finally:
        root = logging.getLogger()
        for h in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
            if Path(h.baseFilename).parent == again:
                root.removeHandler(h)
                h.close()


def test_only_the_newest_sessions_are_kept(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    for day in range(1, 6):
        (root / f"2026-10-0{day}_10-00").mkdir(parents=True)
    folder = start_session(tmp_path, keep=3, now=NOW)
    try:
        assert sorted(p.name for p in root.iterdir()) == [
            "2026-10-04_10-00",
            "2026-10-05_10-00",
            "2026-10-20_16-32",
        ]
    finally:
        log_root = logging.getLogger()
        for h in [h for h in log_root.handlers if isinstance(h, logging.FileHandler)]:
            if Path(h.baseFilename).parent == folder:
                log_root.removeHandler(h)
                h.close()


def test_a_claude_request_reads_like_a_chat() -> None:
    row = {
        "ts": NOW.timestamp(),
        "wake_word": "hey_claudio",
        "wake_score": 1.0,
        "outcome": "ok",
        "route": "claude",
        "text": "Quanto dista Napoli?",
        "reply": "Da dove parti?",
        "m1_ms": 2698.0,
        "m2_ms": 4521.9,
        "stt_ms": 211.5,
        "claude_first_ms": 950,
        "model": "claude-haiku-5-5",
        "input_tokens": 324,
        "output_tokens": 101,
        "cost": 0.00008,
        "wake_check": "Ehi Claudio.",
    }
    text = format_row(row)
    assert "## 16:32:44 · hey_claudio 1.00 · ok · claude" in text
    assert "**JenJo:** Quanto dista Napoli?" in text
    assert "**Claudio:** Da dove parti?" in text
    assert "M1 2698 ms" in text and "Claude prima parola 950 ms" in text
    assert "claude-haiku-5-5 324 in / 101 out = 0.00008 $" in text
    assert "wake_check: Ehi Claudio." in text


def test_metrics_rows_reach_the_conversation(tmp_path: Path, session: Path) -> None:
    writer = MetricsWriter(tmp_path / "metrics.jsonl", ConversationLog(session))
    inter = Interaction(info={"wake_word": "hey_claudio", "outcome": "ok", "text": "che ore sono"})
    writer.write(inter)
    assert "**JenJo:** che ore sono" in (session / CONVERSATION_FILE).read_text(encoding="utf-8")


def test_console_log_rotates(session: Path) -> None:
    from logging.handlers import RotatingFileHandler

    handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, RotatingFileHandler) and Path(h.baseFilename).parent == session
    ]
    assert len(handlers) == 1 and handlers[0].maxBytes > 0 and handlers[0].backupCount >= 1


def test_metrics_file_rotates_and_keeps_the_newest(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path, max_bytes=100, backups=2)
    for _ in range(6):
        writer.write(Interaction(info={"wake_word": "hey_claudio", "outcome": "ok"}))
    assert path.exists() and path.with_name("metrics.jsonl.1").exists()
    assert path.with_name("metrics.jsonl.2").exists()
    assert not path.with_name("metrics.jsonl.3").exists()  # only `backups` copies stay
