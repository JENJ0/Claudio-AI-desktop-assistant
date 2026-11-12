"""logs/sessions/<date_time>/: the console and the conversation of one run, ready to read."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from claudio.metrics import Interaction, MetricsWriter
from claudio.session_log import (
    CONSOLE_FILE,
    CONVERSATION_FILE,
    ConversationLog,
    drop_old_lines,
    format_row,
    purge_old_logs,
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


def test_a_full_row_is_written_in_a_fixed_order() -> None:
    """Heading, conversation lines, then one italic line of figures: order and separators."""
    row = {
        "ts": NOW.timestamp(),
        "wake_word": "claudio",
        "wake_score": 0.9,
        "outcome": "ok",
        "route": "router",
        "command": "apri chrome",
        "text": "apri chrome",
        "reply": "Fatto.",
        "tools": ["apri Chrome", "digita ciao"],
        "shown": "dir C:\\",
        "m1_ms": 0.0,  # zero is shown: only None hides a figure
        "m2_ms": 12.4,
        "stt_ms": 150.6,
        "claude_first_ms": 0,
        "claude_calls": [{"ttft_ms": 610}, {}],
        "api_rtt_ms": 220,
        "api_rtt_new_connection": True,
        "model": "claude-haiku-5-5",
        "input_tokens": 100,
        "output_tokens": 20,
        "cost": 0,
        "claude_error": "boom",
        "barge_in": True,
        "wake_check": "Ehi Claudio.",
        "closed_list_from": "a",
        "retried_from": "b",
        "recovered_from": "c",
        "audio_file": "last.wav",
    }
    assert format_row(row) == (
        "\n## 16:32:44 · claudio 0.90 · ok · router (apri chrome)\n"
        "\n"
        "**JenJo:** apri chrome\n"
        "**Claudio:** Fatto.\n"
        "**Azioni:** apri Chrome; digita ciao\n"
        "**Comando proposto:** `dir C:\\`\n"
        "\n"
        "_M1 0 ms · M2 12 ms · STT 151 ms · Claude prima parola 0 ms"
        " · TTFT 610 · - ms (rete 220 ms, connessione nuova)"
        " · claude-haiku-5-5 100 in / 20 out = 0.00000 $"
        " · claude_error: boom · barge_in: True · wake_check: Ehi Claudio."
        " · closed_list_from: a · retried_from: b · recovered_from: c"
        " · audio: last.wav_\n"
    )


def test_empty_figures_are_left_out_of_a_row() -> None:
    """Text, score and the flags are truthy checks: empty or zero values add nothing."""
    row = {
        "ts": NOW.timestamp(),
        "wake_word": "claudio",
        "wake_score": 0.0,
        "text": "",
        "claude_error": "",
        "audio_file": "",
    }
    assert format_row(row) == "\n## 16:32:44 · claudio · ?\n\n"


def test_metrics_rows_reach_the_conversation(tmp_path: Path, session: Path) -> None:
    writer = MetricsWriter(tmp_path / "metrics.jsonl", ConversationLog(session))
    inter = Interaction(info={"wake_word": "hey_claudio", "outcome": "ok", "text": "che ore sono"})
    writer.write(inter)
    assert "**JenJo:** che ore sono" in (session / CONVERSATION_FILE).read_text(encoding="utf-8")


def test_a_near_miss_stays_in_the_metrics_and_out_of_the_conversation(
    tmp_path: Path, session: Path
) -> None:
    writer = MetricsWriter(tmp_path / "metrics.jsonl", ConversationLog(session))
    near = Interaction(info={"wake_word": "claudio", "wake_score": 0.31, "outcome": "near_miss"})
    writer.write(near)
    assert '"outcome": "near_miss"' in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")
    assert "near_miss" not in (session / CONVERSATION_FILE).read_text(encoding="utf-8")


def test_a_false_wake_still_reaches_the_conversation(tmp_path: Path, session: Path) -> None:
    writer = MetricsWriter(tmp_path / "metrics.jsonl", ConversationLog(session))
    false_wake = Interaction(
        info={"wake_word": "claudio", "outcome": "false_wake", "text": "Hey, ciao, come stai?"}
    )
    writer.write(false_wake)
    written = (session / CONVERSATION_FILE).read_text(encoding="utf-8")
    assert "false_wake" in written and "Hey, ciao, come stai?" in written


def test_a_row_that_cannot_be_rendered_does_not_stop_the_metrics_write(
    tmp_path: Path, session: Path
) -> None:
    writer = MetricsWriter(tmp_path / "metrics.jsonl", ConversationLog(session))
    inter = Interaction(info={"wake_word": "hey_claudio", "outcome": "ok", "claude_calls": [7]})
    row = writer.write(inter)  # format_row cannot read a call that is not a dict
    assert row["outcome"] == "ok"
    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").count("\n") == 1


def circular() -> list[object]:
    loop: list[object] = []
    loop.append(loop)
    return loop


@pytest.mark.parametrize(
    "value",
    [np.float32(-20.5), {"a"}, Path("x"), circular()],
    ids=["numpy", "set", "path", "circular"],
)
def test_a_row_json_cannot_encode_does_not_stop_the_request(
    tmp_path: Path, session: Path, value: object
) -> None:
    """np.float32, a set, a Path: TypeError; a circular list: ValueError. Either wedged Claudio."""
    writer = MetricsWriter(tmp_path / "metrics.jsonl", ConversationLog(session))
    inter = Interaction(
        info={"wake_word": "hey_claudio", "outcome": "ok", "text": "che ore sono", "odd": value}
    )
    row = writer.write(inter)
    assert row["outcome"] == "ok"
    assert "**JenJo:** che ore sono" in (session / CONVERSATION_FILE).read_text(encoding="utf-8")
    written = tmp_path / "metrics.jsonl"
    assert not written.exists() or written.read_text(encoding="utf-8") == ""  # no half a line


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


def test_each_claude_call_shows_its_first_word_and_headers() -> None:
    """IT-1 5.2: a request with a tool round shows the first token of both calls, then the
    network alone."""
    calls = [
        {"model": "claude-haiku-5-5", "ttft_ms": 610, "total_ms": 700},
        {"model": "claude-haiku-5-5", "ttft_ms": 560, "word_ms": 560, "total_ms": 800},
    ]
    row = {"ts": NOW.timestamp(), "outcome": "ok", "route": "claude", "claude_calls": calls}
    assert "TTFT 610 · 560 ms" in format_row(row)
    assert "TTFT 610 · 560 ms (rete 220 ms)" in format_row({**row, "api_rtt_ms": 220})
    new = {**row, "api_rtt_ms": 305, "api_rtt_new_connection": True}
    assert "(rete 305 ms, connessione nuova)" in format_row(new)


NOW_TS = 1_792_000_000.0
CUTOFF = NOW_TS - 30 * 86400


def _iso_line(ts: float, msg: str = "x") -> bytes:
    stamp = datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds")
    return (json.dumps({"ts": stamp, "level": "INFO", "msg": msg}) + "\n").encode()


def _metrics_line(ts: float) -> bytes:
    return (json.dumps({"ts": ts, "outcome": "ok"}) + "\n").encode()


def test_drop_old_lines_cuts_the_old_head(tmp_path: Path) -> None:
    path = tmp_path / "claudio.log.jsonl"
    recent = _iso_line(NOW_TS - 86400, "recent")
    path.write_bytes(_iso_line(CUTOFF - 200000) + _iso_line(CUTOFF - 100) + recent)
    assert drop_old_lines(path, CUTOFF) == 2
    assert path.read_bytes() == recent


def test_drop_old_lines_reads_metrics_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    recent = _metrics_line(NOW_TS - 100)
    path.write_bytes(_metrics_line(CUTOFF - 5) + recent)
    assert drop_old_lines(path, CUTOFF) == 1
    assert path.read_bytes() == recent


def test_drop_old_lines_drops_a_broken_line_among_old_ones(tmp_path: Path) -> None:
    path = tmp_path / "claudio.log.jsonl"
    recent = _iso_line(NOW_TS - 100)
    path.write_bytes(_iso_line(CUTOFF - 9) + b"garbage{\n" + _iso_line(CUTOFF - 5) + recent)
    assert drop_old_lines(path, CUTOFF) == 3
    assert path.read_bytes() == recent


def test_drop_old_lines_keeps_an_unreadable_line_after_the_old_ones(tmp_path: Path) -> None:
    path = tmp_path / "claudio.log.jsonl"
    rest = b"garbage{\n" + _iso_line(NOW_TS - 100)
    path.write_bytes(_iso_line(CUTOFF - 5) + rest)
    assert drop_old_lines(path, CUTOFF) == 1
    assert path.read_bytes() == rest


def test_drop_old_lines_keeps_unreadable_files(tmp_path: Path) -> None:
    path = tmp_path / "claudio.log.jsonl"
    content = b"garbage{\nmore garbage\n"
    path.write_bytes(content)
    assert drop_old_lines(path, CUTOFF) == 0
    assert path.read_bytes() == content


def test_drop_old_lines_leaves_recent_and_missing_files_alone(tmp_path: Path) -> None:
    path = tmp_path / "claudio.log.jsonl"
    path.write_bytes(_iso_line(NOW_TS - 100))
    before = path.stat().st_mtime_ns
    assert drop_old_lines(path, CUTOFF) == 0
    assert path.stat().st_mtime_ns == before
    assert drop_old_lines(tmp_path / "missing.jsonl", CUTOFF) == 0


def test_drop_old_lines_survives_a_file_held_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "claudio.log.jsonl"
    content = _iso_line(CUTOFF - 5) + _iso_line(NOW_TS)
    path.write_bytes(content)

    def locked(*args: object, **kwargs: object) -> None:
        raise PermissionError("held open")

    monkeypatch.setattr(os, "replace", locked)
    assert drop_old_lines(path, CUTOFF) == 0
    assert path.read_bytes() == content
    assert list(tmp_path.iterdir()) == [path]


def test_purge_old_logs_removes_rotated_logs_and_trims_metrics(tmp_path: Path) -> None:
    old_age = NOW_TS - 40 * 86400
    old_rotated = tmp_path / "claudio.log.jsonl.1"
    new_rotated = tmp_path / "claudio.log.jsonl.2"
    old_rotated.write_text("old")
    new_rotated.write_text("new")
    os.utime(old_rotated, (old_age, old_age))
    os.utime(new_rotated, (NOW_TS, NOW_TS))
    metrics = tmp_path / "metrics.jsonl"
    recent = _metrics_line(NOW_TS - 100)
    metrics.write_bytes(_metrics_line(old_age) + recent)
    purge_old_logs(tmp_path, 30, now=NOW_TS)
    assert not old_rotated.exists()
    assert new_rotated.exists()
    assert metrics.read_bytes() == recent


def test_main_trims_the_log_before_logging_opens_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from claudio import __main__ as entry

    log_file = tmp_path / "claudio.log.jsonl"
    recent = _iso_line(time.time() - 100, "recent")
    log_file.write_bytes(_iso_line(time.time() - 40 * 86400, "old") + recent)
    seen: list[bytes] = []

    def stub(*args: object, **kwargs: object) -> None:
        seen.append(log_file.read_bytes())
        raise RuntimeError("stop here")

    monkeypatch.setenv("CLAUDIO_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("CLAUDIO_LOGGING__RETENTION_DAYS", "30")
    monkeypatch.setattr(entry, "setup_logging", stub)
    with pytest.raises(RuntimeError, match="stop here"):
        entry.main(["--check"])
    assert seen == [recent]
