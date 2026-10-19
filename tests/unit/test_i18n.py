"""Every program message has an English version with the same placeholders."""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from claudio import i18n
from claudio.i18n import EN, TranslateFilter, set_language, tr

SRC = Path(i18n.__file__).parent
LOG_METHODS = {"debug", "info", "warning", "error", "exception"}
NOTHING_TO_TRANSLATE = {"%s", "%s %s", "CLAUDIO %s", "WARN %s: %s"}
PERCENT = re.compile(r"%[-#0 +]*\d*(?:\.\d+)?[sdrf]")
NAMES = re.compile(r"\{(\w+)\}")


def messages() -> Iterator[tuple[str, str]]:
    """(where, text) of every log message and tr() text written as a literal in src/."""
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn, first = node.func, node.args[0]
            is_log = (
                isinstance(fn, ast.Attribute)
                and fn.attr in LOG_METHODS
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "log"
            )
            is_tr = isinstance(fn, ast.Name) and fn.id == "tr"
            if is_log or is_tr:
                where = f"{path.relative_to(SRC)}:{node.lineno}"
                assert isinstance(first, ast.Constant), f"{where}: message must be a literal"
                yield where, first.value


@pytest.fixture
def english() -> Iterator[None]:
    set_language("en")
    yield
    set_language("it")


def test_every_message_is_translated() -> None:
    found = list(messages())
    assert len(found) > 60
    missing = [f"{w}: {t!r}" for w, t in found if t not in EN and t not in NOTHING_TO_TRANSLATE]
    assert not missing, "\n".join(missing)


@pytest.mark.parametrize(("italian", "english"), list(EN.items()))
def test_translation_keeps_the_placeholders(italian: str, english: str) -> None:
    assert PERCENT.findall(italian) == PERCENT.findall(english)
    assert sorted(NAMES.findall(italian)) == sorted(NAMES.findall(english))


def test_no_stale_translations() -> None:
    used = {text for _, text in messages()}
    assert not set(EN) - used


def test_tr_and_the_log_filter(english: None) -> None:
    assert tr("Modello {name} mancante in {folder}", name="x", folder="m") == "Model x missing in m"
    record = logging.makeLogRecord({"msg": "Wake word: %s", "args": ("claudio",)})
    record.msg = "Wake word sotto soglia: %s %.2f (soglia %.2f)"
    record.args = ("claudio", 0.31, 0.5)
    assert TranslateFilter().filter(record)
    assert record.getMessage() == "Wake word below threshold: claudio 0.31 (threshold 0.50)"


def test_italian_is_the_default() -> None:
    assert i18n.current() == "it"
    assert (
        tr("Modello {name} mancante in {folder}", name="x", folder="m") == "Modello x mancante in m"
    )
    with pytest.raises(ValueError, match="language"):
        set_language("fr")


def test_errors_follow_the_language(english: None, tmp_path: Path) -> None:
    from claudio.tts.speaker import voice_path

    with pytest.raises(FileNotFoundError, match="Piper voice 'x' missing"):
        voice_path(tmp_path, "x")
