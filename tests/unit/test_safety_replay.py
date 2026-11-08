"""The scene guard and the non-yes summary of the final safety test."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import safety_replay as sr


def ok(ran: bool) -> sr.Result:
    return sr.Result(ran, ran, False, [], False)


def test_a_scene_that_raises_becomes_its_error_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any) -> sr.Result:
        raise RuntimeError("whisper si è rotto")

    monkeypatch.setattr(sr, "run_scene", boom)
    result = sr.run_scene_guarded(None, Path("a.wav"), Path("b.wav"), "p")  # type: ignore[arg-type]
    assert result.error == "RuntimeError: whisper si è rotto"
    assert not result.ran and result.fails_a_non_yes


def test_a_clean_scene_is_returned_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sr, "run_scene", lambda *_: ok(False))
    result = sr.run_scene_guarded(None, Path("a.wav"), Path("b.wav"), "p")  # type: ignore[arg-type]
    assert result.error == "" and not result.fails_a_non_yes


def test_the_summary_splits_old_phrases_from_new_traps() -> None:
    old, new = Path("twin/vecchia__0.wav"), Path("twin/trappola__0.wav")
    results = [(old, ok(True)), (old, ok(False)), (new, sr.error_result(ValueError("x")))]
    line = sr.old_new_summary(results, {"vecchia": 2, "trappola": 1})
    assert line == "Non-sì che hanno eseguito o sono andati in errore: vecchie 1/2, nuove 1/1"


def report_for(results: list[tuple[str, str, sr.Result]]) -> list[str]:
    groups: dict[str, list[tuple[Path, sr.Result]]] = {}
    for group, stem, result in results:
        groups.setdefault(group, []).append((Path(f"{group}/{stem}.wav"), result))
    non_yes = [item for name in ("twin", "offtopic") for item in groups.get(name, [])]
    takes_of = {"vecchia": 2, "trappola": 1, "ho_fame": 1}
    return sr.report_lines(groups, non_yes, groups.get("yes", []), [], takes_of)


def test_an_errored_non_yes_is_an_error_line_and_a_failure_in_group_and_totals() -> None:
    boom = sr.error_result(RuntimeError("scena rotta"))
    lines = report_for(
        [
            ("twin", "vecchia__0", ok(False)),
            ("offtopic", "ho_fame__0", boom),
            ("offtopic", "ho_fame__1", ok(False)),
            ("yes", "si__0", ok(True)),
        ]
    )
    text = "\n".join(lines)
    assert "Non-sì che hanno eseguito o sono andati in errore: 1/3" in text
    assert "offtopic/ho_fame__0: RuntimeError: scena rotta" in text
    assert "  offtopic  1/2" in text
    assert "  offtopic  ho_fame" in text and "1/2" in text.split("ho_fame")[-1].splitlines()[0]
    assert "vecchie 0/1, nuove 1/2" in text
    assert "  yes       1/1" in text


def test_a_run_without_errors_prints_no_error_section() -> None:
    lines = report_for([("twin", "vecchia__0", ok(False)), ("yes", "si__0", ok(True))])
    assert not any("errore (" in line for line in lines)
