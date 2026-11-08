"""The safety-v2 recorder: one open stream, rejected quiet takes, resume, files by group."""

from __future__ import annotations

import sys
import types
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import record_samples as rs

PHRASES_FILE = PROJECT_ROOT / "scripts" / "safety_phrases.tsv"
GROUPS = {"yes", "question", "twin", "rewrite", "offtopic", "noise"}


class FakeInputStream:
    """Stands for sounddevice.InputStream: every read gives the next level of `levels`."""

    opened: ClassVar[list[FakeInputStream]] = []
    levels: ClassVar[list[int]] = []
    buffered: ClassVar[int] = 0

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.started = self.stopped = self.closed = False
        self.reads: list[int] = []
        self.read_available = FakeInputStream.buffered
        FakeInputStream.opened.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        self.reads.append(frames)
        self.read_available = 0
        level = FakeInputStream.levels.pop(0) if FakeInputStream.levels else 20000
        return np.full((frames, 1), level, dtype=np.int16), False


@pytest.fixture
def fake_mic(monkeypatch: pytest.MonkeyPatch) -> type[FakeInputStream]:
    FakeInputStream.opened = []
    FakeInputStream.levels = []
    FakeInputStream.buffered = 0
    monkeypatch.setitem(
        sys.modules, "sounddevice", types.SimpleNamespace(InputStream=FakeInputStream)
    )
    return FakeInputStream


def keys(monkeypatch: pytest.MonkeyPatch, *answers: str) -> list[str]:
    """Scripted Enter presses; the list collects the prompts shown."""
    shown: list[str] = []
    pending = iter(answers)

    def fake_input(prompt: str = "") -> str:
        shown.append(prompt)
        return next(pending, "")

    monkeypatch.setattr("builtins.input", fake_input)
    return shown


def prompts_for(tmp_path: Path, rows: str, out: Path) -> list[tuple[str, str, str, float]]:
    tsv = tmp_path / "phrases.tsv"
    tsv.write_text(rows, encoding="utf-8")
    return rs.safety_v2_prompts(tsv, 3.0, out)


def test_one_stream_is_opened_for_the_whole_session(
    fake_mic: type[FakeInputStream], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "safety_v2"
    prompts = prompts_for(
        tmp_path, "no\tnon ancora\ttwin\t1\nno\tsi vede\trewrite\t1\nyes\tcerto\tyes\t1\n", out
    )
    keys(monkeypatch)
    assert rs.run_safety_v2(prompts, out, None) == 3
    assert len(fake_mic.opened) == 1
    stream = fake_mic.opened[0]
    assert stream.started and stream.stopped and stream.closed
    assert stream.reads == [48000] * 3  # three takes, three slices of the same stream


def test_a_take_under_the_peak_check_is_rejected_and_asked_again(
    fake_mic: type[FakeInputStream],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "safety_v2"
    prompts = prompts_for(tmp_path, "no\tnon ancora\ttwin\t1\n", out)
    fake_mic.levels = [100, 20000]  # whisper-quiet, then a real take
    shown = keys(monkeypatch)
    assert rs.run_safety_v2(prompts, out, None) == 1
    assert "troppo basso" in capsys.readouterr().out
    assert len(fake_mic.opened[0].reads) == 2
    assert len([s for s in shown if "[1/1]" in s]) == 2  # the same prompt came back
    data, _ = sf.read(out / "twin" / "non_ancora__0.wav", dtype="int16")
    assert int(np.abs(data).max()) == 20000  # the quiet take was never saved


def test_a_restarted_session_records_only_the_missing_takes(
    fake_mic: type[FakeInputStream], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "safety_v2"
    rows = "no\tnon ancora\ttwin\t2\nyes\tcerto\tyes\t2\n"
    first = prompts_for(tmp_path, rows, out)
    assert len(first) == 4
    keys(monkeypatch, "", "", "q")  # two takes, then "q + Invio"
    assert rs.run_safety_v2(first, out, None) == 2
    again = prompts_for(tmp_path, rows, out)
    assert len(again) == 2
    keys(monkeypatch)
    assert rs.run_safety_v2(again, out, None) == 2
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav"))
    assert names == ["twin/non_ancora__0.wav", "twin/non_ancora__1.wav",
                     "yes/certo__0.wav", "yes/certo__1.wav"]  # fmt: skip


def test_files_land_in_a_folder_per_group(
    fake_mic: type[FakeInputStream], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "safety_v2"
    rows = "no\tmmh\tnoise\t1\nunclear\tsì?\tquestion\t1\nno\tsi vede\trewrite\t1\n"
    keys(monkeypatch)
    rs.run_safety_v2(prompts_for(tmp_path, rows, out), out, None)
    assert {p.parent.name for p in out.rglob("*.wav")} == {"noise", "question", "rewrite"}


def test_the_question_takes_tell_how_to_say_them(tmp_path: Path) -> None:
    prompts = prompts_for(tmp_path, "unclear\tsì?\tquestion\t1\nno\tmmh\tnoise\t1\n", tmp_path)
    said = {stem.split("/")[0]: text for _, text, stem, _ in prompts}
    assert "tono che sale" in said["question"]
    assert "tono che sale" not in said["noise"]


# --- the list of phrases -------------------------------------------------------------------


def test_the_list_makes_145_takes_in_the_planned_groups() -> None:
    rows = rs.load_safety_v2(PHRASES_FILE)
    takes = Counter()
    for row in rows:
        takes[row.group] += row.takes
    assert sum(takes.values()) == 145
    assert takes["yes"] == 44 and takes["question"] == 8
    assert sum(n for g, n in takes.items() if g not in ("yes", "question")) == 93
    assert set(takes) == GROUPS


def test_the_non_yes_takes_are_62_plus_31_single_take_traps() -> None:
    rows = [r for r in rs.load_safety_v2(PHRASES_FILE) if r.group not in ("yes", "question")]
    assert sum(r.takes == 2 for r in rows) == 31
    assert sum(r.takes == 1 for r in rows) == 31


def test_every_group_has_new_traps_and_no_trap_has_a_question_mark() -> None:
    traps = [r for r in rs.load_safety_v2(PHRASES_FILE) if r.takes == 1]
    assert {r.group for r in traps} == {"twin", "rewrite", "offtopic", "noise"}
    assert not [r.phrase for r in traps if "?" in r.phrase]


def test_no_phrase_is_listed_twice() -> None:
    rows = rs.load_safety_v2(PHRASES_FILE)
    assert len({r.phrase for r in rows}) == len(rows)
    stems = [
        (r.group, rs.slug(r.phrase)) for r in rows
    ]  # "sì" and "sì?" share a slug, not a folder
    assert len(stems) == len(set(stems))


def test_what_was_buffered_before_enter_is_drained_not_recorded(
    fake_mic: type[FakeInputStream], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "safety_v2"
    prompts = prompts_for(tmp_path, "no\tnon ancora\ttwin\t1\n", out)
    fake_mic.buffered = 24000  # the stream kept listening while JenJo read the line
    keys(monkeypatch)
    rs.run_safety_v2(prompts, out, None)
    assert fake_mic.opened[0].reads == [24000, 48000]  # dropped first, then the take


def test_a_quiet_noise_take_can_be_accepted_anyway(
    fake_mic: type[FakeInputStream], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "safety_v2"
    prompts = prompts_for(tmp_path, "unclear\tpsst\tnoise\t1\n", out)
    fake_mic.levels = [100]
    keys(monkeypatch, "", "a")
    assert rs.run_safety_v2(prompts, out, None) == 1
    assert [p.name for p in out.rglob("*.wav")] == ["psst__0.wav"]
    assert len(fake_mic.opened[0].reads) == 1  # kept, not recorded again


def test_a_quiet_phrase_take_is_never_accepted_anyway(
    fake_mic: type[FakeInputStream], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "safety_v2"
    prompts = prompts_for(tmp_path, "no\tnon ancora\ttwin\t1\n", out)
    fake_mic.levels = [100, 100]
    keys(monkeypatch, "", "a", "q")
    assert rs.run_safety_v2(prompts, out, None) == 0
    assert not list(out.rglob("*.wav"))


def test_the_question_takes_are_spread_through_the_session(tmp_path: Path) -> None:
    prompts = rs.safety_v2_prompts(PHRASES_FILE, 3.0, tmp_path)
    assert len(prompts) == 145
    where = [i for i, (_, _, stem, _) in enumerate(prompts) if stem.startswith("question/")]
    assert len(where) == 8
    assert max(b - a for a, b in pairwise(where)) > 3  # not in one block
    assert not any(c - a < 3 for a, c in zip(where, where[2:], strict=False))  # no 3 in a row


def test_missing_takes_names_what_the_set_lacks_and_what_it_should_not_have(
    tmp_path: Path,
) -> None:
    tsv = tmp_path / "p.tsv"
    tsv.write_text("no\tnon ancora\ttwin\t2\nyes\tcerto\tyes\t1\n", encoding="utf-8")
    out = tmp_path / "safety_v2"
    for name in ("twin/non_ancora__0.wav", "yes/certo__0.wav", "yes/altro__0.wav"):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(b"")
    assert rs.missing_takes(tsv, out) == (["twin/non_ancora__1.wav"], ["yes/altro__0.wav"])
    empty = tmp_path / "empty"
    empty.mkdir()
    assert len(rs.missing_takes(PHRASES_FILE, empty)[0]) == 145


@pytest.mark.parametrize("done", [0, 1, 4, 7, 8])
def test_a_restart_asks_exactly_the_missing_takes_of_an_8_take_row(
    fake_mic: type[FakeInputStream], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, done: int
) -> None:
    out = tmp_path / "safety_v2"
    rows = "unclear\tsì?\tquestion\t8\n"
    first = prompts_for(tmp_path, rows, out)
    keys(monkeypatch, *[""] * done, "q")
    rs.run_safety_v2(first, out, None)
    saved = {p.stem for p in (out / "question").glob("*.wav")} if done else set()
    assert len(saved) == done
    again = prompts_for(tmp_path, rows, out)
    assert {Path(stem).name for _, _, stem, _ in again} == {f"s__{n}" for n in range(8)} - saved


def test_a_clipped_noise_take_is_always_asked_again(
    fake_mic: type[FakeInputStream], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "safety_v2"
    prompts = prompts_for(tmp_path, "unclear\tpsst\tnoise\t1\n", out)
    fake_mic.levels = [32767, 32767]
    keys(monkeypatch, "", "a", "q")
    assert rs.run_safety_v2(prompts, out, None) == 0
    assert len(fake_mic.opened[0].reads) == 2  # "a" did not keep it: it was recorded again


@pytest.mark.parametrize("loader", ["load_safety_v2", "load_safety_phrases"])
@pytest.mark.parametrize(
    ("text", "line", "what"),
    [
        ("maybe\tmmh\tnoise\t1\n", "1", "yes, no, unclear"),
        ("# c\n\nno\t \tnoise\t1\n", "3", "frase vuota"),
    ],
)
def test_a_bad_phrase_line_is_named_with_its_number_and_the_problem(
    tmp_path: Path, loader: str, text: str, line: str, what: str
) -> None:
    tsv = tmp_path / "p.tsv"
    tsv.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=rf"riga {line}\b.*{what}"):
        getattr(rs, loader)(tsv)


def test_a_safety_v2_line_with_too_few_columns_is_named(tmp_path: Path) -> None:
    tsv = tmp_path / "p.tsv"
    tsv.write_text("no\tmmh\tnoise\t1\nno\tsolo due\n", encoding="utf-8")  # 2 of 4 columns
    with pytest.raises(ValueError, match=r"riga 2\b.*colonne"):
        rs.load_safety_v2(tsv)


def test_a_takes_column_that_is_not_a_number_is_named(tmp_path: Path) -> None:
    tsv = tmp_path / "p.tsv"
    tsv.write_text("no\tmmh\tnoise\tdue\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"riga 1\b.*prese"):
        rs.load_safety_v2(tsv)
