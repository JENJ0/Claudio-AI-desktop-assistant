"""Folder and file search that finds what JenJo means (25/10: "spesso non trova ancora"):
accents and apostrophes, names Whisper spells the English way, filler words, the list of
every folder name read in the background."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from claudio.config import PcSettings
from claudio.pc import files
from claudio.pc.tools import PcTools, ToolCall
from tests.unit.test_pc_tools import FakeDesktop


@pytest.fixture
def home(tmp_path: Path) -> Path:
    (tmp_path / "Desktop" / "UNIVERSITA'" / "TESI").mkdir(parents=True)
    (tmp_path / "Desktop" / "PROGETTI" / "Claudio-AI-desktop-assistant").mkdir(parents=True)
    (tmp_path / "Documents" / "Project_Work").mkdir(parents=True)
    (tmp_path / "Games" / "Rocket League").mkdir(parents=True)
    (tmp_path / "AppData" / "Local" / "Progetti").mkdir(parents=True)  # skipped
    return tmp_path


@pytest.mark.parametrize(
    ("text", "folded"),
    [
        ("UNIVERSITA'", "universita"),
        ("Università", "universita"),
        ("Project_Work", "project work"),
        ("  Rocket   League ", "rocket league"),
        ("l’esame", "lesame"),
    ],
)
def test_fold(text: str, folded: str) -> None:
    assert files.fold(text) == folded


@pytest.mark.parametrize(
    ("heard", "real"),
    [("projecti", "PROGETTI"), ("progetti", "Progetti"), ("photo", "foto"), ("kiave", "chiave")],
)
def test_names_that_sound_the_same(heard: str, real: str) -> None:
    assert files.sound(heard) == files.sound(real)


def test_filler_words_are_dropped_from_a_search() -> None:
    assert files.search_terms("cartella progetti") == ["progetti"]
    assert files.search_terms("la cartella della tesi") == ["tesi"]
    assert files.search_terms("cartella") == ["cartella"]  # nothing else: kept


@pytest.mark.parametrize(
    ("query", "name"),
    [
        ("università", "UNIVERSITA'"),
        ("projecti", "PROGETTI"),
        ("project work", "Project_Work"),
        ("rocket", "Rocket League"),
    ],
)
def test_walk_matches_accents_spellings_and_separators(home: Path, query: str, name: str) -> None:
    walked = files.walk(files.search_terms(query), [home], "folder", 5, 2.0, similar=3)
    assert name in [f.path.name for f in walked.found]


def test_one_word_like_a_word_of_a_longer_name(home: Path) -> None:
    walked = files.walk(["roket"], [home], "folder", 5, 2.0, similar=3)
    assert [f.path.name for f in walked.found] == ["Rocket League"]  # sounds the same
    walked = files.walk(["rocked"], [home], "folder", 5, 2.0, similar=3)
    assert walked.found == [] and walked.similar[0].path.name == "Rocket League"


def read_names(home: Path) -> files.FolderNames:
    names = files.FolderNames(lambda: [home], refresh_s=900, max_folders=1000)
    names.read_now()
    return names


def test_folder_names_find_every_folder(home: Path) -> None:
    names = read_names(home)
    listed = {f.path.name for f in names.folders() or []}
    assert {"TESI", "PROGETTI", "Rocket League", "Claudio-AI-desktop-assistant"} <= listed
    assert "AppData" not in listed and "Local" not in listed
    assert [f.path.name for f in names.find(["tesi"], 5, 3).found] == ["TESI"]
    assert [f.path.name for f in names.find(["claudio", "assistant"], 5, 3).found] == [
        "Claudio-AI-desktop-assistant"
    ]


def test_folder_names_offer_similar_names(home: Path) -> None:
    names = read_names(home)
    found = names.find(["desi"], 5, 3)
    assert found.found == [] and found.similar[0].path.name == "TESI"
    assert names.find(["zzqx"], 5, 3).similar == []


def test_folder_names_by_name(home: Path) -> None:
    names = read_names(home)
    assert [p.name for p in names.named("tesi")] == ["TESI"]
    assert [p.name for p in names.named("Projecti")] == ["PROGETTI"]  # same sound
    assert names.named("zzqx") == []


def test_folder_names_stop_at_the_limit(home: Path) -> None:
    names = files.FolderNames(lambda: [home], refresh_s=900, max_folders=2)
    names.read_now()
    assert len(names.folders() or []) <= 3  # one folder's children may pass the limit


def test_a_stale_list_is_read_again_in_the_background(home: Path) -> None:
    now = [0.0]
    names = files.FolderNames(lambda: [home], refresh_s=10, max_folders=1000, clock=lambda: now[0])
    names.read_now()
    (home / "Nuova").mkdir()
    assert "Nuova" not in {f.path.name for f in names.folders() or []}
    now[0] = 20.0
    names.folders()  # starts the new read
    names.read_now()  # waits for it, then reads once more
    assert "Nuova" in {f.path.name for f in names.folders() or []}


@pytest.fixture
def tools(home: Path) -> PcTools:
    desk = FakeDesktop()
    desk.index = []
    t = PcTools(desk, PcSettings(), home=home)
    t.folders.read_now()
    yield t
    t.close()


async def test_search_finds_a_folder_from_the_list(tools: PcTools) -> None:
    out = await tools.run(
        ToolCall("1", "search_files", {"query": "cartella università", "kind": "folder"})
    )
    assert out.text.startswith("1 found:") and "UNIVERSITA'" in out.text
    assert ("index", ["università"]) not in tools.desktop.done  # folder found: no index query


async def test_search_still_walks_for_a_folder_made_after_the_list(
    tools: PcTools, home: Path
) -> None:
    (home / "Desktop" / "Nuova cartella esame").mkdir()
    out = await tools.run(ToolCall("1", "search_files", {"query": "esame", "kind": "folder"}))
    assert "Nuova cartella esame" in out.text


async def test_search_offers_similar_names_from_the_whole_list(tools: PcTools) -> None:
    out = await tools.run(ToolCall("1", "search_files", {"query": "desi", "kind": "folder"}))
    assert "Similar names" in out.text and "TESI" in out.text


async def test_open_path_finds_a_misspelled_folder(tools: PcTools, home: Path) -> None:
    out = await tools.run(ToolCall("1", "open_path", {"path": "~\\Projecti"}))
    assert not out.error
    assert ("open", (home / "Desktop" / "PROGETTI").resolve()) in tools.desktop.done


async def test_open_path_ignores_accents(tools: PcTools, home: Path) -> None:
    out = await tools.run(ToolCall("1", "open_path", {"path": "~\\Desktop\\Università"}))
    assert not out.error
    assert ("open", (home / "Desktop" / "UNIVERSITA'").resolve()) in tools.desktop.done


def test_spaces_do_not_count(home: Path) -> None:
    """26/10: "BB Port Windows" and "BBPortWindows" for Desktop\\BLOODBORNE\\bbport-windows."""
    (home / "Desktop" / "BLOODBORNE" / "bbport-windows").mkdir(parents=True)
    for query in ("BB Port Windows", "BBPortWindows"):
        walked = files.walk(files.search_terms(query), [home], "folder", 5, 2.0)
        assert [f.path.name for f in walked.found] == ["bbport-windows"]


def test_a_name_holding_some_words_is_similar(tmp_path: Path) -> None:
    """25/10: "Project Work" found nothing; the folder is UNIVERSITA'\\PROGETTO IUM."""
    (tmp_path / "UNIVERSITA'" / "PROGETTO IUM").mkdir(parents=True)
    (tmp_path / "UNIVERSITA'" / "PSS").mkdir()
    walked = files.walk(["Project", "Work"], [tmp_path], "folder", 5, 2.0, similar=3)
    assert walked.found == [] and walked.similar[0].path.name == "PROGETTO IUM"


# --- IT-2 R1: the walk, step by step ----------------------------------------------------------


def names(found: list[files.Found]) -> list[str]:
    return [f.path.name for f in found]


def test_only_the_best_similar_names_are_kept_best_first(tmp_path: Path) -> None:
    """More candidates than `similar`: the closest ones stay, closest first; equal scores
    are told apart by the path, the later one first."""
    for name in ("tesixyz", "tesi", "tesa", "tess", "teso"):  # 76.9, then four times 75.0
        (tmp_path / name).mkdir()
    walked = files.walk(["tesx"], [tmp_path], "folder", 5, 2.0, similar=3)
    assert walked.found == []
    assert names(walked.similar) == ["tesixyz", "tess", "teso"]


def test_equal_scores_in_different_folders_go_by_path(tmp_path: Path) -> None:
    for parent in ("a", "b"):
        (tmp_path / parent / "tesi").mkdir(parents=True)
    walked = files.walk(["tesx"], [tmp_path], "folder", 5, 2.0, similar=2)
    assert [f.path.parent.name for f in walked.similar] == ["b", "a"]
    walked = files.walk(["tesx"], [tmp_path], "folder", 5, 2.0, similar=1)
    assert [f.path.parent.name for f in walked.similar] == ["b"]


class FakeEntry:
    """What os.scandir yields, with a stat() that may fail (a file gone mid-walk)."""

    def __init__(self, path: Path, mtime: float | None = 1.0) -> None:
        self.name = path.name
        self.path = str(path)
        self._mtime = mtime

    def stat(self) -> SimpleNamespace:
        if self._mtime is None:
            raise OSError("gone")
        return SimpleNamespace(st_mtime=self._mtime)


def list_folders(monkeypatch: pytest.MonkeyPatch, tree: dict[Path, list[FakeEntry]]) -> None:
    monkeypatch.setattr(
        files, "_children", lambda folder: [(e, True) for e in tree.get(folder, [])]
    )


def test_a_hit_that_cannot_be_read_is_neither_found_nor_similar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "tesi").mkdir()
    (tmp_path / "tesi esame").mkdir()
    list_folders(
        monkeypatch,
        {tmp_path: [FakeEntry(tmp_path / "tesi", None), FakeEntry(tmp_path / "tesi esame", 5.0)]},
    )
    walked = files.walk(["tesi"], [tmp_path], "folder", 5, 2.0, similar=3)
    assert names(walked.found) == ["tesi esame"]
    only = {tmp_path: [FakeEntry(tmp_path / "tesi", None)]}
    list_folders(monkeypatch, only)
    walked = files.walk(["tesi"], [tmp_path], "folder", 5, 2.0, similar=3)
    assert walked.found == [] and walked.similar == []  # it matched: it is not "like" anything


def test_a_similar_name_that_vanished_is_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "tess").mkdir()  # "tesa" is not on disk any more by the time it is dated
    list_folders(
        monkeypatch, {tmp_path: [FakeEntry(tmp_path / "tesa"), FakeEntry(tmp_path / "tess")]}
    )
    walked = files.walk(["tesx"], [tmp_path], "folder", 5, 2.0, similar=3)
    assert names(walked.similar) == ["tess"]


def test_an_expired_budget_walks_nothing(tmp_path: Path) -> None:
    (tmp_path / "tesi").mkdir()
    assert files.walk(["tesi"], [tmp_path], "folder", 5, -1.0, similar=3) == files.Walk()


def test_the_deadline_is_checked_per_folder_not_per_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clock runs out while the first folder is read: all of its entries are still looked
    at, the next folder is not."""
    first, second = tmp_path / "first", tmp_path / "second"
    for folder in (first, second):
        (folder / "tesi uno").mkdir(parents=True)
        (folder / "tesi due").mkdir()
    readings = iter([0.0, 0.5])  # the deadline is 1.0; then the first folder is let in

    def clock() -> float:
        return next(readings, 99.0)  # past the deadline from then on

    monkeypatch.setattr(files, "time", SimpleNamespace(monotonic=clock))
    walked = files.walk(["tesi"], [first, second], "folder", 5, 1.0)
    assert sorted(names(walked.found)) == ["tesi due", "tesi uno"]
    assert {f.path.parent for f in walked.found} == {first}


def link_to(target: Path, link: Path) -> None:
    """A symlink, or a junction on Windows (no privilege needed); skips when neither works."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("cannot make a link here")
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=False)
        if made.returncode:
            pytest.skip("cannot make a junction here")


def test_links_are_not_followed(tmp_path: Path) -> None:
    """A junction leads to a folder seen elsewhere, or loops: its content is not listed again."""
    (tmp_path / "real" / "tesi").mkdir(parents=True)
    link_to(tmp_path / "real", tmp_path / "shortcut")
    walked = files.walk(["tesi"], [tmp_path], "folder", 5, 2.0)
    assert [f.path for f in walked.found] == [tmp_path / "real" / "tesi"]


def test_folders_are_walked_even_when_only_files_are_wanted_but_not_the_skipped_ones(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs" / "deep").mkdir(parents=True)
    (tmp_path / "docs" / "deep" / "tesi.pdf").write_bytes(b"x")
    for skipped in ("node_modules", ".hidden"):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / "tesi.pdf").write_bytes(b"x")
    (tmp_path / "tesi").mkdir()  # a folder is not a document
    walked = files.walk(["tesi"], [tmp_path], "document", 5, 2.0)
    assert [f.path for f in walked.found] == [tmp_path / "docs" / "deep" / "tesi.pdf"]


def test_a_matching_folder_is_still_walked_for_the_ones_inside_it(tmp_path: Path) -> None:
    (tmp_path / "tesi" / "tesi vecchia").mkdir(parents=True)
    walked = files.walk(["tesi"], [tmp_path], "folder", 5, 2.0)
    assert sorted(names(walked.found)) == ["tesi", "tesi vecchia"]
