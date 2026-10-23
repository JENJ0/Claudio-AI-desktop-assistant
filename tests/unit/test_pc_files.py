"""Which paths Claudio may touch, folder search without the index, moving files (v0.3.1)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from claudio.pc import files
from claudio.pc.files import PathPolicy, PathRefused


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "JenJo"
    (home / "Downloads").mkdir(parents=True)
    (home / "Documents").mkdir()
    (home / "Downloads" / "fattura luce.pdf").write_text("x")
    (home / "Downloads" / "setup.exe").write_text("x")
    (tmp_path / "outside.txt").write_text("x")
    return home


@pytest.fixture
def policy(home: Path) -> PathPolicy:
    return PathPolicy(["~"], home)


def test_tilde_and_absolute_paths_inside_home(policy: PathPolicy, home: Path) -> None:
    assert policy.check("~" + os.sep + "Downloads") == (home / "Downloads").resolve()
    assert policy.check("~/Downloads/fattura luce.pdf").name == "fattura luce.pdf"
    assert policy.check(str(home / "Documents")) == (home / "Documents").resolve()


@pytest.mark.parametrize("bad", ["", "  ", "Downloads", "\\\\server\\share\\x", "//server/x"])
def test_refused_paths(policy: PathPolicy, bad: str) -> None:
    with pytest.raises(PathRefused):
        policy.check(bad)


def test_outside_home_is_refused(policy: PathPolicy, home: Path) -> None:
    with pytest.raises(PathRefused, match="outside"):
        policy.check(str(home.parent / "outside.txt"))
    with pytest.raises(PathRefused, match="outside"):
        policy.check("~/../outside.txt")  # .. resolved before the check


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_a_link_out_of_home_is_refused(policy: PathPolicy, home: Path) -> None:
    link = home / "link"
    try:
        link.symlink_to(home.parent)
    except OSError:
        pytest.skip("symlinks need privileges here")
    with pytest.raises(PathRefused, match="outside"):
        policy.check("~/link/outside.txt")


def test_anywhere(home: Path) -> None:
    free = PathPolicy(["*"], home)
    assert free.check(str(home.parent / "outside.txt")).name == "outside.txt"


def test_programs_are_not_opened(policy: PathPolicy) -> None:
    files.check_openable(policy.check("~/Downloads/fattura luce.pdf"))
    files.check_openable(policy.check("~/Downloads"))
    with pytest.raises(PathRefused, match="program"):
        files.check_openable(policy.check("~/Downloads/setup.exe"))


def test_search_terms_are_cleaned() -> None:
    assert files.search_terms("fattura' OR 1=1; --") == ["fattura", "OR", "1", "1"]
    assert files.search_terms("report_2026.pdf") == ["report_2026.pdf"]
    with pytest.raises(PathRefused):
        files.search_terms("'%;")


def test_walk_search(policy: PathPolicy, home: Path) -> None:
    (home / "AppData").mkdir()
    (home / "AppData" / "fattura cache.pdf").write_text("x")
    found = files.walk_search(["FATTURA"], policy.search_roots(), None, 5, 2.0)
    assert [f.path.name for f in found] == ["fattura luce.pdf"]  # AppData skipped
    assert files.walk_search(["fattura"], policy.search_roots(), "picture", 5, 2.0) == []
    folders = files.walk_search(["down"], policy.search_roots(), "folder", 5, 2.0)
    assert [f.path.name for f in folders] == ["Downloads"]


def test_move_into_a_folder(policy: PathPolicy, home: Path) -> None:
    src = policy.check("~/Downloads/fattura luce.pdf")
    target = files.move(src, "~/Documents", policy)
    assert target == (home / "Documents" / "fattura luce.pdf").resolve() and target.exists()
    assert not src.exists()


def test_move_never_overwrites(policy: PathPolicy, home: Path) -> None:
    (home / "Documents" / "fattura luce.pdf").write_text("old")
    src = policy.check("~/Downloads/fattura luce.pdf")
    with pytest.raises(PathRefused, match="already exists"):
        files.move(src, "~/Documents", policy)
    assert (home / "Documents" / "fattura luce.pdf").read_text() == "old"


def test_move_out_of_home_or_of_home_itself_is_refused(policy: PathPolicy, home: Path) -> None:
    src = policy.check("~/Downloads/fattura luce.pdf")
    with pytest.raises(PathRefused):
        files.move(src, str(home.parent), policy)
    with pytest.raises(PathRefused):
        files.move(policy.check("~"), "~/Documents", policy)
    with pytest.raises(PathRefused, match="inside itself"):
        files.move(policy.check("~/Downloads"), "~/Downloads/sub", policy)


def test_move_and_rename(policy: PathPolicy, home: Path) -> None:
    src = policy.check("~/Downloads/fattura luce.pdf")
    target = files.move(src, "~/Documents/luce ottobre.pdf", policy)
    assert target.name == "luce ottobre.pdf"


def test_shorten(home: Path) -> None:
    assert files.shorten(home / "Downloads" / "a.pdf", home) == os.sep.join(
        ["~", "Downloads", "a.pdf"]
    )
    assert files.shorten(Path("/other/a.pdf"), home) == str(Path("/other/a.pdf"))
