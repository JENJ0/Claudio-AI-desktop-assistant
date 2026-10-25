"""The names on the PC in Whisper's prompt (stt/vocabulary.py)."""

from __future__ import annotations

import os
from pathlib import Path

from claudio.stt.vocabulary import (
    Vocabulary,
    folder_names,
    pick,
    prompt_text,
    shortcut_names,
    spoken_form,
    usable,
)


def make_tree(root: Path) -> tuple[Path, Path]:
    desktop, docs = root / "Desktop", root / "Documents"
    for folder in (
        desktop / "UNIVERSITA'" / "TESI",
        desktop / "UNIVERSITA'" / "PSS",
        desktop / "BLOODBORNE" / "bbport-windows" / "deep" / "too-deep",
        desktop / "PROGETTI" / "Claudio-AI-desktop-assistant" / "src",
        desktop / ".hidden",
        desktop / "node_modules",
        docs / "Ricette",
    ):
        folder.mkdir(parents=True)
    (desktop / "Rocket League.url").write_text("[InternetShortcut]")
    (desktop / "notes.txt").write_text("not a shortcut")
    return desktop, docs


def test_folders_two_levels_deep_without_generic_or_long_names(tmp_path: Path) -> None:
    desktop, docs = make_tree(tmp_path)
    names = {n.text for n in folder_names([desktop, docs])}
    assert {"UNIVERSITA'", "TESI", "PSS", "BLOODBORNE", "bbport-windows", "PROGETTI"} <= names
    assert "Ricette" in names
    assert "deep" not in names  # third level
    assert "Claudio-AI-desktop-assistant" not in names  # four parts: nobody says that
    assert ".hidden" not in names and "node_modules" not in names


def test_programs_are_the_shortcuts(tmp_path: Path) -> None:
    desktop, _docs = make_tree(tmp_path)
    menu = tmp_path / "Start Menu" / "Steam"
    menu.mkdir(parents=True)
    (menu / "Uninstall Steam.lnk").write_text("")
    (menu / "Steam.lnk").write_text("")
    names = {n.text for n in shortcut_names([desktop, tmp_path / "Start Menu"])}
    assert names == {"Rocket League", "Steam"}


def test_spoken_form_and_usable_names() -> None:
    assert spoken_form("TESI") == "Tesi" and spoken_form("UNIVERSITA'") == "Universita'"
    assert spoken_form("PSS") == "PSS" and spoken_form("bbport-windows") == "bbport-windows"
    assert usable("Rocket League") and not usable("2024") and not usable("Nuova cartella")
    assert not usable("x" * 40)


def test_most_recent_first_and_each_name_once(tmp_path: Path) -> None:
    desktop, docs = make_tree(tmp_path)
    old = desktop / "PROGETTI"
    os.utime(old, (1, 1))
    picked = pick(folder_names([desktop, docs]), limit=50, seen={"pss"})
    assert picked[-1] == "Progetti"  # the oldest
    assert "PSS" not in picked  # already among JenJo's words
    assert len(pick(folder_names([desktop, docs]), limit=3, seen=set())) == 3


def test_prompt_text_in_the_language() -> None:
    assert prompt_text("it", ["Dainese"], ["Tesi", "PSS"], ["Rocket League"]) == (
        "Cartelle: Dainese, Tesi, PSS. Programmi: Rocket League."
    )
    assert prompt_text("en", [], [], ["Steam"]) == "Programs: Steam."
    assert prompt_text("it", [], [], []) == ""


def test_vocabulary_reads_in_the_background_and_refreshes(tmp_path: Path) -> None:
    desktop, docs = make_tree(tmp_path)
    now = [0.0]
    vocabulary = Vocabulary(
        lambda: [desktop, docs], lambda: [desktop], words=["Dainese"], clock=lambda: now[0]
    )
    vocabulary.read_now()
    text = vocabulary.text("it")
    assert text.startswith("Cartelle: Dainese, ") and "Tesi" in text
    assert text.endswith("Programmi: Rocket League.")
    (desktop / "NUOVA").mkdir()
    assert "Nuova" not in vocabulary.text("it")  # still the list read before
    now[0] = 601.0
    vocabulary.read_now()
    assert "Nuova" in vocabulary.text("it")


def test_before_the_first_reading_only_jenjos_words(tmp_path: Path) -> None:
    vocabulary = Vocabulary(lambda: [tmp_path / "missing"], lambda: [], words=["Dainese"])
    vocabulary.refresh = lambda: None  # type: ignore[method-assign]
    assert vocabulary.text("it") == "Cartelle: Dainese."
