"""The names on JenJo's PC that Whisper should know: folders and programs.

Whisper reads its prompt as the words said just before, so a name in the prompt is written
right when it is said: "Tesi" instead of "desi" or "attesi", "PSS" instead of "PZ" (logs of
25/10). The names come from the PC itself (Desktop and Documents folders, two levels; the
shortcuts on the Desktop and in the Start menu), the most recent first, plus the words JenJo
lists in [stt] vocabulary_words. The prompt stays short: Whisper keeps only its last 223
tokens, and a longer one is slower and invents more on noise.

Reading the folders takes a few milliseconds; it happens in a thread, at most once every
`refresh_s`, so a request never waits for it."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

LABELS = {
    "it": ("Cartelle", "Programmi"),
    "en": ("Folders", "Programs"),
}
SHORTCUTS = (".lnk", ".url")
# Shortcuts that are not programs anyone asks for by name.
JUNK_WORDS = frozenset(
    {"uninstall", "disinstalla", "readme", "leggimi", "help", "guida", "website", "sito"}
    | {"documentation", "documentazione", "license", "licenza", "manual", "manuale"}
)
# Folders every PC has, or that only hold program files: no help to Whisper.
SKIP_FOLDERS = frozenset(
    {"appdata", "node_modules", ".git", "$recycle.bin", "__pycache__", ".venv", "venv"}
    | {"desktop.ini", "my music", "my pictures", "my videos", "nuova cartella", "new folder"}
    | {"src", "tests", "build", "dist", "bin", "obj", "lib", "include", "logs", "temp", "tmp"}
)
MAX_NAME_CHARS = 28
MAX_NAME_PARTS = 3  # "bbport-windows" yes, "Claudio-AI-desktop-assistant" no
_PARTS = re.compile(r"[\s_\-.]+")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class Name:
    text: str
    modified: float


def spoken_form(name: str) -> str:
    """How the name is written in the prompt: an all-caps word of 4+ letters is a word in
    capitals ("TESI", "UNIVERSITA'") and goes in as a word ("Tesi"); shorter ones are
    acronyms ("PSS") and stay."""
    if name.isupper() and len(re.sub(r"\W", "", name)) >= 4:
        return name.capitalize()
    return name


def usable(name: str) -> bool:
    """A name someone would say: has letters, is short, and is not generic."""
    if not name or name.startswith((".", "$", "~")) or len(name) > MAX_NAME_CHARS:
        return False
    if name.casefold() in SKIP_FOLDERS or not _LETTER.search(name):
        return False
    parts = [p for p in _PARTS.split(name) if p]
    if len(parts) > MAX_NAME_PARTS:
        return False
    return not JUNK_WORDS.intersection(p.casefold() for p in parts)


def folder_names(roots: Iterable[Path], depth: int = 2) -> list[Name]:
    """The folders under each root, down to `depth` levels (Desktop\\UNIVERSITA'\\TESI)."""
    found: list[Name] = []
    level = [(Path(r), 1) for r in roots]
    while level:
        folder, d = level.pop(0)
        try:
            with os.scandir(folder) as entries:
                listed = list(entries)
        except OSError:
            continue
        for entry in listed:
            try:
                # Junctions ("Dati applicazioni") lead to folders seen elsewhere, or loop.
                if not entry.is_dir(follow_symlinks=False) or entry.is_symlink():
                    continue
                modified = entry.stat(follow_symlinks=False).st_mtime
            except OSError:
                continue
            if entry.name.casefold() in SKIP_FOLDERS or entry.name.startswith("."):
                continue
            if usable(entry.name):
                found.append(Name(entry.name, modified))
            if d < depth:
                level.append((Path(entry.path), d + 1))
    return found


def shortcut_names(folders: Iterable[Path], depth: int = 2) -> list[Name]:
    """The programs: shortcuts (.lnk, .url) in these folders, one level of subfolders too
    (the Start menu groups them: "Steam\\Rocket League.url")."""
    found: list[Name] = []
    level = [(Path(f), 1) for f in folders]
    while level:
        folder, d = level.pop(0)
        try:
            with os.scandir(folder) as entries:
                listed = list(entries)
        except OSError:
            continue
        for entry in listed:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if d < depth:
                        level.append((Path(entry.path), d + 1))
                    continue
                stem, ext = os.path.splitext(entry.name)
                if ext.lower() not in SHORTCUTS or not usable(stem):
                    continue
                found.append(Name(stem, entry.stat().st_mtime))
            except OSError:
                continue
    return found


def pick(names: Sequence[Name], limit: int, seen: set[str]) -> list[str]:
    """The `limit` most recently changed names, each once (case does not count)."""
    out: list[str] = []
    for name in sorted(names, key=lambda n: n.modified, reverse=True):
        key = name.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(spoken_form(name.text))
        if len(out) >= limit:
            break
    return out


def prompt_text(
    language: str, words: Sequence[str], folders: Sequence[str], programs: Sequence[str]
) -> str:
    """ "Cartelle: Tesi, PSS. Programmi: Rocket League." (JenJo's own words go first)."""
    folder_label, program_label = LABELS.get(language, LABELS["en"])
    parts = []
    first = [*words, *folders]
    if first:
        parts.append(f"{folder_label}: {', '.join(first)}.")
    if programs:
        parts.append(f"{program_label}: {', '.join(programs)}.")
    return " ".join(parts)


def default_folders(home: Path) -> list[Path]:
    """Desktop and Documents, where they really are (often moved to OneDrive)."""
    from claudio.router.actions import KNOWN_FOLDERS, known_folder

    desktop = known_folder("{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}") or home / "Desktop"
    documents = known_folder(KNOWN_FOLDERS["documents"]) or home / "Documents"
    return [desktop, documents]


def default_shortcut_folders(desktop: Path) -> list[Path]:
    """The Desktop and the two Start menus (the user's and everyone's)."""
    folders = [desktop]
    for var in ("APPDATA", "PROGRAMDATA"):
        base = os.environ.get(var)
        if base:
            folders.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return folders


class Vocabulary:
    """The prompt's names, read in a thread and kept for `refresh_s`."""

    def __init__(
        self,
        folders: Callable[[], Sequence[Path]],
        shortcuts: Callable[[], Sequence[Path]],
        words: Sequence[str] = (),
        max_folders: int = 25,
        max_programs: int = 12,
        refresh_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._folders = folders
        self._shortcuts = shortcuts
        self.words = [w.strip() for w in words if w.strip()]
        self.max_folders = max_folders
        self.max_programs = max_programs
        self.refresh_s = refresh_s
        self.clock = clock
        self._names: tuple[list[str], list[str]] | None = None
        self._read_at = float("-inf")
        self._lock = threading.Lock()
        self._reading = False

    def refresh(self) -> None:
        """Start a reading in the background (nothing if one is running)."""
        with self._lock:
            if self._reading:
                return
            self._reading = True
        threading.Thread(target=self.read_now, name="vocabulary", daemon=True).start()

    def read_now(self) -> None:
        start = time.perf_counter()
        try:
            seen = {w.casefold() for w in self.words}
            folders = pick(folder_names(self._folders()), self.max_folders, seen)
            programs = pick(shortcut_names(self._shortcuts()), self.max_programs, seen)
            self._names = (folders, programs)
            self._read_at = self.clock()
            log.info(
                "Vocabolario di Whisper: %d cartelle, %d programmi (%.0f ms)",
                len(folders),
                len(programs),
                (time.perf_counter() - start) * 1000,
            )
        except Exception:
            log.exception("Lettura dei nomi per Whisper non riuscita")
        finally:
            with self._lock:
                self._reading = False

    def text(self, language: str) -> str:
        """The names for Whisper's prompt; JenJo's words only until the first reading ends.
        An old list is still used while a new one is read."""
        if self.clock() - self._read_at > self.refresh_s:
            self.refresh()
        folders, programs = self._names or ([], [])
        return prompt_text(language, self.words, folders, programs)
