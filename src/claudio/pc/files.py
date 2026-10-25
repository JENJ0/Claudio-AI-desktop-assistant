"""Which paths Claudio may touch, and the file actions that need no Windows API.

Every path Claude sends is resolved (symlinks and junctions followed) and must sit inside one
of the allowed roots (`[pc] roots`, by default JenJo's user folder). "*" allows any local
drive; network paths (\\\\server\\share) never. Opening a runnable file (.exe, .bat, .lnk...)
is refused: launching programs goes through `open_app`, which only knows the Start menu."""

from __future__ import annotations

import heapq
import os
import re
import shutil
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

ANYWHERE = "*"
# Opening one of these runs code: never a level-1 "open".
RUNNABLE = frozenset(
    {".exe", ".bat", ".cmd", ".com", ".ps1", ".psm1", ".psd1", ".vbs", ".vbe", ".js", ".jse"}
    | {".wsf", ".wsh", ".msi", ".msp", ".mst", ".scr", ".hta", ".cpl", ".lnk", ".url", ".reg"}
    | {".jar", ".pif", ".appref-ms", ".application", ".msc", ".inf", ".scf", ".gadget"}
    | {".settingcontent-ms", ".appx", ".msix", ".appinstaller", ".dll", ".sys", ".py", ".pyw"}
)
KINDS = ("document", "picture", "music", "video", "folder")
# Windows Search's System.Kind values for each kind (folders are matched by item type).
INDEX_KINDS = {"document": "document", "picture": "picture", "music": "music", "video": "video"}
_TERM = re.compile(r"[^\w.\- ]+", re.UNICODE)  # what a search may contain: no quotes, no SQL
MAX_TERMS = 5
SKIP_DIRS = frozenset({"appdata", "node_modules", ".git", "$recycle.bin", "__pycache__", ".venv"})
# rapidfuzz ratio of a name to a search that found nothing: "desi" and "attesi" -> "TESI" (75,
# 80). Below this the name does not sound like the search.
SIMILAR_CUTOFF = 70


class PathRefused(ValueError):
    """The path is outside the allowed roots, missing, or not allowed for this action. The
    message goes back to Claude as the tool's error."""


@dataclass(frozen=True)
class Found:
    path: Path
    modified: float  # seconds since the epoch

    def describe(self, home: Path) -> str:
        day = datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d")
        return f"{shorten(self.path, home)} (modified {day})"


class PathPolicy:
    def __init__(self, roots: Sequence[str], home: Path | None = None) -> None:
        self.home = (home or Path.home()).resolve()
        self.anywhere = ANYWHERE in roots
        self.roots = [self.expand(r) for r in roots if r != ANYWHERE]

    def expand(self, raw: str) -> Path:
        """ "~\\Downloads" -> C:\\Users\\JenJo\\Downloads; forward slashes accepted."""
        text = raw.strip().strip('"').strip()
        if text == "~" or text.startswith(("~/", "~\\")):
            text = str(self.home) + text[1:]
        return Path(os.path.expandvars(text)).expanduser()

    def check(self, raw: str, *, must_exist: bool = True) -> Path:
        """The resolved path, if Claudio may use it. Raises PathRefused."""
        if not raw or not raw.strip():
            raise PathRefused("empty path")
        if raw.strip().startswith(("\\\\", "//")):
            raise PathRefused("network paths are not allowed")
        path = self.expand(raw)
        if not path.is_absolute():
            raise PathRefused("use an absolute path or one starting with ~")
        try:
            path = path.resolve(strict=must_exist)
        except (OSError, RuntimeError) as exc:
            raise PathRefused(f"not found: {raw}") from exc
        if not self.allowed(path):
            raise PathRefused(f"outside the allowed folders: {raw}")
        return path

    def allowed(self, path: Path) -> bool:
        if self.anywhere:
            return True
        return any(path == r or path.is_relative_to(r) for r in self._resolved_roots())

    def _resolved_roots(self) -> list[Path]:
        out = []
        for root in self.roots:
            try:
                out.append(root.resolve())
            except OSError:
                continue
        return out

    def search_roots(self) -> list[Path]:
        if self.anywhere:
            return [Path(f"{d}:\\") for d in "CDEFG" if Path(f"{d}:\\").exists()] or [Path("/")]
        return [r for r in self._resolved_roots() if r.is_dir()]


def check_openable(path: Path) -> None:
    if path.is_file() and path.suffix.lower() in RUNNABLE:
        raise PathRefused(f"{path.name} is a program or script: open_path does not run programs")


def search_terms(query: str) -> list[str]:
    """The words of a search, cleaned of anything but letters, digits, dots, dashes and
    underscores (the Windows Search SQL is built from them). Raises PathRefused if none."""
    terms = [t for t in _TERM.sub(" ", query).split() if t.strip(".-")]
    if not terms:
        raise PathRefused("nothing to search for")
    return terms[:MAX_TERMS]


@dataclass
class Walk:
    found: list[Found] = field(default_factory=list)
    # Nothing found: names that sound like the search (a misheard "desi" for "Tesi"), best
    # first. Empty when something was found or `similar` was 0.
    similar: list[Found] = field(default_factory=list)


def walk_search(
    terms: Iterable[str],
    roots: Sequence[Path],
    kind: str | None,
    limit: int,
    budget_s: float,
) -> list[Found]:
    """Without the Windows index: walk the roots for names holding every term, within
    `budget_s` seconds (most recent first)."""
    return walk(terms, roots, kind, limit, budget_s).found


def walk(
    terms: Iterable[str],
    roots: Sequence[Path],
    kind: str | None,
    limit: int,
    budget_s: float,
    similar: int = 0,
) -> Walk:
    """Walk the roots breadth first (Desktop\\UNIVERSITA\\TESI before the depths of a
    project) for names holding every term, within `budget_s` seconds, most recent first.
    With nothing found, the `similar` names that sound most like the search."""
    wanted = [t.casefold() for t in terms]
    query = " ".join(wanted)
    deadline = time.monotonic() + budget_s
    found: list[Found] = []
    close: list[tuple[float, str, Path]] = []  # a min-heap of the best `similar` scores
    queue: deque[Path] = deque(roots)
    while queue and time.monotonic() <= deadline:
        folder = queue.popleft()
        try:
            with os.scandir(folder) as entries:
                listed = list(entries)
        except OSError:
            continue
        for entry in listed:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                # Junctions ("Dati applicazioni", Documents\My Music...) lead to folders seen
                # elsewhere, or loop: walking them spent the whole budget.
                if is_dir and _is_link(entry):
                    continue
            except OSError:
                continue
            name = entry.name
            if is_dir:
                if name.casefold() in SKIP_DIRS or name.startswith("."):
                    continue
                queue.append(Path(entry.path))
            if is_dir != (kind == "folder") or not _kind_ok(name, kind):
                continue
            low = name.casefold()
            if all(t in low for t in wanted):
                try:
                    found.append(Found(Path(entry.path), entry.stat().st_mtime))
                except OSError:
                    continue
            elif similar and not found:
                stem = low if is_dir else os.path.splitext(low)[0]
                score = fuzz.ratio(query, stem)
                if score >= SIMILAR_CUTOFF:
                    item = (score, entry.path, Path(entry.path))
                    if len(close) < similar:
                        heapq.heappush(close, item)
                    elif item > close[0]:
                        heapq.heapreplace(close, item)
    found.sort(key=lambda f: f.modified, reverse=True)
    if found:
        return Walk(found[:limit])
    alike = []
    for _score, _key, path in sorted(close, reverse=True):
        try:
            alike.append(Found(path, path.stat().st_mtime))
        except OSError:
            continue
    return Walk([], alike)


_EXTENSIONS = {
    "document": {".pdf", ".doc", ".docx", ".odt", ".txt", ".rtf", ".md", ".xls", ".xlsx", ".ppt"}
    | {".pptx", ".csv", ".ods", ".odp"},
    "picture": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff"},
    "music": {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"},
    "video": {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"},
}


REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: a symlink or a junction (Windows)


def _is_link(entry: os.DirEntry[str]) -> bool:
    """A symlink or a Windows junction (Python 3.11 has no DirEntry.is_junction)."""
    if entry.is_symlink():
        return True
    attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & REPARSE_POINT)


def _kind_ok(name: str, kind: str | None) -> bool:
    if kind is None or kind == "folder":
        return True
    return Path(name).suffix.lower() in _EXTENSIONS.get(kind, set())


def move(src: Path, dst_raw: str, policy: PathPolicy) -> Path:
    """Move `src` to `dst_raw` (a folder: into it, keeping the name), never overwriting.
    Returns where it went. Raises PathRefused."""
    if src in policy.roots or src == policy.home or src.parent == src:
        raise PathRefused("that folder cannot be moved")
    dst = policy.expand(dst_raw)
    if dst.is_dir():
        dst = dst / src.name
    parent = policy.check(str(dst.parent))
    if not parent.is_dir():
        raise PathRefused(f"the destination folder does not exist: {dst.parent}")
    target = parent / dst.name
    if not policy.allowed(target):
        raise PathRefused(f"outside the allowed folders: {dst_raw}")
    if target.exists():
        raise PathRefused(f"already exists, not overwritten: {target.name}")
    if src.is_dir() and target.is_relative_to(src):
        raise PathRefused("a folder cannot be moved inside itself")
    shutil.move(str(src), str(target))
    return target


def shorten(path: Path, home: Path) -> str:
    """C:\\Users\\JenJo\\Downloads\\a.pdf -> ~\\Downloads\\a.pdf: fewer tokens, same path."""
    try:
        return "~" + os.sep + str(path.relative_to(home))
    except ValueError:
        return str(path)
