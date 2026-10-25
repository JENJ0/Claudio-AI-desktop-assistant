"""Which paths Claudio may touch, and the file actions that need no Windows API.

Every path Claude sends is resolved (symlinks and junctions followed) and must sit inside one
of the allowed roots (`[pc] roots`, by default JenJo's user folder). "*" allows any local
drive; network paths (\\\\server\\share) never. Opening a runnable file (.exe, .bat, .lnk...)
is refused: launching programs goes through `open_app`, which only knows the Start menu."""

from __future__ import annotations

import os
import re
import shutil
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
# Where a name said without its folder ("apri la cartella Progetti") is looked for, in order
# (v0.3.2: JenJo had to dictate ~\Desktop\PROGETTI). The Desktop may live in OneDrive.
NEAR_HOME = ("", "Desktop", "Documents", "Downloads", "OneDrive/Desktop", "OneDrive/Documents")
SKIP_DIRS = frozenset(
    {"appdata", "node_modules", ".git", "$recycle.bin", "__pycache__", ".venv", "venv"}
    | {"miniconda3", "anaconda3", "npm-cache", "site-packages", "curseforge", "saved games"}
)


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

    def find_near(self, raw: str) -> Path | None:
        """A path that does not exist, given as a bare name or as ~\\name: the item with that
        name (any case) in the user folder, on the Desktop, in Documents or Downloads."""
        text = raw.strip().strip('"').strip().replace("/", "\\")
        home = str(self.home).replace("/", "\\").casefold() + "\\"
        if text.startswith("~\\"):
            text = text[2:]
        elif text.casefold().startswith(home):  # C:\Users\JenJo\Progetti
            text = text[len(home) :]
        if not text or "\\" in text or ":" in text:
            return None  # a real path that is wrong: not guessed
        wanted = text.casefold()
        for sub in NEAR_HOME:
            folder = self.home / sub if sub else self.home
            try:
                for child in folder.iterdir():
                    if child.name.casefold() == wanted and self.allowed(child.resolve()):
                        return child.resolve()
            except OSError:
                continue
        return None

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


def index_path(url: str) -> Path | None:
    """The real path of a Windows Search result from its System.ItemUrl ("file:C:/Users/...").
    Not System.ItemPathDisplay: on an Italian Windows it reads "C:\\Utenti\\..." (25/10: every
    folder found was then outside ~ and dropped). Other stores (mapi:, iehistory:) are skipped."""
    if not url.casefold().startswith("file:"):
        return None
    rest = url[len("file:") :].lstrip("/")
    return Path(rest) if rest else None


def walk_search(
    terms: Iterable[str],
    roots: Sequence[Path],
    kind: str | None,
    limit: int,
    budget_s: float,
    first: Sequence[Path] = (),
) -> list[Found]:
    """Walk the folders for names holding every term, within `budget_s` seconds. Breadth
    first, starting from `first` (Desktop, Documents...): a folder JenJo keeps at hand, like
    Desktop\\UNIVERSITA'\\TESI, is found before the walk drowns in a big tree (24/10: miniconda3).
    Exact names come first, then the most recent."""
    wanted = [t.casefold() for t in terms]
    exact = " ".join(wanted)
    deadline = time.monotonic() + budget_s
    found: list[Found] = []
    queue: deque[Path] = deque(p for p in [*first, *roots] if p.is_dir())
    seen: set[Path] = set()
    while queue and time.monotonic() < deadline:
        folder = queue.popleft()
        if folder in seen:
            continue
        seen.add(folder)
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            name = entry.name
            if is_dir and name.casefold() not in SKIP_DIRS and not name.startswith((".", "$")):
                queue.append(Path(entry.path))
            if is_dir != (kind == "folder"):
                continue
            low = name.casefold()
            if all(t in low for t in wanted) and _kind_ok(name, kind):
                try:
                    found.append(Found(Path(entry.path), entry.stat().st_mtime))
                except OSError:
                    continue
    found.sort(key=lambda f: (Path(f.path).stem.casefold() != exact, -f.modified))
    return found[:limit]


_EXTENSIONS = {
    "document": {".pdf", ".doc", ".docx", ".odt", ".txt", ".rtf", ".md", ".xls", ".xlsx", ".ppt"}
    | {".pptx", ".csv", ".ods", ".odp"},
    "picture": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff"},
    "music": {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"},
    "video": {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"},
}


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
