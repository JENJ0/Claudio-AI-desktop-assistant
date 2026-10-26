"""Which paths Claudio may touch, and the file actions that need no Windows API.

Every path Claude sends is resolved (symlinks and junctions followed) and must sit inside one
of the allowed roots (`[pc] roots`, by default JenJo's user folder). "*" allows any local
drive; network paths (\\\\server\\share) never. Opening a runnable file (.exe, .bat, .lnk...)
is refused: launching programs goes through `open_app`, which only knows the Start menu."""

from __future__ import annotations

import heapq
import logging
import os
import re
import shutil
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

log = logging.getLogger(__name__)

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
# (v0.3.2: JenJo had to dictate ~\\Desktop\\PROGETTI). The Desktop may live in OneDrive.
NEAR_HOME = ("", "Desktop", "Documents", "Downloads", "OneDrive/Desktop", "OneDrive/Documents")
SKIP_DIRS = frozenset(
    {"appdata", "node_modules", ".git", "$recycle.bin", "__pycache__", ".venv", "venv"}
    | {"miniconda3", "anaconda3", "npm-cache", "site-packages", "curseforge", "saved games"}
)
# rapidfuzz ratio of a name to a search that found nothing: "desi" and "attesi" -> "TESI" (75,
# 80). Below this the name does not sound like the search.
SIMILAR_CUTOFF = 70
# Words Claude sometimes leaves in a search ("cartella progetti"): no name holds them, so the
# search found nothing. Dropped when another word remains.
FILLER = frozenset(
    {"cartella", "cartelle", "folder", "folders", "file", "files", "documento", "il", "lo", "la"}
    | {"i", "gli", "le", "di", "del", "della", "dei", "delle", "the", "of", "my", "mia", "mio"}
)
# Spellings that sound the same, applied in order after `fold`: Whisper writes Italian names
# the English way ("Projecti" for PROGETTI, 25/10).
_SOUNDS = (
    ("ph", "f"), ("ck", "c"), ("ch", "c"), ("k", "c"), ("ct", "tt"), ("j", "g"), ("y", "i"),
    ("w", "v"), ("x", "cs"), ("h", ""),
)  # fmt: skip
_DOUBLE = re.compile(r"(.)\1+")
_SEPARATORS = re.compile(r"[_.\-]+")


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
    kept = [t for t in terms if t.casefold() not in FILLER]
    return (kept or terms)[:MAX_TERMS]


def fold(text: str) -> str:
    """Lowercase, without accents, apostrophes or separators: "UNIVERSITA'" and "università"
    are both "universita", "Project_Work" is "project work"."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    plain = plain.replace("'", "").replace("\u2019", "").replace("`", "")
    return " ".join(_SEPARATORS.sub(" ", plain).split())


def sound(text: str) -> str:
    """How a folded name sounds, roughly: "projecti" and "progetti" are both "progeti"."""
    out = fold(text)
    for written, said in _SOUNDS:
        out = out.replace(written, said)
    return _DOUBLE.sub(r"\1", out)


def prepare(terms: Iterable[str]) -> list[tuple[str, str]]:
    """The terms folded and as they sound, once per search (not once per name)."""
    return [(fold(t), sound(t)) for t in terms]


def matches(terms: Sequence[tuple[str, str]], folded: str, sounds: str) -> bool:
    """Every prepared term is in the name, written or (from 3 letters) said the same way."""
    return _held(terms, folded, sounds) == len(terms)


def _held(terms: Sequence[tuple[str, str]], folded: str, sounds: str) -> int:
    """How many terms the name holds. Spaces do not count: "BBPortWindows" is in
    "bbport-windows" (26/10)."""
    joined, joined_sounds = folded.replace(" ", ""), sounds.replace(" ", "")
    return sum(
        1
        for f, s in terms
        if f.replace(" ", "") in joined or (len(s) >= 3 and s.replace(" ", "") in joined_sounds)
    )


def likeness(query: str, folded: str, sounds: str, terms: Sequence[tuple[str, str]] = ()) -> float:
    """0-100: how much a name (folded, and its sound) is like a search that found nothing.
    One word may be like one word of a longer name ("roket" -> "Rocket League"), and a name
    holding some of the prepared `terms` is like it too ("Project Work": "PROGETTO IUM")."""
    q = fold(query)
    score = max(fuzz.ratio(q, folded), fuzz.ratio(sound(query), sounds))
    if len(terms) > 1:
        held = _held([t for t in terms if len(t[0]) >= 3], folded, sounds)
        if held:
            score = max(score, 60 + 40 * held / len(terms))
    words = folded.split()
    if len(words) > 1:
        score = max(score, fuzz.token_sort_ratio(q, folded))
        long_words = [w for w in words if len(w) >= 3]
        if " " not in q and len(q) >= 4 and long_words:
            score = max(score, *(fuzz.ratio(q, w) for w in long_words))
    return score


SHORTCUTS = {".lnk", ".url"}


def shortcuts(folders: Iterable[Path], deep: bool) -> list[tuple[str, Path]]:
    """(name, file) of the shortcuts in these folders. Games from Steam or Epic are .url
    shortcuts that the Start menu app list leaves out (25/10: Rocket League)."""
    out: list[tuple[str, Path]] = []
    for folder in folders:
        try:
            items = folder.rglob("*") if deep else folder.iterdir()
            for item in items:
                if item.suffix.lower() in SHORTCUTS and item.is_file():
                    name = item.stem
                    if not name.casefold().startswith(("uninstall", "disinstalla")):
                        out.append((name, item))
        except OSError:
            continue
    return out


def index_path(url: str) -> Path | None:
    """The real path of a Windows Search result from its System.ItemUrl ("file:C:/Users/...").
    Not System.ItemPathDisplay: on an Italian Windows it reads "C:\\Utenti\\..." (25/10: every
    folder found was then outside ~ and dropped). Other stores (mapi:, iehistory:) are skipped."""
    if not url.casefold().startswith("file:"):
        return None
    rest = url[len("file:") :].lstrip("/")
    return Path(rest) if rest else None


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
    terms = list(terms)
    query = " ".join(terms)
    wanted = prepare(terms)
    deadline = time.monotonic() + budget_s
    found: list[Found] = []
    close: list[tuple[float, str, Path]] = []  # a min-heap of the best `similar` scores
    queue: deque[Path] = deque(roots)
    while queue and time.monotonic() <= deadline:
        folder = queue.popleft()
        for entry, is_dir in _children(folder):
            name = entry.name
            if is_dir:
                if name.casefold() in SKIP_DIRS or name.startswith("."):
                    continue
                queue.append(Path(entry.path))
            if is_dir != (kind == "folder") or not _kind_ok(name, kind):
                continue
            stem = name if is_dir else os.path.splitext(name)[0]
            folded, sounds = fold(stem), sound(stem)
            if matches(wanted, folded, sounds):
                try:
                    found.append(Found(Path(entry.path), entry.stat().st_mtime))
                except OSError:
                    continue
            elif similar and not found:
                score = likeness(query, folded, sounds, wanted)
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


def _children(folder: Path) -> list[tuple[os.DirEntry[str], bool]]:
    """(entry, is a folder) for what `folder` holds, without links: junctions ("Dati
    applicazioni", Documents\\My Music...) lead to folders seen elsewhere, or loop (walking
    them spent the whole budget)."""
    try:
        with os.scandir(folder) as entries:
            listed = list(entries)
    except OSError:
        return []
    out = []
    for entry in listed:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
            if is_dir and _is_link(entry):
                continue
        except OSError:
            continue
        out.append((entry, is_dir))
    return out


@dataclass(frozen=True)
class Folder:
    path: Path
    folded: str
    sounds: str


class FolderNames:
    """The name of every folder under the roots, read in the background and again after
    `refresh_s`: a folder is found in milliseconds wherever it is, and a misheard name is
    compared with all of them, not only with those a 3-second walk reached (25/10: "spesso
    non trova"). Files stay with the Windows index and the walk. Never the only answer: a
    folder made after the last read is not here, so the callers still walk."""

    def __init__(
        self,
        roots: Callable[[], Sequence[Path]],
        refresh_s: float,
        max_folders: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._roots = roots
        self.refresh_s = refresh_s
        self.max_folders = max_folders
        self.clock = clock
        self._folders: list[Folder] | None = None
        self._read_at = -float("inf")
        self._reading = threading.Lock()

    def refresh(self) -> None:
        """Read the folders again in a background thread, unless a read is running."""
        if not self._reading.acquire(blocking=False):
            return
        threading.Thread(target=self._read, name="folder-names", daemon=True).start()

    def read_now(self) -> None:
        """Read the folders on this thread (tests, or a first search that can wait)."""
        with self._reading:
            self._folders = self._list()
            self._read_at = self.clock()

    def folders(self) -> list[Folder] | None:
        """The folders last read (None before the first read ends); a stale list starts a
        new read and is still returned."""
        if self.clock() - self._read_at > self.refresh_s:
            self.refresh()
        return self._folders

    def find(self, terms: Sequence[str], limit: int, similar: int) -> Walk:
        """Folders whose name holds every term, newest first; with none, the `similar`
        names most like the search."""
        folders = self._folders or []
        wanted = prepare(terms)
        hits = [f for f in folders if matches(wanted, f.folded, f.sounds)]
        if hits:
            return Walk(_dated(f.path for f in hits)[:limit])
        if not similar:
            return Walk()
        query = " ".join(terms)
        scored = [(likeness(query, f.folded, f.sounds, wanted), f.path) for f in folders]
        best = heapq.nlargest(similar, (s for s in scored if s[0] >= SIMILAR_CUTOFF))
        return Walk([], _dated((p for _, p in best), newest_first=False))

    def named(self, name: str) -> list[Path]:
        """Folders called `name` (any case, accents or apostrophes); else those that sound
        exactly like it."""
        folders = self._folders or []
        f, s = fold(name), sound(name)
        same = [x.path for x in folders if x.folded == f]
        return same or [x.path for x in folders if len(s) >= 3 and x.sounds == s]

    def _read(self) -> None:
        try:
            folders = self._list()
            self._folders, self._read_at = folders, self.clock()
        except Exception:
            log.exception("Elenco delle cartelle non letto")
            self._read_at = self.clock()  # not again at every search
        finally:
            self._reading.release()

    def _list(self) -> list[Folder]:
        start = time.monotonic()
        out: list[Folder] = []
        queue: deque[Path] = deque(self._roots())
        while queue and len(out) < self.max_folders:
            for entry, is_dir in _children(queue.popleft()):
                name = entry.name
                if not is_dir or name.casefold() in SKIP_DIRS or name.startswith("."):
                    continue
                path = Path(entry.path)
                queue.append(path)
                out.append(Folder(path, fold(name), sound(name)))
        log.info(
            "Elenco delle cartelle: %d in %d ms%s",
            len(out),
            round((time.monotonic() - start) * 1000),
            " (limite raggiunto)" if queue else "",
        )
        return out


def _dated(paths: Iterable[Path], newest_first: bool = True) -> list[Found]:
    """Found items for the paths that still exist; newest first, or in the given order."""
    out = []
    for path in paths:
        try:
            out.append(Found(path, path.stat().st_mtime))
        except OSError:
            continue  # deleted since the folders were read
    if newest_first:
        out.sort(key=lambda f: f.modified, reverse=True)
    return out


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
