"""CLAUDIO.md: Claudio's personality, read once at start and sent after the system prompt.

JenJo's copy lives in data/ (never touched by updates or Git); the first start makes it from
the default in config/. Read once, so the system prompt stays frozen and cached for the run."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_FILE = Path(__file__).resolve().parents[3] / "config" / "CLAUDIO.md"
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)  # notes for JenJo: never sent (they cost)


def load_personality(path: Path, max_chars: int, default: Path = DEFAULT_FILE) -> str:
    """The personality text, stripped and cut to `max_chars` (at a line end); "" when off or
    unreadable: Claudio then answers with the neutral tone of the system prompt."""
    if max_chars <= 0:
        return ""
    if not path.exists() and default.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(default, path)
            log.info("Personalità creata in %s (modificala quando vuoi)", path)
        except OSError as exc:
            log.warning("Non riesco a creare %s (%s): uso %s", path, exc, default)
            path = default
    try:
        text = _COMMENT.sub("", path.read_text(encoding="utf-8")).strip()
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Personalità non letta (%s): tono neutro", exc)
        return ""
    if len(text) > max_chars:
        cut = text[:max_chars]
        text = cut[: cut.rfind("\n")].rstrip() if "\n" in cut else cut
        log.warning(
            "%s supera %d caratteri: tengo solo l'inizio (ogni carattere costa a ogni richiesta)",
            path,
            max_chars,
        )
    return text
