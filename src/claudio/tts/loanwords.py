"""English words Jack would read the Italian way ("subprime" -> "supprime"). JenJo's table
[tts.jack_words] says how to respell them and stays the authority; a word that looks foreign and
is not in it is logged once as a candidate, with a proposal from Wiktionary's Italian IPA when
there is one (data/respell_en.tsv, built by scripts/build_lexicon_it.py). The proposals are only
heard, never applied: scripts/jack_variants.py --words WORD says each way to write it."""

from __future__ import annotations

import logging
import re
from collections.abc import Container
from itertools import pairwise
from pathlib import Path

log = logging.getLogger(__name__)

PROPOSALS = Path(__file__).resolve().parents[3] / "data" / "respell_en.tsv"
LOG_LIMIT = 300  # words logged per run: a long session does not fill the log

_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_ACCENTED = frozenset("àèéìíòóùúÀÈÉÌÍÒÓÙÚ")
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")
# Two consonants in a row that Italian spells: "accendi", "sbaglio", "altro", "campo", "pneumatico".
_ITALIAN_PAIRS = frozenset(
    {"gn", "ch", "gh", "cq", "ps", "pn", "pt", "tm", "dm", "cn", "gm", "bd", "bn", "bs"}
)
_H_WORDS = frozenset({"ho", "hai", "ha", "hanno"})


def _italian_pair(first: str, second: str) -> bool:
    return (
        first == second
        or first in "lmnrs"
        or (second in "lr" and first in "bcdfgptv")
        or first + second in _ITALIAN_PAIRS
    )


def looks_foreign(word: str) -> bool:
    """True if `word` is not spelled the Italian way: j, k, w, x, y, "ou", an h not after c or
    g, a final consonant other than l, n, r, or two consonants Italian never puts together
    ("subprime", "update", "chat", "Windows"). Italian-looking English ("file") passes."""
    w = word.lower()
    if any(c in "jkwxy" for c in w) or "ou" in w:
        return True
    if "h" in w and w not in _H_WORDS and re.search(r"(?<![cg])h", w):
        return True
    if w[-1] in _CONSONANTS and w[-1] not in "lnr":
        return True
    return any(
        a in _CONSONANTS and b in _CONSONANTS and not _italian_pair(a, b) for a, b in pairwise(w)
    )


def load_proposals(path: Path = PROPOSALS) -> dict[str, str]:
    """English word -> a way to write it for Jack ("online" -> "onlàin"); {} if missing."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    pairs = (line.split("\t") for line in lines if line and not line.startswith("#"))
    return {cells[0]: cells[1] for cells in pairs if len(cells) == 2}


class UnknownWords:
    """Logs once each word Jack may misread: foreign-looking or an English loanword with a
    proposal, and not respelled by JenJo's table ("Jack: parola sconosciuta 'subprime'")."""

    def __init__(self, proposals: dict[str, str] | None = None, limit: int = LOG_LIMIT) -> None:
        self.proposals = load_proposals() if proposals is None else proposals
        self.limit = limit
        self.seen: set[str] = set()

    def note(self, text: str, known: Container[str] = ()) -> list[str]:
        """Logs the new candidates in `text` (spoken text, numbers already words) and returns
        them. `known`: lowercase words of the table, keys and respellings."""
        found = []
        for word in _WORD.findall(text):
            key = word.lower()
            if key in self.seen or key in known or len(key) < 2 or _ACCENTED & set(word):
                continue
            if word.isupper():  # an acronym, spelled letter by letter
                continue
            proposal = self.proposals.get(key)
            if proposal is None and not looks_foreign(key):
                continue
            if len(self.seen) >= self.limit:
                break
            self.seen.add(key)
            found.append(key)
            hint = f" (proposta: {proposal!r})" if proposal else ""
            log.info("Jack: parola sconosciuta %r%s, candidata per [tts.jack_words]", key, hint)
        return found
