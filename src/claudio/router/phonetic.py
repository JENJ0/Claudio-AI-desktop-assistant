"""How a phrase sounds in Italian, for matching commands Whisper wrote with the wrong letters.

Whisper sometimes writes a short word it did not catch as the letters it heard: "Claudio
QDT" for "Claudio, chiuditi" (cu-di-ti), "QD Rocket League" for "chiudi Rocket League" (26/10
logs). Other times it picks another spelling of the same sounds ("kiudi", "ciudi"). Comparing
the sounds instead of the letters catches the whole family, not one example: capital letter
groups are read as Italian letter names, then spellings that sound the same become one key
("chiu" and "cu" -> "ku", "gh" -> "g", double letters -> single, a silent "h" goes)."""

from __future__ import annotations

import re

from claudio.router.normalizer import strip_accents

# The Italian names of the letters, as Whisper's capitals are said.
LETTER_NAMES = {
    "A": "a",
    "B": "bi",
    "C": "ci",
    "D": "di",
    "E": "e",
    "F": "effe",
    "G": "gi",
    "H": "acca",
    "I": "i",
    "J": "i",
    "K": "cappa",
    "L": "elle",
    "M": "emme",
    "N": "enne",
    "O": "o",
    "P": "pi",
    "Q": "cu",
    "R": "erre",
    "S": "esse",
    "T": "ti",
    "U": "u",
    "V": "vu",
    "W": "vu",
    "X": "ics",
    "Y": "i",
    "Z": "zeta",
}
# 2 to 4 capitals alone, as Whisper writes a word it spelled ("QDT", "QD", "PZ").
_CAPITALS = re.compile(r"\b[A-Z]{2,4}\b")
_RULES = (
    (re.compile(r"ch(?=[ie])"), "k"),  # chiudi, che: hard c
    (re.compile(r"gh(?=[ie])"), "g"),
    (re.compile(r"k[iy](?=[aou])"), "k"),  # "kiu" of chiudi said fast is "ku" (Q = cu)
    (re.compile(r"qu|q|c(?=[aouhlrtk ]|$)|ck|x"), "k"),
    (re.compile(r"h"), ""),  # silent in Italian
    (re.compile(r"[yj]"), "i"),
    (re.compile(r"w"), "v"),
    (re.compile(r"(\w)\1+"), r"\1"),  # double letters sound almost the same
)


def spell_capitals(text: str) -> str:
    """ "Claudio QDT" -> "Claudio cuditi": a group of capitals becomes the word it sounds
    like. Real acronyms (PSS) become sounds too: this is only for matching, never shown."""
    return _CAPITALS.sub(lambda m: "".join(LETTER_NAMES[c] for c in m.group()), text)


def sound_key(normalized: str) -> str:
    """The sound of an already normalized phrase (lowercase, no accents or punctuation)."""
    key = strip_accents(normalized.lower())
    for pattern, replacement in _RULES:
        key = pattern.sub(replacement, key)
    # "kiuditi" and "kuditi" now meet; "k" before e/i was "ch", after the rules it stays "k".
    key = re.sub(r"ki(?=[aou])", "k", key)
    return " ".join(key.split())
