"""Text as Jack's voice should read it: the fine-tuned Qwen3 learned from spoken Italian, so
symbols, acronyms, Markdown and emoji are guessed (often in another language). They become
plain Italian words first, then digits_to_words_it spells the numbers."""

from __future__ import annotations

import re
import unicodedata

from claudio.tts.number_words import digits_to_words_it

# Italian letter names, for acronyms read letter by letter ("PDF" -> "pi di effe").
_LETTERS = {
    "A": "a",
    "B": "bi",
    "C": "ci",
    "D": "di",
    "E": "e",
    "F": "effe",
    "G": "gi",
    "H": "acca",
    "I": "i",
    "J": "i lunga",
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
    "W": "doppia vu",
    "X": "ics",
    "Y": "ipsilon",
    "Z": "zeta",
}
# Acronyms said as words, or that a vowel rule would get wrong.
_ACRONYMS = {
    "OK": "okay",
    "AI": "a i",
    "IA": "i a",
    "USB": "u esse bi",
    "CPU": "ci pi u",
    "GPU": "gi pi u",
    "RAM": "ram",
    "WIFI": "uaifai",
    "UE": "u e",
    "USA": "usa",
    "ONU": "onu",
    "NASA": "nasa",
    "FIAT": "fiat",
    "IVA": "iva",
}
_SYMBOLS = [
    (re.compile(r"\s*%"), " per cento"),
    (re.compile(r"\s*‰"), " per mille"),
    (re.compile(r"\s*°\s*C\b"), " gradi"),
    (re.compile(r"\s*°"), " gradi"),
    (re.compile(r"\bkm/h\b"), "chilometri orari"),
    (re.compile(r"\s*&\s*"), " e "),
    (re.compile(r"(?<=\d)\s*\+\s*(?=\d)"), " più "),
    (re.compile(r"(?<=\d)\s*=\s*(?=\d)"), " uguale "),
    (re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)"), " per "),
    (re.compile(r"(?<=\d)\s*/\s*(?=\d)"), " su "),
    (re.compile(r"…"), "..."),
    (re.compile(r"[–—]"), ", "),
]
# "€ 5" / "5 €" / "$5": the number first, then the currency word.
_CURRENCY_BEFORE = re.compile(r"([€$£])\s*(\d+(?:[.,]\d+)?)")
_CURRENCY_AFTER = re.compile(r"(\d)\s*([€$£])")
_CURRENCY = {"€": "euro", "$": "dollari", "£": "sterline"}
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MARKDOWN = re.compile(r"\*\*|__|`+|^\s{0,3}#{1,6}\s+|^\s*[-*•]\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(?<!\w)[*_](\S[^*_]*?)[*_](?!\w)")
_CAPS = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")
_VOWELS = set("AEIOU")
_SPACES = re.compile(r"[ \t]{2,}")


def _strip_emoji(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if unicodedata.category(ch) not in ("So", "Sk", "Cs", "Co") and ch not in "‍️︎"
    )


def _caps(match: re.Match[str]) -> str:
    """An acronym is spelled; a shouted word ("BASTA") is just lowered."""
    word = match.group(0)
    if word in _ACRONYMS:
        return _ACRONYMS[word]
    letters = [c for c in word if c.isalpha()]
    if len(letters) <= 4 and not _VOWELS.intersection(letters):  # PDF, HTML, TV
        return " ".join(_LETTERS.get(c, c) for c in word)
    return word.lower()


def to_spoken_it(text: str) -> str:
    """ "Il **PC** è al 50%! 😎" -> "Il pi ci è al cinquanta per cento!" """
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("il link", text)
    text = _MARKDOWN.sub("", text)
    text = _EMPHASIS.sub(r"\1", text)
    text = _CURRENCY_BEFORE.sub(lambda m: f"{m.group(2)} {_CURRENCY[m.group(1)]}", text)
    text = _CURRENCY_AFTER.sub(lambda m: f"{m.group(1)} {_CURRENCY[m.group(2)]}", text)
    for pattern, words in _SYMBOLS:
        text = pattern.sub(words, text)
    text = _strip_emoji(text)  # after the symbols: "°" is a symbol to Unicode too
    text = _CAPS.sub(_caps, text)
    text = digits_to_words_it(text)
    return _SPACES.sub(" ", text).strip()
