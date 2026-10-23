"""Digits to Italian words before the TTS: a bare "22" or "51" gives the multilingual voice
nothing to tell Italian from Spanish, and it sometimes reads it in Spanish."""

from __future__ import annotations

import re

_UNITS = [
    "zero",
    "uno",
    "due",
    "tre",
    "quattro",
    "cinque",
    "sei",
    "sette",
    "otto",
    "nove",
    "dieci",
    "undici",
    "dodici",
    "tredici",
    "quattordici",
    "quindici",
    "sedici",
    "diciassette",
    "diciotto",
    "diciannove",
]
_TENS = [
    "_",
    "_",
    "venti",
    "trenta",
    "quaranta",
    "cinquanta",
    "sessanta",
    "settanta",
    "ottanta",
    "novanta",
]
_NUMBER = re.compile(r"(?<![\w.,])(\d+)(?:([.,])(\d+))?(?![\w])")
_TIME = re.compile(r"(?<![\w:])(\d{1,2}):(\d{2})(?![\w:])")


def _below_100(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    tens, unit = divmod(n, 10)
    word = _TENS[tens]
    if unit == 0:
        return word
    if unit in (1, 8):  # ventuno, ventotto: the vowel drops
        word = word[:-1]
    return word + ("tré" if unit == 3 else _UNITS[unit])


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _below_100(rest)
    word = "cento" if hundreds == 1 else _UNITS[hundreds] + "cento"
    if rest == 0:
        return word
    if rest // 10 == 8 or rest == 8:  # centottanta, centotto
        word = word[:-1]
    return word + _below_100(rest)


def int_to_words(n: int) -> str:
    """0 to 999 999 999 in Italian; anything bigger is read digit by digit."""
    if n >= 1_000_000_000:
        return " ".join(_UNITS[int(d)] for d in str(n))
    if n < 1000:
        return _below_1000(n)
    parts: list[str] = []
    millions, rest = divmod(n, 1_000_000)
    thousands, units = divmod(rest, 1000)
    if millions:
        parts.append("un milione" if millions == 1 else f"{_below_1000(millions)} milioni")
    if thousands:
        parts.append("mille" if thousands == 1 else f"{_below_1000(thousands)}mila")
    if units:
        parts.append(_below_1000(units))
    return " ".join(parts)


def _replace_number(match: re.Match[str]) -> str:
    whole, sep, frac = match.group(1), match.group(2), match.group(3)
    words = int_to_words(int(whole))
    if not sep:
        return words
    if sep == "." and len(frac) == 3 and len(whole) <= 3:  # 1.500: thousands, not decimals
        return int_to_words(int(whole + frac))
    return f"{words} virgola {' '.join(_UNITS[int(d)] for d in frac)}"


def _replace_time(match: re.Match[str]) -> str:
    hours, minutes = int(match.group(1)), int(match.group(2))
    words = int_to_words(hours)
    return words if minutes == 0 else f"{words} e {int_to_words(minutes)}"


def digits_to_words_it(text: str) -> str:
    """ "Sono le 20 e 13" -> "Sono le venti e tredici"; "10:30" -> "dieci e trenta"."""
    return _NUMBER.sub(_replace_number, _TIME.sub(_replace_time, text))
