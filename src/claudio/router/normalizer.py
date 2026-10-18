"""Text normalisation shared by command phrases and transcriptions."""

from __future__ import annotations

import re
import unicodedata

ARTICLES = {"il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una", "un'"}
FILLERS = {
    "claudio",
    "hey",
    "ehi",
    "ei",
    "ok",
    "okay",
    "allora",
    "senti",
    "dai",
    "adesso",
    "puoi",
    "potresti",
    "riesci",
    "a",
    "mi",
    "grazie",
    "prego",
    "subito",
}
POLITE_PHRASES = ("per favore", "per piacere", "per cortesia", "se puoi", "gentilmente")
# Polite/infinitive verb forms -> the imperative used in commands.toml.
VERB_FORMS = {
    "aprire": "apri",
    "aprimi": "apri",
    "apra": "apri",
    "aprilo": "apri",
    "alzare": "alza",
    "alzami": "alza",
    "abbassare": "abbassa",
    "abbassami": "abbassa",
    "avviare": "avvia",
    "avviami": "avvia",
    "lanciare": "lancia",
    "togliere": "togli",
    "aumentare": "aumenta",
    "diminuire": "diminuisci",
    "mostrare": "mostra",
    "mostrami": "mostra",
    "dirmi": "dimmi",
    "silenziare": "silenzia",
}
_NON_WORD = re.compile(r"[^a-z0-9]+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str, drop_fillers: bool = True) -> str:
    """Lowercase, no accents or punctuation; optionally drop articles and courtesy words."""
    text = strip_accents(text.lower())
    text = _NON_WORD.sub(" ", text).strip()
    if not drop_fillers:
        return " ".join(text.split())
    for phrase in POLITE_PHRASES:
        text = re.sub(rf"\b{phrase}\b", " ", text)
    words = [VERB_FORMS.get(w, w) for w in text.split()]
    return " ".join(w for w in words if w not in ARTICLES and w not in FILLERS)


NUMBER_WORDS = {
    "uno": 1,
    "due": 2,
    "tre": 3,
    "quattro": 4,
    "cinque": 5,
    "sei": 6,
    "sette": 7,
    "otto": 8,
    "nove": 9,
    "dieci": 10,
    "quindici": 15,
    "venti": 20,
    "trenta": 30,
    "cinquanta": 50,
}
_NUMBER = re.compile(r"\b(?:di\s+)?(\d+|" + "|".join(NUMBER_WORDS) + r")\b(?:\s+volte)?")


def extract_number(norm_text: str) -> tuple[str, int | None]:
    """Remove the first number ("di 10", "tre volte") from normalised text and return it."""
    m = _NUMBER.search(norm_text)
    if not m:
        return norm_text, None
    raw = m.group(1)
    value = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
    rest = (norm_text[: m.start()] + " " + norm_text[m.end() :]).split()
    return " ".join(rest), value
