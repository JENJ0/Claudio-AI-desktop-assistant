"""Text normalisation shared by command phrases and transcriptions, per language."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Wake words (and how Whisper may spell them) heard at the start of a phrase said in one breath.
WAKE_WORDS = ("claudio", "jarvis")
GREETINGS = {"hey", "ehi", "ei", "hei", "ehy", "hi"}
# Close to "claudio" and still not it. "Claudia" is not here on purpose: the wake word models no
# longer refuse it (refusing it taught them to refuse "Claudio" too), so the router drops it
# like any other spelling.
NOT_WAKE_WORDS = {"audio"}
WAKE_SEARCH = 3  # Whisper writes a fast "hey" as "e", "e il", "e i": the name can be 3rd


@dataclass(frozen=True)
class Language:
    """What a language's commands can do without: articles, courtesy words, verb forms."""

    articles: frozenset[str]
    fillers: frozenset[str]
    polite: tuple[str, ...]  # multi-word courtesy phrases, removed first
    verb_forms: dict[str, str]  # polite/infinitive forms -> the imperative in commands.toml
    numbers: dict[str, int]
    number_affixes: tuple[str, str]  # what may come before and after a number (regex)
    number_pattern: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        prefix, suffix = self.number_affixes
        words = "|".join(self.numbers)
        pattern = re.compile(rf"\b{prefix}(\d+|{words})\b{suffix}")
        object.__setattr__(self, "number_pattern", pattern)


IT_NUMBERS = {
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
EN_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "fifty": 50,
}
LANGUAGES: dict[str, Language] = {
    "it": Language(
        articles=frozenset({"il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una", "un'"}),
        fillers=frozenset(
            {
                "claudio",
                "jarvis",
                "hey",
                "hei",
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
                "scusa",
                "scusami",
            }
        ),
        polite=("per favore", "per piacere", "per cortesia", "se puoi", "gentilmente"),
        verb_forms={
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
            "metterti": "mettiti",
            "svegliarti": "svegliati",
            "riattivarti": "riattivati",
            "spegnerti": "spegniti",
        },
        numbers=IT_NUMBERS,
        number_affixes=(r"(?:di\s+)?", r"(?:\s+volte)?"),  # "alza il volume di 10", "tre volte"
    ),
    "en": Language(
        articles=frozenset({"the", "a", "an"}),
        fillers=frozenset(
            {
                "claudio",
                "jarvis",
                "hey",
                "hi",
                "ok",
                "okay",
                "so",
                "now",
                "just",
                "please",
                "can",
                "could",
                "would",
                "will",
                "you",
                "thanks",
                "sorry",
                "excuse",
            }
        ),
        polite=("for me", "if you can", "thank you", "would you mind"),
        verb_forms={"opening": "open", "turning": "turn"},
        numbers=EN_NUMBERS,
        number_affixes=(r"(?:by\s+)?", r"(?:\s+times)?"),  # "turn it up by 10", "three times"
    ),
}
DEFAULT = "it"
_NON_WORD = re.compile(r"[^a-z0-9]+")
_FILLERS = frozenset().union(*(rules.fillers for rules in LANGUAGES.values()))


def language(code: str) -> Language:
    """The rules of `code`; languages without their own (yet) use the Italian ones."""
    return LANGUAGES.get(code, LANGUAGES[DEFAULT])


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str, drop_fillers: bool = True, lang: str = DEFAULT) -> str:
    """Lowercase, no accents or punctuation; optionally drop articles and courtesy words."""
    text = strip_accents(text.lower())
    text = _NON_WORD.sub(" ", text).strip()
    if not drop_fillers:
        return " ".join(text.split())
    rules = language(lang)
    for phrase in rules.polite:
        text = re.sub(rf"\b{phrase}\b", " ", text)
    words = [rules.verb_forms.get(w, w) for w in strip_wake_words(text.split())]
    return " ".join(w for w in words if w not in rules.articles and w not in rules.fillers)


def is_wake_word(word: str) -> bool:
    """'jarvis', 'giarvis', 'clodio'...: close enough to a wake word, and not a real word."""
    from rapidfuzz import fuzz

    if word in GREETINGS or word in WAKE_WORDS:
        return True
    return word not in NOT_WAKE_WORDS and max(fuzz.ratio(word, w) for w in WAKE_WORDS) >= 75


def _before_wake_word(word: str) -> bool:
    """What may come before the wake word: a greeting, what Whisper makes of a fast "hey"
    ("e", "il", "hai", "ejci") or a courtesy word ("ok", "allora", "scusa", "please")."""
    from rapidfuzz.distance import Levenshtein

    if word in GREETINGS or len(word) <= 3 or word in _FILLERS:
        return True
    return min(Levenshtein.distance(word, greeting) for greeting in GREETINGS) <= 2


def strip_wake_words(words: list[str]) -> list[str]:
    """Drop the wake phrase at the start ("ehi giarvis apri chrome" -> "apri chrome"), also
    when Whisper mangled the greeting before it ("e il jarviso che ore sono"). After a
    command the name is not a start: "svegliati claudio" keeps "svegliati"."""
    start = 0
    for i, word in enumerate(words[:WAKE_SEARCH]):
        if word in GREETINGS or not is_wake_word(word):
            continue
        if all(_before_wake_word(w) for w in words[start:i]):
            start = i + 1
    while start < len(words) and is_wake_word(words[start]):
        start += 1
    return words[start:]


def only_wake_words(text: str) -> bool:
    """True when the transcript is just the wake word: the command is still to come."""
    words = normalize(text, drop_fillers=False).split()
    return bool(words) and not strip_wake_words(words)


def heard_wake_word(text: str) -> bool:
    """True when the transcript has the wake word itself, anywhere ("hey claudio", "ok
    claudio che ore sono"); a lone greeting or "audio" is not it."""
    words = normalize(text, drop_fillers=False).split()
    return any(w not in GREETINGS and is_wake_word(w) for w in words)


def extract_number(norm_text: str, lang: str = DEFAULT) -> tuple[str, int | None]:
    """Remove the first number ("di 10", "tre volte") from normalised text and return it."""
    rules = language(lang)
    m = rules.number_pattern.search(norm_text)
    if not m:
        return norm_text, None
    raw = m.group(1)
    value = int(raw) if raw.isdigit() else rules.numbers[raw]
    rest = (norm_text[: m.start()] + " " + norm_text[m.end() :]).split()
    return " ".join(rest), value
