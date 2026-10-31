"""Where the stress falls in the Italian words Jack would read wrong. Qwen3-TTS reads plain text
and stresses a form it does not know on the second-to-last syllable: "accenditi" became
"accendete". Words stressed earlier (sdrucciole: accènditi, àbitano) get their accent written;
all the others stay as they are, because a written accent on a regular word made it worse
("sòcio" was heard as "soccio", A/B of 30/10).

The lexicon comes from Wiktionary (kaikki.org, CC BY-SA 4.0), built once by
scripts/build_lexicon_it.py into data/stress_it.tsv. A verb with pronouns attached
("accenditi", "dimmelo") is split: the verb's stress plus the pronoun syllables."""

from __future__ import annotations

import logging
import re
from collections.abc import Container, Iterable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

LEXICON = Path(__file__).resolve().parents[3] / "data" / "stress_it.tsv"
KEEP = "-"  # second column of a word that must stay as written (homograph, not a verb + pronoun)
HOST = "h"  # third column of a verb form that takes pronouns: "accendi" -> "accènditi"
# Third column of an imperative + "ti" that is also a participle: accented only where a clause
# starts ("Fermati!", "Socio, calmati") and left as written elsewhere ("si sono fermati").
AT_START = "c"

_PLAIN = str.maketrans("àèéìíòóùúÀÈÉÌÍÒÓÙÚ", "aeeiioouuAEEIIOOUU")
_ACCENTED = frozenset("àèéìíòóùú")
_STRONG = frozenset("aeo")
_VOWELS = frozenset("aeiou")
# Unstressed pronouns attached to a verb: one syllable each, two at most ("dammelo").
_ONE = ("mi", "ti", "si", "ci", "vi", "lo", "la", "li", "le", "gli", "ne")
_TWO = tuple(
    a + b for a in ("me", "te", "se", "ce", "ve", "glie") for b in ("lo", "la", "li", "le", "ne")
)
_CLITICS = sorted([(c, 1) for c in _ONE] + [(c, 2) for c in _TWO], key=lambda c: -len(c[0]))
# "da'", "di'", "fa'", "sta'", "va'" double the pronoun's consonant: "dammi", "vattene".
_SHORT_IMPERATIVES = {"da": "dà", "di": "dì", "fa": "fà", "sta": "stà", "va": "và"}
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
# Where a clause starts: after these marks; after a comma only behind a vocative or an
# interjection ("Socio, fermati", not "chiusi, salvati"); after an opener that itself starts the
# clause ("E ricordati", "Ora fermati", not "sono ora collegati").
_SENTENCE_MARKS = frozenset('.;:!?…"«“(\n')
_VOCATIVES = frozenset(
    {"socio", "eroe", "claudio", "jenjo", "capo", "amico", "ok", "okay", "dai", "ehi", "su"}
    | {"allora", "ecco", "senti", "tranquillo", "bene", "perfetto", "fatto", "no"}
)
_OPENERS = frozenset(
    {"e", "ma", "o", "oppure", "poi", "ora", "adesso", "dai", "su", "allora", "quindi", "però"}
    | {"ok", "okay", "ecco", "ehi"}
)
_IN_NUMBER = re.compile(r"(?<=\d)[.,:](?=\d)")  # "1.250", "3,5", "10:30": not a clause mark


def plain(word: str) -> str:
    """The word without accents: "accènditi" -> "accenditi"."""
    return word.translate(_PLAIN)


def stress_index(word: str) -> int | None:
    """Where the only written accent is, or None (no accent, or more than one)."""
    found = [i for i, c in enumerate(word.lower()) if c in _ACCENTED]
    return found[0] if len(found) == 1 else None


def syllables_after(word: str, index: int) -> int:
    """Syllables after the stressed vowel at `index`: 0 tronca, 1 piana, 2 sdrucciola. In a run
    of vowels each a, e, o is a syllable and i, u only glide, unless the run has nothing else
    ("àquila", "Itàlia" -> 2, 1); right after the stress i and u join it ("àuto" -> 1); u before
    a, e, o after a consonant other than q and g is a syllable ("stàtua", "contìnuo" -> 2)."""
    w = plain(word.lower())
    count = 0
    i = index + 1
    joins_stress = True
    while i < len(w):
        if w[i] not in _VOWELS:
            joins_stress = False
            i += 1
            continue
        end = i
        while end < len(w) and w[end] in _VOWELS:
            end += 1
        run = w[i:end]
        strong = sum(c in _STRONG for c in run)
        count += strong if strong or joins_stress else 1
        if not joins_stress and run[:1] == "u" and run[1:2] in _STRONG and w[i - 1] not in "qg":
            count += 1
        joins_stress = False
        i = end
    return count


def put_accent(word: str, index: int, accented: str) -> str:
    """`word` with the vowel at `index` replaced by `accented`, in the case it was written."""
    if word[index].isupper():
        accented = accented.upper()
    return word[:index] + accented + word[index + 1 :]


@dataclass
class StressLexicon:
    """`words`: plain lowercase word -> the word with its accent, or KEEP. `hosts`: verb forms
    that take pronouns (imperatives, infinitives without the final e, gerunds) -> accented.
    `at_start`: words accented only where a clause starts (see AT_START)."""

    words: dict[str, str] = field(default_factory=dict)
    hosts: dict[str, str] = field(default_factory=dict)
    at_start: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = LEXICON) -> StressLexicon:
        """The lexicon in `path`; empty (accents off) if the file is missing or unreadable."""
        lexicon = cls()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("Accenti di Jack spenti: lessico non leggibile (%s)", exc)
            return lexicon
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            cells = line.split("\t")
            if len(cells) == 3 and cells[2] == HOST:
                lexicon.hosts[cells[0]] = cells[1]
            elif len(cells) == 3 and cells[2] == AT_START:
                lexicon.at_start[cells[0]] = cells[1]
            elif len(cells) == 2:
                lexicon.words[cells[0]] = cells[1]
        return lexicon

    def stressed(self, word: str, starts_clause: bool = False) -> str | None:
        """`word` with its accent written if Jack would stress it wrong (stress before the
        second-to-last syllable), else None. `word` has no accent of its own."""
        key = word.lower()
        if key in self.at_start:
            accented: str | None = self.at_start[key] if starts_clause else None
        else:
            accented = self.words.get(key)
            if accented is None:
                accented = self.with_pronouns(key)
        if not accented or accented == KEEP:
            return None
        index = stress_index(accented)
        if index is None or len(accented) != len(word):
            return None
        if index == 0 and accented[0] == "ù":  # "ùtile", "ùltimo": a token the voice garbles
            return None
        return put_accent(word, index, accented[index])

    def with_pronouns(self, key: str) -> str | None:
        """A verb with pronouns attached, accented if that moves its stress before the
        second-to-last syllable: "accenditi" -> "accènditi", "dimmelo" -> "dìmmelo". Two ways
        to split it with different stresses ("màndatelo" / "mandàtelo") leave it as written."""
        if not self.hosts:  # no lexicon, no accents: not even on "dammelo"
            return None
        readings: set[tuple[int, str, bool]] = set()
        for clitic, syllables in _CLITICS:
            if not key.endswith(clitic) or len(key) <= len(clitic) + 1:
                continue
            host = key[: -len(clitic)]
            accented = self.hosts.get(host)
            if accented is None and host[:-1] in _SHORT_IMPERATIVES and host[-1] == clitic[0]:
                accented = _SHORT_IMPERATIVES[host[:-1]] + host[-1]  # dammi, vattene
            index = stress_index(accented) if accented else None
            if accented is None or index is None:
                continue
            early = syllables_after(accented, index) + syllables >= 2
            readings.add((index, accented[index], early))
        if len({index for index, _, _ in readings}) != 1:
            return None
        index, vowel, early = readings.pop()
        return put_accent(key, index, vowel) if early else None


def mark_stress(text: str, lexicon: StressLexicon, leave: Container[str] = ()) -> str:
    """Accents written on the words of `text` stressed before the second-to-last syllable.
    Words that already have an accent, and those in `leave` (lowercase), stay as written."""
    out: list[str] = []
    end = 0
    previous: str | None = None
    previous_starts = False
    for match in _WORD.finditer(text):
        word, gap = match.group(0), text[end : match.start()]
        marks = _IN_NUMBER.sub("", gap)
        if previous is None or any(c in _SENTENCE_MARKS for c in marks):
            starts_clause = True
        elif "," in marks:
            starts_clause = previous.lower() in _VOCATIVES
        else:
            starts_clause = previous_starts and previous.lower() in _OPENERS
        if word.lower() not in leave and not any(c in _ACCENTED for c in word.lower()):
            word = lexicon.stressed(word, starts_clause) or word
        out += [gap, word]
        end, previous, previous_starts = match.end(), match.group(0), starts_clause
    out.append(text[end:])
    return "".join(out)


def words_of(texts: Iterable[str]) -> set[str]:
    """The lowercase words in `texts`: the respellings mark_stress must leave alone."""
    return {w.lower() for text in texts for w in _WORD.findall(text)}
