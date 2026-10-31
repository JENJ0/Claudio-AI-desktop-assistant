"""Build Jack's pronunciation data from kaikki.org's extract of the Italian words in the English
Wiktionary (CC BY-SA 4.0): data/stress_it.tsv, the accents Jack writes (claudio.tts.stress_it),
and data/respell_en.tsv, proposals for English loanwords (claudio.tts.loanwords).
Run once; the download (77 MB compressed) stays out of git, in models/kaikki/:

  curl -L -o models/kaikki/kaikki.org-dictionary-Italian.jsonl.gz ^
      https://kaikki.org/dictionary/Italian/kaikki.org-dictionary-Italian.jsonl.gz
  uv run python scripts/build_lexicon_it.py models/kaikki/kaikki.org-dictionary-Italian.jsonl.gz

Where the stress is: the syllabification ("ac‧cèn‧di‧ti", 103k entries), the accented
conjugation tables ("accèndo", 825k forms), and for plurals, feminines and participles the
stress of the word they come from. The IPA, when there is one, decides how many syllables
follow the stress. A word with two stresses (àncora/ancóra) is left as written.
English loanwords: their Italian IPA written as Jack reads it ("/onˈlajn/" -> "onlàin")."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claudio.tts.loanwords import PROPOSALS  # noqa: E402
from claudio.tts.number_words import digits_to_words_it  # noqa: E402
from claudio.tts.stress_it import (  # noqa: E402
    AT_START,
    HOST,
    KEEP,
    LEXICON,
    StressLexicon,
    plain,
    stress_index,
    syllables_after,
)

SOURCE_URL = "https://kaikki.org/dictionary/Italian/kaikki.org-dictionary-Italian.jsonl.gz"
_LETTERS = re.compile(r"[a-zàèéìíòóùú]+")
# Accents are written only on words spelled the Italian way: no j, k, w, x, y and a vowel at
# the end. English words are respelled instead ([tts.jack_words]).
_NATIVE = re.compile(r"[a-zàèéìíòóùú]*[aeiouàèéìíòóùú]")
_FOREIGN = re.compile(r"[jkwxy]")
_SKIP_TAGS = {"table-tags", "inflection-template", "auxiliary", "romanization"}
# An entry only archaic, dialectal or wrong lends no stress: "machina" made "màchine".
_OLD_TAGS = {"archaic", "obsolete", "dialectal", "misspelling", "nonstandard"}
# A plural or feminine keeps its lemma's stress; a superlative, diminutive or adverb does not
# ("famóso" but "famosìssimo", "bène" but "benóne").
_INFLECTION_TAGS = {"plural", "singular", "masculine", "feminine"}
_DERIVATION_TAGS = {"superlative", "diminutive", "augmentative", "pejorative", "endearing"}
_BORROWED = {"bor", "bor+", "ubor", "lbor"}  # etymology templates of a loanword
# IPA vowels as Italian letters, plain and stressed (open è/ò, closed é/ó).
_VOWELS = dict(zip("aeɛioɔuəæʌɪʊɐɜɒyø", "aeeiooueeaiuaeoue", strict=True))
_STRESSED = dict(zip("aeɛioɔuəæʌɪʊɐɜɒyø", "àéèìóòùèèàìùàèòùè", strict=True))
_FRONT = set("eɛiɪəæɜyj")  # c and g before these are soft: "ch", "gh" keep them hard
_AFFRICATES = {"t͡ʃ": "ʧ", "d͡ʒ": "ʤ", "t͡s": "ʦ", "d͡z": "ʣ", "tʃ": "ʧ", "dʒ": "ʤ"}
_SAME = set("pbtdfvsmnlh")
_AFFRICATE_OF = {"t": "ʧʦ", "d": "ʤʣ"}  # "t.t͡ʃ" is a long ʧ, written "cc"
_SIMPLE = {
    "ɲ": "gn",
    "ŋ": "n",
    "z": "s",
    "j": "i",
    "w": "u",
    "ʒ": "j",
    "θ": "t",
    "ð": "d",
    "x": "h",
    "ɹ": "r",
    "ɾ": "r",
    "ʁ": "r",
    "r": "r",
    "ʦ": "z",
    "ʣ": "z",
}
# Words Jack says right as written and worse with the accent (scripts/stress_ab.py, 30/10):
# "càlmati" heard "chiamati", "cialmati", "calmate" 3 times in 5, "calmati" 5/5 without.
_SAID_WORSE = {"calmati"}
# Imperative + "ti" that is also a participle, accented where a clause starts: the ones Jack says
# to JenJo. Any other ("copiati", "aggiornati") is left as written, read as the participle.
_IMPERATIVES_SAID = {
    "accomodati",
    "alzati",
    "arrangiati",
    "divertiti",
    "fermati",
    "fidati",
    "girati",
    "guardati",
    "preparati",
    "rilassati",
    "ricordati",
    "riposati",
    "sbrigati",
    "scordati",
    "sdraiati",
    "svegliati",
    "vestiti",
    "voltati",
}
# Common English words Wiktionary gives no Italian IPA for, written by hand (JenJo asked for
# "on", "off", 30/10): proposals to hear like the others.
_HAND_PROPOSALS = {
    "on": "òn",
    "off": "òf",
    "link": "lìnk",
    "app": "èpp",
    "click": "clìk",
    "cloud": "clàud",
    "account": "accàunt",
    "router": "ràuter",
    "offline": "oflàin",
    "update": "apdèit",
    "default": "difòlt",
    "feedback": "fìdbek",
    "hardware": "àrduer",
    "tablet": "tàblet",
    "streaming": "strìming",
    "like": "làik",
}
# Verb forms that take pronouns at the end: imperatives (not the negative or formal ones),
# infinitives and gerunds.
_IMPERATIVES = (
    {"second-person", "singular"},
    {"first-person", "plural"},
    {"second-person", "plural"},
)


@dataclass
class Word:
    stressed: Counter[str] = field(default_factory=Counter)  # accented spellings seen
    # Syllables after the stress in the IPA: they win over the spelling ("stàtua" is
    # stà-tu-a, "àuto" àu-to).
    ipa_after: set[int] = field(default_factory=set)
    plain_word: bool = False  # seen other than as "verb + pronouns"


@dataclass
class Lexicon:
    words: dict[str, str] = field(default_factory=dict)  # accented, or KEEP
    at_start: dict[str, str] = field(default_factory=dict)
    hosts: dict[str, str] = field(default_factory=dict)
    stats: Counter[str] = field(default_factory=Counter)


def entries(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            yield json.loads(line)


def syllables_of(parts: list[str]) -> list[str]:
    """The syllables of a Wiktionary syllabification, given as one "ac‧cèn‧di‧ti" or as parts,
    without a label: ["(traditional", "careful style) u", "ten", "sì", "le"] -> u ten sì le."""
    text = "‧".join(parts)
    if "(" in text or ")" in text:
        text = text.rsplit(")", 1)[1] if ")" in text else ""
    return [s.strip() for s in text.split("‧") if s.strip()]


def from_syllables(parts: list[str]) -> str | None:
    """The accented word in a Wiktionary syllabification ("ac‧cèn‧di‧ti" -> "accènditi")."""
    word = "".join(syllables_of(parts))
    return word if stress_index(word) is not None else None


def ipa_syllables_after(ipa: str) -> int | None:
    """Syllables after the primary stress in a one-word IPA ("/atˈt͡ʃɛn.di.ti/" -> 2); None if
    the IPA is not split into syllables."""
    ipa = ipa.strip("/[]")
    if " " in ipa or ipa.count("ˈ") != 1 or not {".", "ˌ"} & set(ipa):
        return None
    after = ipa.split("ˈ", 1)[1]
    return after.count(".") + after.count("ˌ")  # a secondary stress also starts a syllable


def host_kind(tags: set[str]) -> str | None:
    """ "infinitive", "gerund" or "imperative" when the form takes pronouns, else None."""
    if "infinitive" in tags or "gerund" in tags:
        return "infinitive" if "infinitive" in tags else "gerund"
    positive = "imperative" in tags and not tags & {"negative", "formal", "third-person"}
    if positive and any(person <= tags for person in _IMPERATIVES):
        return "imperative"
    return None


def as_host(accented: str, kind: str) -> str:
    """The form pronouns attach to: an infinitive loses its final e ("accèndere" ->
    "accènder", "pórre" -> "pór")."""
    if kind != "infinitive":
        return accented
    return accented[:-2] if plain(accented).endswith("rre") else accented[:-1]


def by_stress(spellings: Counter[str]) -> list[str]:
    """The most frequent spelling for each stressed vowel ("utènsile", "utensìle")."""
    best: dict[int | None, str] = {}
    for spelling, _ in spellings.most_common():
        best.setdefault(stress_index(spelling), spelling)
    return list(best.values())


def derived(source: str, form: str) -> str | None:
    """`form` with the stress of the word it comes from ("àncora" -> "àncore", "lavàto" ->
    "lavàti"), if the two are spelled the same up to the stressed vowel."""
    index = stress_index(source)
    if index is None or len(form) <= index + 1:
        return None
    if plain(source)[: index + 1] != form[: index + 1]:
        return None
    return form[:index] + source[index] + form[index + 1 :]


def borrowed_from_english(entry: dict[str, Any]) -> bool:
    for template in entry.get("etymology_templates") or []:
        args = template.get("args") or {}
        if template.get("name") in _BORROWED and args.get("2") == "en":
            return True
        if template.get("name") == "ety" and str(args.get("3", "")).startswith("en:"):
            return True
    return False


def ipa_phonemes(ipa: str) -> list[tuple[str, bool]]:
    """(symbol, stressed) for each sound of a one-word IPA; a long consonant counts twice."""
    text = ipa.strip("/[]")
    for affricate, symbol in _AFFRICATES.items():
        text = text.replace(affricate, symbol)
    text = text.replace("i̯", "j").replace("u̯", "w")
    text = "".join(c for c in text if not unicodedata.combining(c))
    out: list[tuple[str, bool]] = []
    stress = False
    for c in text:
        if c == "ˈ":
            stress = True
        elif c == "ː":
            if out and out[-1][0] not in _VOWELS:
                out.append((out[-1][0], False))
        elif c not in "ˌ.‿":
            out.append((c, stress and c in _VOWELS))
            stress = stress and c not in _VOWELS
    return out


def respell_ipa(ipa: str) -> str | None:
    """The IPA of an English loanword in Italian letters, accent on the stress: "/ˈpas.sword/"
    -> "pàssuord", "/komˈpju.ter/" -> "compiùter". None for an IPA it cannot write."""
    if " " in ipa.strip() or "ˈ" not in ipa:
        return None
    sounds = ipa_phonemes(ipa)
    letters = []
    for n, (sound, stressed) in enumerate(sounds):
        before = sounds[n - 1][0] if n else None
        later = [s for s, _ in sounds[n + 1 :]]
        if sound in "ɲʎʃ" and later[:1] == [sound]:
            later = later[1:]  # a long "ʃ.ʃ" is written once: what follows decides the letters
        after = later[0] if later else None
        soft, vowel = after in _FRONT, after in _VOWELS
        if sound in "ɲʎʃ" and before == sound:
            continue  # "gn", "gl", "sc" are already long: "agnòstico", not "agngnòstico"
        if sound in "td" and after is not None and after in _AFFRICATE_OF[sound]:
            letters.append({"ʧ": "c", "ʤ": "g", "ʦ": "z", "ʣ": "z"}[after])  # "cc", "zz"
            continue
        if sound in "ʧʤ" and not vowel:
            return None  # "match", "bridge": Italian letters cannot end on that sound
        if sound in _VOWELS:
            letters.append(_STRESSED[sound] if stressed else _VOWELS[sound])
        elif sound == "k":
            letters.append("ch" if soft else "k" if after is None else "c")
        elif sound in "ɡg":
            letters.append("gh" if soft else "g")
        elif sound in "ʧʤʃ":
            base = {"ʧ": "c", "ʤ": "g", "ʃ": "sc"}[sound]
            if soft:
                letters.append(base)
            elif vowel:
                letters.append(base + "i")  # "ciao", "giù", "sciare"
            else:
                letters.append("sh")  # "flash": only ʃ gets here
        elif sound == "ʎ":
            letters.append("gl" if soft else "gli")
        elif sound in _SIMPLE:
            letters.append(_SIMPLE[sound])
        elif sound in _SAME:
            letters.append(sound)
        else:
            return None
    word = "".join(letters)
    return word if stress_index(word) is not None else None


def one_stress(spellings: Counter[str]) -> str | None:
    """The most frequent spelling, if all of them stress the same vowel."""
    if len({stress_index(s) for s in spellings}) != 1:
        return None
    return spellings.most_common(1)[0][0]


class Collector:
    def __init__(self) -> None:
        self.words: dict[str, Word] = defaultdict(Word)
        self.hosts: dict[str, Counter[str]] = defaultdict(Counter)
        self.compound: set[str] = set()
        self.pending: list[tuple[str, list[str]]] = []  # forms whose stress is their source's
        self.checked = 0  # syllables_after against Wiktionary's syllables
        self.mismatches: list[str] = []
        self.proposals: dict[str, str] = {}  # English loanword -> respelled from its IPA

    def add_reading(self, key: str, accented: str) -> None:
        """A way to stress `key`. An accent on the last vowel is written in every text
        ("numerò", "città"), so it is another word than the one without it."""
        if stress_index(accented) != len(accented) - 1:
            self.words[key].stressed[accented] += 1

    def add_entry(self, entry: dict[str, Any]) -> None:
        key = entry.get("word", "").lower()
        senses = entry.get("senses") or []
        if not _LETTERS.fullmatch(key) or (
            senses and all(set(s.get("tags") or []) & _OLD_TAGS for s in senses)
        ):
            return
        tags = [set(s.get("tags") or []) for s in senses]
        if any("compound-of" in t for t in tags):
            self.compound.add(key)
        else:
            self.words[key].plain_word = True
        for hyphenation in entry.get("hyphenations") or []:
            parts = hyphenation.get("parts") or []
            accented = from_syllables(parts)
            if accented is None or plain(accented).lower() != key:
                continue
            self.add_reading(key, accented.lower())
            self.check_rule(accented, parts)
        for sound in entry.get("sounds") or []:
            after = ipa_syllables_after(sound.get("ipa", ""))
            if after is not None:
                self.words[key].ipa_after.add(after)
        self.add_pending(key, entry.get("pos"), senses)
        for form in entry.get("forms") or []:
            self.add_form(form, entry.get("pos"), key)
        if borrowed_from_english(entry) and entry.get("pos") != "name":
            self.add_proposal(key, entry)

    def add_proposal(self, key: str, entry: dict[str, Any]) -> None:
        for sound in entry.get("sounds") or []:
            if "proscribed" in (sound.get("tags") or []):
                continue
            respelled = respell_ipa(sound.get("ipa", ""))
            if respelled:
                self.proposals.setdefault(key, respelled)
                return

    def add_pending(self, key: str, pos: str | None, senses: list[dict[str, Any]]) -> None:
        """A plural or feminine of a noun, an adjective or a participle takes every stress of
        its source, later. Not a superlative, diminutive or augmentative."""
        for sense in senses:
            tags = set(sense.get("tags") or [])
            if "form-of" not in tags or tags & _DERIVATION_TAGS:
                continue
            if not tags & {"plural", "feminine"} or (pos == "verb" and "participle" not in tags):
                continue
            sources = [f["word"].lower() for f in sense.get("form_of") or [] if "word" in f]
            if sources:
                self.pending.append((key, sources))

    def add_form(self, form: dict[str, Any], pos: str | None, lemma: str) -> None:
        tags = set(form.get("tags") or [])
        text = form.get("form", "").lower()
        if tags & _SKIP_TAGS or not _LETTERS.fullmatch(text):
            return
        key = plain(text)
        self.words[key].plain_word = True
        if stress_index(text) is None:
            if pos != "verb" and tags and tags <= _INFLECTION_TAGS:
                self.pending.append((key, [lemma]))  # "àncore": the stresses of "ancora"
            return
        accented = text
        self.add_reading(key, accented)
        if pos != "verb":
            return
        if {"participle", "past"} <= tags and accented.endswith("o"):
            for inflected in (accented[:-1] + ending for ending in "iae"):  # capìto: capìti
                self.words[plain(inflected)].plain_word = True
                self.add_reading(plain(inflected), inflected)
        kind = host_kind(tags)
        if kind is not None and stress_index(accented) != len(accented) - 1:
            host = as_host(accented, kind)
            self.hosts[plain(host)][host] += 1

    def resolve_pending(self) -> None:
        """Each form takes every stress its source has in Wiktionary: "utensili" both
        utènsili and utensìli, so it is left as written like "utensile"."""
        for key, sources in self.pending:
            for source in sources:
                word = self.words.get(source)
                for spelling in by_stress(word.stressed) if word else []:
                    stressed = derived(spelling, key)
                    if stressed is not None:
                        self.add_reading(key, stressed)

    def check_rule(self, accented: str, parts: list[str]) -> None:
        syllables = syllables_of(parts)
        stressed = [i for i, s in enumerate(syllables) if stress_index(s) is not None]
        index = stress_index(accented)
        if len(syllables) < 2 or len(stressed) != 1 or index is None:
            return  # written in one block: nothing to compare
        self.checked += 1
        rule = syllables_after(accented, index)
        if (rule >= 2) != (len(syllables) - stressed[0] - 1 >= 2):  # only where marking changes
            self.mismatches.append("‧".join(syllables))


def sdrucciola(key: str, word: Word, stats: Counter[str]) -> str | None:
    """The accented spelling of `key` if it is stressed before the second-to-last syllable,
    spelled the Italian way and has one stress only."""
    accented = one_stress(word.stressed)
    if accented is None:
        stats["parole con due accenti (lasciate)"] += 1
        return None
    index = stress_index(accented)
    assert index is not None
    early = syllables_after(accented, index) >= 2
    ipa = {after >= 2 for after in word.ipa_after}
    if len(ipa) > 1:
        stats["IPA con due accenti (lasciate)"] += 1
        return None
    # Wiktionary's IPA splits "au" and "eu" after the stress (/ˈɛ.u.ro/), said as one syllable.
    falling = key[index + 1 : index + 2] == "u" and key[index + 2 : index + 3] not in "aeiou"
    if ipa and ipa != {early} and not falling:
        early = not early
        stats["sillabe corrette dall'IPA"] += 1
    if early and _NATIVE.fullmatch(key) and not _FOREIGN.search(key):
        return accented
    return None


def build(collector: Collector) -> Lexicon:
    out = Lexicon()
    for key, spellings in collector.hosts.items():
        host = one_stress(spellings)
        if host is not None:
            out.hosts[key] = host
        else:
            out.stats["verbi con due accenti (senza pronomi)"] += 1
    for key, word in collector.words.items():
        if word.stressed and word.plain_word:
            accented = sdrucciola(key, word, out.stats)
            if accented is not None:
                out.words[key] = accented
    out.stats["sdrucciole"] = len(out.words)
    split = StressLexicon(words=dict(out.words), hosts=out.hosts).with_pronouns
    for key in collector.compound:  # verb + pronouns whose syllables disagree with the split
        word = collector.words.get(key)
        if word is None or word.plain_word or not word.stressed:
            continue
        accented = one_stress(word.stressed)
        index = stress_index(accented) if accented else None
        guess = split(key)
        if accented is None or index is None:
            if guess is not None:  # two stresses in Wiktionary: left as written
                out.words[key] = KEEP
                out.stats["verbo + pronomi dalle sillabe"] += 1
            continue
        if syllables_after(accented, index) >= 2 and guess != accented:
            out.words[key] = accented
        elif syllables_after(accented, index) < 2 and guess is not None:
            out.words[key] = KEEP
        else:
            continue
        out.stats["verbo + pronomi dalle sillabe"] += 1
    for key, word in collector.words.items():  # real words that look like verb + pronouns
        guess = split(key) if word.plain_word else None
        if guess is None or out.words.get(key) == guess:
            continue
        # "Fermati!" or "si sono fermati": only the imperatives Jack says; "copiati", "salvati"
        # open Claudio's reports as participles ("Chiusa, socio." in the 30/10 session).
        if key in _IMPERATIVES_SAID:
            out.at_start[key] = guess
            out.words.pop(key, None)
            out.stats["imperativo + ti o participio (accento a inizio frase)"] += 1
        else:  # "aprile", "capitali"
            out.words[key] = KEEP
            out.stats["parole che sembrano verbo + pronome (lasciate)"] += 1
    for key in _SAID_WORSE | number_words():
        if key in out.words or key in out.at_start:
            out.at_start.pop(key, None)
            out.words[key] = KEEP
    out.stats["verbi che prendono pronomi"] = len(out.hosts)
    return out


def number_words() -> set[str]:
    """The words digits_to_words_it writes ("tredici", "sedici"): Jack says them every day, so
    they stay as written also when Claude types them in letters."""
    return {w for n in range(2101) for w in digits_to_words_it(str(n)).split()}


def write(path: Path, lexicon: Lexicon, source: Path) -> None:
    lines = [
        "# Accents Jack writes (claudio.tts.stress_it), built by scripts/build_lexicon_it.py",
        f"# on {date.today().isoformat()} from {SOURCE_URL} ({source.name}).",
        "# Data: Wiktionary contributors, https://en.wiktionary.org (via kaikki.org, wiktextract).",
        "# License: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/ - this file",
        "# is an adaptation: only stress positions, homographs and pronoun-taking verb forms kept.",
        f"# word<TAB>accented | word<TAB>{KEEP} (leave as written) | word<TAB>accented<TAB>"
        f"{AT_START} (only where a clause starts) | verb form<TAB>accented<TAB>{HOST}",
    ]
    lines += [f"{key}\t{value}" for key, value in sorted(lexicon.words.items())]
    lines += [f"{key}\t{value}\t{AT_START}" for key, value in sorted(lexicon.at_start.items())]
    lines += [f"{key}\t{value}\t{HOST}" for key, value in sorted(lexicon.hosts.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_proposals(path: Path, proposals: dict[str, str], source: Path) -> None:
    lines = [
        "# English loanwords respelled for Jack from Wiktionary's Italian IPA: proposals to hear",
        "# (scripts/jack_variants.py --words WORD), never applied. Built by",
        f"# scripts/build_lexicon_it.py on {date.today().isoformat()} from {source.name}.",
        "# Data: Wiktionary contributors (via kaikki.org); CC BY-SA 4.0.",
        "# word<TAB>proposal",
    ]
    lines += [f"{key}\t{value}" for key, value in sorted(proposals.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help="kaikki.org-dictionary-Italian.jsonl(.gz)")
    parser.add_argument("--out", type=Path, default=LEXICON)
    parser.add_argument("--proposals", type=Path, default=PROPOSALS)
    args = parser.parse_args()
    collector = Collector()
    for n, entry in enumerate(entries(args.source), 1):
        collector.add_entry(entry)
        if n % 200_000 == 0:
            print(f"  {n} voci...")
    collector.resolve_pending()
    lexicon = build(collector)
    write(args.out, lexicon, args.source)
    proposals = _HAND_PROPOSALS | collector.proposals
    write_proposals(args.proposals, proposals, args.source)
    print(f"Proposte per parole inglesi: {len(proposals)} in {args.proposals}")
    agree = collector.checked - len(collector.mismatches)
    print(f"Sillabe: la regola decide come Wiktionary in {agree}/{collector.checked} parole")
    for name, value in lexicon.stats.most_common():
        print(f"{name}: {value}")
    rows = len(lexicon.words) + len(lexicon.at_start) + len(lexicon.hosts)
    print(f"Scritto {args.out} ({rows} righe, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
