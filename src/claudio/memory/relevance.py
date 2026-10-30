"""Which memories a phrase is about, found locally (free, ~1 ms for hundreds of memories).

Words are compared by their stem, the word without its final vowels ("sorella" and "sorelle"
are both "sorell"), and a stem may differ by a letter or two: Whisper writes "sorela" for
"sorella". Function words and the commonest verbs do not count. It finds what shares words
with the phrase, not what shares a meaning ("mamma" is not "madre"): the meaning is Claude's
job, which sees the memories in the prompt (prompt.py)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from claudio.memory.store import Memory
from claudio.router.normalizer import normalize

# Normalised (no accents). Italian and English: articles, prepositions, pronouns, the commonest
# verbs and adverbs. Wrongly kept words only cost a weaker ranking, never a wrong deletion
# (forget asks when two memories are equally close).
_STOPWORDS = (
    "il lo la i gli le un uno una l gl d di a da in con su per tra fra del dello della dei degli"
    " delle al allo alla ai agli alle dal dallo dalla dai dagli dalle nel nello nella nei negli"
    " nelle col sul sullo sulla sui sugli sulle e ed o od ma pero che chi cui non ne ci vi si se"
    " io tu lui lei noi voi loro me te mi ti mio mia miei mie tuo tua tuoi tue suo sua suoi sue"
    " nostro nostra nostri nostre vostro vostra questo questa questi queste quello quella quelli"
    " quelle cosa cose come dove quando quanto quanta perche piu meno molto poco tutto tutti"
    " tutta tutte anche ancora gia sempre mai poi ora adesso qui li la oggi ieri domani bene"
    " sono sei e siamo siete era ero ho hai ha abbiamo avete hanno avevo aveva fa fai faccio"
    " fare fatto posso puoi puo potrei vuoi voglio vorrei devo devi deve dire dici dimmi detto"
    " va vai vado sa sai so stato stata essere avere"
    " the an of to and or but in on at for with from by about as is are was were be been am"
    " i you he she it we they my your his her its our their me him us them this that these"
    " those what which who how when where why do does did have has had not no yes so very"
    " just also too can could would should will"
)
STOPWORDS = frozenset(_STOPWORDS.split())
_VOWELS_END = re.compile(r"[aeiouy]+$")
MIN_STEM = 3
FUZZY_FROM = 5  # stems this long may differ by a letter or two (rapidfuzz ratio)
FUZZY_RATIO = 85.0
PREFIX_FROM = 4  # "compleann" and "compleanno"... one stem may be the start of the other


def stems(text: str) -> frozenset[str]:
    """The content words of `text`, as stems."""
    out = set()
    for word in normalize(text, drop_fillers=False).split():
        if word in STOPWORDS:
            continue
        if word.isdigit():
            out.add(word)
            continue
        stem = _VOWELS_END.sub("", word) if len(word) > MIN_STEM else word
        if len(stem) >= MIN_STEM:
            out.add(stem)
    return frozenset(out)


def close(a: str, b: str) -> bool:
    """The same word, give or take an ending or a misheard letter."""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    if len(short) >= PREFIX_FROM and long.startswith(short):
        return True
    return len(short) >= FUZZY_FROM and fuzz.ratio(a, b) >= FUZZY_RATIO


def shared(query: Iterable[str], memory: frozenset[str]) -> int:
    """How many of the query's stems the memory has."""
    return sum(1 for q in query if any(close(q, m) for m in memory))


@dataclass(frozen=True)
class Scored:
    memory: Memory
    hits: int  # query stems found in the memory
    coverage: float  # hits / query stems: 1.0 = every word of the query is in it


def rank(query: str, memories: Sequence[Memory]) -> list[Scored]:
    """The memories that share a word with `query`, closest first (more hits, then a
    shorter memory: more of it is about the query; then the newer one)."""
    wanted = stems(query)
    if not wanted:
        return []
    found = []
    for memory in memories:
        own = stems(memory.text)
        hits = shared(wanted, own)
        if hits:
            scored = Scored(memory, hits, hits / len(wanted))
            found.append((hits, -len(own), memory.created, scored))
    found.sort(key=lambda item: item[:3], reverse=True)
    return [item[3] for item in found]
