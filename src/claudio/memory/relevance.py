"""Which memories a phrase is about, found locally (free, ~1 ms for hundreds of memories).

Two strengths of match. By stem, the word without its final vowels ("sorella" and "sorelle"
are both "sorell"), give or take a misheard letter ("sorela"): wide, good to find candidates
and to pick facts for a request. By whole word: "Anna" and "anni", "Marco" and "marzo",
"nonno" and "nonna" share a stem but are different words; only a whole-word match lets
"dimentica X" forget without asking. Function words, the commonest verbs and the wake word
do not count. It finds what shares words with the phrase, not what shares a meaning ("mamma"
is not "madre"): the meaning is Claude's job, which sees the memories in the prompt."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from claudio.memory.store import Memory
from claudio.router.normalizer import GREETINGS, WAKE_WORDS, normalize

# Normalised (no accents). Italian and English: articles, prepositions, pronouns, the commonest
# verbs and adverbs.
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
    " va vai vado sa sai so stato stata essere avere ok okay senti ascolta scusa dai allora"
    " the an of to and or but in on at for with from by about as is are was were be been am"
    " i you he she it we they my your his her its our their me him us them this that these"
    " those what which who how when where why do does did have has had not no yes so very"
    " just also too can could would should will please"
)
STOPWORDS = frozenset(_STOPWORDS.split()) | frozenset(WAKE_WORDS) | GREETINGS
_VOWELS_END = re.compile(r"[aeiouy]+$")
MIN_STEM = 3
FUZZY_FROM = 5  # stems this long may differ by a letter or two (rapidfuzz ratio)
FUZZY_RATIO = 85.0
PREFIX_FROM = 4  # "compleann" and "compleanno"... one stem may be the start of the other
WORD_RATIO = 85.0  # whole words: "sorella"/"sorelle" (86) and "sorela" (92) yes, "anna"/"anni" no
WORD_FUZZY_FROM = 5


def words(text: str) -> tuple[str, ...]:
    """The content words of `text`, normalised, in order."""
    return tuple(
        w
        for w in normalize(text, drop_fillers=False).split()
        if w not in STOPWORDS and (w.isdigit() or len(w) >= MIN_STEM)
    )


def stem(word: str) -> str:
    if word.isdigit() or len(word) <= MIN_STEM:
        return word
    cut = _VOWELS_END.sub("", word)
    return cut if len(cut) >= MIN_STEM else word


def stems(text: str) -> frozenset[str]:
    """The content words of `text`, as stems."""
    return frozenset(stem(w) for w in words(text))


def close(a: str, b: str) -> bool:
    """The same stem, give or take an ending or a misheard letter."""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    if len(short) >= PREFIX_FROM and long.startswith(short):
        return True
    return len(short) >= FUZZY_FROM and fuzz.ratio(a, b) >= FUZZY_RATIO


def same_word(a: str, b: str) -> bool:
    """The same whole word, give or take a misheard letter (never another gender or name)."""
    if a == b:
        return True
    return min(len(a), len(b)) >= WORD_FUZZY_FROM and fuzz.ratio(a, b) >= WORD_RATIO


@dataclass(frozen=True)
class Scored:
    memory: Memory
    hits: int  # query stems found in the memory
    coverage: float  # hits / query stems: 1.0 = every word of the query is in it
    exact: float = 0.0  # the same, by whole words: 1.0 = no word matched only by its stem
    named: bool = False  # a hit that is not a bare number


def rank(query: str, memories: Sequence[Memory]) -> list[Scored]:
    """The memories that share a stem with `query`, closest first (more hits, then more
    whole-word hits, then a shorter memory: more of it is about the query; then the newer)."""
    wanted = words(query)
    wanted_stems = {stem(w): w for w in wanted}
    if not wanted_stems:
        return []
    found = []
    for memory in memories:
        own_words = words(memory.text)
        own_stems = {stem(w) for w in own_words}
        hit_stems = [s for s in wanted_stems if any(close(s, m) for m in own_stems)]
        if not hit_stems:
            continue
        whole = sum(1 for s in hit_stems if any(same_word(wanted_stems[s], w) for w in own_words))
        scored = Scored(
            memory,
            len(hit_stems),
            len(hit_stems) / len(wanted_stems),
            whole / len(wanted_stems),
            any(not s.isdigit() for s in hit_stems),
        )
        found.append((len(hit_stems), whole, -len(own_stems), memory.created, scored))
    found.sort(key=lambda item: item[:4], reverse=True)
    return [item[4] for item in found]
