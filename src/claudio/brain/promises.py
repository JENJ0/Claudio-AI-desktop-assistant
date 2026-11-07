"""Does a reply promise a PC action in the first person? Used only to log it: plain patterns can
be wrong, so the reply itself is never changed."""

from __future__ import annotations

import re

# shortcut: plain patterns; a first-person story ("Ho aperto la porta del castello") and
# "salvo" (except) cannot be told apart. Upgrade only if the logged false positives hurt.
_PRESENT = (
    r"apro|scrivo|chiudo|cerco(?!\s+di\b)|avvio|lancio|sposto|cancello|premo|digito|copio"
    r"|incollo|salvo|eseguo"
)
_CLITIC = r"(?:(?:mi|ti|ci|vi|me|te|ce|ve|gli)?(?:lo|la|li|le|ne)|mi|ti|ci|vi)?"
_PAST = (
    r"ho\s+(?:(?:già|appena|anche|subito)\s+)?"
    r"(?:apert|scritt|chius|cercat|avviat|lanciat|spostat|cancellat|premut|digitat|copiat"
    r"|incollat|salvat|eseguit)[oaie]"
)
_PROGRESSIVE = (
    r"sto\s+(?:aprendo|scrivendo|chiudendo|cercando|avviando|lanciando|spostando|cancellando"
    rf"|premendo|digitando|copiando|incollando|salvando|eseguendo){_CLITIC}"
)
_INFINITIVE = (
    r"(?:apr|scriv|chiud|cerc|avvi|lanci|spost|cancell|prem|digit|copi|incoll|salv|esegu)"
    rf"(?:ire|ere|are|ir|er|ar){_CLITIC}"
)
_INTENTION = rf"(?:dovrei|devo|vado\s+a|vado\s+ad|sto\s+per|vorrei)\s+{_INFINITIVE}"
_NOW = re.compile(rf"\b(?:{_PRESENT}|{_PROGRESSIVE}|{_INTENTION})\b", re.IGNORECASE)
_DONE = re.compile(rf"\b{_PAST}\b", re.IGNORECASE)
_NOT = re.compile(r"\bnon\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")


def promises_action(text: str, ran: bool = False) -> bool:
    """True when a sentence says "I open/write/..." (now, done or about to). A sentence with
    "non" before the verb, or a question, does not count. `ran`: a tool already ran in this
    request, so "Ho aperto..." is the closing sentence, not a promise; only present and
    intention forms still count."""
    patterns = (_NOW,) if ran else (_NOW, _DONE)
    for sentence in _SENTENCE.findall(text):
        if "?" in sentence:
            continue
        for pattern in patterns:
            found = pattern.search(sentence)
            if found and not _NOT.search(sentence[: found.start()]):
                return True
    return False
