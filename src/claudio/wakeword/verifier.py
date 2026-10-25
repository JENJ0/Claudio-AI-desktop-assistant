"""Second check on a wake: did the person really say the name?

The wake word model fires on phrases that start like "Hey Claudio": "Hey, ciao, come stai?"
scored 0.98. Whisper already gets the audio before the detection (the wake word included),
so the transcript tells whether the name was there. A wake is kept when the name is among the
first words, or when the request is a known local command anyway (said fast, the name often
comes out clipped: "e ore sono"). A rejected wake ends in silence, not "non ho capito", and
never reaches the LLM: a false alarm then costs nothing."""

from __future__ import annotations

from claudio.router.normalizer import GREETINGS, WAKE_SEARCH, is_wake_word, normalize

# Words before the name that still count as "at the start": greetings, a fast "hey" split by
# Whisper ("e il claudio"), "ok", "scusa".
VERIFY_WORDS = WAKE_SEARCH + 1


def name_at_start(text: str) -> bool:
    """True when one of the first words is the wake word ("ehi claudio", "ok clodio apri")."""
    words = normalize(text, drop_fillers=False).split()[:VERIFY_WORDS]
    return any(w not in GREETINGS and is_wake_word(w) for w in words)


def wake_confirmed(text: str, matched_command: bool) -> bool:
    """Keep the wake when the name was heard at the start or the request is a local command.
    An empty transcript is not judged here: the caller already handles silence."""
    return not text.strip() or matched_command or name_at_start(text)


# Whisper's spelling of "Claudio" with the "C" cut off, or said softly ("l'audio", "audio").
CLIPPED_NAMES = frozenset({"audio"})
CLIPPED_LEAD = frozenset({"l", "e"})  # "l'audio", "e l'audio" (a fast "hey")


def clipped_name(text: str) -> bool:
    """True when the transcript starts like a clipped name ("L'audio?", "l'audio apri..."): a
    cut-off name can only be at the start of the audio Whisper gets."""
    words = normalize(text, drop_fillers=False).split()
    while words and (words[0] in GREETINGS or words[0] in CLIPPED_LEAD):
        words = words[1:]
    return bool(words) and words[0] in CLIPPED_NAMES
