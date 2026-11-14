"""Claudio's fixed replies: the lines he says word for word again and again. Shared by
scripts/jack_takes.py (Jack's takes chosen in advance) and the ElevenLabs cache."""

from __future__ import annotations

from collections.abc import Iterable

from claudio.config import Settings
from claudio.router.matcher import load_commands
from claudio.safety.confirmation import ASK_LEVEL_2, ASK_LEVEL_3, CANCELLED, NOT_UNDERSTOOD
from claudio.tts.sentence_splitter import SentenceSplitter

READY = "Sono pronto."  # said once everything has loaded (__main__.announce_ready)


def fixed_replies(settings: Settings) -> list[str]:
    """The Italian replies and questions of commands.toml plus the confirmation lines and
    READY: no duplicates, no placeholders, none longer than one piece."""
    commands = load_commands(settings.resolve(settings.router.commands_file))
    lines = [READY, ASK_LEVEL_2, ASK_LEVEL_3, CANCELLED, NOT_UNDERSTOOD]
    for command in commands:
        lines += [command.reply, command.confirm]
    limit = settings.tts.jack_group_chars or 220
    return [t for t in dict.fromkeys(lines) if t and "{" not in t and len(t) <= limit]


def pieces(texts: Iterable[str], first_clause_chars: int) -> frozenset[str]:
    """Every text whole plus the pieces the speaker would hand the synth, all stripped."""
    found: set[str] = set()
    for text in texts:
        splitter = SentenceSplitter(first_clause_chars=first_clause_chars)
        found.add(text.strip())
        found.update(p.strip() for p in [*splitter.feed(text), *splitter.flush()])
    found.discard("")
    return frozenset(found)


def fixed_pieces(settings: Settings) -> frozenset[str]:
    return pieces(fixed_replies(settings), settings.tts.first_clause_chars)
