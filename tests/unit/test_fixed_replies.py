"""The fixed replies shared by scripts/jack_takes.py and the ElevenLabs cache."""

import sys
from pathlib import Path

from claudio.config import load_settings
from claudio.router.matcher import load_commands
from claudio.safety.confirmation import ASK_LEVEL_2, CANCELLED
from claudio.tts import fixed
from claudio.tts.sentence_splitter import SentenceSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import jack_takes  # noqa: E402


def test_fixed_replies_are_the_usual_lines() -> None:
    settings = load_settings()
    replies = fixed.fixed_replies(settings)
    commands = load_commands(settings.resolve(settings.router.commands_file))
    plain = next(c.reply for c in commands if c.reply and "{" not in c.reply)
    assert {fixed.READY, ASK_LEVEL_2, CANCELLED, plain} <= set(replies)
    assert not any("{" in text for text in replies)
    assert len(replies) == len(set(replies))


def test_pieces_split_like_the_speaker() -> None:
    text = "Mi spengo. Per riaccendermi di' Claudio, accenditi."
    splitter = SentenceSplitter(first_clause_chars=25)
    expected = {p.strip() for p in [*splitter.feed(text), *splitter.flush()]}
    found = fixed.pieces([text], 25)
    assert text in found
    assert len(expected) > 1
    assert expected <= found
    assert fixed.READY in fixed.fixed_pieces(load_settings())


def test_jack_takes_uses_the_shared_list() -> None:
    settings = load_settings()
    assert jack_takes.fixed_replies(settings, 0) == fixed.fixed_replies(settings)
