"""Commands matched by how they sound (router/phonetic.py): Whisper's letters for a word it
did not catch ("Claudio QDT" for "Claudio, chiuditi", 26/10 logs) and other spellings."""

from __future__ import annotations

import pytest

from claudio.router.matcher import CommandMatcher
from claudio.router.phonetic import sound_key, spell_capitals


def test_capitals_are_read_as_italian_letter_names() -> None:
    assert spell_capitals("Claudio QDT.") == "Claudio cuditi."
    assert spell_capitals("QD Rocket League") == "cudi Rocket League"
    # One capital or a long word in capitals is a word, not letters.
    assert spell_capitals("A casa, TESTO") == "A casa, TESTO"


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("chiuditi", "cuditi"),  # "QDT"
        ("chiudi", "kiudi"),
        ("ghiaccio", "giaccio"),
        ("cartella", "kartela"),
        ("hotel", "otel"),
    ],
)
def test_spellings_that_sound_the_same_have_one_key(one: str, other: str) -> None:
    assert sound_key(one) == sound_key(other)


def test_different_sounds_keep_different_keys() -> None:
    assert sound_key("chiudi") != sound_key("chiedi")
    assert sound_key("cena") != sound_key("chena")  # soft c is not hard c


@pytest.mark.parametrize(
    ("text", "command_id"),
    [
        ("Claudio QDT.", "shutdown"),
        ("QDT.", "shutdown"),
        ("QD la finestra", "close_window"),
        ("kiudi la finestra", "close_window"),
        ("che ore sono", "tell_time"),  # the letters still match first
    ],
)
def test_commands_heard_as_letters_or_other_spellings(
    matcher: CommandMatcher, text: str, command_id: str
) -> None:
    match = matcher.match(text)
    assert match is not None and match.command.id == command_id


@pytest.mark.parametrize("text", ["PSS", "apri la cartella PSS", "DVD", "cosa vedi"])
def test_acronyms_and_other_phrases_are_not_commands(matcher: CommandMatcher, text: str) -> None:
    assert matcher.match(text) is None
