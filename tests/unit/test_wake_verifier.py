"""The wake verifier keeps real wakes and drops phrases that only start like the name."""

from __future__ import annotations

import pytest

from claudio.wakeword.verifier import name_at_start, wake_confirmed


@pytest.mark.parametrize(
    "text",
    [
        "Hey Claudio, apri Chrome",
        "Ehi Claudio che ore sono",
        "Claudio, volume su",
        "E il Clodio, apri download",  # fast "hey" split by Whisper, name misspelt
        "Ok Claudio",
        "Hey Claudio, what time is it?",
    ],
)
def test_name_at_start(text: str) -> None:
    assert name_at_start(text)


@pytest.mark.parametrize(
    "text",
    [
        "Hey, ciao, come stai?",  # the false wake that scored 0.98
        "Ciao a tutti",
        "Hey, audio su",
        "Oggi parliamo di musica con Claudio",  # the name, but not at the start
        "",
    ],
)
def test_name_not_at_start(text: str) -> None:
    assert not name_at_start(text)


def test_command_without_name_is_kept() -> None:
    # Said in one breath the name gets clipped: "che ore sono" is still a real request.
    assert wake_confirmed("E ore sono", matched_command=True)


def test_false_wake_is_dropped() -> None:
    assert not wake_confirmed("Hey, ciao, come stai?", matched_command=False)


def test_silence_is_not_judged() -> None:
    assert wake_confirmed("  ", matched_command=False)
