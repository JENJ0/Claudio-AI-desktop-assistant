"""CLAUDIO's own voice heard back through the speakers (JenJo's log, 26/10 19:41)."""

from __future__ import annotations

import pytest

from claudio.echo import is_echo, recent_reply, similarity

REPLY = (
    "Sono le 19 e 42. Il tempo passa anche per il padrone dell'universo, che è in linea. "
    "Cos'altro ti serve, campione?"
)


@pytest.mark.parametrize(
    "heard",
    [
        "Se non le diciamo",  # "Sono le 19 e 42"
        "dell'università e lì",  # "dell'universo, che è in linea"
        "Sono le diciannove e quarantadue.",
        "il padrone dell'universo",
        "Cosa altro ti serve campione",
    ],
)
def test_the_echo_is_recognised(heard: str) -> None:
    assert is_echo(heard, REPLY, REPLY, 70)


@pytest.mark.parametrize(
    "heard",
    [
        "Claudio, che ore sono?",
        "aspetta, raccontami una barzelletta",
        "apri la cartella download",
        "Claudio apri Chrome",
        "che tempo fa domani a Milano",
    ],
)
def test_a_real_request_is_not_the_echo(heard: str) -> None:
    assert not is_echo(heard, REPLY, REPLY, 70)


def test_zero_turns_the_letter_check_off() -> None:
    assert not is_echo("Se non le diciamo", REPLY, REPLY, 0)
    assert is_echo("sono le 19 e 42", REPLY, REPLY, 0)  # the words are still checked


def test_nothing_to_compare() -> None:
    assert not is_echo("", REPLY, REPLY, 70)
    assert not is_echo("che ore sono", "", "", 70)
    assert similarity("", REPLY) == 0.0


def test_only_the_last_sentences_count() -> None:
    played = ["Uno.", "Due.", "Tre."]
    assert recent_reply(played, "ignored") == "Due. Tre."
    assert recent_reply([], "x" * 500) == "x" * 240


def test_a_long_story_does_not_swallow_requests() -> None:
    """Only the stretch being played is compared: a long story has letters for everything."""
    story = " ".join(f"Frase numero {n} della storia del robot che scappa." for n in range(40))
    recent = recent_reply(["Frase numero 38 della storia.", "Frase numero 39 della storia."], story)
    assert not is_echo("Claudio, che ore sono?", recent, story, 70)
