"""CLAUDIO's own voice heard back through the speakers (JenJo's log, 26/10 19:41)."""

from __future__ import annotations

import pytest

from claudio.echo import echo_around_name, is_echo, recent_reply, similarity

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


# A story about CLAUDIO (H2 12/11): the name is in the reply too.
HERO = [
    "C'era una volta un eroe bellissimo di nome Claudio.",
    "Era così bello che gli specchi chiedevano un autografo.",
    "Trovò un socio che cercava da tre ore la porta che aveva davanti.",
    "Claudio lo salvò, naturalmente.",
]
SAID = " ".join(HERO)  # all of it has played
NOW = " ".join(HERO[-2:])  # what plays now


@pytest.mark.parametrize(
    "heard",
    [
        "Claudio, che ore sono?",
        "Ehi Claudio, che ore sono?",
        "Che ore sono, Claudio?",
        "Claudio, volume su",
        "Claudio, pausa",
        "Claudio, no",  # "Claudio lo" in the story: one letter apart, other words
        "Claudio, sei bellissimo",  # "bellissimo" is in the story, not next to the name
    ],
)
def test_jenjo_says_other_words_around_the_name(heard: str) -> None:
    assert not echo_around_name(heard, SAID, NOW)


@pytest.mark.parametrize(
    "heard",
    [
        "Claudio lo salvò naturalmente",
        "Claudio lo salva naturalmente",  # misheard
        "Claudio l'ha salvato naturalmente",  # misheard, 87
        "Claudio salvò naturalmente",  # Whisper dropped "lo"
        "Claudio, naturalmente",  # dropped two words, 73: just over the bar
        "un eroe bellissimo di nome Claudio",  # the first sentence, heard late
    ],
)
def test_the_echo_says_the_story_words_around_the_name(heard: str) -> None:
    assert echo_around_name(heard, SAID, NOW)


@pytest.mark.parametrize("heard", ["Claudio?", "Ehi Claudio"])
def test_a_lone_name_is_the_echo_only_when_what_plays_says_it(heard: str) -> None:
    assert echo_around_name(heard, SAID, NOW)
    assert not echo_around_name(heard, SAID, " ".join(HERO[1:3]))


def test_one_word_around_the_name_is_compared_with_what_plays_now() -> None:
    """A long story says "Claudio si", "Claudio non" somewhere: one word would always match."""
    said = "Claudio si alzò presto. Claudio non aveva paura. Poi Claudio continuò a camminare."
    now = "Claudio non aveva paura. Poi Claudio continuò a camminare."
    assert not echo_around_name("Claudio, sì", said, now)
    # Accepted limit: said while "Claudio si alzò" plays, the yes is taken for its echo.
    assert echo_around_name("Claudio, sì", said, "Claudio si alzò presto.")


def test_a_name_not_said_yet_is_no_echo() -> None:
    assert not echo_around_name("Claudio lo salvò naturalmente", HERO[1], HERO[1])
    assert not echo_around_name("lo salvò naturalmente", SAID, NOW)  # no name heard
