import pytest

from claudio.router.normalizer import extract_number, normalize, only_wake_words, strip_accents


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Apri Chrome!", "apri chrome"),
        ("Claudio, apri il Blocco note, per favore.", "apri blocco note"),
        ("Che giorno è oggi?", "che giorno e oggi"),
        ("Volume giù", "volume giu"),
        ("Togli l'audio", "togli audio"),
        ("Potresti aprire la calcolatrice?", "apri calcolatrice"),
        ("Ehi, mi apri i Download?", "apri download"),
        ("Hey Jarvis, che ore sono?", "che ore sono"),  # wake word said in the same breath
        ("Ehi Giarvis apri la calcolatrice", "apri calcolatrice"),
        ("Hey Clodio, alza il volume", "alza volume"),
        ("   ", ""),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_normalize_keep_fillers() -> None:
    assert normalize("Grazie per la visione!", drop_fillers=False) == "grazie per la visione"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hey Jarvis.", True),
        ("Claudio!", True),
        ("Ehi Giarvis", True),
        ("Hey Jarvis, apri Chrome", False),
        ("Audio", False),
        ("Claudia", False),
        ("", False),
    ],
)
def test_only_wake_words(text: str, expected: bool) -> None:
    assert only_wake_words(text) is expected


def test_strip_accents() -> None:
    assert strip_accents("perché più là è") == "perche piu la e"


@pytest.mark.parametrize(
    ("text", "rest", "number"),
    [
        ("alza volume di 10", "alza volume", 10),
        ("alza volume tre volte", "alza volume", 3),
        ("abbassa volume", "abbassa volume", None),
    ],
)
def test_extract_number(text: str, rest: str, number: int | None) -> None:
    assert extract_number(text) == (rest, number)
