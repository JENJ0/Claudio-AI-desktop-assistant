import pytest

from claudio.router.normalizer import extract_number, normalize, strip_accents


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
        ("   ", ""),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_normalize_keep_fillers() -> None:
    assert normalize("Grazie per la visione!", drop_fillers=False) == "grazie per la visione"


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
