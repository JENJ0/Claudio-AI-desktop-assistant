import pytest

from claudio.tts.number_words import digits_to_words_it, int_to_words


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (13, "tredici"),
        (21, "ventuno"),
        (23, "ventitré"),
        (28, "ventotto"),
        (51, "cinquantuno"),
        (100, "cento"),
        (108, "centotto"),
        (180, "centottanta"),
        (1000, "mille"),
        (2000, "duemila"),
        (2026, "duemila ventisei"),
        (1_000_000, "un milione"),
        (3_500_000, "tre milioni cinquecentomila"),
    ],
)
def test_int_to_words(n: int, words: str) -> None:
    assert int_to_words(n) == words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sono le 20 e 13.", "Sono le venti e tredici."),
        ("Alle 22 e 51 in Italia.", "Alle ventidue e cinquantuno in Italia."),
        ("Sono le 10:30.", "Sono le dieci e trenta."),
        ("Sono le 9:00.", "Sono le nove."),
        ("Costa 3,5 euro.", "Costa tre virgola cinque euro."),
        ("Pesa 1.500 grammi.", "Pesa mille cinquecento grammi."),
        ("Il 5. giorno.", "Il cinque. giorno."),
        ("Nessuna cifra qui.", "Nessuna cifra qui."),
        ("Codice A12.", "Codice A12."),
    ],
)
def test_digits_to_words_it(text: str, expected: str) -> None:
    assert digits_to_words_it(text) == expected
