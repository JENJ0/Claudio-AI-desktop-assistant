import pytest

from claudio.tts.spoken_text import to_spoken_it


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Batteria al 35%.", "Batteria al trentacinque per cento."),
        ("Costa 5 €, anzi €12.", "Costa cinque euro, anzi dodici euro."),
        ("Fuori ci sono 18°C.", "Fuori ci sono diciotto gradi."),
        ("Il PC e la TV.", "Il pi ci e la ti vu."),
        ("Ti mando il PDF, OK?", "Ti mando il pi di effe, okay?"),
        ("Collega la chiavetta USB.", "Collega la chiavetta u esse bi."),
        ("BASTA, ho detto NO!", "basta, ho detto no!"),
        ("Fatto **subito**, eroe. 😎", "Fatto subito, eroe."),
        ("- primo punto", "primo punto"),
        ("Guarda [qui](https://example.com) o www.sito.it", "Guarda qui o il link"),
        ("Rock & roll", "Rock e roll"),
        ("3 + 4 = 7", "tre più quattro uguale sette"),
        ("Aspetta… ecco.", "Aspetta... ecco."),
        ("Sono le 10:30.", "Sono le dieci e trenta."),
        ("Un testo normale, senza nulla.", "Un testo normale, senza nulla."),
    ],
)
def test_text_becomes_italian_words(text: str, spoken: str) -> None:
    assert to_spoken_it(text) == spoken


def test_a_snake_case_name_keeps_its_underscores() -> None:
    assert to_spoken_it("il file note_spesa.txt") == "il file note_spesa.txt"
