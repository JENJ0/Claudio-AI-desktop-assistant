import pytest

from claudio.tts.sentence_splitter import SentenceSplitter, split_stream


def split_all(text: str) -> list[str]:
    s = SentenceSplitter()
    return s.feed(text) + s.flush()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Apro Chrome.", ["Apro Chrome."]),
        ("Ciao! Come stai? Bene.", ["Ciao!", "Come stai?", "Bene."]),
        ("Il dott. Rossi arriva alle 10.", ["Il dott. Rossi arriva alle 10."]),
        ("Mele, pere ecc. sono frutta. Sì.", ["Mele, pere ecc. sono frutta.", "Sì."]),
        ("Costa 3.5 euro. Ok.", ["Costa 3.5 euro.", "Ok."]),
        ("Sono le 10:30. Fatto.", ["Sono le 10:30.", "Fatto."]),
        ("Davvero?! Sì...  Va bene.", ["Davvero?!", "Sì...", "Va bene."]),
        ("Ha detto «ciao.» Poi è uscito.", ["Ha detto «ciao.»", "Poi è uscito."]),
        ("Lo scrisse G. Verdi. Fine.", ["Lo scrisse G. Verdi.", "Fine."]),
        ("Prima riga\nSeconda riga", ["Prima riga", "Seconda riga"]),
        ("", []),
    ],
)
def test_split(text: str, expected: list[str]) -> None:
    assert split_all(text) == expected


def test_streamed_pieces_emit_sentence_as_soon_as_complete() -> None:
    s = SentenceSplitter()
    assert s.feed("Apro la car") == []
    assert s.feed("tella. Ecco") == ["Apro la cartella."]
    assert s.feed(" fatto") == []
    assert s.flush() == ["Ecco fatto"]


def test_waits_when_period_is_last_char() -> None:
    s = SentenceSplitter()
    assert s.feed("Costa 3.") == []  # could be "3.5"
    assert s.feed("5 euro. Poi") == ["Costa 3.5 euro."]


def test_long_sentence_is_cut_at_comma() -> None:
    s = SentenceSplitter(max_chars=40)
    out = s.feed("Questa è una frase molto lunga, che continua senza mai finire e ancora")
    assert out == ["Questa è una frase molto lunga,"]


async def test_split_stream() -> None:
    async def gen():
        for piece in ["Ciao", ". Sono ", "Claudio", "."]:
            yield piece

    assert [s async for s in split_stream(gen())] == ["Ciao.", "Sono Claudio."]


def test_first_clause_ends_at_a_comma_once_long_enough() -> None:
    s = SentenceSplitter(first_clause_chars=15)
    assert s.feed("Ciao JenJo, sto bene") == []  # the first comma comes too early
    assert s.feed(", grazie per averlo") == ["Ciao JenJo, sto bene,"]
    assert s.feed(" chiesto, davvero. Tu") == ["grazie per averlo chiesto, davvero."]  # whole
    assert s.flush() == ["Tu"]


def test_first_clause_skips_decimal_commas() -> None:
    s = SentenceSplitter(first_clause_chars=10)
    assert s.feed("La temperatura è 3,5 gradi") == []
    assert s.feed(", poco.") == ["La temperatura è 3,5 gradi,"]


def test_first_clause_waits_for_what_follows_the_comma() -> None:
    s = SentenceSplitter(first_clause_chars=5)
    assert s.feed("Sono le 3,") == []  # "3,5" or "3, poi": the next chunk decides
    assert s.feed("5.") == []
    assert s.flush() == ["Sono le 3,5."]


def test_a_full_sentence_before_the_comma_wins() -> None:
    s = SentenceSplitter(first_clause_chars=5)
    assert s.feed("Certo. Allora, vediamo, ") == ["Certo."]  # only the first piece is cut
