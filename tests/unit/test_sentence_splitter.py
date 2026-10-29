import asyncio

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


def test_sentences_glued_without_a_space_are_split() -> None:
    """Replies with tools join their turns as "Cerco il nome.Nessuna cartella": two sentences."""
    from claudio.tts.sentence_splitter import SentenceSplitter

    splitter = SentenceSplitter()
    out = splitter.feed("Cerco il nome.Nessuna cartella, socio. ") + splitter.flush()
    assert out == ["Cerco il nome.", "Nessuna cartella, socio."]
    splitter = SentenceSplitter()
    assert splitter.feed("Gli U.S.A. e 3.5 restano. ") + splitter.flush() == [
        "Gli U.S.A. e 3.5 restano."
    ]


async def _sentences(*items: str):
    for item in items:
        yield item


async def _collect(gen) -> list[str]:
    return [x async for x in gen]


async def test_group_stream_joins_sentences_up_to_the_limit() -> None:
    from claudio.tts.sentence_splitter import group_stream

    got = await _collect(group_stream(_sentences("Ciao.", "Come stai?", "Bene."), 600))
    assert got == ["Ciao. Come stai? Bene."]
    got = await _collect(group_stream(_sentences("Aaaa.", "Bbbb.", "Cccc."), 11))
    assert got == ["Aaaa. Bbbb.", "Cccc."]


async def test_group_stream_keeps_a_long_sentence_whole_and_skips_empty_replies() -> None:
    from claudio.tts.sentence_splitter import group_stream

    long = "x" * 50 + "."
    assert await _collect(group_stream(_sentences("Hi.", long, "Ok."), 20)) == ["Hi.", long, "Ok."]
    assert await _collect(group_stream(_sentences(), 20)) == []


async def _timed(*items: tuple[float, str]):
    """Sentences that arrive late, as Claude writes them: (delay before it, sentence)."""
    for delay, sentence in items:
        await asyncio.sleep(delay)
        yield sentence


async def test_group_stream_starts_with_what_has_arrived_and_groups_the_rest() -> None:
    from claudio.tts.sentence_splitter import group_stream

    pieces = group_stream(
        _timed((0, "Primo pezzo di risposta."), (0.05, "Due."), (0, "Tre."), (0, "Quattro.")),
        220,
        first_chars=10,
    )
    first = await asyncio.wait_for(pieces.__anext__(), 0.03)  # not waiting for the rest
    assert first == "Primo pezzo di risposta."
    await asyncio.sleep(0.1)  # the first piece is being said: the rest of the reply arrives
    assert [p async for p in pieces] == ["Due. Tre. Quattro."]


async def test_group_stream_first_piece_waits_for_first_chars_or_the_end() -> None:
    from claudio.tts.sentence_splitter import group_stream

    got = await _collect(group_stream(_timed((0, "Ah."), (0.02, "Chrome è aperto.")), 220, 10))
    assert got == ["Ah. Chrome è aperto."]  # a lone "Ah." would be a generation of its own
    assert await _collect(group_stream(_timed((0, "Fatto.")), 220, 60)) == ["Fatto."]


async def test_group_stream_passes_on_a_failing_text_stream() -> None:
    from claudio.tts.sentence_splitter import group_stream

    async def broken():
        yield "Ciao."
        raise RuntimeError("rete")

    with pytest.raises(RuntimeError):
        await _collect(group_stream(broken(), 220))
