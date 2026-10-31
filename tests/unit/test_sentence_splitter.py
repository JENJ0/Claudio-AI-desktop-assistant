import asyncio

import pytest

from claudio.protocols import CUT_SHORT
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


CLOSING = " Mi fermo qui: se vuoi, continuo."


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        # 30/10: Claude stopped at the token limit inside "Attraverso": "Attra" was said.
        (["Fine. Attra", CUT_SHORT], ["Fine."]),
        (["Fine.", CUT_SHORT], ["Fine."]),  # a sentence ending right at the cut is said
        (["Fine. Attra", CUT_SHORT, CLOSING], ["Fine.", CLOSING.strip()]),
        ([CUT_SHORT, CLOSING], [CLOSING.strip()]),
        (["Uno. Due", ". Tre"], ["Uno.", "Due.", "Tre"]),  # no marker: the rest is said
    ],
)
async def test_a_cut_reply_drops_its_unfinished_sentence(
    chunks: list[str], expected: list[str]
) -> None:
    assert [s async for s in split_stream(_raw(*((0, c) for c in chunks)))] == expected
    # Jack's path (group_stream reads the raw chunks itself, IT-1 latenza) drops it too.
    from claudio.tts.sentence_splitter import group_stream

    pieces = await _collect(group_stream(_raw(*((0, c) for c in chunks)), 220))
    assert " ".join(pieces) == " ".join(expected)  # pieces group what has arrived


def test_a_cut_reply_said_again_has_no_fragment() -> None:
    """ "ripeti" and "Dicevo: ..." say a reply again from its chunks: never "Attra"."""
    from claudio.protocols import CUT_SHORT
    from claudio.tts.sentence_splitter import joined_reply

    parts = ["C'era una volta Jack. ", "Poi arrivò. Attra", CUT_SHORT, " Mi fermo qui."]
    assert joined_reply(parts) == "C'era una volta Jack. Poi arrivò. Mi fermo qui."
    assert joined_reply(["Ciao,\n", "socio.  "]) == "Ciao,\nsocio."  # untouched without a cut


def test_drop_unfinished_keeps_what_the_cut_completes() -> None:
    s = SentenceSplitter()
    assert s.feed("Ecco. Dove si trova Roma? In Ita") == ["Ecco.", "Dove si trova Roma?"]
    assert s.drop_unfinished() == []
    assert s.feed(CLOSING) == [] and s.flush() == [CLOSING.strip()]
    assert SentenceSplitter().drop_unfinished() == []


async def _sentences(*items: str):
    """Claude's text, one sentence per chunk (the space after it ends the sentence)."""
    for item in items:
        yield f"{item} "


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
    """Text that arrives late, as Claude writes it: (delay before it, chunk). A chunk ending
    in "." gets the space that ends its sentence."""
    for delay, chunk in items:
        await asyncio.sleep(delay)
        yield f"{chunk} " if chunk.endswith(".") else chunk


async def _raw(*items: tuple[float, str]):
    """Claude's chunks exactly as given, each after its delay."""
    for delay, chunk in items:
        await asyncio.sleep(delay)
        yield chunk


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


async def test_group_stream_first_piece_leaves_when_claude_pauses_for_a_tool() -> None:
    from claudio.tts.sentence_splitter import group_stream

    # 30/10 14:07: "Ci guardo, socio." then three tool calls; the next text 2.1 s later.
    pieces = group_stream(
        _timed((0, "Ci guardo, socio."), (1.0, "Trovata sul desktop, la apro.")),
        220,
        first_chars=40,
        pause_s=0.1,
    )
    first = await asyncio.wait_for(pieces.__anext__(), 0.6)  # not after the tools
    assert first == "Ci guardo, socio."
    assert [p async for p in pieces] == ["Trovata sul desktop, la apro."]


async def test_group_stream_a_sentence_still_being_written_is_no_pause() -> None:
    from claudio.tts.sentence_splitter import group_stream

    # A long second sentence streams in for longer than the pause (16 words x 30 ms = 0.48 s
    # against 0.25 s), but every word is new text: one piece, as before.
    rest = (
        "Accomodati, socio, che il narratore più bravo dell'universo sta per cominciare "
        "la sua storia di oggi."
    )
    words = [(0.03, f" {word}") for word in rest.split()]
    assert len(words) * 0.03 > 0.25
    got = await _collect(group_stream(_timed((0, "Ah."), *words), 220, 40, pause_s=0.25))
    assert got == [f"Ah. {rest}"]


async def test_group_stream_whitespace_is_no_new_text() -> None:
    from claudio.tts.sentence_splitter import group_stream

    # The brain ends the sentence with a space once the tools are called (the text stopped
    # 0.2 s earlier): the piece leaves at once, the pause has already passed.
    pieces = group_stream(
        _timed((0, "Ci guardo"), (0, ", socio"), (0, "!"), (0.4, " "), (0.6, "Fatto.")),
        220,
        first_chars=40,
        pause_s=0.4,
    )
    loop = asyncio.get_running_loop()
    start = loop.time()
    assert await pieces.__anext__() == "Ci guardo, socio!"
    assert loop.time() - start < 0.7  # ~0.4 s, not 0.4 + 0.4
    assert [p async for p in pieces] == ["Fatto."]


async def test_group_stream_a_pause_also_releases_the_sentence_the_splitter_held() -> None:
    from claudio.tts.sentence_splitter import group_stream

    # Review of 5.1: "Ci guardo subito." waits in the splitter for what follows (it could
    # be "3." before "5") until the brain's space, ~0.35 s later when the tool's arguments
    # are long. Both sentences go in one piece, before that space.
    pieces = group_stream(
        _raw((0, "Certo, socio."), (0, " Ci guardo subito."), (0.8, " "), (0.4, "Fatto.")),
        220,
        first_chars=40,
        pause_s=0.1,
    )
    loop = asyncio.get_running_loop()
    start = loop.time()
    assert await pieces.__anext__() == "Certo, socio. Ci guardo subito."
    assert loop.time() - start < 0.6  # the pause, not the space at 0.8 s
    assert [p async for p in pieces] == ["Fatto."]


async def test_group_stream_text_after_a_pause_is_grouped_again() -> None:
    from claudio.tts.sentence_splitter import group_stream

    # Review of 5.1: after the tools, the next round's sentences go together again (a
    # lone "Trovata." would be a generation of its own).
    got = await _collect(
        group_stream(
            _raw(
                (0, "Ci guardo, socio. "),
                (1.0, "Trovata. "),
                (0.02, "La apro adesso, socio. "),
                (0.02, "Altro?"),
            ),
            220,
            first_chars=40,
            pause_s=0.4,
        )
    )
    assert got == ["Ci guardo, socio.", "Trovata. La apro adesso, socio. Altro?"]


def test_the_splitter_gives_up_a_held_sentence_only_when_it_is_complete() -> None:
    s = SentenceSplitter()
    assert s.feed("Costa 3.") == [] and s.ended() == ["Costa 3."]
    s = SentenceSplitter()
    assert s.feed("Ha detto «ciao.»") == [] and s.ended() == ["Ha detto «ciao.»"]
    for unfinished in ("Cerco il fil", "Chiedi al dott.", "Lo scrisse G."):
        s = SentenceSplitter()
        assert s.feed(unfinished) == [] and s.ended() == []
        assert s.flush() == [unfinished]  # still there for the rest of the reply


async def test_group_stream_without_pause_waits_for_first_chars() -> None:
    from claudio.tts.sentence_splitter import group_stream

    got = await _collect(group_stream(_timed((0, "Ci guardo."), (0.1, "Fatto.")), 220, 40))
    assert got == ["Ci guardo. Fatto."]


async def test_group_stream_passes_on_a_failing_text_stream() -> None:
    from claudio.tts.sentence_splitter import group_stream

    async def broken():
        yield "Ciao."
        raise RuntimeError("rete")

    with pytest.raises(RuntimeError):
        await _collect(group_stream(broken(), 220))
