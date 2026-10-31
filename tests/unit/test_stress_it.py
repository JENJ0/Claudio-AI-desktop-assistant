import logging

import pytest

from claudio.tts.stress_it import (
    KEEP,
    LEXICON,
    StressLexicon,
    mark_stress,
    plain,
    put_accent,
    stress_index,
    syllables_after,
    words_of,
)


@pytest.mark.parametrize(
    ("word", "after"),
    [
        ("accènditi", 2),
        ("àbitano", 3),  # bisdrucciola
        ("àquila", 2),  # "qui" is one syllable
        ("Itàlia", 1),  # "lia" too
        ("àuto", 1),  # "au" joins the stress
        ("pàio", 1),
        ("farmacìa", 1),
        ("àrea", 2),  # e and a are two
        ("dìglielo", 2),
        ("città", 0),
        ("accèndimelo", 3),
    ],
)
def test_syllables_after_the_stress(word: str, after: int) -> None:
    index = stress_index(word)
    assert index is not None
    assert syllables_after(word, index) == after


def test_the_written_accent_and_its_place() -> None:
    assert stress_index("accènditi") == 3
    assert stress_index("perché") == 5
    assert stress_index("socio") is None
    assert stress_index("àncóra") is None  # two accents: no single stress
    assert plain("Àbitano accènditi") == "Abitano accenditi"
    assert put_accent("Abitano", 0, "à") == "Àbitano"
    assert put_accent("accenditi", 3, "è") == "accènditi"


def _lexicon() -> StressLexicon:
    return StressLexicon(
        words={"abitano": "àbitano", "brindisi": "brìndisi", "aprile": KEEP},
        hosts={
            "accendi": "accèndi",
            "accender": "accènder",
            "accendendo": "accendèndo",
            "accendete": "accendéte",
            "apri": "àpri",
            "aprir": "aprìr",
            "ferma": "férma",
        },
        at_start={"fermati": "férmati"},
    )


@pytest.mark.parametrize(
    ("word", "stressed"),
    [
        ("accenditi", "accènditi"),  # the A/B of 30/10: heard "accendete" without the accent
        ("accendimelo", "accèndimelo"),
        ("accendermi", "accèndermi"),  # infinitive without its final e
        ("accendendolo", "accendèndolo"),
        ("accendetelo", "accendételo"),
        ("aprilo", "àprilo"),
        ("aprirmelo", "aprìrmelo"),  # two pronouns after a stressed last syllable
        ("dammelo", "dàmmelo"),  # da' doubles the consonant
        ("vattene", "vàttene"),
        ("Abitano", "Àbitano"),  # the case stays
        ("BRINDISI", "BRÌNDISI"),
        ("aprirlo", None),  # still second-to-last: left as written
        ("dammi", None),
        ("dagli", None),
        ("aprile", None),  # a real word, not "apri" + "le"
        ("socio", None),  # not in the lexicon: the voice already says it right
        ("fermati", None),  # only where a clause starts
    ],
)
def test_words_stressed_before_the_second_to_last_syllable(word: str, stressed: str | None) -> None:
    assert _lexicon().stressed(word) == stressed


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Fermati!", "Férmati!"),
        ("Socio, fermati.", "Socio, férmati."),  # a comma after a vocative
        ("Dai, fermati.", "Dai, férmati."),
        ("Ora fermati. E fermati!", "Ora férmati. E férmati!"),  # an opener that starts it
        ("Si sono fermati qui.", "Si sono fermati qui."),
        ("Sono ora fermati.", "Sono ora fermati."),  # "ora" inside the clause
        ("Salvati e fermati.", "Salvati e fermati."),  # a list of participles
        ("Chiusi, fermati e salvati.", "Chiusi, fermati e salvati."),
        ("Ne ho 1.250 fermati.", "Ne ho 1.250 fermati."),  # the point of a number
    ],
)
def test_an_imperative_that_is_also_a_participle_is_accented_where_a_clause_starts(
    text: str, spoken: str
) -> None:
    assert mark_stress(text, _lexicon()) == spoken


def test_a_verb_and_pronouns_split_two_ways_is_left_as_written() -> None:
    lexicon = StressLexicon(hosts={"manda": "mànda", "mandate": "mandàte"})
    assert lexicon.stressed("mandatelo") is None  # màndatelo (2sg + te + lo) or mandàtelo
    assert lexicon.stressed("mandami") == "màndami"


def test_a_u_before_a_vowel_after_a_consonant_is_a_syllable() -> None:
    assert syllables_after("stàtua", 2) == 2
    assert syllables_after("contìnuo", 4) == 2
    assert syllables_after("àcqua", 0) == 1  # after q and g it glides
    assert syllables_after("lìngua", 1) == 1


def test_no_accent_on_a_starting_u() -> None:
    lexicon = StressLexicon(words={"utile": "ùtile", "ospiti": "òspiti"}, hosts={"x": "x"})
    assert lexicon.stressed("Utile") is None  # "ùtile" was heard "etile" (A/B 30/10)
    assert lexicon.stressed("ospiti") == "òspiti"


def test_mark_stress_leaves_accented_words_and_the_table_alone() -> None:
    lexicon = _lexicon()
    text = "Socio, accenditi: abitano a Brindisi, accènditi già."
    assert mark_stress(text, lexicon) == "Socio, accènditi: àbitano a Brìndisi, accènditi già."
    assert mark_stress("abitano qui", lexicon, leave={"abitano"}) == "abitano qui"
    assert mark_stress("", lexicon) == ""


def test_words_of_the_table_respellings() -> None:
    assert words_of(["dèsk-top", "pì cì", "Clòd"]) == {"dèsk", "top", "pì", "cì", "clòd"}


def test_the_lexicon_file_is_read_by_kind(tmp_path) -> None:
    path = tmp_path / "stress_it.tsv"
    path.write_text(
        "# comment\n"
        "abitano\tàbitano\n"
        "aprile\t-\n"
        "fermati\tférmati\tc\n"
        "accendi\taccèndi\th\n"
        "broken line\n",
        encoding="utf-8",
    )
    lexicon = StressLexicon.load(path)
    assert lexicon.words == {"abitano": "àbitano", "aprile": KEEP}
    assert lexicon.at_start == {"fermati": "férmati"}
    assert lexicon.hosts == {"accendi": "accèndi"}


def test_a_missing_lexicon_turns_the_accents_off(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        lexicon = StressLexicon.load(tmp_path / "missing.tsv")
    assert lexicon.words == {} and lexicon.hosts == {}
    assert mark_stress("Socio, accenditi e dammelo.", lexicon) == "Socio, accenditi e dammelo."
    assert "Accenti di Jack spenti" in caplog.text


@pytest.fixture(scope="module")
def real() -> StressLexicon:
    if not LEXICON.exists():
        pytest.skip("data/stress_it.tsv not built")
    return StressLexicon.load()


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Socio, accenditi.", "Socio, accènditi."),
        ("Ok, riaccendermi?", "Ok, riaccèndermi?"),
        ("Accomodati, eroe.", "Accòmodati, eroe."),
        ("Ci siamo accomodati.", "Ci siamo accomodati."),
        ("Dimmelo e vattene.", "Dìmmelo e vàttene."),
        ("Abitano a Brindisi.", "Àbitano a Brìndisi."),
        ("Socio, ancora tu?", "Socio, ancora tu?"),  # àncora / ancóra: left as written
        ("Lo faccio subito.", "Lo faccio subito."),  # sùbito / subìto
        ("È tutto pronto in cartella.", "È tutto pronto in cartella."),
        ("Ad aprile, socio.", "Ad aprile, socio."),
        ("Costa un euro.", "Costa un euro."),  # èu-ro, not è-u-ro
        ("Calmati, socio.", "Calmati, socio."),  # "càlmati" was heard "chiamati" (A/B 30/10)
        ("Ultimo file utile.", "Ultimo file utile."),  # a starting "ù" is garbled
        # Found by the review of 31/10: each was accented on the wrong vowel before.
        ("Ci siamo capiti.", "Ci siamo capiti."),  # capìti (capire) / càpiti (capitare)
        ("Sto benone, è famosissimo.", "Sto benone, è famosissimo."),  # not bène, famóso
        ("Il bastoncino del fannullone.", "Il bastoncino del fannullone."),
        ("Sono le tredici, cioè sedici.", "Sono le tredici, cioè sedici."),  # number words
        ("Il machine learning.", "Il machine learning."),  # not the archaic "màchine"
        ("Gli utensili e i micrometri.", "Gli utensili e i micrometri."),  # two stresses
        ("Gli obiettivi perseguiti.", "Gli obiettivi perseguiti."),
        ("Ecco i risultati. Risultati veri.", "Ecco i risultati. Risultati veri."),  # a noun
        ("Rilassati! Si sono rilassati.", "Rilàssati! Si sono rilassati."),  # a Jack line
        ("Aggrappati! Sono aggrappati.", "Aggrappati! Sono aggrappati."),  # not in the list
        ("I dispositivi sono ora collegati.", "I dispositivi sono ora collegati."),
        ("Copiati, rinominati e ordinati.", "Copiati, rinominati e ordinati."),
        ("Ho trovato 1.250 risultati.", "Ho trovato 1.250 risultati."),
        ("La statua e il residuo.", "La stàtua e il resìduo."),
    ],
)
def test_the_real_lexicon(real: StressLexicon, text: str, spoken: str) -> None:
    assert mark_stress(text, real) == spoken
