import logging

import pytest

from claudio.tts.loanwords import PROPOSALS, UnknownWords, load_proposals, looks_foreign


@pytest.mark.parametrize(
    "word",
    ["subprime", "update", "chat", "off", "Windows", "desktop", "hotel", "smartphone", "Spotify"],
)
def test_foreign_spellings(word: str) -> None:
    assert looks_foreign(word)


@pytest.mark.parametrize(
    "word",
    ["accenditi", "socio", "atmosfera", "tecnico", "pneumatico", "cartella", "ho", "chiave", "per"],
)
def test_italian_spellings(word: str) -> None:
    assert not looks_foreign(word)


def test_unknown_words_are_logged_once_with_their_proposal(caplog) -> None:
    unknown = UnknownWords({"online": "onlàin"})
    with caplog.at_level(logging.INFO, logger="claudio.tts.loanwords"):
        found = unknown.note("Il subprime è online, il subprime no.")
        again = unknown.note("Ancora subprime.")
    assert found == ["subprime", "online"]  # "online" looks Italian: known by its proposal
    assert again == []
    assert "parola sconosciuta 'subprime'" in caplog.text
    assert "proposta: 'onlàin'" in caplog.text


def test_the_table_acronyms_and_accented_words_are_not_candidates() -> None:
    unknown = UnknownWords({})
    table = {"desktop", "dèsk", "top"}  # key and respelling "dèsk-top"
    assert unknown.note("Il PDF del desktop su dèsk-top e Chrome", known=table) == ["chrome"]


def test_the_log_stops_at_its_limit() -> None:
    unknown = UnknownWords({}, limit=2)
    assert unknown.note("chat web app link") == ["chat", "web"]
    assert unknown.note("cloud") == []


def test_proposals_file(tmp_path) -> None:
    path = tmp_path / "respell_en.tsv"
    path.write_text("# header\nonline\tonlàin\nbad line\n", encoding="utf-8")
    assert load_proposals(path) == {"online": "onlàin"}
    assert load_proposals(tmp_path / "missing.tsv") == {}


def test_the_built_proposals_agree_with_jenjos_picks_by_ear() -> None:
    if not PROPOSALS.exists():
        pytest.skip("data/respell_en.tsv not built")
    proposals = load_proposals()
    # JenJo chose these by ear on 24-25/10 ([tts.jack_words]); Wiktionary's IPA gives the same.
    assert {k: proposals[k] for k in ("browser", "computer", "mouse", "password")} == {
        "browser": "bràuser",
        "computer": "compiùter",
        "mouse": "màus",
        "password": "pàssuord",
    }
