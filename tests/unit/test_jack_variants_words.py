import sys

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import jack_variants


def test_words_from_the_command_line_get_the_table_and_the_proposal() -> None:
    spellings = jack_variants.spellings_for(["File", "computer", "subprime"], {"file": "fàiil"})
    assert spellings["File"][:2] == ["File", "fàiil"]  # as written, then JenJo's table
    assert spellings["computer"] == ["computer", "compiùter"]  # from /komˈpju.ter/
    assert spellings["subprime"] == ["subprime"]  # nothing known: only as written


def test_wer_counts_the_words_said_not_the_accents_written_for_the_voice() -> None:
    assert jack_variants.words("Socio, accenditi e poi fermati.") == [
        "socio",
        "accenditi",
        "e",
        "poi",
        "fermati",
    ]
    assert jack_variants.wer("Fermati, socio: accenditi.", "fermati socio accenditi") == 0.0


def test_spellings_reach_the_worker_as_written() -> None:
    import claudio.tts.jack as jack_module

    with jack_variants.sent_as_written():
        assert jack_module.to_spoken_it("Socio, accenditi.", {"socio": "sòcio"}) == (
            "Socio, accenditi."
        )
    assert jack_module.to_spoken_it("Socio, accenditi.") == "Socio, accènditi."
