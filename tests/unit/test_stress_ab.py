import sys

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import stress_ab


def test_each_word_is_said_as_written_and_with_its_accent() -> None:
    jobs = stress_ab.make_jobs(["pronomi", "piane"], {})
    accenditi = [j for j in jobs if j["word"] == "accenditi"]
    assert [(j["arm"], j["text"]) for j in accenditi] == [
        ("scritta", "Socio, ho detto: accenditi. Chiaro?"),
        ("accentata", "Socio, ho detto: accènditi. Chiaro?"),
    ]
    socio = [j for j in jobs if j["word"] == "socio"]
    assert [j["arm"] for j in socio] == ["uguale"]  # a piana: mark_stress leaves it
    assert len({j["name"] for j in jobs}) == len(jobs)


def test_what_whisper_heard_in_place_of_the_word() -> None:
    sentence = "Socio, ho detto: accènditi. Chiaro?"
    assert stress_ab.heard_as("accenditi", sentence, "Socio, ho detto accendete. Chiaro?") == (
        "accendete"
    )
    assert stress_ab.heard_as("accenditi", sentence, "Socio ho detto accenditi chiaro") == (
        "accenditi"
    )
    assert stress_ab.heard_as("accenditi", sentence, "Socio, ho detto. Chiaro?") == "(niente)"
