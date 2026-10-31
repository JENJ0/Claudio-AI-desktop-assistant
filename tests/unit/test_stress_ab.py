import re
import sys

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import stress_ab


def test_each_word_is_said_as_written_and_with_its_accent() -> None:
    jobs = stress_ab.make_jobs(["pronomi", "piane"], {})
    accenditi = [j for j in jobs if j["targets"] == ["accenditi"]]
    assert [(j["arm"], j["text"]) for j in accenditi] == [
        ("scritta", "Socio, ho detto: accenditi. Chiaro?"),
        ("accentata", "Socio, ho detto: accènditi. Chiaro?"),
    ]
    socio = [j for j in jobs if j["targets"] == ["socio"]]
    assert [j["arm"] for j in socio] == ["uguale"]  # a piana: mark_stress leaves it
    assert len({j["name"] for j in jobs}) == len(jobs)


def test_the_real_sentences_keep_their_words_and_get_the_accents() -> None:
    # Left as written since the A/B of 30/10: "càlmati" and a starting "ù" sound worse.
    left = {"calmati", "ultimo", "utile", "unico"}
    jobs = stress_ab.make_jobs([], {}, real=True)
    for job in jobs:
        said = stress_ab.tokens(job["text"])
        for target in job["targets"]:
            assert target in said  # the accent does not change the letters
            written = re.search(rf"\b{target}\b", job["text"].lower()) is not None
            assert written == (job["arm"] != "accentata" or target in left), (target, job["arm"])


def test_what_whisper_heard_in_place_of_the_word() -> None:
    sentence = "Socio, ho detto: accènditi. Chiaro?"
    assert stress_ab.heard_as("accenditi", sentence, "Socio, ho detto accendete. Chiaro?") == (
        "accendete"
    )
    assert stress_ab.heard_as("accenditi", sentence, "Socio ho detto accenditi chiaro") == (
        "accenditi"
    )
    assert stress_ab.heard_as("accenditi", sentence, "Socio, ho detto. Chiaro?") == "(niente)"


def test_scores_count_every_target_with_every_seed() -> None:
    jobs = [
        {"arm": "scritta", "targets": ["a", "b"], "heard": {7: {"a": "a", "b": "x"}}},
        {"arm": "uguale", "targets": ["c"], "heard": {7: {"c": "c"}}},
    ]
    assert stress_ab.score(jobs, [7], "scritta") == (2, 3)
    assert stress_ab.score(jobs, [7], "accentata") == (1, 1)
