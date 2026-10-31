import sys
from pathlib import Path

import pytest

from claudio.config import PROJECT_ROOT
from claudio.tts.loanwords import load_proposals
from claudio.tts.stress_it import KEEP, StressLexicon

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_lexicon_it as build_it

IMPERATIVE = ["imperative", "second-person", "singular"]
PARTICIPLE = ["participle", "past"]
# A few entries shaped like kaikki.org's, each one a case of the build.
ENTRIES = [
    {
        "word": "accendere",
        "pos": "verb",
        "hyphenations": [{"parts": ["ac‧cèn‧de‧re"]}],
        "sounds": [{"ipa": "/atˈt͡ʃɛn.de.re/"}],
        "forms": [
            {"form": "accèndere", "tags": ["infinitive"], "source": "conjugation"},
            {"form": "accèndi", "tags": IMPERATIVE, "source": "conjugation"},
            {"form": "accèndono", "tags": ["present", "plural", "third-person"]},
            {"form": "accendèndo", "tags": ["gerund"], "source": "conjugation"},
            {"form": "accèndere", "tags": ["imperative", "negative", "second-person"]},
            {"form": "avére", "tags": ["auxiliary"]},
        ],
        "senses": [{"glosses": ["to light"]}],
    },
    {"word": "ancora", "pos": "adv", "hyphenations": [{"parts": ["an‧có‧ra"]}], "senses": [{}]},
    {
        "word": "ancora",
        "pos": "noun",
        "hyphenations": [{"parts": ["àn‧co‧ra"]}],
        "forms": [{"form": "ancore", "tags": ["plural"]}],
        "senses": [{}],
    },
    {
        "word": "tavolo",
        "pos": "noun",
        "hyphenations": [{"parts": ["tà", "vo", "lo"]}],  # syllables as separate parts
        "forms": [{"form": "tavoli", "tags": ["plural"]}],
        "senses": [{}],
    },
    {"word": "numerare", "pos": "verb", "forms": [{"form": "numerò", "tags": ["past"]}]},
    {"word": "numero", "pos": "noun", "hyphenations": [{"parts": ["nù‧me‧ro"]}], "senses": [{}]},
    {
        "word": "fermare",
        "pos": "verb",
        "forms": [{"form": "férma", "tags": IMPERATIVE}, {"form": "fermàto", "tags": PARTICIPLE}],
    },
    {
        "word": "fermati",  # Wiktionary lists the imperative: accented where a clause starts
        "pos": "verb",
        "hyphenations": [{"parts": ["fér‧ma‧ti"]}],
        "senses": [{"tags": ["form-of", *IMPERATIVE], "form_of": [{"word": "fermarsi"}]}],
    },
    {
        "word": "risultare",
        "pos": "verb",
        "forms": [
            {"form": "risùlta", "tags": IMPERATIVE},
            {"form": "risultàto", "tags": PARTICIPLE},
        ],
    },
    {
        "word": "risultati",  # a noun that splits as "risulta" + "ti": no imperative listed
        "pos": "noun",
        "hyphenations": [{"parts": ["ri‧sul‧tà‧ti"]}],
        "senses": [{"tags": ["form-of", "plural"], "form_of": [{"word": "risultato"}]}],
    },
    {
        "word": "capitare",
        "pos": "verb",
        "forms": [{"form": "càpiti", "tags": ["present", "second-person", "singular"]}],
    },
    {"word": "capire", "pos": "verb", "forms": [{"form": "capìto", "tags": PARTICIPLE}]},
    {"word": "aprile", "pos": "noun", "hyphenations": [{"parts": ["a‧prì‧le"]}], "senses": [{}]},
    {"word": "aprire", "pos": "verb", "forms": [{"form": "àpri", "tags": IMPERATIVE}]},
    {
        "word": "famoso",
        "pos": "adj",
        "hyphenations": [{"parts": ["fa‧mó‧so"]}],
        "forms": [
            {"form": "famosa", "tags": ["feminine"]},
            {"form": "famosissimo", "tags": ["superlative"]},  # famosìssimo, not famósissimo
        ],
        "senses": [{}],
    },
    {"word": "bene", "pos": "adv", "hyphenations": [{"parts": ["bè‧ne"]}], "senses": [{}]},
    {
        "word": "benone",
        "pos": "adv",
        "senses": [{"tags": ["augmentative", "form-of"], "form_of": [{"word": "bene"}]}],
    },
    {
        "word": "machina",  # archaic: it lends "màchine" to the English "machine" no more
        "pos": "noun",
        "hyphenations": [{"parts": ["mà‧chi‧na"]}],
        "forms": [{"form": "machine", "tags": ["plural"]}],
        "senses": [{"tags": ["alt-of", "archaic", "dialectal"]}],
    },
    {
        "word": "utensile",  # two stresses, the second behind a label
        "pos": "noun",
        "hyphenations": [
            {"parts": ["u", "tèn", "si", "le"]},
            {"parts": ["(traditional"]},
            {"parts": ["careful style) u", "ten", "sì", "le"]},
        ],
        "forms": [{"form": "utensili", "tags": ["plural"]}],
        "senses": [{}],
    },
    {"word": "tredici", "pos": "num", "hyphenations": [{"parts": ["tré‧di‧ci"]}], "senses": [{}]},
    {
        "word": "statua",
        "pos": "noun",
        "hyphenations": [{"parts": ["stà‧tu‧a"]}],
        "sounds": [{"ipa": "/ˈsta.tu.a/"}],
        "senses": [{}],
    },
    {
        "word": "euro",
        "pos": "noun",
        "hyphenations": [{"parts": ["è‧u‧ro"]}],
        "sounds": [{"ipa": "/ˈɛ.u.ro/"}],
        "senses": [{}],
    },
    {
        "word": "manager",
        "pos": "noun",
        "hyphenations": [{"parts": ["mà‧na‧ger"]}],
        "sounds": [{"ipa": "/ˈma.na.d͡ʒer/"}],
        "etymology_templates": [{"name": "bor+", "args": {"1": "it", "2": "en"}}],
        "senses": [{}],
    },
]


@pytest.fixture(scope="module")
def collector() -> build_it.Collector:
    collector = build_it.Collector()
    for entry in ENTRIES:
        collector.add_entry(entry)
    collector.resolve_pending()
    return collector


@pytest.fixture(scope="module")
def built(collector: build_it.Collector) -> build_it.Lexicon:
    return build_it.build(collector)


def test_sdrucciole_from_syllables_and_conjugation_tables(built: build_it.Lexicon) -> None:
    assert built.words["accendere"] == "accèndere"
    assert built.words["accendono"] == "accèndono"
    assert "accendi" not in built.words  # accèn-di: second-to-last, left as written
    assert built.words["numero"] == "nùmero"  # "numerò" is written with its accent
    assert built.words["tavoli"] == "tàvoli"  # a plural keeps its lemma's stress


def test_verb_forms_that_take_pronouns(built: build_it.Lexicon) -> None:
    assert built.hosts["accender"] == "accènder"
    assert built.hosts["accendi"] == "accèndi"
    assert built.hosts["accendendo"] == "accendèndo"
    assert built.hosts["apri"] == "àpri"


def test_homographs_and_lookalikes_are_left_as_written(built: build_it.Lexicon) -> None:
    assert "ancora" not in built.words  # àncora / ancóra
    assert "ancore" not in built.words  # a form takes every stress of its lemma
    assert built.words["aprile"] == KEEP  # not "àpri" + "le"
    assert "capiti" not in built.words  # càpiti (capitare) / capìti (participle of capire)
    assert "utensili" not in built.words  # utènsile / utensìle (traditional)


def test_an_imperative_and_participle_is_accented_only_where_a_clause_starts(
    built: build_it.Lexicon,
) -> None:
    assert built.at_start == {"fermati": "férmati"}  # also the participle "fermàti"
    assert built.words["risultati"] == KEEP  # a noun: Wiktionary lists no "risulta + ti"


def test_derived_words_keep_no_wrong_stress(built: build_it.Lexicon) -> None:
    assert "famosissimo" not in built.words  # a superlative: not the lemma's "famóso"
    assert "benone" not in built.words  # an augmentative: not "bène"
    assert "machine" not in built.words  # from an archaic entry only
    assert built.words["tredici"] == KEEP  # number words stay as written


def test_the_ipa_decides_the_syllables(built: build_it.Lexicon) -> None:
    assert built.words["statua"] == "stàtua"  # stà-tu-a
    assert "euro" not in built.words  # èu-ro: Wiktionary's /ˈɛ.u.ro/ splits a diphthong


def test_syllables_without_their_labels() -> None:
    assert build_it.syllables_of(["ac‧cèn‧di‧ti"]) == ["ac", "cèn", "di", "ti"]
    assert build_it.syllables_of(["u", "tèn", "si", "le"]) == ["u", "tèn", "si", "le"]
    assert build_it.syllables_of(["careful style) u", "ten", "sì", "le"]) == [
        "u",
        "ten",
        "sì",
        "le",
    ]
    assert build_it.syllables_of(["(traditional"]) == []


def test_the_rule_is_checked_on_syllables_given_as_parts(collector: build_it.Collector) -> None:
    assert collector.checked == 16  # every syllabification, tavolo's parts too
    assert collector.mismatches == ["è‧u‧ro"]  # Wiktionary splits the diphthong


def test_english_loanwords_get_a_proposal_not_an_accent(
    built: build_it.Lexicon, collector: build_it.Collector
) -> None:
    assert "manager" not in built.words
    assert collector.proposals == {"manager": "mànager"}


def test_the_written_files_load_back(
    built: build_it.Lexicon, collector: build_it.Collector, tmp_path: Path
) -> None:
    path = tmp_path / "stress_it.tsv"
    build_it.write(path, built, Path("kaikki.jsonl.gz"))
    lexicon = StressLexicon.load(path)
    assert lexicon.stressed("accenditi") == "accènditi"
    assert lexicon.stressed("Fermati", starts_clause=True) == "Férmati"
    assert "CC BY-SA 4.0" in path.read_text(encoding="utf-8")
    proposals = tmp_path / "respell_en.tsv"
    build_it.write_proposals(proposals, collector.proposals, Path("kaikki.jsonl.gz"))
    assert load_proposals(proposals) == {"manager": "mànager"}


@pytest.mark.parametrize(
    ("ipa", "respelled"),
    [
        ("/ˈpas.sword/", "pàssuord"),
        ("/komˈpju.ter/", "compiùter"),
        ("/ˈt͡ʃat/", "ciàt"),
        ("/wiˈkɛnd/", "uichènd"),
        ("/ˈmi.tinɡ/", "mìting"),
        ("/lɔˈɡin/", "loghìn"),
        ("/ˈbraw.zer/", "bràuser"),
        ("/ˈfa.il/", "fàil"),
        ("/ˈsɔf.twɛr/", "sòftuer"),
        ("/ˈkat/", "càt"),
        ("/aɲˈɲɔ.sti.ko/", "agnòstico"),  # a long ɲ is one "gn"
        ("/ˈpuʃ.ʃer/", "pùscer"),
        ("/e.kwa.lidˈd͡za.re/", "ecualizzàre"),  # "d.d͡z" is a long z
        ("/ˈsɛn.dwit͡ʃ/", None),  # Italian letters cannot end on ʧ
        ("/kat/", None),  # no stress mark
        ("/ˈɡud ˈbaj/", None),  # two words
        ("/ˈq/", None),  # a sound it cannot write
    ],
)
def test_ipa_respelled_in_italian_letters(ipa: str, respelled: str | None) -> None:
    assert build_it.respell_ipa(ipa) == respelled


@pytest.mark.parametrize(
    ("ipa", "after"),
    [("/atˈt͡ʃɛn.di.ti/", 2), ("/ˈnum.meˌro/", 2), ("/ˈso.t͡ʃo/", 1), ("/ˈmaus/", None)],
)
def test_ipa_syllables_after_the_stress(ipa: str, after: int | None) -> None:
    assert build_it.ipa_syllables_after(ipa) == after
