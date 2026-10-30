"""Long-term memory (Phase 4): the store, the local search, the spoken requests, the block
Claude sees and the tools Claude may use."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claudio.memory.prompt import FACTS, HEADER, MORE, RULES, MemoryPrompt
from claudio.memory.relevance import close, rank, stems
from claudio.memory.store import MemoryRefused, MemoryStore, sensitive
from claudio.memory.tools import FORGET, REMEMBER, MemoryToolError, MemoryTools
from claudio.memory.voice import CHOICE_TTL_S, Intent, MemoryVoice, parse, without_wake_words
from claudio.safety.confirmation import LEVEL_CONFIRM, LEVEL_SAFE

DAY = 1_793_000_000.0  # 2026-10-26, a fixed "now" for the stored memories


class Clock:
    def __init__(self, t: float = DAY) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db", clock=Clock())


def remember(store: MemoryStore, *texts: str) -> None:
    for text in texts:
        store.add(text)


# --- the store ----------------------------------------------------------------------------


def test_memories_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "data" / "memory.db"
    first = MemoryStore(path, clock=Clock())
    first.add("mia sorella si chiama Anna.")
    first.add("chiamami capo", "rule")
    again = MemoryStore(path)
    assert [(m.text, m.kind, m.source) for m in again.items()] == [
        ("mia sorella si chiama Anna", "fact", "jenjo"),
        ("chiamami capo", "rule", "jenjo"),
    ]
    assert again.items()[0].day == "2026-10-26"


def test_a_known_memory_is_not_stored_twice(store: MemoryStore) -> None:
    first, new = store.add("Mia sorella si chiama Anna")
    again, new_again = store.add("mia sorella si chiama anna!")
    assert new and not new_again and again == first and len(store) == 1


@pytest.mark.parametrize(
    "text",
    [
        "la password del wifi è gattino99",
        "il PIN del bancomat è 1234",
        "il mio IBAN è IT60 X054 2811 1010 0000 0123 456",
        "la carta 4111 1111 1111 1111",
        "il codice dell'allarme è 7890",
        "my api key is abc",
    ],
)
def test_passwords_and_codes_are_never_kept(store: MemoryStore, text: str) -> None:
    assert sensitive(text)
    with pytest.raises(MemoryRefused) as refused:
        store.add(text)
    assert refused.value.reason == "sensitive" and len(store) == 0


@pytest.mark.parametrize(
    "text", ["il numero di Marco è 333 123 4567", "ho 2 gatti", "abito al numero 12"]
)
def test_ordinary_numbers_are_kept(text: str) -> None:
    assert not sensitive(text)


def test_empty_too_long_and_full_are_refused(tmp_path: Path) -> None:
    small = MemoryStore(tmp_path / "m.db", max_items=1, max_chars=20)
    for text, reason in (("  . ", "empty"), ("x" * 21, "too_long")):
        with pytest.raises(MemoryRefused) as refused:
            small.add(text)
        assert refused.value.reason == reason
    small.add("uno")
    with pytest.raises(MemoryRefused) as refused:
        small.add("due")
    assert refused.value.reason == "full"


def test_remove_clear_last_added_and_version(store: MemoryStore) -> None:
    remember(store, "a uno", "b due", "c tre")
    assert store.version == 3
    last = store.last_added()
    assert last is not None and last.text == "c tre"
    assert [m.text for m in store.remove([last.id, 999])] == ["c tre"]
    assert getattr(store.last_added(), "text", None) == "b due"
    assert store.remove([999]) == [] and store.version == 4  # nothing removed: no change
    assert store.clear() == 2 and len(store) == 0 and store.last_added() is None
    assert store.version == 5


def test_last_added_is_only_this_run(tmp_path: Path) -> None:
    MemoryStore(tmp_path / "m.db").add("di ieri")
    assert MemoryStore(tmp_path / "m.db").last_added() is None  # "dimenticalo" never reaches back


def test_a_wrong_database_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    path.write_bytes(b"not a database at all" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        MemoryStore(path)


# --- the local search -----------------------------------------------------------------------


def test_stems_keep_the_content_words() -> None:
    assert stems("Mia sorella si chiama Anna") == {"sorell", "chiam", "ann"}
    assert stems("il mio gatto ha 3 anni") == {"gatt", "3", "ann"}
    assert stems("che cosa è il") == frozenset()


def test_close_words() -> None:
    assert close("sorell", "sorell")
    assert close("compleann", "compleanno")  # one is the start of the other
    assert close("sorel", "sorell")  # Whisper's "sorela"
    assert not close("gatt", "cane")
    assert not close("ann", "anni")  # too short to be a prefix match


def test_rank_puts_the_closest_first(store: MemoryStore) -> None:
    remember(store, "mia sorella abita a Torino", "mia sorella si chiama Anna", "il gatto è nero")
    found = rank("come si chiama mia sorella", store.items())
    assert [s.memory.text for s in found] == [
        "mia sorella si chiama Anna",
        "mia sorella abita a Torino",
    ]
    assert found[0].hits == 2 and found[0].coverage == 1.0
    assert rank("che ore sono", store.items()) == []


# --- what JenJo says ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("said", "action", "text"),
    [
        ("Claudio, ricordati che mia sorella è Anna.", "remember", "mia sorella è Anna."),
        ("Ehi Claudio ricordati bene che il wifi è lento, grazie", "remember", "il wifi è lento"),
        ("puoi ricordarti che il gatto si chiama Leo?", "remember", "il gatto si chiama Leo?"),
        ("Non dimenticare che giovedì ho il dentista", "remember", "giovedì ho il dentista"),
        (
            "Segnati che la cartella Tesi è sul desktop",
            "remember",
            "la cartella Tesi è sul desktop",
        ),
        ("Ricordati che da ora in poi mi chiami capo.", "rule", "mi chiami capo."),
        ("D'ora in poi chiamami capo", "rule", "chiamami capo"),
        ("Da ora in poi rispondimi in modo breve.", "rule", "rispondimi in modo breve."),
        ("ricordati sempre di salutarmi", "rule", "salutarmi"),
        ("Dimentica che mia sorella si chiama Anna.", "forget", "mia sorella si chiama Anna."),
        ("Dimenticati di chiamarmi capo", "forget", "chiamarmi capo"),
        ("Cosa ricordi di mia sorella?", "recall", "mia sorella?"),
    ],
)
def test_memory_requests_are_understood(said: str, action: str, text: str) -> None:
    assert parse(said) == Intent(action, text, "it")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("said", "action", "text"),
    [
        ("Remember that my sister is Anna.", "remember", "my sister is Anna."),
        ("From now on call me boss.", "rule", "call me boss."),
        ("Forget that my sister is Anna", "forget", "my sister is Anna"),
        ("What do you remember about my sister?", "recall", "my sister?"),
    ],
)
def test_english_requests_are_understood(said: str, action: str, text: str) -> None:
    assert parse(said) == Intent(action, text, "en")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "said",
    [
        "ricordati di comprare il latte",  # a reminder: Claude says it cannot set them yet
        "Ricordami di chiamare la mamma",
        "ti ricordi che ieri ti ho detto una cosa?",  # a question
        "cosa sai di Napoleone",  # general knowledge
        "dimentica tutto",  # a command (commands.toml), with its "confermo"
        "Dimentica.",
        "forget it",
        "apri chrome",
        "la pausa pranzo è alle 13",
    ],
)
def test_other_phrases_are_not_memory_requests(said: str) -> None:
    assert parse(said) is None


def test_the_wake_word_is_cut_and_the_spelling_kept() -> None:
    assert without_wake_words("Ehi Claudio, Ricordati che È così") == "Ricordati che È così"
    assert without_wake_words("Claudio") == ""
    assert without_wake_words("ricordati che") == "ricordati che"


# --- the spoken replies ---------------------------------------------------------------------


@pytest.fixture
def voice(store: MemoryStore) -> MemoryVoice:
    return MemoryVoice(store, clock=Clock(0.0))


def reply(voice: MemoryVoice, said: str) -> str:
    answer = voice.handle(said)
    assert answer is not None, said
    return answer.text


def test_remember_reads_it_back(voice: MemoryVoice) -> None:
    assert reply(voice, "Claudio, ricordati che mia sorella si chiama Anna.") == (
        "Me lo ricordo: mia sorella si chiama Anna."
    )
    assert reply(voice, "ricordati che mia sorella si chiama Anna") == "Lo sapevo già."
    assert reply(voice, "da ora in poi chiamami capo") == "D'accordo, da ora in poi: chiamami capo."
    assert [m.kind for m in voice.store.items()] == ["fact", "rule"]
    long = "ricordati che " + "la casa al mare ha le chiavi nel vaso blu " * 3
    assert reply(voice, long) == "Fatto, me lo ricordo."
    assert reply(voice, "remember that my cat is Leo") == "I'll remember: my cat is Leo."


def test_remember_refuses_a_password(voice: MemoryVoice) -> None:
    assert reply(voice, "ricordati che la password del wifi è gatto").startswith(
        "Questo non lo salvo"
    )
    assert len(voice.store) == 0


def test_forget_the_clear_one(voice: MemoryVoice) -> None:
    remember(voice.store, "mia sorella si chiama Anna", "mia sorella abita a Torino")
    assert reply(voice, "dimentica che mia sorella abita a Torino") == (
        "Dimenticato: mia sorella abita a Torino."
    )
    assert [m.text for m in voice.store.items()] == ["mia sorella si chiama Anna"]
    assert reply(voice, "dimentica il colore preferito") == "Non ricordo niente del genere."


def test_forget_asks_which_and_takes_the_answer(voice: MemoryVoice) -> None:
    remember(voice.store, "mia sorella si chiama Anna", "mia sorella abita a Torino")
    assert reply(voice, "dimentica mia sorella") == (
        "Ne ricordo 2: mia sorella si chiama Anna; mia sorella abita a Torino. Quale dimentico?"
    )
    assert voice.awaiting
    assert reply(voice, "il secondo") == "Dimenticato: mia sorella abita a Torino."
    assert not voice.awaiting and len(voice.store) == 1


@pytest.mark.parametrize(
    ("answer", "left"),
    [
        ("quella di Torino", ["mia sorella si chiama Anna"]),
        ("tutti e due", []),
        ("nessuno", ["mia sorella si chiama Anna", "mia sorella abita a Torino"]),
    ],
)
def test_the_answer_to_which(voice: MemoryVoice, answer: str, left: list[str]) -> None:
    remember(voice.store, "mia sorella si chiama Anna", "mia sorella abita a Torino")
    reply(voice, "dimentica mia sorella")
    reply(voice, answer)
    assert [m.text for m in voice.store.items()] == left


def test_another_request_drops_the_question(voice: MemoryVoice) -> None:
    remember(voice.store, "mia sorella si chiama Anna", "mia sorella abita a Torino")
    reply(voice, "dimentica mia sorella")
    assert voice.handle("che ore sono") is None  # the router answers it
    assert not voice.awaiting
    assert voice.handle("il primo") is None  # too late: not an answer any more
    assert len(voice.store) == 2


def test_the_question_expires(store: MemoryStore) -> None:
    clock = Clock(0.0)
    voice = MemoryVoice(store, clock=clock)
    remember(store, "mia sorella si chiama Anna", "mia sorella abita a Torino")
    reply(voice, "dimentica mia sorella")
    clock.t = CHOICE_TTL_S + 1
    assert not voice.awaiting
    assert voice.handle("il primo") is None and len(store) == 2


def test_recall_reads_what_matches(voice: MemoryVoice) -> None:
    remember(voice.store, "mia sorella si chiama Anna", "il gatto è nero")
    assert reply(voice, "cosa ricordi di mia sorella?") == (
        "Mi hai detto: mia sorella si chiama Anna."
    )
    assert voice.handle("cosa ricordi di Napoleone") is None  # Claude answers


def test_the_fixed_commands(voice: MemoryVoice) -> None:
    assert voice.run("list") == "Non ricordo ancora niente. Dimmi: ricordati che, e poi la cosa."
    assert voice.run("forget_last") == "Non ho niente di nuovo da dimenticare."
    remember(voice.store, "uno", "due", "tre", "quattro")
    assert voice.run("list") == "Ricordo 4 cose. Le ultime: quattro; tre; due."
    assert voice.run("forget_last") == "Dimenticato: quattro."
    assert voice.run("wipe") == "Fatto: ho dimenticato tutto."
    assert len(voice.store) == 0
    voice.store.add("solo")
    assert voice.run("list", "en") == "I remember one thing: solo."
    with pytest.raises(ValueError):
        voice.run("nonsense")


def test_rules_are_read_with_their_da_ora_in_poi(voice: MemoryVoice) -> None:
    voice.store.add("chiamami capo", "rule")
    assert voice.run("list") == "Ricordo una cosa: da ora in poi chiamami capo."
    assert reply(voice, "dimenticati di chiamarmi capo") == (
        "Dimenticato: da ora in poi chiamami capo."
    )


# --- what Claude sees -----------------------------------------------------------------------


def test_no_memories_no_block(store: MemoryStore) -> None:
    prompt = MemoryPrompt(store)
    assert prompt.block() == "" and prompt.relevant("mia sorella") == ""


def test_the_block_lists_rules_and_facts(store: MemoryStore) -> None:
    store.add("mia sorella si chiama Anna")
    store.add("chiamami capo", "rule")
    block = MemoryPrompt(store).block()
    assert block == "\n".join(
        [
            HEADER,
            RULES,
            "[2 | 2026-10-26] chiamami capo",
            FACTS,
            "[1 | 2026-10-26] mia sorella si chiama Anna",
        ]
    )


def test_the_block_is_rebuilt_only_after_a_change(store: MemoryStore) -> None:
    prompt = MemoryPrompt(store)
    store.add("uno")
    first = prompt.block()
    assert prompt.block() is first  # same bytes: the cache reads it back
    store.add("due")
    assert prompt.block() != first and "due" in prompt.block()


def test_facts_that_do_not_fit_come_with_a_matching_request(store: MemoryStore) -> None:
    remember(store, "il wifi del garage si chiama Tana", "mia sorella si chiama Anna")
    store.add("il gatto è nero")
    room = len("[3 | 2026-10-26] il gatto è nero") + 1
    prompt = MemoryPrompt(store, block_chars=room, relevant=3)
    block = prompt.block()
    assert "il gatto è nero" in block and "Anna" not in block and block.endswith(MORE)
    assert prompt.relevant("qual è la password del wifi del garage") == (
        "[memory: [1 | 2026-10-26] il wifi del garage si chiama Tana]"
    )
    assert prompt.relevant("che ore sono") == ""
    assert prompt.relevant("il gatto") == ""  # already in the block


def test_only_relevant_mode_keeps_the_rules(store: MemoryStore) -> None:
    store.add("chiamami capo", "rule")
    store.add("mia sorella si chiama Anna")
    prompt = MemoryPrompt(store, block_chars=0)
    assert "chiamami capo" in prompt.block() and "Anna" not in prompt.block()
    assert "Anna" in prompt.relevant("come si chiama mia sorella")


# --- Claude's tools -------------------------------------------------------------------------


def test_tool_definitions(store: MemoryStore) -> None:
    plain, learning = MemoryTools(store), MemoryTools(store, learn=True)
    assert [d["name"] for d in plain.definitions] == [REMEMBER, FORGET]
    for d in plain.definitions:
        assert d["input_schema"]["additionalProperties"] is False
    assert "without being asked" not in plain.definitions[0]["description"]
    assert "without being asked" in learning.definitions[0]["description"]


def test_tool_levels(store: MemoryStore) -> None:
    tools = MemoryTools(store)
    assert tools.level(REMEMBER, {"text": "x"}) == LEVEL_SAFE
    assert tools.level(REMEMBER, {"text": "x", "kind": "rule"}) == LEVEL_CONFIRM
    assert tools.level(FORGET, {"ids": [1]}) == LEVEL_CONFIRM


def test_claude_remembers_and_forgets(store: MemoryStore) -> None:
    tools = MemoryTools(store)
    assert tools.run(REMEMBER, {"text": "JenJo tifa Napoli"}).startswith("Saved as [1]")
    assert store.items()[0].source == "claude"
    assert tools.run(REMEMBER, {"text": "JenJo tifa Napoli"}) == "Already in your memory as [1]."
    assert tools.describe(FORGET, {"ids": [1]}, "it") == 'dimentico "JenJo tifa Napoli"'
    assert tools.run(FORGET, {"ids": [1, 7]}) == "Forgotten: [1] JenJo tifa Napoli"
    with pytest.raises(MemoryToolError):
        tools.run(FORGET, {"ids": [1]})
    with pytest.raises(MemoryToolError, match="never kept"):
        tools.run(REMEMBER, {"text": "il PIN è 1234"})
    with pytest.raises(MemoryToolError):
        tools.run("nonsense", {})


def test_a_rule_from_claude_is_read_out_before_the_yes(store: MemoryStore) -> None:
    tools = MemoryTools(store)
    args = {"text": "chiamami capo", "kind": "rule"}
    assert tools.describe(REMEMBER, args, "it") == 'da ora in poi "chiamami capo"'
    tools.run(REMEMBER, args)
    assert store.items()[0].kind == "rule"


def test_forget_ids_are_checked(store: MemoryStore) -> None:
    tools = MemoryTools(store)
    store.add("uno")
    with pytest.raises(MemoryToolError):
        tools.run(FORGET, {"ids": "1"})  # not a list: nothing is forgotten
    with pytest.raises(MemoryToolError):
        tools.run(FORGET, {"ids": [True]})
    assert len(store) == 1
