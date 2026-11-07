"""promises_action: a reply that says it will act (or did) in the first person."""

from __future__ import annotations

import pytest

from claudio.brain.promises import promises_action


@pytest.mark.parametrize(
    "reply",
    [
        "Ecco, dovrei aprire la cartella AMD, socio.",
        "Ora scrivo il piano.",
        "Lo sto scrivendo.",
        "Ho aperto il Blocco note.",
        "Apro Chrome.",
        "Lo scrivo subito.",
        "Vado ad aprire Spotify.",
        "Vado a cercare il file.",
        "Aperta la cartella. Ora chiudo Chrome.",
        "Devo salvarlo prima.",
        "L'ho aperta.",
        "L'ho chiusa, socio.",
        "L'ho salvata.",
        "Le ho copiate.",
        "Li ho spostati.",
        "Ho già aperto il file.",
        "L'ho appena scritto.",
        "Ho anche salvato.",
        "Vado ad aprirtelo.",
        "Devo scrivertelo.",
        "Sto aprendola.",
    ],
)
def test_promises_action_true(reply: str) -> None:
    assert promises_action(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "Non posso aprire quella cartella.",
        "Vuoi che lo scriva?",
        "La luna dista 384 mila chilometri.",
        "Aprire una finestra fa bene.",
        "Cosa apro?",
        "Cerco di capire.",
        "Non l'ho aperto.",
        "Posso chiudere Chrome?",
        "Cosa apro?!",
        "",
    ],
)
def test_promises_action_false(reply: str) -> None:
    assert not promises_action(reply)


def test_a_story_in_the_first_person_is_not_told_apart() -> None:
    # shortcut: patterns cannot tell "Ho aperto la porta del castello" from a PC action;
    # the feature only logs, the coordinator measures the false positives.
    assert promises_action("Ho aperto la porta del castello.")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Ho aperto il Blocco note, socio.", False),  # the closing sentence after a tool
        ("Aperto, socio. Ora scrivo il piano.", True),  # a new promise after a tool
        ("Dovrei chiudere Chrome.", True),
    ],
)
def test_after_a_tool_ran_past_forms_do_not_count(reply: str, expected: bool) -> None:
    assert promises_action(reply, ran=True) is expected
