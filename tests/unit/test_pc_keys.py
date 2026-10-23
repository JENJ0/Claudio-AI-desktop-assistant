"""Key combinations from Claude to pywinauto (v0.3.1)."""

from __future__ import annotations

import pytest

from claudio.pc.keys import BadKey, canonical, escape_text, parse_combos, spoken, to_send_keys


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("ctrl+c", "ctrl+c"),
        ("Shift+Ctrl+T", "ctrl+shift+t"),
        ("PageDown", "page_down"),
        ("pgdn", "page_down"),
        ("ctrl++", "ctrl+plus"),
        ("ctrl+-", "ctrl+minus"),
        ("win", "win"),
        ("win+d", "win+d"),
        ("esc", "escape"),
        ("invio", "enter"),
        ("F5", "f5"),
        ("alt + tab", "alt+tab"),
    ],
)
def test_canonical(given: str, expected: str) -> None:
    assert canonical(given) == expected


@pytest.mark.parametrize("bad", ["", "ctrl", "ctrl+a+b", "hyper+x", "f13", "ctrl+é"])
def test_bad_keys_are_refused(bad: str) -> None:
    with pytest.raises(BadKey):
        canonical(bad)


@pytest.mark.parametrize(
    ("combo", "sent"),
    [
        ("ctrl+c", "^c"),
        ("ctrl+shift+t", "^+t"),
        ("alt+f4", "%{F4}"),
        ("page_down", "{PGDN}"),
        ("ctrl+plus", "^{+}"),
        ("win+d", "{VK_LWIN down}d{VK_LWIN up}"),
        ("win", "{VK_LWIN down}{VK_LWIN up}"),
        ("enter", "{ENTER}"),
    ],
)
def test_to_send_keys(combo: str, sent: str) -> None:
    assert to_send_keys(combo) == sent


def test_combo_list_limits() -> None:
    with pytest.raises(BadKey):
        parse_combos([])
    with pytest.raises(BadKey):
        parse_combos(["down"] * 11)
    assert parse_combos(["Ctrl+T", "down"]) == ["ctrl+t", "down"]


def test_text_is_typed_literally() -> None:
    """send_keys' syntax characters never act as keys: "50% (circa)" is typed as written."""
    assert escape_text("50% (circa) a+b ^_^ {x} ~") == (
        "50{%} {(}circa{)} a{+}b {^}_{^} {{}x{}} {~}"
    )
    assert escape_text("riga 1\nriga 2\tfine") == "riga 1{ENTER}riga 2{TAB}fine"
    assert escape_text("perché è così") == "perché è così"


def test_spoken() -> None:
    assert spoken("ctrl+shift+t", "it") == "control maiuscolo T"
    assert spoken("page_down", "en") == "page down"
