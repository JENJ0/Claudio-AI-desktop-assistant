"""Key combinations as Claude writes them ("ctrl+shift+t", "page_down") and as pywinauto's
`send_keys` wants them ("^+t", "{PGDN}"). Pure functions: tested on any OS."""

from __future__ import annotations

import re

MODIFIERS = {"ctrl": "^", "control": "^", "alt": "%", "shift": "+", "win": "win"}
MODIFIER_ORDER = ("ctrl", "alt", "shift", "win")
# Canonical name -> send_keys code. Letters and digits are sent as themselves.
NAMED = {
    "enter": "{ENTER}",
    "tab": "{TAB}",
    "escape": "{ESC}",
    "space": "{SPACE}",
    "backspace": "{BACKSPACE}",
    "delete": "{DELETE}",
    "insert": "{INSERT}",
    "up": "{UP}",
    "down": "{DOWN}",
    "left": "{LEFT}",
    "right": "{RIGHT}",
    "home": "{HOME}",
    "end": "{END}",
    "page_up": "{PGUP}",
    "page_down": "{PGDN}",
    "plus": "{+}",
    "minus": "-",
    "print_screen": "{PRTSC}",
    **{f"f{n}": f"{{F{n}}}" for n in range(1, 13)},
}
ALIASES = {
    "esc": "escape",
    "return": "enter",
    "invio": "enter",
    "del": "delete",
    "canc": "delete",
    "pgup": "page_up",
    "pageup": "page_up",
    "pgdn": "page_down",
    "pagedown": "page_down",
    "pag_su": "page_up",
    "pag_giu": "page_down",
    "spazio": "space",
    "su": "up",
    "giu": "down",
    "sinistra": "left",
    "destra": "right",
    "+": "plus",
    "-": "minus",
    "windows": "win",
    "cmd": "win",
    "control": "ctrl",
    "maiusc": "shift",
}
_SPECIAL = set("+^%~(){}[]")  # send_keys syntax: typed literally only inside braces
MAX_COMBOS = 10


class BadKey(ValueError):
    """A key name Claudio does not know: nothing is pressed."""


def canonical(combo: str) -> str:
    """ "Shift+Ctrl+T" -> "ctrl+shift+t"; "ctrl++" -> "ctrl+plus". Raises BadKey."""
    raw = re.sub(r"\s*\+\s*", "+", combo.strip().lower()).replace(" ", "_")
    if not raw:
        raise BadKey("empty key")
    if raw.endswith("++"):  # "ctrl++": the plus key itself
        parts = [*raw[:-2].split("+"), "plus"] if raw != "++" else ["plus"]
    else:
        parts = raw.split("+") if raw != "+" else ["plus"]
    parts = [ALIASES.get(p, p) for p in parts if p]
    mods = [p for p in parts if p in MODIFIERS]
    keys = [p for p in parts if p not in MODIFIERS]
    if len(keys) > 1:
        raise BadKey(f"one key per combination: {combo!r}")
    for key in keys:
        if key not in NAMED and not (len(key) == 1 and key.isalnum() and key.isascii()):
            raise BadKey(f"unknown key: {key!r}")
    if not keys and mods != ["win"]:  # a lone modifier does nothing (except the Start menu)
        raise BadKey(f"a modifier needs a key: {combo!r}")
    ordered = [m for m in MODIFIER_ORDER if m in mods]
    return "+".join(ordered + keys)


def to_send_keys(combo: str) -> str:
    """A canonical combination in pywinauto's `send_keys` syntax."""
    parts = combo.split("+")
    key = parts[-1] if parts[-1] not in MODIFIERS else ""
    mods = [p for p in parts if p in MODIFIERS]
    code = NAMED.get(key, key)
    prefix = "".join(MODIFIERS[m] for m in mods if m != "win")
    sent = prefix + code
    if "win" in mods:
        sent = "{VK_LWIN down}" + sent + "{VK_LWIN up}"
    return sent


def parse_combos(keys: list[str]) -> list[str]:
    """Claude's list of combinations, canonical. Raises BadKey on a bad or too long list."""
    if not keys:
        raise BadKey("no keys")
    if len(keys) > MAX_COMBOS:
        raise BadKey(f"at most {MAX_COMBOS} combinations")
    return [canonical(k) for k in keys]


def escape_text(text: str) -> str:
    """Text typed as-is: send_keys' special characters wrapped in braces, new lines as Enter."""
    out = []
    for ch in text.replace("\r\n", "\n"):
        if ch in _SPECIAL:
            out.append("{" + ch + "}")
        elif ch == "\n":
            out.append("{ENTER}")
        elif ch == "\t":
            out.append("{TAB}")
        else:
            out.append(ch)
    return "".join(out)


def spoken(combo: str, language: str) -> str:
    """ "ctrl+shift+t" -> "control maiuscolo T": how the confirmation question says it."""
    words = {
        "it": {"ctrl": "control", "shift": "maiuscolo", "win": "Windows", "plus": "più"},
        "en": {"ctrl": "control", "shift": "shift", "win": "Windows", "plus": "plus"},
    }.get(language, {})
    out = []
    for part in combo.split("+"):
        word = words.get(part, part.replace("_", " "))
        out.append(word.upper() if len(word) == 1 else word)
    return " ".join(out)
