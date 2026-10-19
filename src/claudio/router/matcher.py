"""Match a transcription to a local command: exact first, then fuzzy (rapidfuzz)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from claudio.router.normalizer import extract_number, normalize


@dataclass(frozen=True)
class Command:
    id: str
    phrases: tuple[str, ...]
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    exact: bool = False  # only an exact phrase counts: "spegni" must not shut CLAUDIO down


@dataclass(frozen=True)
class Match:
    command: Command
    score: float
    number: int | None = None


def load_commands(path: Path) -> list[Command]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    commands = []
    seen: set[str] = set()
    for raw in data.get("command", []):
        cmd = Command(
            id=raw["id"],
            phrases=tuple(raw["phrases"]),
            action=raw["action"],
            params=dict(raw.get("params", {})),
            reply=raw.get("reply", ""),
            exact=bool(raw.get("exact", False)),
        )
        if cmd.id in seen:
            raise ValueError(f"{path}: comando duplicato '{cmd.id}'")
        if not cmd.phrases:
            raise ValueError(f"{path}: il comando '{cmd.id}' non ha frasi")
        seen.add(cmd.id)
        commands.append(cmd)
    return commands


class CommandMatcher:
    def __init__(self, commands: list[Command], threshold: float = 85) -> None:
        self.threshold = threshold
        self._by_phrase: dict[str, Command] = {}
        for cmd in commands:
            for phrase in cmd.phrases:
                norm = normalize(phrase)
                other = self._by_phrase.get(norm)
                if other and other.id != cmd.id:
                    raise ValueError(f"Frase '{phrase}' ambigua: {other.id} e {cmd.id}")
                self._by_phrase[norm] = cmd
        self._choices = [p for p, cmd in self._by_phrase.items() if not cmd.exact]

    def match(self, text: str) -> Match | None:
        norm, number = extract_number(normalize(text))
        if not norm:
            return None
        if norm in self._by_phrase:
            return Match(self._by_phrase[norm], 100.0, number)
        best = process.extractOne(
            norm, self._choices, scorer=fuzz.ratio, score_cutoff=self.threshold
        )
        if best is None:
            return None
        phrase, score, _ = best
        return Match(self._by_phrase[phrase], float(score), number)
