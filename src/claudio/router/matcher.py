"""Match a transcription to a local command: exact first, then fuzzy (rapidfuzz).

Each command has its phrases and reply in Italian (`phrases`, `reply`). `keywords` are words a
fuzzy match must contain ("che tempo fa" is close to "che giorno e" but has no "giorno").

Last, in Italian, the phrases are compared by how they sound (phonetic.py): Whisper's "Claudio
QDT" is "chiuditi" spelled as letters, "kiudi" is "chiudi"."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from claudio.config import DEFAULT_LANGUAGE
from claudio.router.normalizer import extract_number, last_clauses, normalize
from claudio.router.normalizer import language as language_rules
from claudio.router.phonetic import sound_key, spell_capitals

ARTICLES = language_rules(DEFAULT_LANGUAGE).articles


@dataclass(frozen=True)
class Command:
    id: str
    phrases: tuple[str, ...]
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    exact: bool = False  # only an exact phrase counts: "spegni" must not shut CLAUDIO down
    keywords: frozenset[str] | None = None  # a fuzzy match must contain one of them
    level: int = 1  # safety level: 1 runs at once, 2 asks yes/no, 3 asks an explicit "confermo"
    confirm: str = ""  # the question asked at level 2-3


@dataclass(frozen=True)
class Match:
    command: Command
    score: float
    number: int | None = None
    language: str = DEFAULT_LANGUAGE


@dataclass(frozen=True)
class Near:
    """A command a phrase comes close to without matching it: its closest phrase (as written
    in commands.toml) and how close, 0-100, by letters or by sound."""

    command: Command
    phrase: str
    score: float


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
            keywords=_keywords(raw["keywords"]) if "keywords" in raw else None,
            level=int(raw.get("level", 1)),
            confirm=raw.get("confirm", ""),
        )
        if cmd.level not in (1, 2, 3):
            raise ValueError(
                f"{path}: il comando '{cmd.id}' ha un livello non valido ({cmd.level})"
            )
        if cmd.id in seen:
            raise ValueError(f"{path}: comando duplicato '{cmd.id}'")
        if not cmd.phrases:
            raise ValueError(f"{path}: il comando '{cmd.id}' non ha frasi")
        seen.add(cmd.id)
        commands.append(cmd)
    return commands


def _keywords(words: list[str]) -> frozenset[str]:
    return frozenset(normalize(w, drop_fillers=False) for w in words)


class CommandMatcher:
    def __init__(self, commands: list[Command], threshold: float = 85) -> None:
        self.threshold = threshold
        self._by_phrase: dict[str, Command] = {}
        self._written: dict[str, str] = {}  # phrase -> as in the file
        for cmd in commands:
            for phrase in cmd.phrases:
                norm = normalize(phrase)
                self._written.setdefault(norm, phrase)
                other = self._by_phrase.get(norm)
                if other and other.id != cmd.id:
                    raise ValueError(f"Frase '{phrase}' ambigua: {other.id} e {cmd.id}")
                self._by_phrase[norm] = cmd
        self._choices = [p for p, cmd in self._by_phrase.items() if not cmd.exact]
        # How each phrase sounds -> its command. Two phrases sounding alike are one command
        # (they were written as variants of the same words).
        self._by_sound: dict[str, Command] = {}
        self._sound_phrase: dict[str, str] = {}  # sound -> phrase
        for phrase, cmd in self._by_phrase.items():
            self._by_sound.setdefault(sound_key(phrase), cmd)
            self._sound_phrase.setdefault(sound_key(phrase), phrase)
        # The same sounds with the spaces gone: Whisper splits a word it did not know
        # ("a cenditi", "spegni ti") or joins two.
        self._by_joined_sound = {
            k.replace(" ", ""): cmd for k, cmd in reversed(self._by_sound.items())
        }
        self._sound_choices = [k for k, cmd in self._by_sound.items() if not cmd.exact]

    def command(self, command_id: str) -> Command | None:
        """The command with this id; None when there is none."""
        for cmd in self._by_phrase.values():
            if cmd.id == command_id:
                return cmd
        return None

    def match(self, text: str, language: str = DEFAULT_LANGUAGE) -> Match | None:
        # A stop goes before the fuzzy match: "apri la calcolatrice, stop" is close to a command.
        found = self._exact(text) or self._stop_at_end(text)
        if found is not None:
            return found
        norm, number = extract_number(normalize(text))
        if norm and self._choices:
            words = set(norm.split())
            for phrase, score, _ in process.extract(
                norm, self._choices, scorer=fuzz.ratio, score_cutoff=self.threshold, limit=5
            ):
                cmd = self._by_phrase[phrase]
                if cmd.keywords is None or words & cmd.keywords:
                    return Match(cmd, float(score), number)
        return self._match_sound(text)

    def _exact(self, text: str) -> Match | None:
        """`text` is a command's phrase: that beats a fuzzy match."""
        raw = normalize(text, drop_fillers=False).split()
        norm, number = extract_number(normalize(text))
        cmd = self._by_phrase.get(norm) if norm else None
        if cmd is None:
            return None
        # An exact-only command ("spegniti") must be all that was said: not cut off
        # on an article ("spegni il...") nor followed by a number ("di dieci").
        if cmd.exact and (number is not None or (raw and raw[-1] in ARTICLES)):
            return None
        return Match(cmd, 100.0, number)

    def _stop_at_end(self, text: str) -> Match | None:
        """A longer phrase said to Claudio whose last clause is a stop: what came before it is
        not done, and Claude is not asked ("No, Claudio, non parlare. Claudio, stop.", 01/11)."""
        for clause in last_clauses(text):
            found = self._exact(clause)
            if found is not None and found.command.action == "stop":
                return found
        return None

    def _match_sound(self, text: str) -> Match | None:
        """The same, comparing how the phrases sound: an exact command ("chiuditi") only when
        the sounds are the same ("QDT" = cu-di-ti), the others also when they are close."""
        spelled = spell_capitals(text)
        raw = normalize(spelled, drop_fillers=False).split()
        norm, number = extract_number(normalize(spelled))
        if not self._by_sound or not norm:
            return None
        key = sound_key(norm)
        cmd = self._by_sound.get(key) or self._by_joined_sound.get(key.replace(" ", ""))
        if cmd is not None and not (
            cmd.exact and (number is not None or (raw and raw[-1] in ARTICLES))
        ):
            return Match(cmd, 100.0, number)
        heard = set(key.split())
        for phrase, score, _ in process.extract(
            key, self._sound_choices, scorer=fuzz.ratio, score_cutoff=self.threshold
        ):
            cmd = self._by_sound[phrase]
            if cmd.keywords is None or heard & {sound_key(w) for w in cmd.keywords}:
                return Match(cmd, float(score), number)
        return None

    def nearest(
        self, texts: Sequence[str], language: str = DEFAULT_LANGUAGE, limit: int = 4
    ) -> list[Near]:
        """The commands `texts` (a transcription and Whisper's other guesses) come closest to
        without matching, best first, one entry per command. Letters and sound are both
        compared, and exact-only commands count too: "Claudio spegni ti" is 94 from
        "spegniti", which never matches fuzzily. Keywords are not required: this only
        lists candidates for Claude to judge (brain/guess.py)."""
        best: dict[str, Near] = {}

        def keep(cmd: Command, phrase: str, score: float) -> None:
            if cmd.id not in best or score > best[cmd.id].score:
                best[cmd.id] = Near(cmd, phrase, float(score))

        for text in texts:
            norm, _ = extract_number(normalize(text))
            if not norm or not self._by_phrase:
                continue
            for phrase, score, _ in process.extract(
                norm, list(self._by_phrase), scorer=fuzz.ratio, limit=limit * 3
            ):
                keep(self._by_phrase[phrase], self._written[phrase], score)
            spelled, _ = extract_number(normalize(spell_capitals(text)))
            if not self._by_sound or not spelled:
                continue
            for key, score, _ in process.extract(
                sound_key(spelled), list(self._by_sound), scorer=fuzz.ratio, limit=limit * 3
            ):
                keep(self._by_sound[key], self._written[self._sound_phrase[key]], score)
        return sorted(best.values(), key=lambda near: -near.score)[:limit]
