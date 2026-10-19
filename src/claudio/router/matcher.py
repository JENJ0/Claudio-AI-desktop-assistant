"""Match a transcription to a local command: exact first, then fuzzy (rapidfuzz).

Each command has its phrases and reply in Italian (`phrases`, `reply`) and optionally in
other languages (`phrases_en`, `reply_en`...). The language Whisper heard is tried first,
then the others, so a request still works when the language was misjudged."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from claudio.config import DEFAULT_LANGUAGE
from claudio.i18n import tr
from claudio.router.normalizer import extract_number, normalize

_TRANSLATED = re.compile(r"^(phrases|reply)_([a-z]{2})$")


@dataclass(frozen=True)
class Command:
    id: str
    phrases: tuple[str, ...]  # in DEFAULT_LANGUAGE
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    exact: bool = False  # only an exact phrase counts: "spegni" must not shut CLAUDIO down
    other_phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    other_replies: dict[str, str] = field(default_factory=dict)

    @property
    def languages(self) -> tuple[str, ...]:
        return (DEFAULT_LANGUAGE, *self.other_phrases)

    def phrases_in(self, language: str) -> tuple[str, ...]:
        return (
            self.phrases if language == DEFAULT_LANGUAGE else self.other_phrases.get(language, ())
        )

    def reply_in(self, language: str) -> str:
        """The reply in `language`, else the Italian one."""
        return self.other_replies.get(language, self.reply)

    def param_in(self, name: str, language: str) -> Any:
        """params["<name>_<language>"], else params["<name>"]."""
        return self.params.get(f"{name}_{language}", self.params.get(name))


@dataclass(frozen=True)
class Match:
    command: Command
    score: float
    number: int | None = None
    language: str = DEFAULT_LANGUAGE  # of the phrase that matched


def load_commands(path: Path) -> list[Command]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    commands = []
    seen: set[str] = set()
    for raw in data.get("command", []):
        other_phrases: dict[str, tuple[str, ...]] = {}
        other_replies: dict[str, str] = {}
        for key, value in raw.items():
            if m := _TRANSLATED.match(key):
                if m.group(1) == "phrases":
                    other_phrases[m.group(2)] = tuple(value)
                else:
                    other_replies[m.group(2)] = str(value)
        cmd = Command(
            id=raw["id"],
            phrases=tuple(raw["phrases"]),
            action=raw["action"],
            params=dict(raw.get("params", {})),
            reply=raw.get("reply", ""),
            exact=bool(raw.get("exact", False)),
            other_phrases=other_phrases,
            other_replies=other_replies,
        )
        if cmd.id in seen:
            raise ValueError(tr("{path}: comando duplicato '{command}'", path=path, command=cmd.id))
        if not cmd.phrases:
            raise ValueError(
                tr("{path}: il comando '{command}' non ha frasi", path=path, command=cmd.id)
            )
        seen.add(cmd.id)
        commands.append(cmd)
    return commands


class CommandMatcher:
    def __init__(self, commands: list[Command], threshold: float = 85) -> None:
        self.threshold = threshold
        self._by_phrase: dict[str, dict[str, Command]] = {}  # language -> phrase -> command
        for cmd in commands:
            for language in cmd.languages:
                table = self._by_phrase.setdefault(language, {})
                for phrase in cmd.phrases_in(language):
                    norm = normalize(phrase, lang=language)
                    other = table.get(norm)
                    if other and other.id != cmd.id:
                        raise ValueError(
                            tr(
                                "Frase '{phrase}' ambigua: {first} e {second}",
                                phrase=phrase,
                                first=other.id,
                                second=cmd.id,
                            )
                        )
                    table[norm] = cmd
        self._choices = {
            language: [p for p, cmd in table.items() if not cmd.exact]
            for language, table in self._by_phrase.items()
        }

    @property
    def languages(self) -> list[str]:
        return list(self._by_phrase)

    def match(self, text: str, language: str = DEFAULT_LANGUAGE) -> Match | None:
        order = [language, *(lang for lang in self._by_phrase if lang != language)]
        texts = {lang: extract_number(normalize(text, lang=lang), lang) for lang in order}
        for lang in order:  # an exact phrase in any language beats a fuzzy one
            norm, number = texts[lang]
            if norm and norm in self._by_phrase.get(lang, {}):
                return Match(self._by_phrase[lang][norm], 100.0, number, lang)
        for lang in order:
            norm, number = texts[lang]
            if not norm or not self._choices.get(lang):
                continue
            best = process.extractOne(
                norm, self._choices[lang], scorer=fuzz.ratio, score_cutoff=self.threshold
            )
            if best is not None:
                phrase, score, _ = best
                return Match(self._by_phrase[lang][phrase], float(score), number, lang)
        return None
