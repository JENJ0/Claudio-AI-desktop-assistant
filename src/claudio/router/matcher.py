"""Match a transcription to a local command: exact first, then fuzzy (rapidfuzz).

Each command has its phrases and reply in Italian (`phrases`, `reply`) and optionally in
other languages (`phrases_en`, `reply_en`...). The language Whisper heard is tried first,
then the others, so a request still works when the language was misjudged. `keywords_xx`
are words a fuzzy match in that language must contain ("what's the weather" is close to
"what's the date" but has no "date").

Last, in Italian, the phrases are compared by how they sound (phonetic.py): Whisper's "Claudio
QDT" is "chiuditi" spelled as letters, "kiudi" is "chiudi"."""

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
from claudio.router.normalizer import language as language_rules
from claudio.router.phonetic import sound_key, spell_capitals

_TRANSLATED = re.compile(r"^(phrases|reply|keywords|confirm)_([a-z]{2})$")
SOUND_LANGUAGES = ("it",)  # the sound rules are Italian


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
    keywords: dict[str, frozenset[str]] = field(default_factory=dict)  # per language
    level: int = 1  # safety level: 1 runs at once, 2 asks yes/no, 3 asks an explicit "confermo"
    confirm: str = ""  # the question asked at level 2-3 (Italian; confirm_xx for the others)
    other_confirms: dict[str, str] = field(default_factory=dict)

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

    def confirm_in(self, language: str) -> str:
        """The question for `language`, else the Italian one (may be empty: the generic one)."""
        return self.other_confirms.get(language, self.confirm)

    def has_reply_in(self, language: str) -> bool:
        return language == DEFAULT_LANGUAGE or language in self.other_replies

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
        other_confirms: dict[str, str] = {}
        keywords: dict[str, frozenset[str]] = {}
        if "keywords" in raw:
            keywords[DEFAULT_LANGUAGE] = _keywords(raw["keywords"], DEFAULT_LANGUAGE)
        for key, value in raw.items():
            if m := _TRANSLATED.match(key):
                kind, lang = m.groups()
                if kind == "phrases":
                    other_phrases[lang] = tuple(value)
                elif kind == "reply":
                    other_replies[lang] = str(value)
                elif kind == "confirm":
                    other_confirms[lang] = str(value)
                else:
                    keywords[lang] = _keywords(value, lang)
        cmd = Command(
            id=raw["id"],
            phrases=tuple(raw["phrases"]),
            action=raw["action"],
            params=dict(raw.get("params", {})),
            reply=raw.get("reply", ""),
            exact=bool(raw.get("exact", False)),
            other_phrases=other_phrases,
            other_replies=other_replies,
            keywords=keywords,
            level=int(raw.get("level", 1)),
            confirm=raw.get("confirm", ""),
            other_confirms=other_confirms,
        )
        if cmd.level not in (1, 2, 3):
            raise ValueError(
                tr(
                    "{path}: il comando '{command}' ha un livello non valido ({level})",
                    path=path,
                    command=cmd.id,
                    level=cmd.level,
                )
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


def _keywords(words: list[str], lang: str) -> frozenset[str]:
    return frozenset(normalize(w, drop_fillers=False, lang=lang) for w in words)


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
        # How each phrase sounds -> its command. Two phrases sounding alike are one command
        # (they were written as variants of the same words).
        self._by_sound: dict[str, dict[str, Command]] = {}
        for language in SOUND_LANGUAGES:
            sounds = self._by_sound[language] = {}
            for phrase, cmd in self._by_phrase.get(language, {}).items():
                sounds.setdefault(sound_key(phrase), cmd)
        self._sound_choices = {
            language: [k for k, cmd in sounds.items() if not cmd.exact]
            for language, sounds in self._by_sound.items()
        }

    @property
    def languages(self) -> list[str]:
        return list(self._by_phrase)

    def match(self, text: str, language: str = DEFAULT_LANGUAGE) -> Match | None:
        order = [language, *(lang for lang in self._by_phrase if lang != language)]
        texts = {lang: extract_number(normalize(text, lang=lang), lang) for lang in order}
        raw = normalize(text, drop_fillers=False).split()
        for lang in order:  # an exact phrase in any language beats a fuzzy one
            norm, number = texts[lang]
            cmd = self._by_phrase.get(lang, {}).get(norm) if norm else None
            if cmd is None:
                continue
            # An exact-only command ("shut down") must be all that was said: not cut off
            # on an article ("shut down the...") nor followed by a number ("by ten").
            if cmd.exact and (
                number is not None or (raw and raw[-1] in language_rules(lang).articles)
            ):
                continue
            return Match(cmd, 100.0, number, lang)
        for lang in order:
            norm, number = texts[lang]
            if not norm or not self._choices.get(lang):
                continue
            words = set(norm.split())
            for phrase, score, _ in process.extract(
                norm, self._choices[lang], scorer=fuzz.ratio, score_cutoff=self.threshold, limit=5
            ):
                cmd = self._by_phrase[lang][phrase]
                needed = cmd.keywords.get(lang)
                if needed is None or words & needed:
                    return Match(cmd, float(score), number, lang)
        return self._match_sound(text, order)

    def _match_sound(self, text: str, order: list[str]) -> Match | None:
        """The same, comparing how the phrases sound: an exact command ("chiuditi") only when
        the sounds are the same ("QDT" = cu-di-ti), the others also when they are close."""
        spelled = spell_capitals(text)
        raw = normalize(spelled, drop_fillers=False).split()
        for lang in order:
            sounds = self._by_sound.get(lang)
            if not sounds:
                continue
            norm, number = extract_number(normalize(spelled, lang=lang), lang)
            if not norm:
                continue
            key = sound_key(norm)
            cmd = sounds.get(key)
            if cmd is not None and not (
                cmd.exact
                and (number is not None or (raw and raw[-1] in language_rules(lang).articles))
            ):
                return Match(cmd, 100.0, number, lang)
            heard = set(key.split())
            for phrase, score, _ in process.extract(
                key, self._sound_choices[lang], scorer=fuzz.ratio, score_cutoff=self.threshold
            ):
                cmd = sounds[phrase]
                needed = cmd.keywords.get(lang)
                if needed is None or heard & {sound_key(w) for w in needed}:
                    return Match(cmd, float(score), number, lang)
        return None
