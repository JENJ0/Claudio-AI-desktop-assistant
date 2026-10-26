"""How Claudio speaks, as JenJo asked by voice ("più drammatico", "parla più lento", "voce più
grave", "torna normale"): Haiku's `set_style` tool changes it, and it lasts across restarts.

Two halves, one place:
- the words: `tone`, a few words told to Claude in every request's tag (after the cached
  prefix, so changing it costs nothing extra), followed inside the CLAUDIO.md personality;
- the voice (Jack only): `pace` and `pitch` go to RVC, `energy` to Qwen3's sampling
  temperature, sentence by sentence, so a change is heard from the next sentence on.

Jack is the only voice (a single-speaker model): "cambia voce" changes these, never the
speaker. Kokoro and Piper (fallbacks) keep their settings."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

Energy = Literal["calm", "normal", "lively"]
ENERGIES: tuple[Energy, ...] = ("calm", "normal", "lively")
STYLE_FILE = "style.json"  # in [paths] data_dir: survives restarts and updates
TONE_MAX_CHARS = 80  # sent with every request: a few words, never a second personality
PACE_MIN, PACE_MAX = 0.7, 1.3  # times the configured [tts] jack_speed
PITCH_MAX = 4  # semitones: beyond this RVC turns Jack into a cartoon
SPEED_MIN, SPEED_MAX = 0.6, 1.4  # the worker's absolute pace, whatever the base
# Qwen3's sampling temperature around [tts] jack_temperature: higher = more varied delivery,
# but more mispronounced words (JenJo picked 0.7 for pronunciation, 24/10).
ENERGY_TEMPERATURE = {"calm": -0.1, "normal": 0.0, "lively": 0.1}
TEMPERATURE_MIN, TEMPERATURE_MAX = 0.4, 0.9


@dataclass(frozen=True)
class Style:
    tone: str = ""  # "" = the personality's own tone
    pace: float = 1.0
    pitch: int = 0
    energy: Energy = "normal"

    @property
    def is_default(self) -> bool:
        return self == Style()


class SpeakingStyle:
    """The current style, saved to `path` (None: kept in memory only). Read from the voice's
    thread while the brain writes it: a `Style` is immutable and swapped whole."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.current = self._load()

    def update(self, args: dict[str, Any]) -> Style:
        """Apply a `set_style` call: only the fields it names; `reset` first, if true. Numbers
        out of range are clamped; ValueError (sent back to Claude) on a value that makes no
        sense."""
        style = Style() if args.get("reset") is True else self.current
        changes: dict[str, Any] = {}
        if "tone" in args:
            changes["tone"] = _tone(args["tone"])
        if "pace" in args:
            changes["pace"] = _number(args["pace"], "pace", PACE_MIN, PACE_MAX)
        if "pitch" in args:
            changes["pitch"] = round(_number(args["pitch"], "pitch", -PITCH_MAX, PITCH_MAX))
        if "energy" in args:
            if args["energy"] not in ENERGIES:
                raise ValueError(f"energy must be one of {', '.join(ENERGIES)}")
            changes["energy"] = args["energy"]
        if not changes and args.get("reset") is not True:
            raise ValueError("nothing to change: give tone, pace, pitch, energy or reset")
        self.current = replace(style, **changes)
        self._save()
        log.info("Stile: %s", describe(self.current))
        return self.current

    def tag(self) -> str:
        """For the request's tag: "" when nothing was asked (no tokens)."""
        return "" if self.current.is_default else f"style: {describe(self.current)}"

    def voice(self, base_speed: float, base_temperature: float) -> dict[str, Any]:
        """What Jack's worker gets with each sentence: absolute values, clamped."""
        s = self.current
        speed = min(max(base_speed * s.pace, SPEED_MIN), SPEED_MAX)
        temperature = base_temperature + ENERGY_TEMPERATURE[s.energy]
        if s.energy != "normal":  # never pushes a configured value the other way
            temperature = min(max(temperature, TEMPERATURE_MIN), TEMPERATURE_MAX)
        return {"speed": round(speed, 3), "pitch": s.pitch, "temperature": round(temperature, 3)}

    def _load(self) -> Style:
        if self.path is None or not self.path.exists():
            return Style()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            style = Style()
            self.current = style
            return self.update({"reset": True, **{k: data[k] for k in asdict(style) if k in data}})
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            log.warning("Stile in %s non valido (%s): uso quello normale", self.path, exc)
            return Style()

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(asdict(self.current), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # the style still works for this run
            log.warning("Stile non salvato in %s: %s", self.path, exc)


def describe(style: Style) -> str:
    """The non-default fields, e.g. 'tone "drammatico", pace 0.85, pitch -2' ("normal")."""
    parts = []
    if style.tone:
        parts.append(f'tone "{style.tone}"')
    if style.pace != 1.0:
        parts.append(f"pace {style.pace:g}")
    if style.pitch:
        parts.append(f"pitch {style.pitch:+d}")
    if style.energy != "normal":
        parts.append(f"energy {style.energy}")
    return ", ".join(parts) or "normal"


def _tone(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("tone must be text")
    tone = " ".join(value.replace('"', "'").split())  # one line: it goes in the tag
    if len(tone) > TONE_MAX_CHARS:
        raise ValueError(f"tone too long: a few words, at most {TONE_MAX_CHARS} characters")
    return tone


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number")
    return round(min(max(float(value), low), high), 2)  # "molto più lento": the slowest
