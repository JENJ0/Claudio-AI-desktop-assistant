"""Takes of Jack worth keeping (data/jack_takes.json).

With one fixed seed for every piece, a text that once came out badly came out badly forever
(30/10: "Chiudo la chat." high every time, "accenditi" heard as "Accendete" four times out of
four). So each text keeps its own seed: the one of a take that passed the TakeJudge, used
again the next time the same words are said. The fixed replies can also have the take
itself, chosen in advance among several by scripts/jack_takes.py (Whisper included) and
saved as a WAV: played with no TTS at all.

Every key holds a `profile`, all that shapes the voice (checkpoint, sampling, level...):
another checkpoint never reuses old takes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Raised when the worker shapes the audio differently (the level guard, a new converter):
# takes saved before would no longer match a take generated now.
TAKES_VERSION = 1
LIMIT = 2000  # texts remembered; the least recently said go first, saved WAVs never


# The [tts] settings that shape a take (not jack_seed: a remembered seed is a seed).
PROFILE_SETTINGS = (
    "jack_checkpoint",
    "jack_rvc_model",
    "jack_speed",
    "jack_temperature",
    "jack_top_p",
    "jack_pitch_fix",
    "jack_lufs",
)


def voice_profile(tts: Any, language: str = "Italian") -> str:
    """What makes a take sound as it does: those settings as configured (paths relative to
    the models folder, so another checkout of Claudio finds the same takes), the language
    told to Qwen3 and TAKES_VERSION."""
    values = [str(getattr(tts, name)).replace("\\", "/") for name in PROFILE_SETTINGS]
    return json.dumps([TAKES_VERSION, language, values], ensure_ascii=False)


class TakeStore:
    """`path` is the JSON table; saved WAVs go in a jack_cache folder next to it."""

    def __init__(self, path: Path, profile: str, limit: int = LIMIT) -> None:
        self.path = path
        self.cache = path.parent / "jack_cache"
        self.profile = profile
        self.limit = limit
        self._lock = threading.Lock()
        self._takes: dict[str, dict[str, Any]] = self._load()
        self._last = max((float(v.get("used", 0)) for v in self._takes.values()), default=0.0)

    def _stamp(self) -> float:
        """When a take was last used: the clock, but always later than the last stamp (on
        Windows the clock moves in steps of 15.6 ms, and the order decides what goes)."""
        self._last = max(time.time(), self._last + 1e-6)
        return self._last

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            takes = data.get("takes", {})
            return {k: v for k, v in takes.items() if isinstance(v, dict) and "seed" in v}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, AttributeError) as exc:
            log.warning("Tabella delle take di Jack illeggibile (%s): riparto da zero", exc)
            return {}

    def key(self, spoken: str, rvc: str) -> str:
        """The same words, said the same way, by the same voice."""
        text = f"{self.profile}\n{rvc}\n{spoken}"
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]

    def seed(self, key: str) -> int | None:
        with self._lock:
            entry = self._takes.get(key)
            if entry is None:
                return None
            entry["used"] = self._stamp()
            return int(entry["seed"])

    def remember(self, key: str, seed: int, text: str) -> None:
        """A take of `text` with this seed passed the check: keep the seed for next time."""
        with self._lock:
            entry = self._takes.setdefault(key, {})
            if entry.get("seed") == seed:
                entry["used"] = self._stamp()
                return
            entry.update(seed=seed, text=text[:120], used=self._stamp())
            self._trim()
            self._save()

    def audio(self, key: str) -> tuple[np.ndarray, int] | None:
        """The take saved in advance for this text, as int16 PCM and its rate, if any."""
        with self._lock:
            entry = self._takes.get(key)
            if entry is None or not entry.get("wav"):
                return None
            name = entry["wav"]
            entry["used"] = self._stamp()
        try:
            with wave.open(str(self.cache / name), "rb") as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
                return pcm.astype(np.int16), wav.getframerate()
        except (OSError, EOFError, wave.Error) as exc:
            log.warning("Take salvata di Jack illeggibile (%s): la genero", exc)
            return None

    def save_audio(
        self, key: str, seed: int, text: str, audio: np.ndarray, rate: int, **info: Any
    ) -> Path:
        """Keep this take of a fixed reply as a WAV (scripts/jack_takes.py)."""
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / f"{key}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(np.asarray(audio, dtype="<i2").tobytes())
        with self._lock:
            entry = self._takes.setdefault(key, {})
            entry.update(seed=seed, text=text[:120], used=self._stamp(), wav=path.name, **info)
            self._save()
        return path

    def _trim(self) -> None:
        loose = [k for k, v in self._takes.items() if not v.get("wav")]
        excess = len(self._takes) - self.limit
        if excess > 0:
            for k in sorted(loose, key=lambda k: self._takes[k].get("used", 0))[:excess]:
                del self._takes[k]

    def _save(self) -> None:
        """Written whole to a temporary file, then renamed over the table: a crash leaves the
        old table or the new one, never half of it."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            data = {"version": TAKES_VERSION, "takes": self._takes}
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Tabella delle take di Jack non salvata: %s", exc)
