"""Fixed replies spoken by ElevenLabs, kept as WAVs in data/eleven_cache/: generated online
once, then played from disk with no network and no credits. The key holds a `profile` (the
voice and the settings that shape it): another voice never reuses old files."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import wave
from pathlib import Path

import numpy as np

from claudio.config import TtsSettings

log = logging.getLogger(__name__)

# Raised when the audio is shaped differently (a new level guard, a new format).
CACHE_VERSION = 1


def eleven_profile(tts: TtsSettings) -> str:
    """What makes a recording sound as it does: voice, model, voice settings, language,
    audio format and the target level."""
    values = [
        tts.eleven_voice_id,
        tts.eleven_model,
        tts.eleven_stability,
        tts.eleven_similarity,
        "it",
        "pcm_24000",
        tts.jack_lufs,
    ]
    return json.dumps([CACHE_VERSION, values], ensure_ascii=False)


class TakeCache:
    def __init__(self, folder: Path, profile: str) -> None:
        self.folder = folder
        self.profile = profile

    def key(self, text: str) -> str:
        return hashlib.sha1(f"{self.profile}\n{text.strip()}".encode()).hexdigest()[:20]

    def _path(self, text: str) -> Path:
        return self.folder / f"{self.key(text)}.wav"

    def load(self, text: str, rate: int) -> np.ndarray | None:
        """The recording of this text as int16 PCM, or None: missing, unreadable, truncated
        or not mono 16-bit at `rate`."""
        try:
            with wave.open(str(self._path(text)), "rb") as wav:
                frames = wav.getnframes()
                shape = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
                data = wav.readframes(frames)
        except FileNotFoundError:
            return None
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            log.warning("Registrazione in cache illeggibile (%s): la rigenero", exc)
            return None
        if shape != (1, 2, rate) or len(data) != frames * 2:
            log.warning("Registrazione in cache incompleta o diversa: la rigenero")
            return None
        return np.frombuffer(data, dtype="<i2").astype(np.int16)

    def save(self, text: str, pcm: np.ndarray, rate: int) -> None:
        """Written to a temporary file of its own, then renamed: a crash or a second save of
        the same text never leaves half a WAV."""
        path = self._path(text)
        tmp: Path | None = None
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=self.folder, suffix=".tmp")
            tmp = Path(name)
            with os.fdopen(fd, "wb") as handle, wave.open(handle, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("Registrazione non salvata in cache: %s", exc)
            if tmp is not None:
                with contextlib.suppress(OSError):  # a stray .tmp is harmless: load reads .wav
                    tmp.unlink(missing_ok=True)
