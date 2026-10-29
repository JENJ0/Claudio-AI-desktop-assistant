"""Automatic microphone level: after every request Claudio corrects its own input gain.

The digital gain follows the voice's own level on each microphone (the loudest 32 ms of a
phrase, before the gain), smoothed over the phrases, towards speech peaks around TARGET_DBFS:
slowly up, quickly down (a loud voice clips, a quiet one is only quieter), within 0..max_db.
It learns only from phrases in which Whisper heard words (a breath or a bump says nothing
about the voice: on 26/10 an empty phrase at -50 dBFS raised it) and keeps one value per
microphone really in use, so moving from the webcam to the headset does not start the headset
at the webcam's +24 dB (26/10 19:50, saturated at once). Windows' own microphone level is not
touched.

The boosted signal never clips: above LIMIT_KNEE it bends smoothly towards full scale
(soft_limit), so a phrase louder than the others is compressed instead of being torn into
harmonics Whisper cannot read (28/10 16:43, "apri la chat" at +29 dB -> "qui la chat")."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TARGET_DBFS = -20.0  # loudest 32 ms of speech: clear for the wake word and Whisper, no clipping
TOLERANCE_DB = 3.0  # the gain stays while the voice is within this of the target
MAX_UP_DB = 3.0
MAX_DOWN_DB = 6.0
# How much one phrase moves the voice level Claudio expects: a louder one counts more (it is
# what clips), a quieter one less (JenJo's voice on the headset spans 20 dB, 28/10 16:41).
LOUDER_WEIGHT = 0.6
QUIETER_WEIGHT = 0.25
FIRST_MARGIN_DB = 3.0  # a microphone's first phrase sets the gain at once, this far under target
CLIPPED_LIMIT = 0.001  # share of speech samples at full scale that counts as clipping
QUIET_DBFS = -30.0
LIMIT_KNEE = 0.7  # -3 dBFS: a boosted sample above this is bent, not clipped


def soft_limit(x: np.ndarray) -> np.ndarray:
    """Float samples (1.0 = full scale) unchanged up to LIMIT_KNEE, then bent smoothly
    towards full scale (tanh, same slope at the knee): no clipping however loud."""
    over = np.abs(x) > LIMIT_KNEE
    if not over.any():
        return x
    room = 1.0 - LIMIT_KNEE
    bent = np.sign(x) * (LIMIT_KNEE + room * np.tanh((np.abs(x) - LIMIT_KNEE) / room))
    return np.where(over, bent, x)


# A sample this high after soft_limit was at full scale or above before it: clipping.
CLIP_LEVEL = float(soft_limit(np.array([1.0]))[0])


class AutoGain:
    def __init__(
        self,
        gain_db: float,
        apply: Callable[[float], None],
        state_file: Path | None = None,
        microphone: str | int = "",
        max_db: float = 30.0,
    ) -> None:
        self.apply = apply
        self.state_file = state_file
        self.max_db = max_db
        self.start_db = self._clamp(gain_db)  # where a microphone never heard before starts
        # The setting ("" = Windows' microphone): until v0.3 the gain was saved under it, so
        # its value is where a microphone without one of its own starts.
        self.configured = str(microphone)
        self.microphone: str | None = None
        self.gain_db = self.start_db
        self._voice: float | None = None  # the voice level expected, before the gain (dBFS)
        self._warned = False
        self.use(self.configured)

    def _clamp(self, gain_db: float) -> float:
        return min(max(float(gain_db), 0.0), self.max_db)

    def use(self, microphone: str) -> None:
        """The microphone now in use (the capture opened it): its own gain, saved last time,
        or a start and a fresh measure for one never heard before. The same one again (a
        reopened stream) keeps what was learned."""
        microphone = str(microphone)
        if microphone == self.microphone:
            return
        self.microphone = microphone
        data = self._load()
        saved = data.get(microphone)
        learned = isinstance(saved, int | float)
        if not learned:
            saved = data.get(self.configured)
        self.gain_db = self._clamp(saved) if isinstance(saved, int | float) else self.start_db
        self._voice = TARGET_DBFS - self.gain_db if learned else None
        self._warned = False
        if learned:
            log.info("Microfono: guadagno +%.0f dB (salvato dall'ultima volta)", self.gain_db)
        self.apply(self.gain_db)

    def observe(self, stats: Mapping[str, float]) -> None:
        """Stats of a phrase in which Whisper heard words (Endpointer.stats())."""
        speech = stats["speech_dbfs"]
        voice = speech - self.gain_db  # before the boost
        if stats.get("clipped", 0.0) > CLIPPED_LIMIT:
            gain, why = self.gain_db - MAX_DOWN_DB, "voce saturata"
            self._voice = voice if self._voice is None else max(self._voice, voice)
        elif self._voice is None:  # the first phrase on this microphone: straight to it
            self._voice = voice
            gain, why = TARGET_DBFS - FIRST_MARGIN_DB - voice, f"prima frase a {speech:.0f} dBFS"
        else:
            weight = LOUDER_WEIGHT if voice > self._voice else QUIETER_WEIGHT
            self._voice += weight * (voice - self._voice)
            error = TARGET_DBFS - (self._voice + self.gain_db)
            if abs(error) <= TOLERANCE_DB:
                return
            gain = self.gain_db + min(max(error, -MAX_DOWN_DB), MAX_UP_DB)
            why = f"voce a {speech:.0f} dBFS"
        gain = round(self._clamp(gain), 1)
        if gain == self.gain_db:
            if gain >= self.max_db and speech < QUIET_DBFS and not self._warned:
                self._warned = True
                log.warning(
                    "Microfono basso anche con +%.0f dB (voce a %.0f dBFS): alza il livello del "
                    "microfono nelle impostazioni audio di Windows o avvicinati",
                    gain,
                    speech,
                )
            return
        log.info("Microfono: guadagno da +%.0f a +%.0f dB (%s)", self.gain_db, gain, why)
        self.gain_db = gain
        self.apply(gain)
        self._save()

    def _load(self) -> dict[str, float]:
        if self.state_file is None or not self.state_file.is_file():
            return {}
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("File del livello microfono illeggibile: riparto da zero")
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        if self.state_file is None or self.microphone is None:
            return
        try:
            data = self._load()
            data[self.microphone] = self.gain_db
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("Impossibile salvare il livello del microfono: %s", exc)
