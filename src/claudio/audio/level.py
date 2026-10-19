"""Automatic microphone level: after every request Claudio corrects its own input gain.

The digital gain moves towards speech peaks around TARGET_DBFS, slowly up and quickly down
(clipping cannot be repaired later), within 0..max_db. The value is saved per microphone, so
the next start begins where the last one ended. Windows' own microphone level is not touched."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

log = logging.getLogger(__name__)

TARGET_DBFS = -20.0  # loudest 32 ms of speech: clear for the wake word and Whisper, no clipping
TOLERANCE_DB = 4.0  # -24..-16 dBFS: the gain is fine
MAX_UP_DB = 3.0
MAX_DOWN_DB = 6.0
CLIPPED_LIMIT = 0.001  # share of speech samples at full scale that counts as clipping
QUIET_DBFS = -30.0


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
        self.microphone = str(microphone)
        self.max_db = max_db
        saved = self._load().get(self.microphone)
        self.gain_db = float(saved) if isinstance(saved, int | float) else gain_db
        self.gain_db = min(max(self.gain_db, 0.0), max_db)
        self._warned = False
        apply(self.gain_db)
        if saved is not None:
            log.info("Microfono: guadagno +%.0f dB (salvato dall'ultima volta)", self.gain_db)

    def observe(self, stats: Mapping[str, float]) -> None:
        """Stats of a request in which speech was heard (Endpointer.stats())."""
        speech = stats["speech_dbfs"]
        if stats.get("clipped", 0.0) > CLIPPED_LIMIT:
            step, why = -MAX_DOWN_DB, "voce saturata"
        else:
            error = TARGET_DBFS - speech
            if abs(error) <= TOLERANCE_DB:
                return
            step = min(max(error / 2, -MAX_DOWN_DB), MAX_UP_DB)
            why = f"voce a {speech:.0f} dBFS"
        gain = round(min(max(self.gain_db + step, 0.0), self.max_db), 1)
        if gain == self.gain_db:
            if step > 0 and speech < QUIET_DBFS and not self._warned:
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
        if self.state_file is None:
            return
        try:
            data = self._load()
            data[self.microphone] = self.gain_db
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("Impossibile salvare il livello del microfono: %s", exc)
