"""Jack's voice checked as it speaks: the pitch of every piece, so the robotic or too high
takes JenJo hears "a couple of times" (29/10) can be found in the log, with their WAV.

The pitch tracker is a plain YIN in numpy (also used by scripts/jack_variants.py)."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import wave
from collections import deque
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PITCH_RATE = 16_000
PITCH_FRAME = 640  # 40 ms
PITCH_HOP = 160  # 10 ms
PITCH_MIN_HZ, PITCH_MAX_HZ = 60.0, 400.0
YIN_THRESHOLD = 0.15


def pitch_track(audio: np.ndarray, rate: int) -> np.ndarray:
    """f0 in Hz every 10 ms (0 = unvoiced), a plain YIN: enough to spot octave jumps and
    flicker, not a studio pitch tracker."""
    from scipy.signal import resample_poly

    x = audio.astype(np.float64) / 32768
    if rate != PITCH_RATE:
        x = resample_poly(x, PITCH_RATE, rate)
    lag_min = int(PITCH_RATE / PITCH_MAX_HZ)
    lag_max = int(PITCH_RATE / PITCH_MIN_HZ)
    loud = 10 ** (-40 / 20)  # quieter frames are pauses
    f0 = []
    for start in range(0, len(x) - PITCH_FRAME - lag_max, PITCH_HOP):
        frame = x[start : start + PITCH_FRAME + lag_max]
        if np.sqrt(np.mean(frame[:PITCH_FRAME] ** 2)) < loud:
            f0.append(0.0)
            continue
        ref = frame[:PITCH_FRAME]
        shifted = np.lib.stride_tricks.sliding_window_view(frame, PITCH_FRAME)[: lag_max + 1]
        diff = np.sum((shifted - ref) ** 2, axis=1)
        cum = np.cumsum(diff[1:])
        cmnd = np.ones_like(diff)
        cmnd[1:] = diff[1:] * np.arange(1, len(diff)) / np.maximum(cum, 1e-12)
        below = np.flatnonzero(cmnd[lag_min:] < YIN_THRESHOLD)
        if not len(below):
            f0.append(0.0)
            continue
        lag = lag_min + below[0]
        while lag + 1 < len(cmnd) and cmnd[lag + 1] < cmnd[lag]:
            lag += 1  # bottom of that dip
        f0.append(PITCH_RATE / lag)
    return np.array(f0)


def pitch_stats(audio: np.ndarray, rate: int) -> dict[str, float]:
    """`jumps`: octave-sized leaps between neighbouring voiced frames (robotic warble);
    `flicker`: voiced runs under 30 ms (buzz); `range_st`: spread of the melody in semitones
    (low = flat, monotone); `median_hz`: the register the piece is said in."""
    f0 = pitch_track(audio, rate)
    voiced = f0 > 0
    if voiced.sum() < 5:
        return {"jumps": 0, "flicker": 0, "range_st": 0.0, "median_hz": 0.0}
    semis = 12 * np.log2(np.where(voiced, f0, 1.0) / 100)
    both = voiced[1:] & voiced[:-1]
    jumps = int(np.sum(both & (np.abs(np.diff(semis)) > 9)))
    edges = np.diff(np.concatenate([[0], voiced.astype(int), [0]]))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {
        "jumps": jumps,
        "flicker": int(np.sum(runs < 3)),
        "range_st": float(np.std(semis[voiced])),
        "median_hz": float(np.median(f0[voiced])),
    }


def suspicious(row: dict) -> bool:
    return row["jumps"] >= 2 or row["flicker"] >= 4 or 0 < row["range_st"] < 1.5


# A piece this many semitones above the usual register is the "higher, robotic" Jack.
HIGH_SEMITONES = 3.0
HISTORY = 30  # pieces the usual register is the median of
MIN_HISTORY = 3


class VoiceMonitor:
    """Saves every piece Jack says as NNN.wav in `folder` and logs its pitch; a piece off
    the usual register, or with octave jumps, flicker or a flat melody, is logged as a
    warning with its file name. One thread checks the pieces in order: the reply never
    waits for it."""

    def __init__(self, folder: Path, keep: int = 200) -> None:
        self.folder = folder
        self.keep = keep  # oldest WAVs deleted past this (~0.5 MB each)
        self._medians: deque[float] = deque(maxlen=HISTORY)
        self._lock = threading.Lock()
        self._count = 0
        self._todo: queue.Queue[tuple[int, str, np.ndarray, int]] = queue.Queue(maxsize=50)
        self._thread: threading.Thread | None = None

    def check(self, text: str, audio: np.ndarray, rate: int) -> None:
        if len(audio) < rate // 4:
            return
        with self._lock:
            self._count += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._loop, daemon=True, name="jack-voice-check"
                )
                self._thread.start()
            item = (self._count, text, audio, rate)
        # a diagnostic that falls behind drops pieces, it never blocks the voice
        with contextlib.suppress(queue.Full):
            self._todo.put_nowait(item)

    def _loop(self) -> None:
        while True:
            self._run(*self._todo.get())

    def _run(self, n: int, text: str, audio: np.ndarray, rate: int) -> None:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / f"{n:03d}.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(audio.astype("<i2").tobytes())
            for old in sorted(self.folder.glob("*.wav"))[: -self.keep]:
                old.unlink(missing_ok=True)
            stats = pitch_stats(audio, rate)
            verdict = self.judge(stats)
            args = (
                path.name,
                stats["median_hz"],
                stats["range_st"],
                stats["jumps"],
                stats["flicker"],
                f" ({verdict})" if verdict else "",
                text[:60],
            )
            if verdict:
                log.warning(
                    "Voce Jack %s: f0 %.0f Hz, melodia %.1f st, salti %d, tremolii %d%s - %s",
                    *args,
                )
            else:
                log.info(
                    "Voce Jack %s: f0 %.0f Hz, melodia %.1f st, salti %d, tremolii %d%s - %s",
                    *args,
                )
        except Exception as exc:  # a diagnostic must never break the voice
            log.debug("Controllo voce Jack fallito: %s", exc)

    def judge(self, stats: dict[str, float]) -> str:
        """Why the piece sounds wrong, or "" if it does not."""
        reasons = []
        median = stats["median_hz"]
        with self._lock:
            usual = float(np.median(self._medians)) if len(self._medians) >= MIN_HISTORY else 0.0
            if median > 0:
                self._medians.append(median)
        if usual and median > 0 and 12 * np.log2(median / usual) > HIGH_SEMITONES:
            reasons.append(f"registro alto, di solito {usual:.0f} Hz")
        if suspicious(stats):
            reasons.append("intonazione sospetta")
        return ", ".join(reasons)
