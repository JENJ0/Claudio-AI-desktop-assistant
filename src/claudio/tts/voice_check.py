"""Jack's voice checked: every piece he says (VoiceMonitor: its WAV, pitch and loudness in
the log, so the robotic or too high takes JenJo hears "a couple of times", 29/10, can be
found) and, before a take is played, whether it is one to throw away (TakeJudge).

The pitch tracker is a YIN in numpy, all frames at once through the FFT: a 10 s piece takes
a few milliseconds (also used by scripts/jack_variants.py)."""

from __future__ import annotations

import contextlib
import logging
import math
import queue
import re
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from claudio.tts.loudness import clipped_samples, integrated_lufs, peak_dbfs

log = logging.getLogger(__name__)

PITCH_RATE = 16_000
PITCH_FRAME = 640  # 40 ms
PITCH_HOP = 160  # 10 ms
PITCH_MIN_HZ, PITCH_MAX_HZ = 60.0, 400.0
# 0.15 found a voice in 3 % of the frames of "Apro la chat." (30/10) and logged "f0 0 Hz" for
# a take that sounds fine: Jack's rough voice needs a wider dip.
YIN_THRESHOLD = 0.3
QUIET_DBFS = -40.0  # frames under this are pauses
MIN_VOICED = 20  # voiced frames (200 ms) under which the pitch is "n/d", not 0
WINDOW_FRAMES = 150  # 1.5 s: the register of a long piece is also judged window by window


def pitch_track(audio: np.ndarray, rate: int, threshold: float = YIN_THRESHOLD) -> np.ndarray:
    """f0 in Hz every 10 ms (0 = unvoiced or pause): enough to spot a high register, octave
    jumps and flicker, not a studio pitch tracker."""
    from scipy.signal import resample_poly

    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if np.asarray(audio).dtype == np.int16:
        x = x / 32768
    if rate != PITCH_RATE:
        x = resample_poly(x, PITCH_RATE, rate)
    lag_min = int(PITCH_RATE / PITCH_MAX_HZ)
    lag_max = int(PITCH_RATE / PITCH_MIN_HZ)
    span = PITCH_FRAME + lag_max
    if len(x) <= span:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(x, span)[: len(x) - span : PITCH_HOP]
    ref = frames[:, :PITCH_FRAME]
    size = 1 << (span + PITCH_FRAME - 1).bit_length()
    # YIN's difference d(lag) = energy of the frame + energy of the shifted frame - 2 * their
    # correlation; the correlation of all frames at once by FFT.
    corr = np.fft.irfft(np.fft.rfft(frames, size) * np.conj(np.fft.rfft(ref, size)), size, axis=1)[
        :, : lag_max + 1
    ]
    squares = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(frames**2, axis=1)], axis=1)
    shifted = squares[:, PITCH_FRAME : PITCH_FRAME + lag_max + 1] - squares[:, : lag_max + 1]
    diff = np.maximum(shifted[:, :1] + shifted - 2 * corr, 0.0)
    cmnd = np.ones_like(diff)
    lags = np.arange(1, lag_max + 1)
    cmnd[:, 1:] = diff[:, 1:] * lags / np.maximum(np.cumsum(diff[:, 1:], axis=1), 1e-12)
    below = cmnd[:, lag_min:] < threshold
    found = below.any(axis=1)
    first = lag_min + np.argmax(below, axis=1)
    # the bottom of that dip: the first lag after it where the curve stops falling
    rising = np.zeros_like(below)
    rising[:, :-1] = cmnd[:, lag_min + 1 :] >= cmnd[:, lag_min:-1]
    rising[:, -1] = True
    rising &= np.arange(lag_min, lag_max + 1) >= first[:, None]
    lag = lag_min + np.argmax(rising, axis=1)
    loud = np.sqrt(np.mean(ref**2, axis=1)) >= 10 ** (QUIET_DBFS / 20)
    return np.where(found & loud, PITCH_RATE / lag, 0.0)


@dataclass
class VoiceStats:
    """What one take sounds like. `median_hz` is 0 when there is too little voice to tell
    (`voiced` frames < MIN_VOICED): the log says "n/d", not "0 Hz"."""

    median_hz: float = 0.0
    high_hz: float = 0.0  # the highest median of a 1.5 s window: a shout inside a long piece
    range_st: float = 0.0  # spread of the melody in semitones (low = flat, monotone)
    jumps: int = 0  # octave-sized leaps between neighbouring voiced frames (robotic warble)
    flicker: int = 0  # voiced runs under 30 ms (buzz)
    voiced: int = 0
    lufs: float = -math.inf
    peak_dbfs: float = -math.inf
    clipped: int = 0
    seconds: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.voiced >= MIN_VOICED

    def as_dict(self) -> dict[str, float]:
        """For scripts/jack_variants.py: the numbers it averages."""
        return {
            "jumps": self.jumps,
            "flicker": self.flicker,
            "range_st": self.range_st,
            "median_hz": self.median_hz,
            "high_hz": self.high_hz,
            "voiced": self.voiced,
            "lufs": self.lufs,
            "peak_dbfs": self.peak_dbfs,
            "clipped": self.clipped,
        }


def voice_stats(audio: np.ndarray, rate: int) -> VoiceStats:
    f0 = pitch_track(audio, rate)
    voiced = f0 > 0
    stats = VoiceStats(
        voiced=int(voiced.sum()),
        lufs=integrated_lufs(audio, rate),
        peak_dbfs=peak_dbfs(audio),
        clipped=clipped_samples(audio),
        seconds=len(audio) / rate,
    )
    if not stats.known:
        return stats
    semis = 12 * np.log2(np.where(voiced, f0, 1.0) / 100)
    both = voiced[1:] & voiced[:-1]
    edges = np.diff(np.concatenate([[0], voiced.astype(int), [0]]))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    stats.jumps = int(np.sum(both & (np.abs(np.diff(semis)) > 9)))
    stats.flicker = int(np.sum(runs < 3))
    stats.range_st = float(np.std(semis[voiced]))
    stats.median_hz = float(np.median(f0[voiced]))
    windows = [
        f0[i : i + WINDOW_FRAMES]
        for i in range(0, max(len(f0) - WINDOW_FRAMES // 2, 1), WINDOW_FRAMES // 2)
    ]
    medians = [float(np.median(w[w > 0])) for w in windows if np.sum(w > 0) >= MIN_VOICED]
    stats.high_hz = max(medians, default=stats.median_hz)
    return stats


def pitch_stats(audio: np.ndarray, rate: int) -> dict[str, float]:
    """`jumps`, `flicker`, `range_st`, `median_hz` and the rest of voice_stats as numbers."""
    return voice_stats(audio, rate).as_dict()


def suspicious(row: dict) -> bool:
    return row["jumps"] >= 2 or row["flicker"] >= 4 or 0 < row["range_st"] < 1.5


_VOWELS = re.compile(r"[aeiouàèéìíòóùú]+", re.IGNORECASE)


def syllables(text: str) -> int:
    """Italian syllables, about: one per group of vowels."""
    return len(_VOWELS.findall(text))


def semitones(hz: float, usual: float) -> float:
    return 12 * math.log2(hz / usual) if hz > 0 and usual > 0 else 0.0


# Jack's usual register: the median of the 36 takes of 30/10 (checkpoint 1.7B-calmo, seed 7).
# Fixed, not learnt as he speaks: the median of the takes he played drifted down to 115 Hz
# and then flagged half of them.
USUAL_HZ = 132.0
HIGH_ST = 5.0  # a take whose median is this far over USUAL_HZ is the strained, high Jack
WINDOW_ST = 9.0  # ...or any 1.5 s of it this far over: a shout in a long piece (018, 30/10)
LOUD_DB = 6.0  # a take this far over the level target before the level guard was shouted
FLAT_SYLLABLES = 8  # a shorter phrase is flat by nature: "Apro la chat."
LOUDNESS_TOLERANCE_DB = 3.0  # after the level guard, a piece further off the target is logged
PACE_SLACK_S = 0.8  # silence a take may have around its words before it counts as too slow


def _hz(value: float) -> str:
    return f"{value:.0f} Hz" if value > 0 else "n/d"


@dataclass
class _Piece:
    n: int
    text: str
    spoken: str
    audio: np.ndarray
    rate: int
    raw: dict[str, float]


@dataclass
class TakeVerdict:
    reasons: list[str]
    stats: VoiceStats

    @property
    def ok(self) -> bool:
        return not self.reasons


class TakeJudge:
    """Whether a take of Jack is one to throw away and generate again, decided in a few ms
    before it is played: every Qwen3 generation is a new draw, and on 30/10 the same seed
    gave takes from 81 to 296 Hz.

    A take is bad when it is high (its median HIGH_ST over `usual_hz`, or any 1.5 s of it
    WINDOW_ST over), when it came out of Qwen3 more than LOUD_DB over the level target
    (shouted: the level guard evens out the volume, not the shout) or past full scale, and
    when it is said too slowly for its text (babbling on past it: also while it is still
    being generated) or, once whole, too fast. `raw` is the take's own level before the
    level guard, as the worker reports it (raw_lufs, raw_clipped). On the 36 takes of 30/10
    it rejects 8: the five high ones (002, 004, 028, 038, 040), 012, 018 and 039."""

    def __init__(
        self,
        target_lufs: float = -24.0,
        usual_hz: float = USUAL_HZ,
        min_cps: float = 12.0,
        max_cps: float = 24.0,
    ) -> None:
        self.target_lufs = target_lufs
        self.usual_hz = usual_hz
        self.min_cps = min_cps
        self.max_cps = max_cps

    def judge(
        self,
        audio: np.ndarray,
        rate: int,
        spoken: str,
        raw: dict[str, float] | None = None,
        complete: bool = True,
    ) -> TakeVerdict:
        stats = voice_stats(audio, rate)
        stats.extra = dict(raw or {})
        return TakeVerdict(self.reasons(stats, spoken, complete), stats)

    def reasons(self, stats: VoiceStats, spoken: str, complete: bool = True) -> list[str]:
        reasons = []
        if stats.known and (
            semitones(stats.median_hz, self.usual_hz) > HIGH_ST
            or semitones(stats.high_hz, self.usual_hz) > WINDOW_ST
        ):
            usual = f"{self.usual_hz:.0f} Hz"
            reasons.append(f"registro alto ({_hz(stats.high_hz)}, di solito {usual})")
        level = stats.extra.get("raw_lufs", stats.lufs)
        if math.isfinite(level) and level > self.target_lufs + LOUD_DB:
            reasons.append(f"take forte ({level:.0f} LUFS)")
        if stats.extra.get("raw_clipped", stats.clipped):
            reasons.append("take oltre il fondo scala")
        slowest = len(spoken) / self.min_cps + PACE_SLACK_S
        fastest = len(spoken) / self.max_cps
        if stats.seconds > slowest or (complete and 0 < stats.seconds < fastest):
            reasons.append(f"{len(spoken) / max(stats.seconds, 1e-3):.0f} caratteri/s")
        return reasons


class VoiceMonitor:
    """Saves every piece Jack says as NNN.wav in `folder` and logs its pitch and loudness. A
    piece the TakeJudge would reject, or with octave jumps, a flat melody (only past
    FLAT_SYLLABLES) or a level off the target after the level guard is logged as a warning
    with its file name. One thread checks the pieces in order: the reply never waits."""

    def __init__(self, folder: Path, keep: int = 200, judge: TakeJudge | None = None) -> None:
        self.folder = folder
        self.keep = keep  # oldest WAVs deleted past this (~0.5 MB each)
        self.judge_take = judge or TakeJudge()
        self._lock = threading.Lock()
        self._count = 0
        self._todo: queue.Queue[_Piece] = queue.Queue(maxsize=50)
        self._thread: threading.Thread | None = None

    def check(
        self,
        text: str,
        audio: np.ndarray,
        rate: int,
        raw: dict[str, float] | None = None,
        spoken: str | None = None,
    ) -> None:
        """`raw`: the take's own numbers before the level guard (raw_lufs, raw_clipped...);
        `spoken`: the text as Jack read it (digits as words), for its pace."""
        if len(audio) < rate // 4:
            return
        with self._lock:
            self._count += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._loop, daemon=True, name="jack-voice-check"
                )
                self._thread.start()
            item = _Piece(self._count, text, spoken or text, audio, rate, dict(raw or {}))
        # a diagnostic that falls behind drops pieces, it never blocks the voice
        with contextlib.suppress(queue.Full):
            self._todo.put_nowait(item)

    def _loop(self) -> None:
        while True:
            self._run(self._todo.get())

    def _run(self, piece: _Piece) -> None:
        n, text, audio, rate, raw = piece.n, piece.text, piece.audio, piece.rate, piece.raw
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
            stats = voice_stats(audio, rate)
            stats.extra = raw
            verdict = self.judge(stats, text, piece.spoken)
            args = (
                path.name,
                _hz(stats.median_hz),
                f"{stats.range_st:.1f} st" if stats.known else "n/d",
                stats.jumps,
                stats.flicker,
                stats.lufs,
                stats.peak_dbfs,
                raw.get("raw_lufs", stats.lufs),
                f" ({verdict})" if verdict else "",
                text[:60],
            )
            if verdict:
                log.warning(
                    "Voce Jack %s: f0 %s, melodia %s, salti %d, tremolii %d, %.1f LUFS, "
                    "picco %.1f dBFS (take %.1f LUFS)%s - %s",
                    *args,
                )
            else:
                log.info(
                    "Voce Jack %s: f0 %s, melodia %s, salti %d, tremolii %d, %.1f LUFS, "
                    "picco %.1f dBFS (take %.1f LUFS)%s - %s",
                    *args,
                )
        except Exception as exc:  # a diagnostic must never break the voice
            log.debug("Controllo voce Jack fallito: %s", exc)

    def judge(self, stats: VoiceStats, text: str = "", spoken: str | None = None) -> str:
        """Why the piece sounds wrong, or "" if it does not."""
        reasons = self.judge_take.reasons(stats, text if spoken is None else spoken)
        # Flicker is not a reason: on JenJo's PC (29/10, jack_variants R vs K) normal pieces
        # had 0-8 of it, so it flagged nearly every piece; octave jumps marked the robotic one.
        if stats.jumps >= 2:
            reasons.append("salti d'ottava")
        if stats.known and 0 < stats.range_st < 1.5 and syllables(text) > FLAT_SYLLABLES:
            reasons.append("melodia piatta")
        off = stats.lufs - self.judge_take.target_lufs
        if stats.clipped or (math.isfinite(off) and abs(off) > LOUDNESS_TOLERANCE_DB):
            reasons.append("volume")
        return ", ".join(reasons)
