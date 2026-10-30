"""How loud Jack's voice is, by ITU-R BS.1770 (the LUFS of EBU R128), and the level guard
that keeps every piece near one loudness without clipping.

Only numpy and scipy: the Jack worker, which runs in a venv of its own, imports it too."""

from __future__ import annotations

import math

import numpy as np

ABSOLUTE_GATE = -70.0  # LUFS: quieter blocks are silence
RELATIVE_GATE = 10.0  # LU under the level of the blocks above the absolute gate
BLOCK_HOPS = 4  # a 400 ms block is four 100 ms hops (BS.1770: 75 % overlap)
HOP_S = 0.1
SILENT_POWER = 1e-20  # floor of a mean square, so that log10 never sees 0


def k_weighting(rate: int) -> np.ndarray:
    """The two K-weighting biquads of BS.1770 (a +4 dB shelf over ~1.7 kHz, then a high pass
    at 38 Hz) as scipy second-order sections, for any sample rate (libebur128's formulas:
    at 48 kHz they give the standard's table)."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (gain_db / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a0,
        2 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        1.0,
        2 * (k * k - 1) / a0,
        (1 - k / q + k * k) / a0,
    ]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1 + k / q + k * k
    high_pass = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, high_pass])


def to_float(audio: np.ndarray) -> np.ndarray:
    """int16 PCM to -1..1 floats; floats are taken as they are."""
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return audio.astype(np.float64) / 32768
    return audio.astype(np.float64).reshape(-1)


def lufs(power: float | np.ndarray) -> np.ndarray:
    """Loudness of a mean square of K-weighted audio (one channel)."""
    return -0.691 + 10 * np.log10(np.maximum(power, SILENT_POWER))


def hop_powers(audio: np.ndarray, rate: int) -> np.ndarray:
    """Mean square of the K-weighted audio in 100 ms hops (a shorter tail is left out)."""
    from scipy.signal import sosfilt

    x = sosfilt(k_weighting(rate), to_float(audio))
    hop = round(HOP_S * rate)
    n = len(x) // hop
    return np.mean(np.square(x[: n * hop]).reshape(n, hop), axis=1) if n else np.zeros(0)


def block_powers(hops: np.ndarray) -> np.ndarray:
    """400 ms blocks every 100 ms from 100 ms hop powers. Audio under 400 ms is one block,
    so that a short take still has a level."""
    if len(hops) < BLOCK_HOPS:
        return np.array([np.mean(hops)]) if len(hops) else np.zeros(0)
    return np.convolve(hops, np.full(BLOCK_HOPS, 1 / BLOCK_HOPS), mode="valid")


def gated_lufs(blocks: np.ndarray) -> float:
    """Integrated loudness of block powers (BS.1770 gating); -inf when all is silence."""
    loud = lufs(blocks)
    kept = blocks[loud > ABSOLUTE_GATE]
    if not len(kept):
        return -math.inf
    relative = float(lufs(np.mean(kept))) - RELATIVE_GATE
    kept = blocks[(loud > ABSOLUTE_GATE) & (loud > relative)]
    return float(lufs(np.mean(kept)))


def integrated_lufs(audio: np.ndarray, rate: int) -> float:
    """BS.1770 integrated loudness of mono audio (int16 or -1..1 floats); -inf if silent."""
    return gated_lufs(block_powers(hop_powers(audio, rate)))


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(to_float(audio)))) if len(audio) else 0.0
    return 20 * math.log10(peak) if peak > 0 else -math.inf


def clipped_samples(audio: np.ndarray) -> int:
    """Samples at full scale: int16 at ±32767/-32768, floats at or past ±1."""
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return int(np.count_nonzero((audio >= 32767) | (audio <= -32768)))
    return int(np.count_nonzero(np.abs(audio) >= 1.0))


class LevelGuard:
    """Brings each piece of Jack's voice to `target` LUFS as it is generated, chunk by chunk
    (no latency: a chunk is measured before it is played, so it is its own look-ahead).

    The gain follows the loudness of the last `window_s` of the piece (BS.1770 gating, so
    pauses do not count), within `min_gain_db`..`max_gain_db`: long takes that start loud
    and fade by 7 dB (30/10) come out even. A chunk whose loudest 400 ms block goes more
    than `shout_db` over the target is turned down at once (a shout inside a long piece).
    The gain goes down within 50 ms and comes back up at `release_db_s`, always with a ramp;
    then a look-ahead limiter keeps the peaks under `ceiling_dbfs` instead of clipping them.
    Each piece starts afresh: begin().

    On the 44 pieces of the 30/10 test (2 s, 4 dB/s, 8 dB): spread between pieces 4.1 ->
    0.9 dB, largest jump between pieces 15.7 -> 2.6 dB, clipped samples 204 -> 0."""

    LIMIT_S = 0.005  # the limiter's gain ramps over this long around a peak

    def __init__(
        self,
        rate: int,
        target: float = -24.0,
        min_gain_db: float = -15.0,
        max_gain_db: float = 6.0,
        shout_db: float = 8.0,
        release_db_s: float = 4.0,
        ceiling_dbfs: float = -1.0,
        window_s: float = 2.0,
    ) -> None:
        from scipy.signal import sosfilt_zi

        self.rate = rate
        self.target = target
        self.window_s = window_s
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.shout_db = shout_db
        self.release_db_s = release_db_s
        self.ceiling = 10 ** (ceiling_dbfs / 20)
        self.hop = round(HOP_S * rate)
        self.sos = k_weighting(rate)
        self._zi_unit = sosfilt_zi(self.sos)
        self.begin()

    def begin(self) -> None:
        """A new piece: its loudness is measured from its first chunk."""
        self._zi = self._zi_unit * 0.0
        self._pending = np.zeros(0)  # K-weighted samples short of a 100 ms hop
        self._hops: list[float] = []
        self.gain_db: float | None = None  # None until the piece has had speech
        self._limit = 1.0  # the limiter's gain at the end of the last chunk
        self.raw_lufs = -math.inf  # the piece so far, before the gain
        self.raw_peak = 0.0
        self.raw_clipped = 0

    def report(self) -> dict[str, float | None]:
        """The piece so far as it came out of the model, for the protocol (JSON: no -inf)."""
        return {
            "raw_lufs": round(self.raw_lufs, 2) if math.isfinite(self.raw_lufs) else None,
            "raw_peak_dbfs": round(20 * math.log10(self.raw_peak), 2) if self.raw_peak else None,
            "raw_clipped": self.raw_clipped,
            "gain_db": None if self.gain_db is None else round(self.gain_db, 2),
        }

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """One chunk of float audio (-1..1, may go past it) in; the same length out."""
        from scipy.signal import sosfilt

        x = np.asarray(chunk, dtype=np.float64).reshape(-1)
        if not len(x):
            return x.astype(np.float32)
        self.raw_peak = max(self.raw_peak, float(np.max(np.abs(x))))
        self.raw_clipped += int(np.count_nonzero(np.abs(x) >= 1.0))
        weighted, self._zi = sosfilt(self.sos, x, zi=self._zi)
        before = len(self._hops)
        self._add_hops(weighted)
        blocks = block_powers(np.array(self._hops))
        self.raw_lufs = gated_lufs(blocks)
        new_blocks = blocks[max(len(blocks) - (len(self._hops) - before), 0) :]
        start = self.gain_db
        end = self._next_gain(blocks, new_blocks, len(x) / self.rate)
        if start is None:  # no speech before this chunk: nothing to ramp from
            start = end
        out = x * self._ramp(start, end, len(x))
        return self._limit_peaks(out).astype(np.float32)

    def _add_hops(self, weighted: np.ndarray) -> None:
        joined = np.concatenate([self._pending, weighted])
        n = len(joined) // self.hop
        if n:
            squares = np.square(joined[: n * self.hop]).reshape(n, self.hop)
            self._hops.extend(np.mean(squares, axis=1).tolist())
        self._pending = joined[n * self.hop :]

    def _next_gain(self, blocks: np.ndarray, new_blocks: np.ndarray, seconds: float) -> float:
        """The gain (dB) a chunk of `seconds` ends at."""
        recent = gated_lufs(blocks[-max(1, round(self.window_s / HOP_S)) :])
        if math.isinf(recent):  # silence so far: keep what there was
            return 0.0 if self.gain_db is None else self.gain_db
        wanted = self.target - recent
        if len(new_blocks):
            loudest = float(np.max(lufs(new_blocks)))
            wanted = min(wanted, self.target + self.shout_db - loudest)
        wanted = min(max(wanted, self.min_gain_db), self.max_gain_db)
        if self.gain_db is not None and wanted > self.gain_db:
            wanted = min(wanted, self.gain_db + self.release_db_s * seconds)
        self.gain_db = wanted
        return wanted

    def _ramp(self, start_db: float, end_db: float, n: int) -> np.ndarray:
        """Gain per sample. Down (a louder chunk) within the first 50 ms, so the loud part
        is already turned down; up slowly across the whole chunk."""
        if start_db == end_db:
            return np.full(n, 10 ** (start_db / 20))
        span = n if end_db > start_db else min(n, round(0.05 * self.rate))
        db = np.full(n, end_db)
        db[:span] = np.linspace(start_db, end_db, span, endpoint=False)
        return np.asarray(10 ** (db / 20))

    def _limit_peaks(self, x: np.ndarray) -> np.ndarray:
        """Look-ahead limiter within the chunk: the gain needed under each peak is spread
        over LIMIT_S before and after it (a minimum, then a moving average of the same
        width: it never rises above what any peak needs), so it ramps instead of clipping."""
        from scipy.ndimage import minimum_filter1d, uniform_filter1d

        width = max(3, round(self.LIMIT_S * self.rate) | 1)
        need = np.minimum(1.0, self.ceiling / np.maximum(np.abs(x), 1e-12))
        if self._limit < 1.0:  # the last chunk ended under a peak: come back up from there
            back = np.linspace(self._limit, 1.0, min(width, len(x)), endpoint=False)
            need[: len(back)] = np.minimum(need[: len(back)], back)
        if need.min() >= 1.0:
            self._limit = 1.0
            return x
        floor = minimum_filter1d(need, width, mode="nearest")
        gain = uniform_filter1d(floor, width, mode="nearest")
        self._limit = float(gain[-1])
        return np.clip(x * gain, -self.ceiling, self.ceiling)
