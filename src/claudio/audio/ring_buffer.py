"""Fixed-size sample buffer used for the pre-roll before speech starts."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int, dtype: type = np.float32) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._buf: np.ndarray = np.zeros(capacity, dtype=dtype)
        self._write = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def append(self, samples: np.ndarray) -> None:
        if self.capacity == 0:
            return
        samples = samples[-self.capacity :]
        n = len(samples)
        end = self._write + n
        if end <= self.capacity:
            self._buf[self._write : end] = samples
        else:
            first = self.capacity - self._write
            self._buf[self._write :] = samples[:first]
            self._buf[: n - first] = samples[first:]
        self._write = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def get(self) -> np.ndarray:
        """Contents in chronological order (a copy)."""
        if self._size < self.capacity:
            return self._buf[: self._size].copy()
        return np.concatenate((self._buf[self._write :], self._buf[: self._write]))

    def clear(self) -> None:
        self._write = 0
        self._size = 0
