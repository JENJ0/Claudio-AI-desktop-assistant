import numpy as np
import pytest

from claudio.audio.ring_buffer import RingBuffer


def test_partial_fill_keeps_order() -> None:
    rb = RingBuffer(5)
    rb.append(np.array([1, 2], dtype=np.float32))
    assert rb.get().tolist() == [1, 2]
    assert len(rb) == 2


def test_wraps_around_and_keeps_latest() -> None:
    rb = RingBuffer(5)
    for chunk in ([1, 2, 3], [4, 5, 6], [7]):
        rb.append(np.array(chunk, dtype=np.float32))
    assert rb.get().tolist() == [3, 4, 5, 6, 7]


def test_chunk_bigger_than_capacity() -> None:
    rb = RingBuffer(3)
    rb.append(np.arange(10, dtype=np.float32))
    assert rb.get().tolist() == [7, 8, 9]


def test_clear_and_zero_capacity() -> None:
    rb = RingBuffer(3)
    rb.append(np.ones(2, dtype=np.float32))
    rb.clear()
    assert len(rb) == 0 and rb.get().size == 0
    zero = RingBuffer(0)
    zero.append(np.ones(4, dtype=np.float32))
    assert zero.get().size == 0
    with pytest.raises(ValueError):
        RingBuffer(-1)
