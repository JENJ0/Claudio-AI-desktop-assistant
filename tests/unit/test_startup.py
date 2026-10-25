"""Startup: the models load together, and what each one cost is logged."""

from __future__ import annotations

import logging
import time

import pytest

from claudio.__main__ import load_all


def sleeper(value: str, seconds: float = 0.3):
    def load() -> str:
        time.sleep(seconds)
        return value

    return load


def test_loaders_run_at_the_same_time(caplog) -> None:
    caplog.set_level(logging.INFO, logger="claudio")
    start = time.perf_counter()
    loaded = load_all({"whisper": sleeper("w"), "voce": sleeper("v"), "claude": sleeper("c")})
    assert time.perf_counter() - start < 0.8  # 3 x 0.3 s one after the other would be 0.9
    assert loaded == {"whisper": "w", "voce": "v", "claude": "c"}
    assert "whisper 0.3 s" in caplog.text and "voce 0.3 s" in caplog.text


def test_sequential_start_still_works() -> None:
    start = time.perf_counter()
    loaded = load_all({"a": sleeper("a", 0.1), "b": sleeper("b", 0.1)}, parallel=False)
    assert time.perf_counter() - start >= 0.2
    assert loaded == {"a": "a", "b": "b"}


@pytest.mark.parametrize("parallel", [True, False])
def test_a_failing_loader_raises_after_the_others_end(parallel: bool) -> None:
    ended: list[str] = []

    def other() -> str:
        time.sleep(0.2)
        ended.append("other")
        return "ok"

    def broken() -> str:
        raise FileNotFoundError("models/whisper")

    with pytest.raises(FileNotFoundError):
        load_all({"broken": broken, "other": other}, parallel=parallel)
    assert ended == (["other"] if parallel else [])
