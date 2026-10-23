"""AudioCapture with a fake sounddevice: the microphone unplugged and plugged back in."""

from __future__ import annotations

import asyncio
import logging
import sys
import types

import numpy as np
import pytest

from claudio.audio import capture


class FakeSd(types.ModuleType):
    class PortAudioError(Exception):
        pass

    def __init__(self, failures: int) -> None:
        super().__init__("sounddevice")
        self.failures, self.opened, self.rescans = failures, 0, 0
        sd = self

        class InputStream:
            def __init__(self, callback, **kwargs) -> None:
                self.callback = callback

            def __enter__(self):
                sd.opened += 1
                if sd.opened <= sd.failures:
                    raise sd.PortAudioError("Error querying device -1")
                for i in range(3):
                    self.callback(np.full((4, 1), i, np.int16), 4, None, None)
                return self

            def __exit__(self, *exc) -> None:
                return None

        self.InputStream = InputStream

    def _terminate(self) -> None:
        self.rescans += 1

    def _initialize(self) -> None:
        pass


@pytest.fixture
def fake_sd(monkeypatch: pytest.MonkeyPatch):
    def install(failures: int) -> FakeSd:
        sd = FakeSd(failures)
        monkeypatch.setitem(sys.modules, "sounddevice", sd)
        monkeypatch.setattr(capture, "pick_device", lambda spec, kind, default=None: 1)
        monkeypatch.setattr(capture, "describe_device", lambda index, kind: "Webcam C270")
        monkeypatch.setattr(capture, "stream_settings", lambda index: None)
        monkeypatch.setattr(capture, "RETRY_S", 0.0)
        return sd

    return install


async def take(n: int) -> list[np.ndarray]:
    mic = capture.AudioCapture(16000, 4, device="C270")
    frames = []
    async for frame in mic.frames():
        frames.append(frame)
        if len(frames) == n:
            break
    return frames


def test_unplugged_microphone_comes_back(fake_sd, caplog: pytest.LogCaptureFixture) -> None:
    sd = fake_sd(failures=65)  # about a minute unplugged
    with caplog.at_level(logging.INFO, logger=capture.__name__):
        frames = asyncio.run(take(3))
    assert [int(f[0]) for f in frames] == [0, 1, 2]
    assert sd.rescans == 65  # the device list is read again at every try
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    # One warning when it goes missing, one every 30 tries after: not one per second.
    assert len(warnings) == 3
    assert "non disponibile" in warnings[0] and "ancora non disponibile" in warnings[1]
    assert any("di nuovo disponibile" in r.getMessage() for r in caplog.records)


def test_working_microphone_logs_no_warning(fake_sd, caplog: pytest.LogCaptureFixture) -> None:
    fake_sd(failures=0)
    with caplog.at_level(logging.INFO, logger=capture.__name__):
        asyncio.run(take(2))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("di nuovo" in r.getMessage() for r in caplog.records)


def test_follows_the_windows_microphone(fake_sd, monkeypatch, caplog) -> None:
    """No device configured: when Windows switches microphone the stream reopens on it."""
    sd = fake_sd(failures=0)
    monkeypatch.setattr(capture, "FOLLOW_S", 0.0)
    names = iter(["Webcam C270", "Webcam C270", "Cuffie", "Cuffie", "Cuffie", "Cuffie"])
    opened_on: list[str | None] = []

    def pick(spec, kind, default=None):
        opened_on.append(default)
        return 1

    monkeypatch.setattr(capture, "pick_device", pick)
    mic = capture.AudioCapture(16000, 4, device="", windows_default=lambda: next(names, "Cuffie"))

    async def run() -> int:
        n = 0
        async for _ in mic.frames():
            n += 1
            if n == 4:
                return n
        return n

    with caplog.at_level(logging.INFO, logger=capture.__name__):
        asyncio.run(run())
    assert opened_on == ["Webcam C270", "Cuffie"]
    assert sd.opened == 2 and sd.rescans == 1
    assert any("cambiato" in r.getMessage() for r in caplog.records)


def test_configured_microphone_is_not_followed() -> None:
    assert capture.AudioCapture(16000, 4, device="C270").windows_default is None
