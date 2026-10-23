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


# -- input_device = "auto": the microphone that hears the most (JenJo, 23/10) --


def test_choose_input_keeps_windows_one_unless_another_is_clearly_louder() -> None:
    from claudio.audio.devices import choose_input

    assert choose_input({1: -40.0, 2: -35.0}, default=1) == 1  # room noise: no jump
    assert choose_input({1: -40.0, 2: -20.0}, default=1) == 2  # someone talks into 2
    assert choose_input({1: -120.0, 2: -60.0}, default=1) == 2  # 1 is switched off
    assert choose_input({1: -120.0, 2: -120.0}, default=1) == 1  # nothing alive: Windows'
    assert choose_input({}, default=None) is None


def test_candidates_skip_loopbacks_and_prefer_wasapi() -> None:
    from claudio.audio.devices import candidate_inputs

    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    devices = [
        {"name": "Microfono (C270)", "hostapi": 0, "max_input_channels": 1},
        {"name": "Microfono (C270)", "hostapi": 1, "max_input_channels": 1},
        {"name": "Missaggio stereo (Realtek)", "hostapi": 1, "max_input_channels": 2},
        {"name": "Cuffie", "hostapi": 1, "max_input_channels": 0},
        {"name": "Microfono (Headset)", "hostapi": 1, "max_input_channels": 1},
    ]
    assert candidate_inputs(devices, hostapis) == [1, 4]


def test_auto_microphone_is_chosen_again_when_it_goes_silent(fake_sd, monkeypatch) -> None:
    sd = fake_sd(failures=0)
    monkeypatch.setattr(capture, "DEAD_S", 0.0)  # every frame of the fake is quiet
    chosen = iter([3, 5, 5, 5])
    picks: list[int] = []

    def chooser() -> int | None:
        picks.append(next(chosen))
        return picks[-1]

    mic = capture.AudioCapture(16000, 4, device="auto", chooser=chooser)

    async def run() -> None:
        n = 0
        async for _ in mic.frames():
            n += 1
            if n == 2:
                return

    asyncio.run(run())
    assert picks[:2] == [3, 5] and sd.opened >= 2
