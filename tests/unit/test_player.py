"""AudioPlayer following Windows' default output, with a fake sounddevice stream."""

from typing import ClassVar

import numpy as np
import pytest

import claudio.audio.player as player_module
from claudio.audio.player import AudioPlayer

DEVICES = {"Cuffie (Wireless Stereo Headset)": 5, "Odyssey G80SD (NVIDIA High Definition)": 7}


class FakeStream:
    opened: ClassVar[list["FakeStream"]] = []

    def __init__(self, device=None, **kwargs) -> None:
        self.device, self.written, self.closed = device, 0, False
        FakeStream.opened.append(self)

    def start(self) -> None: ...

    def write(self, audio: np.ndarray) -> None:
        self.written += len(audio)

    def stop(self) -> None: ...

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_audio(monkeypatch):
    import sounddevice as sd

    FakeStream.opened = []
    monkeypatch.setattr(sd, "OutputStream", FakeStream)
    monkeypatch.setattr(player_module, "describe_device", lambda index, kind: f"device {index}")
    monkeypatch.setattr(player_module, "stream_settings", lambda index: None)
    monkeypatch.setattr(player_module, "match_endpoint", lambda name, kind: DEVICES.get(name))
    monkeypatch.setattr(
        player_module,
        "pick_output",
        lambda spec, name: DEVICES.get(name, 1) if spec == "" else 9,
    )
    return FakeStream


def test_follows_the_windows_default_between_sentences(fake_audio, caplog) -> None:
    current = ["Cuffie (Wireless Stereo Headset)"]
    player = AudioPlayer(22050, windows_default=lambda: current[0])
    player.start()
    player.write(np.zeros(100, np.int16))
    assert [s.device for s in fake_audio.opened] == [5]

    current[0] = "Odyssey G80SD (NVIDIA High Definition)"  # changed in Windows while running
    player.write(np.zeros(100, np.int16))
    assert [s.device for s in fake_audio.opened] == [5, 7]
    assert fake_audio.opened[0].closed and fake_audio.opened[1].written == 100

    current[0] = "Altoparlanti USB (plugged in later)"  # unknown to PortAudio: stay, warn once
    with caplog.at_level("WARNING"):
        player.write(np.zeros(100, np.int16))
        player.write(np.zeros(100, np.int16))
    assert len(fake_audio.opened) == 2 and fake_audio.opened[1].written == 300
    assert sum("riavvia CLAUDIO" in r.getMessage() for r in caplog.records) == 1


def test_configured_device_is_not_moved(fake_audio) -> None:
    asked: list[int] = []

    def windows_default() -> str:
        asked.append(1)
        return "Odyssey G80SD (NVIDIA High Definition)"

    player = AudioPlayer(22050, device="Cuffie", windows_default=windows_default)
    player.start()
    player.write(np.zeros(100, np.int16))
    assert [s.device for s in fake_audio.opened] == [9] and asked == []


def test_unreadable_default_keeps_playing(fake_audio) -> None:
    player = AudioPlayer(22050, windows_default=lambda: None)  # off Windows, or Core Audio failed
    player.write(np.zeros(100, np.int16))
    player.write(np.zeros(100, np.int16))
    assert [s.device for s in fake_audio.opened] == [1] and fake_audio.opened[0].written == 200


def test_wasapi_default_that_does_not_open_falls_back(fake_audio, monkeypatch) -> None:
    import sounddevice as sd

    class Picky(FakeStream):
        def __init__(self, device=None, **kwargs) -> None:
            if device is not None:
                raise sd.PortAudioError("Invalid device")
            super().__init__(device, **kwargs)

    monkeypatch.setattr(sd, "OutputStream", Picky)
    player = AudioPlayer(22050, windows_default=lambda: "Cuffie (Wireless Stereo Headset)")
    player.start()
    player.write(np.zeros(100, np.int16))
    assert [s.device for s in fake_audio.opened] == [None]
