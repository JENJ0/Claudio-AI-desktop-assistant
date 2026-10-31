"""AudioPlayer following Windows' default output, with a fake sounddevice stream."""

import threading
from typing import ClassVar

import numpy as np
import pytest

import claudio.audio.player as player_module
from claudio.audio.player import DUCK_RAMP_S, AudioPlayer

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


# --- a reply lowered while a voice talks over it (ducking) fades, never steps -------------

RATE = 10000  # a reply is queued in slices of 1000 samples
STEP = 1 / (DUCK_RAMP_S * RATE)  # the most the gain may move from one sample to the next
TRUNCATION = 2e-4  # int16 samples of 10000: each gain read back is off by less than 1e-4


class Speakers:
    """The stream: keeps every sample it is given."""

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def write(self, data: np.ndarray) -> None:
        self.chunks.append(data.copy())

    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks)


def reply_player(now: list[float] | None = None) -> tuple[AudioPlayer, Speakers]:
    """The clock stands still unless `now` moves it: the stream never runs dry."""
    clock = now if now is not None else [0.0]
    player = AudioPlayer(RATE, device=0, clock=lambda: clock[0])
    speakers = Speakers()
    player._stream = speakers
    return player, speakers


def test_lowering_a_reply_fades_instead_of_stepping() -> None:
    """The volume dropped from 1 to 0.3 (-10.5 dB) at a slice edge when a voice started."""
    player, speakers = reply_player()
    cancel = threading.Event()
    player.write(np.full(1000, 10000, np.int16), cancel=cancel)
    player.gain = 0.3  # a voice starts over the reply
    player.write(np.full(2000, 10000, np.int16), cancel=cancel)
    gains = speakers.audio() / 10000
    assert (gains[:1000] == 1.0).all() and gains[1000] < 1.0  # it starts at the next slice
    assert np.abs(np.diff(gains)).max() <= STEP + TRUNCATION
    down = 1000 + round(0.7 * DUCK_RAMP_S * RATE) + 1  # a swing of 0.7 takes 56 ms
    assert (gains[down:] == 0.3).all()


def test_at_full_volume_a_reply_is_untouched() -> None:
    """Bit for bit, also once a duck is over; and a fade never overflows int16."""
    speech = np.random.default_rng(7).integers(-32768, 32768, 3000).astype(np.int16)
    speech[:2] = (-32768, 32767)
    player, speakers = reply_player()
    cancel = threading.Event()
    player.write(speech, cancel=cancel)
    assert np.array_equal(speakers.audio(), speech)
    player.gain = 0.3
    player.write(speech, cancel=cancel)
    player.gain = 1.0  # "not for me": back up, fading
    player.write(speech, cancel=cancel)
    faded = speakers.audio()[3000:].astype(np.int32)
    original = np.tile(speech, 2).astype(np.int32)
    assert (np.abs(faded) <= np.abs(original)).all() and (faded * original >= 0).all()
    speakers.chunks.clear()
    player.write(speech, cancel=cancel)
    player.gain = 2.0  # never louder than the voice was made
    player.write(speech, cancel=cancel)
    assert np.array_equal(speakers.audio(), np.tile(speech, 2))


def test_beeps_are_never_lowered() -> None:
    player, speakers = reply_player()
    player.gain = 0.3
    beep = np.full(1500, 8000, np.int16)
    player.write(beep)  # no cancel: a beep, in one piece
    assert np.array_equal(speakers.audio(), beep)


@pytest.mark.parametrize(("ramp_s", "pieces"), [(DUCK_RAMP_S, [200] * 4), (0.25, [3000])])
def test_a_fade_goes_on_from_where_the_last_slice_left_it(
    monkeypatch, ramp_s: float, pieces: list[int]
) -> None:
    """Jack hands a sentence over in pieces shorter than the fade; a fade longer than one
    slice carries on in the next. Either way it never starts over nor jumps."""
    monkeypatch.setattr(player_module, "DUCK_RAMP_S", ramp_s)
    player, speakers = reply_player()
    cancel = threading.Event()
    player.write(np.full(300, 10000, np.int16), cancel=cancel)
    player.gain = 0.3
    for n in pieces:
        player.write(np.full(n, 10000, np.int16), cancel=cancel)
    gains = speakers.audio() / 10000
    assert np.abs(np.diff(gains)).max() <= 1 / (ramp_s * RATE) + TRUNCATION
    fading = int(((gains > 0.3) & (gains < 1.0)).sum())
    assert abs(fading - 0.7 * ramp_s * RATE) <= 2 and gains[-1] == 0.3


@pytest.mark.parametrize("gain", [1.0, 0.3])
def test_after_a_silence_the_volume_asked_applies_at_once(gain: float) -> None:
    """A reply cut halfway through a fade never makes the next one start lowered: with
    nothing playing there is no sound for the volume to jump in."""
    now = [0.0]
    player, speakers = reply_player(now)
    cancel = threading.Event()
    player.write(np.full(1000, 10000, np.int16), cancel=cancel)
    player.gain = 0.3
    player.write(np.full(300, 10000, np.int16), cancel=cancel)  # cut on the way down
    player.gain = gain  # full volume for the next reply (speak), or lowered in a pause
    now[0] += 1.0  # what was queued has come out: silence
    speakers.chunks.clear()
    player.write(np.full(1000, 10000, np.int16), cancel=cancel)
    assert (speakers.audio() == int(10000 * gain)).all()
