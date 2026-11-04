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
        self.closed = 0
        self.devices: list[int | None] = []  # the device asked for by every stream
        self.values: tuple[int, ...] = (0, 1, 2)  # the frames a stream delivers as it opens
        self.silent = 0  # how many streams, after the failures, open and deliver nothing
        self.status: str | None = None  # told to the callback with every frame
        self.rescan_failures = 0  # how many re-scans fail
        self.initializes = 0  # how many times PortAudio was started again
        self.initialize_failures = 0  # how many of those fail
        self.bad_devices: set[int] = set()  # the ones that never open
        sd = self

        class InputStream:
            def __init__(self, callback, device=None, **kwargs) -> None:
                self.callback, self.device = callback, device
                sd.devices.append(device)

            def __enter__(self):
                sd.opened += 1
                if sd.opened <= sd.failures or self.device in sd.bad_devices:
                    raise sd.PortAudioError("Error querying device -1")
                if sd.opened > sd.failures + sd.silent:
                    for i in sd.values:
                        self.callback(np.full((4, 1), i, np.int16), 4, None, sd.status)
                return self

            def __exit__(self, *exc) -> None:
                sd.closed += 1

        self.InputStream = InputStream

    def _terminate(self) -> None:
        self.rescans += 1
        if self.rescan_failures:
            self.rescan_failures -= 1
            raise self.PortAudioError("Error terminating")

    def _initialize(self) -> None:
        self.initializes += 1
        if self.initialize_failures:
            self.initialize_failures -= 1
            raise self.PortAudioError("Error initializing")


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


async def take(n: int, mic: capture.AudioCapture | None = None) -> list[np.ndarray]:
    mic = mic or capture.AudioCapture(16000, 4, device="C270")
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


def test_the_microphone_opened_is_told_by_its_own_name(fake_sd, monkeypatch) -> None:
    """AutoGain.use: every microphone keeps its own gain, whatever host API opened it."""
    fake_sd(failures=0)
    monkeypatch.setattr(
        capture, "describe_device", lambda index, kind: "Cuffie (Wireless Head) (Windows WASAPI)"
    )
    mic = capture.AudioCapture(16000, 4, device="C270")
    told: list[str] = []
    mic.on_device = lambda name: (told.append(name), mic.set_gain_db(12.0))

    async def first() -> np.ndarray:
        async for frame in mic.frames():
            return frame
        raise AssertionError("no frame")

    asyncio.run(first())
    assert told == ["Cuffie (Wireless Head)"]
    assert mic.gain > 3.9  # the gain it set is the one the frames get


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


def test_auto_microphone_moves_only_to_a_louder_one(fake_sd, monkeypatch) -> None:
    """A quiet microphone is kept unless another hears more, and the switch re-scans nothing
    (sd._terminate() also broke the speaker's stream: PaErrorCode -9988 on 24/10)."""
    sd = fake_sd(failures=0)
    monkeypatch.setattr(capture, "QUIET_S", 0.0)  # every frame of the fake is quiet
    chosen = iter([3, 3, 5, 5, 5, 5])
    asked: list[int | None] = []

    def chooser(current: int | None) -> int | None:
        asked.append(current)
        return next(chosen)

    mic = capture.AudioCapture(16000, 4, device="auto", chooser=chooser)

    async def run() -> None:
        n = 0
        async for _ in mic.frames():
            n += 1
            if n == 4:
                return

    asyncio.run(run())
    assert asked[:3] == [None, 3, 3]  # at start, then twice from microphone 3
    assert sd.opened == 2 and sd.rescans == 0  # 3 kept once, then 5 opened


@pytest.mark.parametrize("louder", [5, 0])  # 0 is a real device
def test_a_louder_microphone_that_fails_to_open_is_not_tried_again(
    fake_sd, monkeypatch, louder: int
) -> None:
    """It is opened as chosen, without asking again; after it fails, the chooser starts over."""
    sd = fake_sd(failures=0)
    sd.bad_devices = {louder}
    monkeypatch.setattr(capture, "QUIET_S", 0.0)  # every frame of the fake is quiet
    chosen = iter([3, louder, 7])
    asked: list[int | None] = []

    def chooser(current: int | None) -> int | None:
        asked.append(current)
        return next(chosen)

    mic = capture.AudioCapture(16000, 4, device="auto", chooser=chooser)
    asyncio.run(asyncio.wait_for(take(2, mic), 5))  # kept on the bad one, it would never end
    assert sd.devices == [3, louder, 7]
    assert asked == [None, 3, None]


def test_auto_probes_after_quiet_s_with_no_loud_frame(fake_sd, monkeypatch) -> None:
    """A loud frame restarts the wait, and so does a probe. 40 s pass between two frames."""
    sd = fake_sd(failures=0)
    sd.values = (500, 0, 0, 0, 0, 0)
    now = [0.0]
    monkeypatch.setattr(capture, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
    asked: list[tuple[int | None, float]] = []

    def chooser(current: int | None) -> int:
        asked.append((current, now[0]))
        return 3

    mic = capture.AudioCapture(16000, 4, device="auto", chooser=chooser)

    async def run() -> None:
        n = 0
        async for _ in mic.frames():
            n += 1
            if n == len(sd.values):
                return
            now[0] += 40.0

    asyncio.run(run())
    assert asked == [(None, 0.0), (3, 120.0), (3, 200.0)]
    assert sd.opened == 1


@pytest.mark.parametrize(
    ("current", "levels", "expected", "logged"),
    [
        (None, {1: -40.0, 2: -20.0}, 2, "1 -40 dBFS, 2 -20 dBFS"),  # someone talks into 2
        (None, {2: -35.0, 1: -40.0}, 1, "1 -40 dBFS, 2 -35 dBFS"),  # room noise: Windows' stays
        (2, {1: -40.0, 2: -35.0}, 2, "1 -40 dBFS, 2 -35 dBFS"),  # the one in use, not Windows'
        (None, {}, 1, "nessuno"),  # nothing could be opened
    ],
)
def test_choose_keeps_the_microphone_in_use_or_windows_one(
    monkeypatch, caplog, current, levels, expected, logged
) -> None:
    monkeypatch.setattr(capture, "default_device", lambda kind: {"input": 1}[kind])
    monkeypatch.setattr(capture, "probe_inputs", lambda rate: {16000: levels}[rate])
    mic = capture.AudioCapture(16000, 4, device="auto")
    with caplog.at_level(logging.INFO, logger=capture.__name__):
        assert mic.chooser(current) == expected
    assert [r.getMessage() for r in caplog.records] == [f"Microfoni sentiti: {logged}"]


# -- what happens when the microphone fails --


def test_a_failed_open_moves_to_portaudio_default_until_windows_changes(
    fake_sd, monkeypatch
) -> None:
    """No device configured: the WASAPI default failed, so PortAudio's own (no device given) is
    used, until Windows changes microphone and the choice is made again."""
    sd = fake_sd(failures=1)
    monkeypatch.setattr(capture, "FOLLOW_S", 0.0)
    names = iter(["Webcam C270", "Webcam C270"])
    mic = capture.AudioCapture(16000, 4, device="", windows_default=lambda: next(names, "Cuffie"))
    asyncio.run(take(1, mic))
    assert sd.devices == [1, None, 1]


def test_a_microphone_that_stops_delivering_is_opened_again(fake_sd, caplog) -> None:
    """No frame for 2 s (TimeoutError): the same device again, not PortAudio's default."""
    sd = fake_sd(failures=0)
    sd.silent = 1
    mic = capture.AudioCapture(16000, 4, device="", windows_default=lambda: "Webcam C270")
    with caplog.at_level(logging.INFO, logger=capture.__name__):
        frames = asyncio.run(take(1, mic))
    assert [int(f[0]) for f in frames] == [0]
    assert sd.devices == [1, 1] and sd.rescans == 1
    assert sd.closed == 2  # the silent one at once, the other as asyncio.run() ends
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "non disponibile" in warnings[0]
    assert any("di nuovo disponibile" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "error",
    [lambda sd: ValueError("nessun microfono"), lambda sd: sd.PortAudioError("host API")],
    ids=["ValueError", "PortAudioError"],
)
def test_a_microphone_that_cannot_be_picked_is_picked_again(fake_sd, monkeypatch, error) -> None:
    """Nothing was opened, so nothing says the WASAPI default is broken: PortAudio's stays out."""
    sd = fake_sd(failures=0)
    failures = [error(sd)]

    def pick(spec, kind, default=None):
        if failures:
            raise failures.pop()
        return 1

    monkeypatch.setattr(capture, "pick_device", pick)
    mic = capture.AudioCapture(16000, 4, device="", windows_default=lambda: "Webcam C270")
    asyncio.run(take(1, mic))
    assert sd.devices == [1] and sd.rescans == 1


def test_a_late_consumer_loses_the_oldest_frames(fake_sd) -> None:
    fake_sd(failures=0)
    mic = capture.AudioCapture(16000, 4, device="C270", queue_size=2)
    frames = asyncio.run(take(2, mic))
    assert [int(f[0]) for f in frames] == [1, 2]
    assert mic.overflows == 1


def test_the_stream_status_is_logged_and_the_frame_kept(fake_sd, caplog) -> None:
    sd = fake_sd(failures=0)
    sd.status = "input overflow"
    with caplog.at_level(logging.DEBUG, logger=capture.__name__):
        frames = asyncio.run(take(3))
    assert [int(f[0]) for f in frames] == [0, 1, 2]
    told = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(told) == 3 and all("input overflow" in message for message in told)


async def test_closing_the_frames_closes_the_stream(fake_sd) -> None:
    sd = fake_sd(failures=0)
    frames = capture.AudioCapture(16000, 4, device="C270").frames()
    await frames.__anext__()
    assert sd.closed == 0
    await frames.aclose()
    assert sd.closed == 1 and sd.opened == 1 and sd.rescans == 0


def test_a_microphone_is_told_only_once_open_and_logged_with_the_gain_it_set(
    fake_sd, caplog
) -> None:
    fake_sd(failures=1)  # the first open fails: nobody hears of that one
    mic = capture.AudioCapture(16000, 4, device="C270")
    told: list[str] = []
    mic.on_device = lambda name: (told.append(name), mic.set_gain_db(12.0))
    with caplog.at_level(logging.INFO, logger=capture.__name__):
        asyncio.run(take(1, mic))
    assert told == ["Webcam C270"]
    assert any("guadagno +12 dB" in r.getMessage() for r in caplog.records)


def test_a_reopened_stream_goes_on_from_the_same_queue(fake_sd, monkeypatch) -> None:
    """The frame a change of microphone is found on is dropped; the ones behind it, already
    queued, still come out ahead of the new stream's. Those old frames are a known oddity, not
    a wish: drain the queue on purpose and this test changes with it."""
    fake_sd(failures=0)
    monkeypatch.setattr(capture, "FOLLOW_S", 0.0)
    names = iter(["Webcam C270"])
    mic = capture.AudioCapture(16000, 4, device="", windows_default=lambda: next(names, "Cuffie"))
    frames = asyncio.run(take(5, mic))
    assert [int(f[0]) for f in frames] == [1, 2, 0, 1, 2]


def test_a_rescan_that_fails_after_a_change_of_microphone_is_tried_again(
    fake_sd, monkeypatch, caplog
) -> None:
    """That re-scan is inside the try: PortAudio failing there is one more microphone problem.
    Which device it reopens on is not asserted: this failure also moves it to PortAudio's own
    default, though the WASAPI one opened fine (a side effect, not a decision)."""
    sd = fake_sd(failures=0)
    sd.rescan_failures = 1
    monkeypatch.setattr(capture, "FOLLOW_S", 0.0)
    names = iter(["Webcam C270"])
    mic = capture.AudioCapture(16000, 4, device="", windows_default=lambda: next(names, "Cuffie"))
    with caplog.at_level(logging.WARNING, logger=capture.__name__):
        frames = asyncio.run(take(1, mic))
    assert len(frames) == 1 and sd.opened == 2 and sd.rescans == 2
    assert ["Microfono non disponibile" in r.getMessage() for r in caplog.records] == [True]


def take_or_error(sd: FakeSd, n: int = 1) -> tuple[list[np.ndarray], Exception | None]:
    """What frames() delivered, and the error that ended it (None: it was still going)."""
    try:
        return asyncio.run(take(n)), None
    except sd.PortAudioError as exc:
        return [], exc


def test_a_failing_terminate_does_not_skip_the_initialize(fake_sd) -> None:
    """PortAudio left uninitialized by an earlier failure makes every _terminate raise: the
    _initialize after it is the only way back, so it must always run."""
    sd = fake_sd(failures=1)
    sd.rescan_failures = 1
    frames, error = take_or_error(sd)
    assert error is None
    assert len(frames) == 1 and sd.initializes == 1


def test_a_failing_initialize_is_tried_again(fake_sd) -> None:
    sd = fake_sd(failures=2)
    sd.initialize_failures = 1
    frames, error = take_or_error(sd)
    assert error is None
    assert len(frames) == 1 and sd.initializes == 2


def test_a_rescan_that_keeps_failing_warns_like_an_unplugged_microphone(
    fake_sd, caplog: pytest.LogCaptureFixture
) -> None:
    sd = fake_sd(failures=65)
    sd.rescan_failures = 65
    sd.initialize_failures = 29  # one more is the limit: the capture ends
    with caplog.at_level(logging.INFO, logger=capture.__name__):
        frames, error = take_or_error(sd, 3)
    assert error is None and len(frames) == 3
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 3  # as with a working re-scan: not one more line per try


def test_a_rescan_that_never_recovers_raises_the_real_error_out_of_frames(
    fake_sd, caplog: pytest.LogCaptureFixture
) -> None:
    """Claudio deaf and mute with a green tray: only a new process can cure it, and the error
    leaving frames() is what makes the supervisor start one."""
    sd = fake_sd(failures=1000)
    sd.initialize_failures = 1000
    with caplog.at_level(logging.WARNING, logger=capture.__name__):
        frames, error = take_or_error(sd)
    assert str(error) == "Error initializing"
    assert frames == [] and sd.initializes == 30
    real = [r.getMessage() for r in caplog.records if "Error initializing" in r.getMessage()]
    assert len(real) == 1


def test_rescans_failing_short_of_the_limit_keep_going_and_a_success_resets_the_count(
    fake_sd,
) -> None:
    sd = fake_sd(failures=30)
    sd.initialize_failures = 29
    mic = capture.AudioCapture(16000, 4, device="C270")
    assert len(asyncio.run(take(1, mic))) == 1
    sd.opened, sd.initialize_failures = 0, 29  # a count not reset would reach the limit at once
    assert len(asyncio.run(take(1, mic))) == 1
    assert sd.initializes == 60


def test_a_terminate_that_always_fails_never_gives_up(fake_sd) -> None:
    """It always raises after a failed _initialize, and the next _initialize may still work."""
    sd = fake_sd(failures=100)
    sd.rescan_failures = 1000
    frames, error = take_or_error(sd)
    assert error is None and len(frames) == 1 and sd.initializes == 100


def test_one_failed_rescan_after_a_long_absence_does_not_give_up(fake_sd) -> None:
    """A microphone unplugged for an hour: its re-scans work, so many retries are no reason."""
    sd = fake_sd(failures=100)
    initialize = sd._initialize

    def fails_once_late() -> None:
        sd.initialize_failures = 1 if sd.initializes == 50 else 0
        initialize()

    sd._initialize = fails_once_late
    frames, error = take_or_error(sd)
    assert error is None and len(frames) == 1 and sd.initializes == 100
