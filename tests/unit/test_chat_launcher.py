import sys
from pathlib import Path

from claudio.__main__ import startup_notices
from claudio.chat.launcher import ChatLauncher, window_command
from claudio.config import load_settings


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.pid = 42

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float) -> int:
        return 0

    def kill(self) -> None:  # pragma: no cover - only when terminate hangs
        self.returncode = -9


def test_window_command_uses_the_same_python_and_config(tmp_path: Path) -> None:
    assert window_command() == [sys.executable, "-m", "claudio", "--ui"]
    config = tmp_path / "my.toml"
    assert window_command(config)[-2:] == ["--config", str(config.resolve())]


def test_launcher_opens_once_reopens_after_user_close_and_closes() -> None:
    started: list[FakeProcess] = []

    def popen(command: object) -> FakeProcess:
        started.append(FakeProcess())
        return started[-1]

    chat = ChatLauncher(["claudio", "--ui"], popen)
    assert not chat.close()  # nothing open yet
    assert chat.open() and chat.is_open
    assert not chat.open()  # already open: no second window
    started[0].returncode = 0  # the user closed it with the X
    assert chat.open() and len(started) == 2
    assert chat.close() and not chat.is_open
    assert started[1].returncode == 0


def test_startup_notices_only_when_degraded() -> None:
    settings = load_settings()
    jack = settings.model_copy(
        update={
            "tts": settings.tts.model_copy(update={"engine": "jack"}),
            "stt": settings.stt.model_copy(update={"device": "auto"}),
        }
    )
    assert startup_notices(jack, jack_running=True, stt_device="cuda") == []
    notices = startup_notices(jack, jack_running=False, stt_device="cpu")
    assert len(notices) == 3 and "Jack" in notices[0] and "log" in notices[-1]
    piper = jack.model_copy(update={"tts": settings.tts.model_copy(update={"engine": "piper"})})
    assert startup_notices(piper, jack_running=False, stt_device="cuda") == []
