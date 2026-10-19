from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from claudio.config import PROJECT_ROOT, Settings, VadSettings
from claudio.router.matcher import CommandMatcher, load_commands

COMMANDS_FILE = PROJECT_ROOT / "config" / "commands.toml"


class FakeVad:
    """Speech probability = 0.9 when the window is loud, 0.0 when silent."""

    window_size = 512

    def __init__(self) -> None:
        self.resets = 0

    def __call__(self, window: np.ndarray) -> float:
        return 0.9 if float(np.abs(window).max()) > 0.1 else 0.0

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture(autouse=True)
def ignore_the_developers_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The .env on a developer's PC (their voice, their microphone) must not change what a
    test expects of the shipped settings. Tests that need a value set it in the environment."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def fake_vad() -> FakeVad:
    return FakeVad()


@pytest.fixture
def vad_settings() -> VadSettings:
    return VadSettings(
        threshold=0.5,
        min_speech_ms=96,
        end_silence_ms=400,
        pre_roll_ms=300,
        no_speech_timeout_s=3.0,
        max_utterance_s=8.0,
    )


@pytest.fixture(scope="session")
def matcher() -> CommandMatcher:
    return CommandMatcher(load_commands(COMMANDS_FILE), threshold=85)


def frames(kind: str, ms: int, block: int = 1280) -> list[np.ndarray]:
    """int16 frames of silence or 'speech' (a loud tone)."""
    n = 16000 * ms // 1000 // block
    if kind == "silence":
        return [np.zeros(block, dtype=np.int16) for _ in range(n)]
    t = np.arange(block) / 16000
    tone = (np.sin(2 * np.pi * 220 * t) * 16000).astype(np.int16)
    return [tone.copy() for _ in range(n)]


@pytest.fixture
def make_frames() -> Callable[[str, int], list[np.ndarray]]:
    return frames
