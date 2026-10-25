"""The "claudio" wake word missed on JenJo's 25/10 test: Whisper heard "l'audio" because the
check got only 1.5 s and the model fires late, on the word after the name."""

from __future__ import annotations

import pytest

from claudio.app import PREWAKE_S, WAKE_CHECK_S, Assistant, State
from claudio.audio.vad import Endpointer
from claudio.metrics import MetricsWriter
from claudio.wakeword.verifier import clipped_name
from tests.unit.test_app import AudioClock, FakeSpeaker, FakeStt, SecondChance
from tests.unit.test_jenjo_test_fixes import ClaudioWake, OnlyCommands, WeakWake

RATE = 16000


@pytest.mark.parametrize(
    ("text", "clipped"),
    [
        ("L'audio?", True),
        ("l'audio apri...", True),
        ("Audio.", True),
        ("Fraudio.", True),  # 26/10, said at 0.88
        ("Radio Italia", False),
        ("Studio", False),
        ("Alza l'audio della TV", False),  # the name is never cut in the middle of the audio
        ("Grazie.", False),
        ("", False),
    ],
)
def test_clipped_name(text: str, clipped: bool) -> None:
    assert clipped_name(text) is clipped


def assistant_with(wake, heard: str, tmp_path, fake_vad, vad_settings):
    check = SecondChance(heard)
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        OnlyCommands(),
        FakeSpeaker(),
        MetricsWriter(tmp_path / "metrics.jsonl"),
        clock=AudioClock(),
        second_chance=check,
        confirm_below=0.45,
        always_confirm=frozenset({"claudio"}),
    )
    return assistant, check


async def run(assistant: Assistant, frames) -> None:
    for frame in frames:
        await assistant.handle_frame(frame)
    await assistant.wait_idle()


async def test_claudio_model_wakes_on_a_clipped_name(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    assistant, _ = assistant_with(ClaudioWake({20}), "L'audio?", tmp_path, fake_vad, vad_settings)
    await run(assistant, make_frames("silence", 1700))
    assert assistant.state is State.LISTENING


async def test_hey_claudio_model_does_not_take_audio_for_the_name(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    """Only the "claudio" model has heard something like the name: elsewhere it is a word."""
    assistant, _ = assistant_with(WeakWake({20}), "L'audio?", tmp_path, fake_vad, vad_settings)
    await run(assistant, make_frames("silence", 1700))
    assert assistant.state is State.IDLE


async def test_the_check_hears_2_5_seconds_and_the_request_keeps_1_5(
    fake_vad, vad_settings, make_frames, tmp_path
) -> None:
    assistant, check = assistant_with(
        ClaudioWake({50}), "Claudio", tmp_path, fake_vad, vad_settings
    )
    await run(assistant, make_frames("silence", 4100))
    assert len(check.heard[0]) == int(WAKE_CHECK_S * RATE)
    assert assistant.state is State.LISTENING
    assert len(assistant._prewake) == int(PREWAKE_S * RATE)
