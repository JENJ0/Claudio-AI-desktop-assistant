"""Group W (14/11): the wake word check logs what really happened.

H2 quater read "Seconda possibilità ... (punteggio 0.28)" as a model under its threshold:
it was "claudio" over its 0.25, checked by Whisper by design (`confirm`), and 0.28 was only
the first frame over the threshold (the replay gave 0.97 one frame later)."""

from __future__ import annotations

import threading

import numpy as np

from claudio.app import Assistant, State
from claudio.audio.vad import Endpointer
from tests.unit.test_app import (
    AudioClock,
    FakeResponder,
    FakeSpeaker,
    FakeStt,
    FakeWake,
    SecondChance,
    run_frames,
)

LOGGER = "claudio.app"
RISING = {20: 0.28, 21: 0.97, 22: 0.95}  # frame -> score: 23:25:43 on 13/11, as replayed


class RisingClaudio(FakeWake):
    """The "claudio" model on one word: over its threshold at 0.28, then 0.97 and 0.95."""

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        super().process(frame)
        score = RISING.get(self.calls)
        return None if score is None else ("claudio", score)


class NoCommands(FakeResponder):
    def understands(self, text: str, memory: bool = True) -> bool:
        return False  # "Grazie." must not pass as a command said without the name


def checked_claudio(fake_vad, vad_settings, heard: str) -> tuple[Assistant, threading.Event]:
    gate = threading.Event()  # Whisper is still listening while frames 21-22 arrive
    assistant = Assistant(
        RisingClaudio(set()),
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        NoCommands(),
        FakeSpeaker(),
        clock=AudioClock(),
        second_chance=SecondChance(heard, gate=gate),
        always_confirm=frozenset({"claudio"}),
    )
    return assistant, gate


async def run_check(assistant: Assistant, gate: threading.Event, make_frames) -> None:
    for frame in make_frames("silence", 2400)[:25]:
        await assistant.handle_frame(frame)
    gate.set()
    await assistant.wait_idle()


async def test_a_failed_check_logs_the_frames_it_dropped(
    fake_vad, vad_settings, make_frames, caplog
) -> None:
    """The evidence T2 waits for: frames over the threshold that Whisper never heard."""
    assistant, gate = checked_claudio(fake_vad, vad_settings, "Grazie.")
    with caplog.at_level("INFO", logger=LOGGER):
        await run_check(assistant, gate, make_frames)
    assert assistant.state is State.IDLE
    line = next(r.getMessage() for r in caplog.records if "niente da fare" in r.getMessage())
    assert line.startswith("Conferma di Whisper: ha sentito 'Grazie.'")
    assert "claudio 0.28 sopra soglia, picco 0.97" in line
    assert "2 blocchi sopra soglia durante il controllo" in line
    assert " ms)" in line


async def test_a_confirmed_wake_keeps_the_peak(fake_vad, vad_settings, make_frames, caplog) -> None:
    assistant, gate = checked_claudio(fake_vad, vad_settings, "Claudio.")
    with caplog.at_level("INFO", logger=LOGGER):
        await run_check(assistant, gate, make_frames)
    assert assistant.state is State.LISTENING and assistant._interaction is not None
    info = assistant._interaction.info
    assert info["wake_score"] == 0.28 and info["wake_peak"] == 0.97
    assert "Conferma di Whisper: ha sentito 'Claudio.' (claudio 0.28 sopra soglia" in caplog.text
    assert "Seconda possibilità" not in caplog.text


async def test_a_near_miss_is_still_a_second_chance(
    fake_vad, vad_settings, make_frames, caplog
) -> None:
    assistant = Assistant(
        FakeWake(set(), miss_on={20}),  # hey_claudio 0.3, under its threshold
        Endpointer(fake_vad, vad_settings),
        FakeStt(),
        FakeResponder(),
        FakeSpeaker(),
        clock=AudioClock(),
        second_chance=SecondChance("Hey Claudio."),
    )
    with caplog.at_level("INFO", logger=LOGGER):
        await run_frames(assistant, make_frames("silence", 2400))
    assert assistant.state is State.LISTENING and assistant._interaction is not None
    assert "wake_peak" not in assistant._interaction.info  # no frame went higher
    assert (
        "Seconda possibilità: Whisper ha sentito 'Hey Claudio.' (hey_claudio 0.30 sotto soglia"
        in caplog.text
    )
