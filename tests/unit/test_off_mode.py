"""Off ("Claudio, spegniti"): silent until "Claudio, accenditi"; "chiuditi" still closes."""

from __future__ import annotations

import numpy as np
import pytest

from claudio.app import Assistant, State
from claudio.audio.vad import Endpointer
from claudio.config import WakeWordSettings
from claudio.protocols import StrictWake
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import LocalResponder
from claudio.system import SystemControl
from claudio.wakeword.detector import best_detection, raised
from tests.unit.test_actions import Recorder
from tests.unit.test_app import AudioClock, FakeSpeaker, FakeStt, OnsetWake, run_requests

# --- the words: Whisper's spellings of the same sounds ------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Claudio, spegniti.",
        "Spegniti",
        "Spengiti",
        "Spegnity.",  # y for i
        "Claudio spegnitti",  # a double letter
        "Spegni ti",  # one word split in two
        "Claudio, smetti di ascoltare.",
    ],
)
def test_off_phrases_and_their_variants(matcher: CommandMatcher, text: str) -> None:
    match = matcher.match(text)
    assert match is not None and match.command.id == "off", text


@pytest.mark.parametrize(
    "text",
    [
        "Claudio, accenditi.",
        "Accenditi",
        "Accendity",
        "Claudio acenditi",  # one c
        "Claudio, accenditti",
        "Accendi ti.",  # split in two
        "Riaccenditi",
        "Torna ad ascoltare",
    ],
)
def test_on_phrases_and_their_variants(matcher: CommandMatcher, text: str) -> None:
    match = matcher.match(text)
    assert match is not None and match.command.id == "on", text


@pytest.mark.parametrize("text", ["Accendi la luce", "Spegni la luce", "Accendi", "Spegni"])
def test_lights_never_switch_claudio(matcher: CommandMatcher, text: str) -> None:
    match = matcher.match(text)
    assert match is None or match.command.id not in ("off", "on", "shutdown"), text


@pytest.mark.parametrize("text", ["Claudio, chiuditi.", "Chiuditi pure", "Claudio QDT."])
def test_chiuditi_still_closes(matcher: CommandMatcher, text: str) -> None:
    match = matcher.match(text)
    assert match is not None and match.command.id == "shutdown"


def test_english_phrases(matcher: CommandMatcher) -> None:
    assert matcher.match("Stop listening", "en").command.id == "off"  # type: ignore[union-attr]
    assert matcher.match("Turn yourself on", "en").command.id == "on"  # type: ignore[union-attr]
    assert matcher.match("Shut down", "en").command.id == "shutdown"  # type: ignore[union-attr]


# --- the state --------------------------------------------------------------------------


def test_on_also_ends_a_pause() -> None:
    system = SystemControl()
    system.apply("pause")
    assert system.apply("on") and not system.paused and not system.off
    assert not system.apply("on")  # nothing left to change


def test_off_and_shutdown_are_separate() -> None:
    system = SystemControl()
    assert system.apply("off") and system.off and not system.stopping
    assert not system.apply("off")


# --- the responder ------------------------------------------------------------------------


async def say(r: LocalResponder, text: str) -> str:
    return "".join([c async for c in r.respond(text, "it")])


async def test_while_off_only_accenditi_is_answered(matcher: CommandMatcher) -> None:
    system = SystemControl()
    rec = Recorder()
    prepared: list[int] = []
    r = LocalResponder(
        matcher, ActionRunner(rec.ctx(), system), on_prepare=lambda: prepared.append(1)
    )
    assert (await say(r, "spegniti")).startswith("Mi spengo") and system.off
    for text in ("apri chrome", "che ore sono", "svegliati", "chiuditi", "basta"):
        assert await say(r, text) == "", text
        assert r.last_route == {"route": "off"}
    r.prepare()
    assert prepared == []  # nothing is got ready for Claude while off
    assert rec.ran == [] and not system.stopping
    assert await say(r, "accenditi") == "Eccomi, sono acceso." and not system.off
    assert await say(r, "apri chrome") == "Apro Chrome."


# --- the wake word threshold --------------------------------------------------------------


def test_strict_thresholds_are_a_little_higher() -> None:
    thresholds = {"hey_claudio": 0.3, "claudio": 0.95}
    strict = raised(thresholds, WakeWordSettings().off_threshold_boost)
    assert strict == pytest.approx({"hey_claudio": 0.4, "claudio": 0.99})
    assert best_detection({"hey_claudio": 0.35}, thresholds) is not None
    assert best_detection({"hey_claudio": 0.35}, strict) is None  # too weak while off


def test_boost_is_validated() -> None:
    with pytest.raises(ValueError):
        WakeWordSettings(off_threshold_boost=-0.1)
    with pytest.raises(ValueError):
        WakeWordSettings(off_threshold_boost=0.6)


# --- the whole assistant ------------------------------------------------------------------


class StrictOnsetWake(OnsetWake):
    """OnsetWake that records whether the assistant asked it to be strict, and a near miss
    ready on every frame (which must never reach Whisper while off)."""

    def __init__(self) -> None:
        super().__init__()
        self.strict = False
        self.strict_seen: list[bool] = []

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        self.strict_seen.append(self.strict)
        return super().process(frame)

    def pop_near_miss(self) -> tuple[str, float] | None:
        return ("claudio", 0.2) if self.strict else None


def build(fake_vad, vad_settings, matcher: CommandMatcher, texts: list[str]):
    system = SystemControl()
    rec = Recorder()
    responder = LocalResponder(matcher, ActionRunner(rec.ctx(), system))
    wake, speaker, stt = StrictOnsetWake(), FakeSpeaker(), FakeStt(texts)
    checks: list[np.ndarray] = []
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        stt,
        responder,
        speaker,
        clock=AudioClock(),
        system=system,
        second_chance=lambda audio: checks.append(audio) or "",
        conversation_s=0,
    )
    return assistant, speaker, system, wake, rec, checks


async def test_off_is_silent_until_accenditi(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    texts = ["Claudio, spegniti.", "Claudio, apri Chrome.", "che ore sono", "Claudio, accenditi."]
    assistant, speaker, system, wake, rec, checks = build(fake_vad, vad_settings, matcher, texts)
    assert isinstance(wake, StrictWake)

    await run_requests(assistant, make_frames, 1)
    assert system.off and speaker.spoken[-1].startswith("Mi spengo")
    beeps_before_off = list(speaker.beeps)

    await run_requests(assistant, make_frames, 2)  # while off: no beep, no reply, no action
    assert speaker.beeps == beeps_before_off
    assert speaker.spoken[1:] == ["", ""]  # nothing was said
    assert rec.ran == [] and checks == []  # no Chrome, no second Whisper pass
    assert [row["outcome"] for row in assistant.history][1:] == ["off", "off"]
    assert True in wake.strict_seen  # the wake word was stricter while off

    await run_requests(assistant, make_frames, 1)
    assert not system.off and speaker.spoken[-1] == "Eccomi, sono acceso."
    assert assistant.state is State.IDLE and not system.stopping
