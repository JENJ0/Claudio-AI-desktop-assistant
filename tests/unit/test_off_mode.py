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
from tests.unit.test_app import (
    AudioClock,
    FakeSpeaker,
    FakeStt,
    FakeWake,
    OnsetWake,
    run_frames,
    run_requests,
)

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
    # Off, "accenditi" is the only command understood: the closed list and the second
    # chance look for it alone, and a phrase without the name is a false wake.
    assert r.understands("Claudio, accenditi.") and not r.understands("apri chrome")
    for text in ("apri chrome", "che ore sono", "svegliati", "chiuditi", "basta"):
        assert await say(r, text) == "", text
        assert r.last_route == {"route": "off"}
    r.prepare()
    assert prepared == []  # nothing is got ready for Claude while off
    assert rec.ran == [] and not system.stopping
    assert await say(r, "accenditi") == "Eccomi, sono acceso." and not system.off
    assert r.understands("apri chrome")
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
    """OnsetWake that records whether the assistant asked it to be strict."""

    def __init__(self) -> None:
        super().__init__()
        self.strict = False
        self.strict_seen: list[bool] = []

    def process(self, frame: np.ndarray) -> tuple[str, float] | None:
        self.strict_seen.append(self.strict)
        return super().process(frame)


class Heard:
    """Whisper on the second chance's audio: one text after another (the last repeats)."""

    def __init__(self, texts: list[str]) -> None:
        self.texts, self.calls = texts, 0

    def __call__(self, audio: np.ndarray) -> str:
        self.calls += 1
        return self.texts[min(self.calls, len(self.texts)) - 1]


def build(
    fake_vad,
    vad_settings,
    matcher: CommandMatcher,
    texts: list[str],
    heard: list[str] | None = None,
    **options,
):
    system = SystemControl()
    rec = Recorder()
    responder = LocalResponder(matcher, ActionRunner(rec.ctx(), system))
    wake, speaker, stt = StrictOnsetWake(), FakeSpeaker(), FakeStt(texts)
    checks = Heard(heard or [""])
    assistant = Assistant(
        wake,
        Endpointer(fake_vad, vad_settings),
        stt,
        responder,
        speaker,
        clock=AudioClock(),
        system=system,
        second_chance=checks,
        conversation_s=0,
        **options,
    )
    return assistant, speaker, system, wake, rec, checks


async def test_off_is_silent_until_accenditi(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """Off, Whisper checks every detection: only the name wakes CLAUDIO, with a soft tick
    (JenJo 30/10), and of what follows only "accenditi" is answered."""
    texts = ["Claudio, spegniti.", "Claudio, apri Chrome.", "Claudio, accenditi."]
    heard = ["Claudio, apri", "che ore sono", "Claudio, accenditi."]
    assistant, speaker, system, wake, rec, checks = build(
        fake_vad, vad_settings, matcher, texts, heard
    )
    assert isinstance(wake, StrictWake)

    await run_requests(assistant, make_frames, 1)
    assert system.off and speaker.spoken[-1].startswith("Mi spengo")
    assert speaker.beeps == ["wake"] and checks.calls == 0  # on: a clear wake needs no check

    await run_requests(assistant, make_frames, 2)  # the name, then the TV without it
    assert checks.calls == 2
    assert speaker.beeps == ["wake", "tick"]  # the tick only for the name Whisper heard
    assert speaker.spoken[1:] == [""] and rec.ran == []  # no reply, no Chrome
    assert [row["outcome"] for row in assistant.history][1:] == ["off"]
    assert True in wake.strict_seen  # the wake word was stricter while off

    await run_requests(assistant, make_frames, 1)
    assert not system.off and speaker.spoken[-1] == "Eccomi, sono acceso."
    assert speaker.beeps == ["wake", "tick", "tick"]
    assert assistant.state is State.IDLE and not system.stopping


async def test_off_without_the_name_whispers_other_guesses_are_searched_for_accenditi(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """30/10 14:02:24: "Claudio, accenditi" heard as "Grazie a te.", while "Dai, accenditi."
    was Whisper's second guess: off, the closed list looks for it, in the second chance and
    in the request."""
    guesses: list[str] = []

    def closed_list(audio: np.ndarray, language: str) -> list[str]:
        guesses.append(language)
        return ["Grazie a te.", "Dai, accenditi."]

    texts = ["Claudio, spegniti.", "Grazie a te."]
    assistant, speaker, system, *_ = build(
        fake_vad, vad_settings, matcher, texts, ["Grazie a te."], closed_list=closed_list
    )
    await run_requests(assistant, make_frames, 1)
    assert system.off
    await run_requests(assistant, make_frames, 1)
    assert not system.off and speaker.spoken[-1] == "Eccomi, sono acceso."
    assert speaker.beeps == ["wake", "tick"] and len(guesses) == 2
    row = assistant.history[-1]
    assert row["second_chance"] == "Dai, accenditi." and row["second_chance_from"] == "Grazie a te."
    assert row["closed_list_from"] == "Grazie a te." and row["text"] == "Dai, accenditi."


async def test_off_accenditi_heard_by_the_second_chance_alone_turns_claudio_on(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """The model fired 1.2 s after "Claudio, accenditi." had ended: too late for the
    lead-in, but the second chance heard the whole command, and that is the request."""
    system = SystemControl()
    system.apply("off")
    rec = Recorder()
    speaker, stt = FakeSpeaker(), FakeStt(["Claudio, accenditi."])
    checks = Heard(["Claudio, accenditi."])
    assistant = Assistant(
        FakeWake({26}),
        Endpointer(fake_vad, vad_settings),
        stt,
        LocalResponder(matcher, ActionRunner(rec.ctx(), system)),
        speaker,
        clock=AudioClock(),
        system=system,
        second_chance=checks,
        conversation_s=0,
    )
    frames = make_frames("silence", 160) + make_frames("speech", 800)
    await run_frames(assistant, frames + make_frames("silence", 2400))
    assert checks.calls == 1 and stt.heard  # the second chance's audio, transcribed
    assert not system.off and speaker.spoken == ["Eccomi, sono acceso."]
    assert speaker.beeps == ["tick"]


async def test_off_noise_never_reaches_whisper(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    """A click or the headset's gate: the voice meter rejects it before Whisper, silently,
    and its audio is kept apart to retrain the wake word on."""
    saved: list[str] = []
    assistant, speaker, system, _wake, _rec, checks = build(
        fake_vad,
        vad_settings,
        matcher,
        ["Claudio, spegniti."],
        ["Claudio, accenditi."],
        voice_meter=lambda audio: 0 if system.off else 500,
        wake_min_voice_ms=96,
        rejected_sink=lambda audio, kind: saved.append(kind) or f"{kind}.wav",
    )
    await run_requests(assistant, make_frames, 2)
    assert system.off and checks.calls == 0 and saved == ["no_voice"]
    assert speaker.beeps == ["wake"]
