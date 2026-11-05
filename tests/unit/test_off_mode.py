"""Off ("Claudio, spegniti"): silent until "Claudio, accenditi"; "chiuditi" still closes."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from claudio.app import Assistant, State
from claudio.audio.recorder import UtteranceRecorder
from claudio.audio.vad import Endpointer
from claudio.chat.store import ChatStore
from claudio.config import NoiseSettings, SttSettings, WakeWordSettings
from claudio.logging_setup import JsonFormatter
from claudio.metrics import Interaction, MetricsWriter
from claudio.protocols import StrictWake
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import LocalResponder
from claudio.session_log import CONSOLE_FORMAT, ConversationLog
from claudio.stt.transcriber import QUICK_NEW_TOKENS, WhisperTranscriber
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
from tests.unit.test_conversation import (
    STORY,
    LongSpeaker,
    barge,
    claude_says,
    speak_over,
    talk,
)
from tests.unit.test_listening_it1 import CueSpeaker, say_over_the_reply, slicing_barge

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
    assert row["second_chance"] == "Dai, accenditi." and row["text"] == "Dai, accenditi."
    # Only the turn-on phrase stays: "Grazie a te." is known to have been heard, and no more.
    assert row["second_chance_from"] == row["closed_list_from"] == "<hidden: off>"


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
    """A click or the headset's gate: the voice meter rejects it before Whisper, silently, and
    while off nothing heard is saved, not even apart to retrain the wake word on."""
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
    assert system.off and checks.calls == 0 and saved == []
    assert speaker.beeps == ["wake"]


async def test_the_wake_word_is_strict_from_the_first_frame_after_switching_off(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    assistant, _, system, wake, *_ = build(fake_vad, vad_settings, matcher, ["x"])
    quiet = make_frames("silence", 80)[0]
    await assistant.handle_frame(quiet)
    system.apply("off")
    await assistant.handle_frame(quiet)
    system.apply("on")
    await assistant.handle_frame(quiet)
    assert wake.strict_seen == [False, True, False]


async def test_off_claude_is_not_told_how_the_phrase_was_heard(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    told: list[object] = []
    assistant, _, system, *_ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["Claudio, apri Chrome."],
        ["Claudio."],
        on_heard=told.append,
    )
    system.apply("off")
    await run_requests(assistant, make_frames, 1)
    assert [row["outcome"] for row in assistant.history] == ["off"] and told == []


def rows(tmp_path) -> list[dict]:
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


async def test_off_without_whisper_only_a_clear_wake_counts(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher, tmp_path
) -> None:
    """Off with no second chance to ask, the strict threshold decides alone: a clear wake gets
    the phrase after it, without a sound; a near miss leaves no row and no WAV."""
    system = SystemControl()
    system.apply("off")
    saved: list[str] = []
    speaker = FakeSpeaker()
    assistant = Assistant(
        FakeWake({4}, miss_on={2}),
        Endpointer(fake_vad, vad_settings),
        FakeStt("Claudio, accenditi."),
        LocalResponder(matcher, ActionRunner(Recorder().ctx(), system)),
        speaker,
        MetricsWriter(tmp_path / "metrics.jsonl"),
        clock=AudioClock(),
        audio_sink=lambda audio, kind: saved.append(kind) or f"{kind}.wav",
        system=system,
    )
    frames = make_frames("silence", 400) + make_frames("speech", 800) + make_frames("silence", 800)
    await run_frames(assistant, frames)
    assert speaker.beeps == [] and speaker.spoken == ["Eccomi, sono acceso."] and not system.off
    assert saved == ["ok"] and [r["outcome"] for r in rows(tmp_path)] == ["ok"]


async def test_going_busy_during_the_closed_list_stops_the_second_chance(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher, tmp_path
) -> None:
    """Off, the closed list looks for "accenditi" after the state was checked: a message from
    the chat window may be answered by the time it finds it. No wake over that, and its seconds
    are not the second chance's."""
    reached, gate = threading.Event(), threading.Event()

    def closed_list(audio: np.ndarray, language: str) -> list[str]:
        reached.set()
        assistant.clock.t += 5.0
        gate.wait(5)
        return ["Dai, accenditi."]

    assistant, speaker, system, *_ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["x"],
        ["Grazie a te."],
        closed_list=closed_list,
        metrics=MetricsWriter(tmp_path / "metrics.jsonl"),
    )
    system.apply("off")
    await assistant.handle_frame(make_frames("speech", 80)[0])
    for _ in range(500):
        if reached.is_set():
            break
        await asyncio.sleep(0.01)
    assert reached.is_set()
    assistant.state = State.RESPONDING
    gate.set()
    await assistant.wait_idle()
    assert assistant.state is State.RESPONDING and speaker.beeps == []
    (row,) = rows(tmp_path)
    assert row["superseded"] is True and row["second_chance_ms"] == 0
    assert "second_chance" not in row and "second_chance_from" not in row  # it did not turn on


async def test_a_failing_closed_list_while_off_is_a_plain_rejection(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher, tmp_path, caplog
) -> None:
    def closed_list(audio: np.ndarray, language: str) -> list[str]:
        raise RuntimeError("CUDA esplosa")

    assistant, speaker, system, _, _, checks = build(
        fake_vad,
        vad_settings,
        matcher,
        ["x"],
        ["Grazie a te."],
        closed_list=closed_list,
        metrics=MetricsWriter(tmp_path / "metrics.jsonl"),
    )
    system.apply("off")
    loud, quiet = make_frames("speech", 80)[0], make_frames("silence", 80)[0]
    with caplog.at_level("ERROR"):
        for frame in (loud, quiet, loud):  # two detections, one after the other has decided
            await assistant.handle_frame(frame)
            await assistant.wait_idle()
    assert checks.calls == 2 and speaker.beeps == [] and assistant.state is State.IDLE
    assert "CUDA esplosa" in caplog.text
    first, second = rows(tmp_path)
    assert first["rejected"] == second["rejected"] == "no_name"
    assert "second_chance" not in first and "second_chance_from" not in first


# --- what stays on disk while off: nothing, except the turn-on ------------------------------

WORD = "zafferano"  # said around Claudio while off: no file may ever hold it
SAID_ON = "tiramisu"  # said once it is on again: the walk below does see every file
EVERY_FILE_WITH_WORDS = {
    "console.log",
    "claudio.log.jsonl",
    "metrics.jsonl",
    "session/conversazione.md",
    "chat.db",
}
HIDDEN_IN_THE_LOG = "'<hidden: off>'"
# What holds a phrase in a row: heard while off, none of them is written.
SAID_FIELDS = {
    "text",
    "second_chance",
    "second_chance_from",
    "closed_list_from",
    "retried_from",
    "continued_from",
    "relisten",
    "barge_in",
}


def seg(text: str, no_speech: float = 0.01, logprob: float = -0.2, ratio: float = 1.0):
    return SimpleNamespace(
        text=text, no_speech_prob=no_speech, avg_logprob=logprob, compression_ratio=ratio
    )


class WhisperModel:
    """The model under the real WhisperTranscriber: each pass writes the next of `phrases`, and
    each short look of the second chance the next of `names`; then nothing."""

    def __init__(self, phrases: Sequence[object], names: Sequence[object] = ()) -> None:
        self.phrases, self.names = list(phrases), list(names)

    def transcribe(self, audio: np.ndarray, **kwargs):
        heard = self.names if kwargs["max_new_tokens"] == QUICK_NEW_TOKENS else self.phrases
        said = heard.pop(0) if heard else ""
        return iter([seg(said) if isinstance(said, str) else said]), None

    def encode(self, features):
        return features


def real_whisper(phrases: Sequence[object], names: Sequence[object], prompt: str = ""):
    stt = WhisperTranscriber.__new__(WhisperTranscriber)
    stt.settings = SttSettings(languages=["it"], prompts={"it": prompt})
    stt.language, stt.language_confidence, stt._encoded, stt._reused = "it", 1.0, None, 0
    stt.last_language, stt.last_segments = "it", []
    stt._model = WhisperModel(phrases, names)
    return stt


@pytest.fixture
def disk(tmp_path: Path) -> Iterator[Path]:
    """A folder with both log files of a run, fed by the `claudio` logger at DEBUG."""
    logger = logging.getLogger("claudio")
    level = logger.level
    logger.setLevel(logging.DEBUG)
    console = logging.FileHandler(tmp_path / "console.log", encoding="utf-8")
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    structured = logging.FileHandler(tmp_path / "claudio.log.jsonl", encoding="utf-8")
    structured.setFormatter(JsonFormatter())  # it also writes the row each log line carries
    logger.addHandler(console)
    logger.addHandler(structured)
    yield tmp_path
    for handler in (console, structured):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level)


def holding(word: str, disk: Path) -> set[str]:
    """The files under `disk` with `word` in their bytes, whatever they are (SQLite too)."""
    needle = word.encode()
    return {
        path.relative_to(disk).as_posix().removesuffix("-wal")
        for path in disk.rglob("*")
        if path.is_file() and needle in path.read_bytes().lower()
    }


def wavs(disk: Path) -> set[str]:
    return {path.name for path in disk.rglob("*.wav")}


def in_the_console(disk: Path) -> str:
    return (disk / "console.log").read_text(encoding="utf-8")


def on_disk(
    disk: Path,
    fake_vad,
    vad_settings,
    matcher: CommandMatcher,
    phrases: Sequence[object] = (),
    names: Sequence[object] | None = None,
    prompt: str = "",
    wake: FakeWake | None = None,
    speaker: FakeSpeaker | None = None,
    **options,
) -> Assistant:
    """The real Assistant, router, Whisper, metrics, conversation, chat and recorders, into
    `disk`: only the microphone, the voice and Whisper's model are fakes. `names`: Whisper's
    second chance is on, and this is what it hears."""
    system = SystemControl()
    stt = real_whisper(phrases, names or (), prompt)
    (disk / "session").mkdir(exist_ok=True)
    options.setdefault(
        "second_chance",
        (lambda audio: stt.transcribe(audio, quick=True)) if names is not None else None,
    )
    return Assistant(
        wake or StrictOnsetWake(),
        Endpointer(fake_vad, vad_settings),
        stt,
        LocalResponder(matcher, ActionRunner(Recorder().ctx(), system)),
        speaker or FakeSpeaker(),
        MetricsWriter(
            disk / "metrics.jsonl",
            ConversationLog(disk / "session"),
            ChatStore(disk / "chat.db", "session"),
        ),
        clock=AudioClock(),
        audio_sink=UtteranceRecorder(disk / "audio"),
        rejected_sink=UtteranceRecorder(disk / "audio" / "rejected"),
        system=system,
        conversation_s=0,
        **options,
    )


async def the_whole_story(assistant: Assistant, make_frames, disk: Path, while_off: int) -> None:
    """ "Spegniti" said while on: kept. `while_off` requests: nothing of them stays, in any
    file. "Accenditi": its text and its audio stay, and Claudio is on. What is said after it
    is in every file that keeps words: the walk does see them."""
    system = assistant.system
    await run_requests(assistant, make_frames, 1)
    assert system.off
    kept = wavs(disk)
    assert len(kept) == 1 and rows(disk)[-1]["text"] == "Claudio, spegniti."

    await run_requests(assistant, make_frames, while_off)
    assert system.off
    assert holding(WORD, disk) == set()
    assert wavs(disk) == kept  # no *_off.wav, no *_timeout.wav, no near miss
    for row in rows(disk)[1:]:
        assert not SAID_FIELDS & set(row), row
        assert "audio_file" not in row

    await run_requests(assistant, make_frames, 1)
    assert not system.off
    assert rows(disk)[-1]["text"] == "Claudio, accenditi."
    assert len(wavs(disk)) == 2 and any(name.endswith("_ok.wav") for name in wavs(disk) - kept)
    assert "'Claudio, accenditi.'" in in_the_console(disk)
    assert holding(WORD, disk) == set()

    await run_requests(assistant, make_frames, 1)
    assert holding(SAID_ON, disk) == EVERY_FILE_WITH_WORDS
    assert len(wavs(disk)) == 3


TURNS_OFF, TURNS_ON = "Claudio, spegniti.", "Claudio, accenditi."
THEN_ON = f"Claudio, una ricetta col {SAID_ON}."


@pytest.mark.parametrize(
    ("while_off", "options"),
    [
        pytest.param([f"Claudio, ti porto lo {WORD}."], {}, id="a phrase with the name"),
        pytest.param([f"Grazie per la visione, {WORD}."], {}, id="whisper: a hallucination"),
        pytest.param([seg(f"{WORD} {WORD}", ratio=3.0)], {}, id="whisper: a repetition"),
        pytest.param([seg(WORD, 0.99, -2.0)], {}, id="whisper: sounds like silence"),
        pytest.param(
            [f"Assaggia lo {WORD}.", f"Claudio, lo {WORD}."],
            {"prompt": f"Ehi Claudio. Apri Chrome, poi assaggia lo {WORD}."},
            id="whisper: a piece of the prompt",
        ),
        pytest.param([f"Lo {WORD} è finito."], {"verify_wake": True}, id="a false wake, dropped"),
        pytest.param(
            [f"Claudio, lo {WORD}."],
            {"noise": NoiseSettings(min_speech_dbfs=-1.0)},
            id="noise, dropped",
        ),
        pytest.param([f"Claudio, lo {WORD} e"], {"continue_s": 0.2}, id="left halfway, then over"),
        pytest.param(
            [f"Claudio, lo {WORD}.", f"Ancora lo {WORD}."],
            {"retry_language": "en"},
            id="tried again in another language",
        ),
        pytest.param(["Claudio."], {}, id="only the name, then nothing"),
    ],
)
async def test_the_normal_path_while_off_leaves_nothing_on_disk(
    disk, fake_vad, vad_settings, make_frames, matcher, while_off, options
) -> None:
    phrases = [TURNS_OFF, *while_off, TURNS_ON, THEN_ON]
    assistant = on_disk(disk, fake_vad, vad_settings, matcher, phrases, **options)
    await the_whole_story(assistant, make_frames, disk, 1)
    assert HIDDEN_IN_THE_LOG in in_the_console(disk)  # a phrase was heard, and hidden


class Guesses:
    """Whisper's n-best: the next list at each call, then nothing."""

    def __init__(self, *lists: list[str]) -> None:
        self.lists = list(lists)

    def __call__(self, audio: np.ndarray, language: str) -> list[str]:
        return self.lists.pop(0) if self.lists else []


async def test_the_closed_list_while_off_leaves_nothing_on_disk(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    said = f"Lo {WORD} è finito."
    phrases = [TURNS_OFF, said, TURNS_ON, THEN_ON]
    guesses = Guesses([said, f"Il {WORD} è finito."])
    assistant = on_disk(disk, fake_vad, vad_settings, matcher, phrases, closed_list=guesses)
    await the_whole_story(assistant, make_frames, disk, 1)


@pytest.mark.parametrize(
    ("names", "phrases", "guesses"),
    [
        pytest.param([f"Lo {WORD} è finito."], [], [[f"Il {WORD}."]], id="no name, no accenditi"),
        pytest.param([f"Claudio, lo {WORD}"], [f"Claudio, lo {WORD}."], [], id="the name"),
        pytest.param([seg(WORD, ratio=3.0)], [], [], id="whisper: a repetition"),
        pytest.param(
            [f"Lo {WORD} è finito."],
            [f"Lo {WORD} è finito."],
            [["Dai, accenditi."], [f"Il {WORD}."]],
            id="accenditi among the guesses, not in the phrase",
        ),
    ],
)
async def test_the_second_chance_while_off_leaves_nothing_on_disk(
    disk, fake_vad, vad_settings, make_frames, matcher, names, phrases, guesses
) -> None:
    assistant = on_disk(
        disk,
        fake_vad,
        vad_settings,
        matcher,
        [TURNS_OFF, *phrases, TURNS_ON, THEN_ON],
        [*names, TURNS_ON],
        closed_list=Guesses(*guesses),
    )
    await the_whole_story(assistant, make_frames, disk, 1)


async def test_what_the_turn_on_was_first_heard_as_does_not_stay(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    """Only the turn-on phrase is kept: Whisper's first hearing of that audio may be the TV,
    with "Dai, accenditi." as its fifth guess. The row still says the closed list did it."""
    first = f"Lo {WORD} è finito."
    assistant = on_disk(
        disk,
        fake_vad,
        vad_settings,
        matcher,
        [TURNS_OFF, first],
        [first],
        closed_list=Guesses([first, "Dai, accenditi."], [first, "Dai, accenditi."]),
    )
    await run_requests(assistant, make_frames, 2)
    assert not assistant.system.off
    assert holding(WORD, disk) == set()
    row = rows(disk)[-1]
    assert row["text"] == row["second_chance"] == "Dai, accenditi."
    assert row["closed_list_from"] == row["second_chance_from"] == "<hidden: off>"
    assert "Lista chiusa: '<hidden: off>' -> 'Dai, accenditi.'" in in_the_console(disk)


@pytest.mark.parametrize(
    ("phrases", "names", "options", "field"),
    [
        pytest.param(
            [TURNS_OFF, TURNS_ON],
            [f"Claudio, lo {WORD}"],
            {},
            "second_chance",
            id="the second chance heard the name and more",
        ),
        pytest.param(
            [TURNS_OFF, f"Lo {WORD}.", TURNS_ON],
            None,
            {"retry_language": "en"},
            "retried_from",
            id="the retry found it",
        ),
    ],
)
async def test_a_turn_on_keeps_its_phrase_and_no_other_words(
    disk, fake_vad, vad_settings, make_frames, matcher, phrases, names, options, field
) -> None:
    assistant = on_disk(disk, fake_vad, vad_settings, matcher, phrases, names, **options)
    await run_requests(assistant, make_frames, 2)
    assert not assistant.system.off
    assert holding(WORD, disk) == set()
    row = rows(disk)[-1]
    assert row["text"] == TURNS_ON and row[field] == "<hidden: off>"


async def test_a_turn_on_after_a_phrase_left_halfway_keeps_its_phrase_and_no_other_words(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant = on_disk(
        disk, fake_vad, vad_settings, matcher, [f"Lo {WORD} e", TURNS_ON], continue_s=1.0
    )
    assistant.system.apply("off")
    frames = (
        make_frames("silence", 160)
        + make_frames("speech", 800)
        + make_frames("silence", 800)
        + make_frames("speech", 480)
        + make_frames("silence", 1600)
    )
    await run_frames(assistant, frames)
    assert not assistant.system.off
    assert holding(WORD, disk) == set()
    (row,) = rows(disk)
    assert row["text"] == TURNS_ON and row["continued_from"] == "<hidden: off>"


async def test_nothing_heard_while_off_is_not_logged_as_a_hidden_phrase(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant = on_disk(disk, fake_vad, vad_settings, matcher, [""])
    assistant.system.apply("off")
    await run_requests(assistant, make_frames, 1)
    console = in_the_console(disk)
    assert "Trascritto (it): ''" in console and HIDDEN_IN_THE_LOG not in console


async def test_a_voice_the_meter_rejects_while_off_saves_no_audio(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant = on_disk(
        disk, fake_vad, vad_settings, matcher, [TURNS_OFF], [], wake_min_voice_ms=96
    )
    assistant.voice_meter = lambda audio: 0 if assistant.system.off else 500
    await run_requests(assistant, make_frames, 2)
    assert assistant.system.off
    assert len(wavs(disk)) == 1  # "spegniti", said while on
    assert rows(disk)[-1]["rejected"] == "no_voice"  # the row stays: only the audio goes


async def test_a_timeout_while_off_leaves_a_row_and_no_audio(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant = on_disk(disk, fake_vad, vad_settings, matcher, wake=FakeWake({1}))
    assistant.system.apply("off")
    await run_frames(assistant, make_frames("silence", 3600))
    assert wavs(disk) == set()
    (row,) = rows(disk)
    assert row["outcome"] == "timeout" and row["wake_word"] == "hey_claudio"


async def test_a_row_heard_while_off_keeps_every_figure(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant = on_disk(
        disk, fake_vad, vad_settings, matcher, [f"Claudio, lo {WORD}."], [f"Claudio, lo {WORD}"]
    )
    assistant.system.apply("off")
    await run_requests(assistant, make_frames, 1)
    (row,) = rows(disk)
    assert row["outcome"] == row["route"] == "off"
    assert row["wake_word"] == "claudio" and row["wake_score"] == 0.9
    assert row["stt_language"] == "it"  # which language, not what was said
    for figure in ("stt_ms", "speech_ms", "second_chance_ms", "voice_ms", "speech_dbfs"):
        assert row[figure] is not None, figure


async def test_the_phrase_that_turns_claudio_off_is_kept(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    """Off by the end of its own request: what matters is the state when it was heard."""
    assistant = on_disk(disk, fake_vad, vad_settings, matcher, [TURNS_OFF])
    await run_requests(assistant, make_frames, 1)
    assert assistant.system.off
    assert rows(disk)[-1]["text"] == TURNS_OFF and len(wavs(disk)) == 1
    assert f"'{TURNS_OFF}'" in in_the_console(disk)
    assert holding("spegniti", disk) == EVERY_FILE_WITH_WORDS


@pytest.mark.parametrize(
    ("over_the_reply", "barge_options"),
    [
        pytest.param(f"Lo {WORD} è finito.", {"needs_name": True}, id="not for Claudio"),
        pytest.param(f"Claudio, lo {WORD}. Basta.", {}, id="a stop"),
        pytest.param(f"Claudio, lo {WORD}.", {}, id="a new request"),
        pytest.param(
            f"Mi spengo. Stop, {WORD}. Per riaccendermi di' Claudio, accenditi.",
            {},
            id="a stop written together with the reply's echo",
        ),
    ],
)
async def test_what_is_said_over_mi_spengo_leaves_nothing_on_disk(
    disk, fake_vad, vad_settings, make_frames, matcher, over_the_reply, barge_options
) -> None:
    """The reply "Mi spengo" is said once Claudio is off: a phrase over it is heard while off,
    in the row of a request that is kept."""
    speaker = LongSpeaker()
    assistant = on_disk(
        disk,
        fake_vad,
        vad_settings,
        matcher,
        [TURNS_OFF, f"Claudio, lo {WORD} è finito."],
        wake=FakeWake({3}),
        speaker=speaker,
        barge_in=barge(vad_settings, [over_the_reply], **barge_options),
    )
    await speak_over(assistant, make_frames, speaker)
    assert assistant.system.off
    assert holding(WORD, disk) == set()
    assert len(wavs(disk)) == 1  # the request's: no barge_ignored in rejected/
    first = rows(disk)[0]
    assert first["text"] == TURNS_OFF
    assert first.get("barge_in", "<hidden: off>") == "<hidden: off>"
    assert HIDDEN_IN_THE_LOG in in_the_console(disk)


async def test_a_phrase_still_going_when_mi_spengo_ends_leaves_nothing_on_disk(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    """Judged at the end of the reply, with Claudio off by then."""
    speaker = CueSpeaker()
    assistant = on_disk(
        disk,
        fake_vad,
        vad_settings,
        matcher,
        [TURNS_OFF],
        wake=FakeWake({3}),
        speaker=speaker,
        barge_in=slicing_barge(vad_settings, lambda audio: WORD),
    )

    def end_the_reply(n: int) -> None:
        if n == 13:  # ~1 s into the 1.6 s phrase: one slice cut, the next one going on
            speaker.done.set()

    await say_over_the_reply(
        assistant, make_frames, speaker, speech_ms=1600, on_frame=end_the_reply
    )
    assert assistant.system.off
    assert holding(WORD, disk) == set() and len(wavs(disk)) == 1
    assert f"non è per me: {HIDDEN_IN_THE_LOG}" in in_the_console(disk)


@pytest.mark.parametrize(
    ("said", "options", "in_the_log"),
    [
        pytest.param(
            f"Claudio, lo {WORD}.", {}, f"Trascritto (it): {HIDDEN_IN_THE_LOG}", id="a phrase"
        ),
        pytest.param(
            "Sono le 14 e 5.",
            {"claude_min_words": 1, "only_addressed": False},
            f"dopo la risposta, la ignoro: {HIDDEN_IN_THE_LOG}",
            id="the reply's echo",
        ),
    ],
)
async def test_a_phrase_heard_after_the_tray_switched_claudio_off_leaves_nothing(
    disk, fake_vad, vad_settings, make_frames, matcher, said, options, in_the_log
) -> None:
    """The conversation window was already open: the state when it was heard is what counts."""

    class TrayOffInTheWindow(FakeSpeaker):
        async def beep(self, kind: str = "wake") -> None:
            await super().beep(kind)
            if len(self.beeps) == 2:
                assistant.system.apply("off")

    speaker = TrayOffInTheWindow()
    assistant = on_disk(
        disk,
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono", said],
        wake=FakeWake({3}),
        speaker=speaker,
        **options,
    )
    assistant.conversation_s = 1.0
    await talk(assistant, make_frames, speaker, requests=2)
    assert assistant.system.off
    assert holding(WORD, disk) == set()
    assert len(wavs(disk)) == 1  # "che ore sono"
    assert in_the_log in in_the_console(disk)
    assert [row.get("text") for row in rows(disk)] == ["che ore sono", None]


async def test_what_whisper_heard_before_the_tray_turned_claudio_on_does_not_stay(
    disk, fake_vad, vad_settings, make_frames, matcher
) -> None:
    """The second chance's audio was heard while off; the state changed during its pass."""

    def hears(audio: np.ndarray) -> str:
        assistant.system.apply("on")
        return f"Claudio, lo {WORD}"

    assistant = on_disk(disk, fake_vad, vad_settings, matcher, second_chance=hears)
    assistant.system.apply("off")
    await assistant.handle_frame(make_frames("speech", 80)[0])
    await assistant.wait_idle()
    assert not assistant.system.off and assistant.state is State.IDLE
    assert holding(WORD, disk) == set() and wavs(disk) == set()
    (row,) = rows(disk)
    assert row["superseded"] is True and "second_chance" not in row


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="an answer"),
        pytest.param({"noise": NoiseSettings(answer_min_dbfs=-1.0)}, id="an answer too far away"),
    ],
)
async def test_a_question_armed_before_the_tray_switched_off_leaves_nothing_on_disk(
    disk, fake_vad, vad_settings, make_frames, matcher, options
) -> None:
    """Such a question is dropped: Claude never hears the answer, and its words stay off the
    disk."""
    asked: list[str] = []

    async def claude(text: str, language: str) -> AsyncGenerator[str, None]:
        asked.append(text)
        yield f"Hai detto {WORD}."

    assistant = on_disk(disk, fake_vad, vad_settings, matcher, [f"Sì, lo {WORD}."], **options)
    responder = assistant.responder
    responder.fallback, responder.fallback_awaiting = claude, lambda: True
    assistant.system.apply("off")
    await run_requests(assistant, make_frames, 1)
    assert asked == []
    assert holding(WORD, disk) == set() and wavs(disk) == set()


@pytest.mark.parametrize(
    "said",
    [
        pytest.param(f"no, lo {WORD} è finito", id="a new request instead"),
        pytest.param(f"detto apri chrome {WORD}", id="sounds like the question itself"),
    ],
)
async def test_an_answer_to_hai_detto_while_off_is_hidden_in_the_log(
    matcher: CommandMatcher, caplog, said: str
) -> None:
    system = SystemControl()
    responder = LocalResponder(matcher, ActionRunner(Recorder().ctx(), system))
    responder.propose("apri chrome", "it")
    system.apply("off")
    with caplog.at_level("INFO"):
        assert await say(responder, said) == ""
    assert WORD not in caplog.text and HIDDEN_IN_THE_LOG in caplog.text


async def test_a_message_typed_while_off_is_kept(disk, fake_vad, vad_settings, matcher) -> None:
    """Written in the chat window by the owner, not said around Claudio: the window shows it. It
    is not answered."""
    assistant = on_disk(disk, fake_vad, vad_settings, matcher)
    assistant.system.apply("off")
    assert await assistant.handle_typed(f"una ricetta col {WORD}", "it") == ""
    assert rows(disk)[0]["route"] == "off"
    assert "chat.db" in holding(WORD, disk)


def test_the_row_of_a_phrase_heard_while_off_has_no_field_with_words() -> None:
    """Every field that holds a phrase, those that cannot run while off today too."""
    said = {
        *SAID_FIELDS,
        "wake_unclear",
        "wake_check",
        "recovered_from",
        "guessed_from",
        "recovery",
        "guess",
        "stt_alternatives",
        "reply",
        "tools",
        "shown",
    }
    figures = {
        "wake_word": "claudio",
        "wake_score": 0.9,
        "outcome": "off",
        "route": "off",
        "stt_language": "it",
        "second_chance_ms": 120,
        "rejected": "no_name",
    }
    inter = Interaction(info={**dict.fromkeys(said, WORD), **figures}, off=True)
    row = inter.summary()
    assert WORD not in json.dumps(row)
    assert figures.items() <= row.items()
    assert inter.info["text"] == WORD  # the decisions still read it
    assert Interaction(info={"text": WORD}).summary()["text"] == WORD


def test_whispers_own_logs_hide_a_second_hearing_while_off(caplog) -> None:
    """Only the command recovery asks for one, never while off: the same rule all the same."""
    stt = real_whisper([f"Grazie per la visione, {WORD}."], [])
    stt._window = lambda audio: audio
    stt.hides_text = lambda: True
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    with caplog.at_level("INFO"):
        assert stt.listen_again(speech, "", "it").text == ""
    assert WORD not in caplog.text and HIDDEN_IN_THE_LOG in caplog.text


def test_whispers_logs_are_as_ever_when_nobody_says_it_is_off(caplog) -> None:
    stt = real_whisper([f"Grazie per la visione, {WORD}."], [])
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    with caplog.at_level("INFO"):
        assert stt.transcribe(speech) == ""
    assert WORD in caplog.text


@pytest.mark.parametrize("asked_by", ["a command", "a guess", "claude"])
async def test_while_off_an_open_question_is_dropped_not_answered(
    matcher: CommandMatcher, asked_by: str
) -> None:
    """The tray switched Claudio off with "Confermi?" waiting: a "sì" runs nothing, not even
    after "accenditi" (05/11: a probe saying just this pressed a real Alt+F4 and closed VS Code)."""
    system, rec = SystemControl(), Recorder()
    armed = [True] if asked_by == "claude" else []
    asked: list[str] = []

    async def claude(text: str, language: str) -> AsyncGenerator[str, None]:
        asked.append(text)
        yield "Ok."

    r = LocalResponder(
        matcher,
        ActionRunner(rec.ctx(), system),
        fallback=claude,
        fallback_awaiting=lambda: bool(armed),
        fallback_cancel=lambda cut: armed.clear(),
    )
    if asked_by == "a command":
        await say(r, "chiudi la finestra")
    elif asked_by == "a guess":
        r.propose("chiudi la finestra", "it")
    assert r.awaiting
    system.apply("off")
    assert await say(r, "sì") == ""
    assert r.last_route == {"route": "off"}
    assert rec.combos == [] and asked == [] and not r.awaiting
    assert (await say(r, "accenditi")).startswith("Eccomi") and not system.off
    await say(r, "sì")  # now an ordinary request: the question is gone
    assert rec.combos == []


async def test_a_story_cut_by_spegniti_does_not_go_on_while_off(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    speaker = LongSpeaker()
    responder = LocalResponder(matcher, ActionRunner(Recorder().ctx()))
    claude_says(responder, STORY)
    assistant = Assistant(
        FakeWake({3}),
        Endpointer(fake_vad, vad_settings),
        FakeStt(["raccontami una storia", "Claudio, spegniti."]),
        responder,
        speaker,
        clock=AudioClock(),
        system=responder.actions.system,
        barge_in=barge(vad_settings, ["Claudio, spegniti."]),
    )
    await speak_over(assistant, make_frames, speaker)
    assert responder.actions.system.off
    assert speaker.spoken[-1].startswith("Mi spengo")  # no "Dicevo: ..." after it


async def test_the_turn_ons_wav_leaves_out_what_was_said_before_its_wake_word(
    fake_vad, vad_settings, make_frames, matcher: CommandMatcher
) -> None:
    saved: list[np.ndarray] = []
    system = SystemControl()
    assistant = Assistant(
        FakeWake({15}),  # 1.2 s of voice before the wake word, said while off
        Endpointer(fake_vad, vad_settings),
        FakeStt("Claudio, accenditi."),
        LocalResponder(matcher, ActionRunner(Recorder().ctx(), system)),
        FakeSpeaker(),
        clock=AudioClock(),
        system=system,
        audio_sink=lambda audio, kind: saved.append(audio) or f"{kind}.wav",
    )
    system.apply("off")
    frames = make_frames("speech", 1200) + make_frames("speech", 400) + make_frames("silence", 800)
    await run_frames(assistant, frames)
    row = assistant.history[-1]
    assert not system.off and row["text"] == "Claudio, accenditi." and row["prewake_ms"] > 0
    assert len(saved) == 1 and len(saved[0]) / 16000 < 1.3  # the phrase alone
