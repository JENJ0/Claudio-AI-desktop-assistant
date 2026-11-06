"""Fast Italian heard as English: a stricter bar for English, and a retry in Italian."""

from __future__ import annotations

import numpy as np
import pytest

from claudio.app import Assistant
from claudio.audio.vad import Endpointer
from claudio.intake import Segment
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import LocalResponder
from claudio.stt.transcriber import pick_language
from tests.unit.test_actions import Recorder
from tests.unit.test_app import AudioClock, FakeSpeaker, FakeWake, run_frames


@pytest.mark.parametrize(
    ("probs", "expected"),
    [
        ({"en": 0.7, "it": 0.3}, "it"),  # 70% English: not enough to leave Italian
        ({"en": 0.9, "it": 0.1}, "en"),  # 90%: really English
        ({"it": 0.65, "en": 0.35}, "it"),  # Italian needs only the usual 60%
    ],
)
def test_english_needs_more_confidence(probs: dict[str, float], expected: str) -> None:
    assert pick_language(probs, ["it", "en"], "it", 0.6, 0.8)[0] == expected


def test_unsure_english_keeps_the_last_language() -> None:
    assert pick_language({"en": 0.7, "it": 0.3}, ["it", "en"], "en", 0.6, 0.8)[0] == "en"


class TwoLanguageStt:
    """First pass: what Whisper guessed (`first`, in `heard`); forced Italian: `italian`."""

    def __init__(self, first: str, heard: str, italian: str) -> None:
        self.first, self.heard, self.italian = first, heard, italian
        self.language = heard
        self.calls: list[str | None] = []

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> str:
        self.calls.append(language)
        self.language = language or self.heard
        return self.italian if language else self.first


def build(fake_vad, vad_settings, matcher: CommandMatcher, stt: TwoLanguageStt, retry: bool):
    rec = Recorder()
    speaker = FakeSpeaker()
    assistant = Assistant(
        FakeWake({3}),
        Endpointer(fake_vad, vad_settings),
        stt,
        LocalResponder(matcher, ActionRunner(rec.ctx())),
        speaker,
        clock=AudioClock(),
        retry_language="it" if retry else None,
    )
    return assistant, speaker, rec


def request(make_frames) -> list[np.ndarray]:
    return make_frames("silence", 640) + make_frames("speech", 1000) + make_frames("silence", 800)


async def test_misheard_english_is_retried_in_italian(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    stt = TwoLanguageStt(
        "Hey Claudio, cue the calculator.", "en", "Hey Claudio, apri la calcolatrice."
    )
    assistant, _speaker, rec = build(fake_vad, vad_settings, matcher, stt, retry=True)
    await run_frames(assistant, request(make_frames))
    assert stt.calls == [None, "it"]
    assert rec.ran == [["calc.exe"]]
    row = assistant.history[-1]
    assert row["command"] == "open_calculator"
    assert row["retried_from"] == "Hey Claudio, cue the calculator."
    assert row["stt_language"] == "it"


async def test_a_failing_retry_keeps_the_first_transcription(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    class FailingRetry(TwoLanguageStt):
        def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> str:
            if language:
                raise RuntimeError("CUDA esplosa")
            return super().transcribe(audio)

    stt = FailingRetry("Hey Claudio, cue the calculator.", "en", "")
    assistant, speaker, _rec = build(fake_vad, vad_settings, matcher, stt, retry=True)
    await run_frames(assistant, request(make_frames))
    assert "error" not in speaker.beeps  # the first transcription goes on to the router
    row = assistant.history[-1]
    assert row["text"] == "Hey Claudio, cue the calculator." and row["stt_language"] == "en"
    assert "retried_from" not in row and row["outcome"] != "error"
    assert not assistant._stt_lock.locked()


async def test_real_english_is_not_retried(fake_vad, vad_settings, make_frames, matcher) -> None:
    stt = TwoLanguageStt("Hey Claudio, open Chrome.", "en", "Ehi Claudio, open crome.")
    assistant, speaker, _rec = build(fake_vad, vad_settings, matcher, stt, retry=True)
    await run_frames(assistant, request(make_frames))
    assert stt.calls == [None]
    assert speaker.languages == ["en"]


async def test_a_retry_not_understood_keeps_the_first_text(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    stt = TwoLanguageStt("What's the weather?", "en", "Che tempo fa?")
    assistant, speaker, _ = build(fake_vad, vad_settings, matcher, stt, retry=True)
    await run_frames(assistant, request(make_frames))
    assert stt.calls == [None, "it"]
    assert speaker.spoken == ["I can't do that yet."]
    assert "retried_from" not in assistant.history[-1]


@pytest.mark.parametrize(
    ("first", "italian", "logprob"),
    [
        ("What's the weather?", "Che tempo fa?", -0.65),  # not understood: the first pass stands
        ("Hey Claudio, cue the calculator.", "apri la calcolatrice", None),  # a command: no doubt
    ],
)
async def test_whispers_doubts_are_those_of_the_first_pass(
    fake_vad, vad_settings, make_frames, matcher, first: str, italian: str, logprob: float | None
) -> None:
    stt = TwoLanguageStt(first, "en", italian)
    assistant, *_ = build(fake_vad, vad_settings, matcher, stt, retry=True)
    assistant.stt_segments = lambda: [Segment(0.05, -0.65 if len(stt.calls) == 1 else -0.05)]
    await run_frames(assistant, request(make_frames))
    assert stt.calls == [None, "it"] and assistant.history[-1].get("stt_logprob") == logprob


async def test_italian_and_disabled_retry_cost_nothing(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    stt = TwoLanguageStt("prenota un volo", "it", "x")
    assistant, *_ = build(fake_vad, vad_settings, matcher, stt, retry=True)
    await run_frames(assistant, request(make_frames))
    stt2 = TwoLanguageStt("cue the calculator", "en", "apri la calcolatrice")
    assistant2, *_ = build(fake_vad, vad_settings, matcher, stt2, retry=False)
    await run_frames(assistant2, request(make_frames))
    assert stt.calls == [None] and stt2.calls == [None]


async def test_a_retry_not_understood_leaves_the_first_pass_prompt_in_the_note(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    from tests.unit.test_app import NotingSink

    class PromptedStt(TwoLanguageStt):
        last_prompt = ""

        def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> str:
            self.last_prompt = "Italian prompt" if language else "English prompt"
            return super().transcribe(audio, language=language)

    stt = PromptedStt("Hey Claudio, cue the calculator.", "en", "Hey Claudio, blah blah blah.")
    assistant, *_ = build(fake_vad, vad_settings, matcher, stt, retry=True)
    assistant.audio_sink = sink = NotingSink()
    await run_frames(assistant, request(make_frames))
    assert stt.calls == [None, "it"]
    assert assistant.history[-1]["stt_language"] == "en"
    (_, meta), *_ = sink.notes
    assert (meta["language"], meta["whisper_prompt"]) == ("en", "English prompt")
    assert "whisper_prompt" not in assistant.history[-1]  # folder names stay out of the rows
