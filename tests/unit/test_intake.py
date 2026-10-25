"""The request filter (intake.py), the short-yes rule, the noise filter in the Assistant and
the log retention."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from claudio.config import NoiseSettings
from claudio.intake import CONVERSATION, Heard, Kind, RequestFilter, Segment, content_words
from claudio.safety.confirmation import Answer, parse_answer
from claudio.session_log import purge_old_logs
from tests.unit.test_app import FakeResponder, FakeStt, build, run_frames

SURE = Segment(0.01, -0.2)
UNSURE = Segment(0.7, -0.3)  # Whisper thinks there was no speech
GARBLED = Segment(0.1, -1.1)  # Whisper is guessing the words
NOISE = NoiseSettings()  # the shipped values


class Commands(FakeResponder):
    """Understands only "apri chrome"."""

    def understands(self, text: str, language: str = "it") -> bool:
        return "chrome" in text.lower()


def decide(text: str, responder=None, f: RequestFilter | None = None, **heard):
    f = f or RequestFilter(noise=NOISE)
    return f.decide(
        text, "it", Heard(**{"wake_word": "hey_claudio", **heard}), responder or Commands()
    )


# -- decide() --


def test_lone_wake_word_listens_again_once() -> None:
    assert decide("Hey Claudio").kind is Kind.RELISTEN
    assert decide("Hey Claudio", relistened=True).kind is Kind.EMPTY
    assert decide("").kind is Kind.EMPTY


def test_false_wake_unless_the_name_was_heard() -> None:
    f = RequestFilter(verify_wake=True)
    assert decide("Hey, ciao, come stai?", f=f).kind is Kind.FALSE_WAKE
    assert decide("Hey, ciao, come stai?", f=f, name_heard=True).kind is Kind.ANSWER
    assert decide("e apri chrome", f=f).kind is Kind.ANSWER  # a command needs no name
    assert decide("ciao come stai", f=f, wake_word=CONVERSATION).kind is not Kind.FALSE_WAKE


def test_conversation_window_without_claude_keeps_only_commands() -> None:
    f = RequestFilter(claude_min_words=None)
    assert decide("e poi gli ho detto di no", f=f, wake_word=CONVERSATION).kind is Kind.NOT_REQUEST
    assert decide("apri chrome", f=f, wake_word=CONVERSATION).kind is Kind.ANSWER


def test_conversation_window_with_claude() -> None:
    f = RequestFilter(claude_min_words=3, only_addressed=False)
    assert decide("e poi gli ho detto di no", f=f, wake_word=CONVERSATION).kind is Kind.ANSWER
    assert decide("ma dai", f=f, wake_word=CONVERSATION).kind is Kind.NOT_REQUEST
    f.only_addressed = True
    assert decide("e poi gli ho detto di no", f=f, wake_word=CONVERSATION).kind is Kind.NOT_REQUEST
    assert decide("dimmi di più sul meteo", f=f, wake_word=CONVERSATION).kind is Kind.ANSWER


@pytest.mark.parametrize("segments", [[UNSURE], [GARBLED], [UNSURE, GARBLED]])
def test_short_phrase_whisper_doubts_is_noise(segments) -> None:
    decision = decide("Tremi un podcast", segments=segments)
    assert decision.kind is Kind.NOISE and decision.reason == "unsure"
    assert decision.dropped


def test_speech_is_not_noise() -> None:
    assert decide("Tremi un podcast", segments=[SURE]).kind is Kind.ANSWER
    assert decide("Tremi un podcast", segments=[UNSURE, SURE]).kind is Kind.ANSWER  # not all
    assert decide("Tremi un podcast").kind is Kind.ANSWER  # no figures: nothing to judge
    long = "che tempo farà domani a Milano secondo te"
    assert decide(long, segments=[UNSURE]).kind is Kind.ANSWER  # only short phrases


def test_quiet_phrase_is_noise() -> None:
    decision = decide("che tempo farà domani a Milano", speech_dbfs=-50.0)
    assert decision.kind is Kind.NOISE and decision.reason == "quiet"
    assert decide("che tempo farà domani a Milano", speech_dbfs=-20.0).kind is Kind.ANSWER


def test_local_commands_are_never_noise() -> None:
    """They are free: a doubtful "apri chrome" still opens Chrome."""
    assert decide("apri chrome", segments=[UNSURE], speech_dbfs=-60.0).kind is Kind.ANSWER


def test_noise_filter_off() -> None:
    off = RequestFilter(noise=NoiseSettings(enabled=False))
    assert decide("Tremi un podcast", f=off, segments=[UNSURE]).kind is Kind.ANSWER
    assert decide("Tremi", f=RequestFilter(), speech_dbfs=-80.0).kind is Kind.ANSWER


def test_quiet_answer_is_passed_on_as_nothing() -> None:
    """The TV's "sì" after "Confermi?": the responder hears nothing and does not run it."""
    responder = Commands()
    responder.awaiting = True
    decision = decide("sì", responder, speech_dbfs=-40.0, wake_word="follow_up")
    assert (
        decision.kind is Kind.ANSWER and decision.text == "" and decision.reason == "quiet_answer"
    )
    near = decide("sì", responder, speech_dbfs=-15.0, wake_word="follow_up")
    assert near.text == "sì" and near.reason == ""
    assert decide("sì", responder, wake_word="follow_up").text == "sì"  # not measured


def test_an_answer_is_never_noise() -> None:
    responder = Commands()
    responder.awaiting = True
    assert decide("no", responder, segments=[UNSURE], wake_word="follow_up").kind is Kind.ANSWER


# -- before decide() --


def test_closed_list_skips_questions_and_long_phrases() -> None:
    f, r = RequestFilter(closed_list_max_words=5), Commands()
    assert f.wants_closed_list("apri crom", "it", r)
    assert not f.wants_closed_list("come ti chiami?", "it", r)  # a question: for Claude
    assert not f.wants_closed_list("apri chrome", "it", r)  # already understood
    assert not f.wants_closed_list("uno due tre quattro cinque sei", "it", r)
    assert not f.wants_closed_list("Hey Claudio", "it", r)


def test_retry_only_in_another_language() -> None:
    f, r = RequestFilter(retry_language="it"), Commands()
    assert f.wants_retry("cue the calculator", "en", r)
    assert not f.wants_retry("cue the calculator", "it", r)
    assert not f.wants_retry("open chrome", "en", r)
    assert not RequestFilter().wants_retry("cue the calculator", "en", r)


# -- the short yes (level 2) --


@pytest.mark.parametrize("text", ["sì", "Sì, va bene", "sì sì, va bene", "Yes, go ahead"])
def test_short_yes(text: str) -> None:
    assert parse_answer(text) is Answer.YES


@pytest.mark.parametrize("text", ["sì, ma poi è andato via", "ok allora ci vediamo domani"])
def test_long_yes_is_the_tv(text: str) -> None:
    assert parse_answer(text) is Answer.UNCLEAR


def test_long_refusal_still_refuses() -> None:
    assert parse_answer("no aspetta un attimo che ci penso") is Answer.NO


# -- in the Assistant --


def request(make_frames):
    return make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800)


async def test_noise_never_reaches_the_responder(fake_vad, vad_settings, make_frames) -> None:
    assistant, _, speaker, _ = build(fake_vad, vad_settings, FakeStt("Tremi un podcast"), {1})
    responder = Commands()
    assistant.responder = responder
    assistant.intake.noise = NOISE
    assistant.stt_segments = lambda: [UNSURE]
    await run_frames(assistant, request(make_frames))
    assert responder.seen == [] and speaker.spoken == []
    assert speaker.beeps == ["wake", "cancel"]
    row = assistant.history[-1]
    assert row["outcome"] == "noise" and row["drop_reason"] == "noise"
    assert row["noise"] == "unsure" and row["stt_no_speech"] == 0.7


async def test_figures_logged_on_every_request(fake_vad, vad_settings, make_frames) -> None:
    assistant, _, speaker, _ = build(fake_vad, vad_settings, FakeStt(), {1})
    assistant.intake.noise = NOISE
    assistant.stt_segments = lambda: [SURE]
    await run_frames(assistant, request(make_frames))
    assert speaker.spoken == ["Apro Chrome."]
    row = assistant.history[-1]
    assert row["stt_logprob"] == -0.2 and "drop_reason" not in row


async def test_far_answer_is_not_counted(fake_vad, vad_settings, make_frames) -> None:
    """The test tone peaks around -9 dBFS: with the floor at -5 it is "far away"."""
    assistant, _, _, responder = build(fake_vad, vad_settings, FakeStt("sì"), {1})
    responder.awaiting = True
    assistant.intake.noise = NoiseSettings(answer_min_dbfs=-5.0)
    await run_frames(assistant, request(make_frames))
    assert responder.seen == [""]
    assert assistant.history[-1]["intake"] == "quiet_answer"


# -- the transcriber's figures --


def test_transcriber_keeps_the_segments_figures() -> None:
    from tests.unit.test_responder_metrics_stt import make_stt

    stt = make_stt()
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    stt.transcribe(speech)
    assert stt.last_segments == [Segment(0.01, -0.2)]
    stt.transcribe(speech, quick=True)  # the second chance leaves them alone
    assert stt.last_segments == [Segment(0.01, -0.2)]
    stt.transcribe(np.zeros(16000, dtype=np.float32))
    assert stt.last_segments == []


# -- retention --


def test_old_logs_are_deleted(tmp_path) -> None:
    now = time.time()
    old_t = now - 40 * 86400
    sessions, audio = tmp_path / "sessions", tmp_path / "audio"
    old_session, new_session, current = sessions / "old", sessions / "new", sessions / "now"
    for folder in (old_session, new_session, current):
        folder.mkdir(parents=True)
        (folder / "console.log").write_text("x")
    audio.mkdir()
    old_wav, new_wav = audio / "a_near_miss.wav", audio / "b_ok.wav"
    old_wav.write_bytes(b"x")
    new_wav.write_bytes(b"x")
    live, rotated = tmp_path / "metrics.jsonl", tmp_path / "metrics.jsonl.2"
    live.write_text("{}")
    rotated.write_text("{}")
    for path in (old_session, current, old_wav, live, rotated):
        os.utime(path, (old_t, old_t))

    assert purge_old_logs(tmp_path, 30, current=current, now=now) == 3
    assert not old_session.exists() and not old_wav.exists() and not rotated.exists()
    assert new_session.exists() and new_wav.exists() and current.exists() and live.exists()
    assert purge_old_logs(tmp_path / "missing", 30) == 0


def test_interjections_are_not_words() -> None:
    assert content_words("Mm-hmm.") == 0
    assert content_words("Ehm, che ore sono?") == 3
    assert content_words("aspetta un attimo") == 3
