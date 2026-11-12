"""The request filter (intake.py), the short-yes rule, the noise filter in the Assistant and
the log retention."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from claudio.config import NoiseSettings
from claudio.intake import (
    CONVERSATION,
    Heard,
    Kind,
    RequestFilter,
    Segment,
    content_words,
    echo_of,
    incomplete,
)
from claudio.safety.confirmation import Answer, parse_answer
from claudio.session_log import purge_old_logs
from tests.unit.test_app import FakeResponder, FakeStt, build, run_frames

SURE = Segment(0.01, -0.2)
UNSURE = Segment(0.7, -0.3)  # Whisper thinks there was no speech
GARBLED = Segment(0.1, -1.1)  # Whisper is guessing the words
NOISE = NoiseSettings()  # the shipped values


class Commands(FakeResponder):
    """Understands only "apri chrome"."""

    def understands(self, text: str, language: str = "it", memory: bool = True) -> bool:
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


@pytest.mark.parametrize(
    "text",
    ["Apri Chrome e cerca...", "Dimmi...", "Claudio, apri la", "Scrivi una mail a", "open the…"],
)
def test_phrases_left_halfway(text: str) -> None:
    assert incomplete(text)


@pytest.mark.parametrize(
    "text",
    [
        "Apri Chrome e cerca Dainese Moto.",
        "Claudio apri la cartella tesi.",
        "Che ore sono?",
        "La",
        "Hai cercato e?",
        "Grazie.",
    ],
)
def test_finished_phrases(text: str) -> None:
    assert not incomplete(text)


def test_a_phrase_left_halfway_waits_once() -> None:
    f = RequestFilter(continue_incomplete=True)
    assert decide("Apri Firefox e cerca...", f=f).kind is Kind.CONTINUE
    assert decide("Apri Firefox e cerca...", f=f, continued=True).kind is Kind.ANSWER
    assert decide("Apri Firefox e cerca...").kind is Kind.ANSWER  # off by default
    assert decide("apri chrome e", f=f).kind is Kind.ANSWER  # a command runs at once
    assert not decide("Dimmi...", f=f).dropped


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


def test_a_short_phrase_with_the_name_is_a_request() -> None:
    """26/10 19:52: "Claudio QDT." (chiuditi) in the window was dropped as the TV."""
    f = RequestFilter(claude_min_words=3, only_addressed=True)
    assert decide("Claudio QDT.", f=f, wake_word=CONVERSATION).kind is Kind.ANSWER
    assert decide("ma dai", f=f, wake_word=CONVERSATION).kind is Kind.NOT_REQUEST


# CLAUDIO's reply and what Whisper made of it in the microphone (logs of 26/10).
CAMPIONE = "Ciao, campione. Sempre un piacere, anche se ripeti come un disco rotto."
UNIVERSO = "Ciao, socio. Eccomi, il genio più bello dell'universo è in linea. Che vuoi fare?"
MISTERO = "Ah, il grande mistero: non dire niente. Strategia audace, socio, quasi zen."
GIOCO = "Se vuoi altro, dimmelo chiaro: non ho capito se intendi di nuovo il gioco."
BLOCCO = "Fatto, il blocco note è pulito. Un foglio bianco, come la mia modestia."


@pytest.mark.parametrize(
    ("heard", "reply"),
    [
        ("ciao cappiore", CAMPIONE),
        ("dell'università e lì.", UNIVERSO),
        ("Ah, che grande vista.", MISTERO),
        ("e come il genio", UNIVERSO),
        ("Non capito se intendo.", GIOCO),
        ("di nuovo il gioco", GIOCO),
        ("sono le 14 e 5", "Sono le 14 e 5, capo."),
    ],
)
def test_the_reply_misheard_is_its_echo(heard: str, reply: str) -> None:
    assert echo_of(heard, reply)


@pytest.mark.parametrize(
    ("heard", "reply"),
    [
        ("Claudio, che di blocco.", BLOCCO),  # the name: JenJo talking
        ("Claudio, chiudi il blocco note", BLOCCO),
        ("aspetta, che ore sono?", MISTERO),
        ("No, raccontala lo stesso", MISTERO),
        ("Basta, grazie.", CAMPIONE),
        ("apri la cartella progetti", UNIVERSO),
    ],
)
def test_words_said_over_the_reply_are_not_its_echo(heard: str, reply: str) -> None:
    assert not echo_of(heard, reply)


def test_the_name_is_echo_only_when_the_reply_says_it() -> None:
    assert echo_of("sono Claudio, il tuo assistente", "Sono Claudio, il tuo assistente.")
    assert not echo_of("", "Sono Claudio.") and not echo_of("ciao", "")


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


@pytest.mark.parametrize("text", ["No, Claudio.", "Sì, Claudio.", "Ok Claudio."])
def test_an_answer_with_the_name_is_an_answer_while_a_question_is_open(text: str) -> None:
    responder = Commands()
    responder.awaiting = True
    assert decide(text, responder).kind is Kind.ANSWER
    assert decide(text, responder, relistened=True).kind is Kind.ANSWER


def test_an_unnamed_answer_after_a_noise_wake_is_a_false_wake() -> None:
    f = RequestFilter(verify_wake=True)
    responder = Commands()
    responder.awaiting = True
    assert decide("sì", responder, f=f).kind is Kind.FALSE_WAKE
    assert decide("Claudio, sì", responder, f=f).kind is Kind.ANSWER
    assert decide("Sì, Claudio.", responder, f=f).kind is Kind.ANSWER
    assert decide("sì", responder, f=f, name_heard=True).kind is Kind.ANSWER
    assert decide("sì", responder, f=f, second_chance=True).kind is Kind.ANSWER
    for window in ("follow_up", CONVERSATION):  # the window right after the question
        assert decide("sì", responder, f=f, wake_word=window).kind is Kind.ANSWER


def test_the_name_alone_still_listens_again_while_a_question_is_open() -> None:
    responder = Commands()
    responder.awaiting = True
    assert decide("Claudio.", responder).kind is Kind.RELISTEN


def test_no_with_the_name_is_unchanged_without_an_open_question() -> None:
    assert decide("No, Claudio.").kind is Kind.RELISTEN
    assert decide("Ok Claudio.").kind is Kind.RELISTEN


def test_an_answer_is_never_noise() -> None:
    responder = Commands()
    responder.awaiting = True
    assert decide("no", responder, segments=[UNSURE], wake_word="follow_up").kind is Kind.ANSWER


# -- Task 3: the answer's confidence and the voice lost while deaf --

WEAK = Segment(0.01, -0.9)  # under the floor of -0.75
FAIR = Segment(0.01, -0.5)


FLOOR = RequestFilter(noise=NoiseSettings(answer_min_logprob=-0.75))  # the floor, switched on


def answer_to_question(text: str, f: RequestFilter | None = None, **heard):
    responder = Commands()
    responder.awaiting = True
    return decide(text, responder, f, wake_word="follow_up", **heard)


def test_the_floor_and_the_deaf_voice_have_their_settings() -> None:
    assert NoiseSettings().answer_min_logprob is None  # off: it stopped no false yes (08/11)
    assert NoiseSettings().name_min_logprob == -0.9
    assert NoiseSettings().answer_deaf_voice_ms == 100


@pytest.mark.parametrize("text", ["sì", "Va bene.", "Confermo.", "Sì, Claudio."])
def test_a_yes_whisper_is_unsure_of_is_passed_on_as_nothing(text: str) -> None:
    decision = answer_to_question(text, FLOOR, segments=[SURE, WEAK])  # the weakest counts
    assert decision.kind is Kind.ANSWER and decision.text == ""
    assert decision.reason == "unsure_yes"


def test_a_yes_above_the_floor_or_without_figures_is_a_yes() -> None:
    assert answer_to_question("sì", FLOOR, segments=[FAIR]).text == "sì"
    assert answer_to_question("sì", FLOOR).text == "sì"  # no segments, no floor


def test_with_no_floor_a_weak_yes_runs() -> None:
    assert answer_to_question("Sì.", segments=[Segment(0.01, -1.5)]).text == "Sì."


def test_a_no_below_the_floor_is_still_a_no() -> None:
    decision = answer_to_question("no", FLOOR, segments=[Segment(0.01, -1.0)])
    assert decision.text == "no" and decision.reason == ""


def test_an_unclear_phrase_below_the_floor_is_passed_on_as_it_is() -> None:
    assert answer_to_question("che ore sono", FLOOR, segments=[WEAK]).text == "che ore sono"


def test_a_weak_yes_is_a_weak_yes_at_level_3_too() -> None:
    decision = answer_to_question("confermo", FLOOR, segments=[WEAK])
    assert decision.text == "" and decision.reason == "unsure_yes"


@pytest.mark.parametrize("text", ["va bene", "no", "no, aspetta", "sì"])
def test_voice_lost_while_deaf_is_no_answer_at_all(text: str) -> None:
    decision = answer_to_question(text, deaf_voice_ms=100)
    assert decision.kind is Kind.ANSWER and decision.text == ""
    assert decision.reason == "deaf_answer"


def test_a_little_voice_while_deaf_is_the_echo_not_a_lost_head() -> None:
    assert answer_to_question("va bene", deaf_voice_ms=99).text == "va bene"
    assert answer_to_question("va bene", deaf_voice_ms=0).text == "va bene"


def test_a_quiet_answer_keeps_its_own_reason() -> None:
    decision = answer_to_question("sì", speech_dbfs=-40.0, segments=[WEAK], deaf_voice_ms=500)
    assert decision.reason == "quiet_answer"


# -- before decide() --


def test_closed_list_skips_questions_and_long_phrases() -> None:
    f, r = RequestFilter(closed_list_max_words=5), Commands()
    assert f.wants_closed_list("apri crom", "it", r)
    assert not f.wants_closed_list("come ti chiami?", "it", r)  # a question: for Claude
    assert not f.wants_closed_list("apri chrome", "it", r)  # already understood
    assert not f.wants_closed_list("uno due tre quattro cinque sei", "it", r)
    assert not f.wants_closed_list("Hey Claudio", "it", r)


# -- the short yes (level 2) --


@pytest.mark.parametrize("text", ["sì", "Sì, va bene", "sì sì, va bene"])
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
    (audio / "rejected").mkdir(parents=True)
    old_wav, new_wav = audio / "a_near_miss.wav", audio / "b_ok.wav"
    old_rejected = audio / "rejected" / "c_no_name.wav"  # a second chance's, kept apart
    for wav in (old_wav, new_wav, old_rejected):
        wav.write_bytes(b"x")
    live, rotated = tmp_path / "metrics.jsonl", tmp_path / "metrics.jsonl.2"
    live.write_text("{}")
    rotated.write_text("{}")
    for path in (old_session, current, old_wav, old_rejected, live, rotated):
        os.utime(path, (old_t, old_t))

    assert purge_old_logs(tmp_path, 30, current=current, now=now) == 4
    assert not old_session.exists() and not old_wav.exists() and not rotated.exists()
    assert not old_rejected.exists()
    assert new_session.exists() and new_wav.exists() and current.exists() and live.exists()
    assert purge_old_logs(tmp_path / "missing", 30) == 0


def test_purge_old_logs_removes_old_notes(tmp_path) -> None:
    now = time.time()
    audio = tmp_path / "audio"
    (audio / "rejected").mkdir(parents=True)
    old, new = audio / "a_ok.json", audio / "rejected" / "b_no_name.json"
    for note in (old, new):
        note.write_text("{}")
    os.utime(old, (now - 40 * 86400,) * 2)
    assert purge_old_logs(tmp_path, 30, now=now) == 1
    assert not old.exists() and new.exists()


def test_interjections_are_not_words() -> None:
    assert content_words("Mm-hmm.") == 0
    assert content_words("Ehm, che ore sono?") == 3
    assert content_words("aspetta un attimo") == 3
