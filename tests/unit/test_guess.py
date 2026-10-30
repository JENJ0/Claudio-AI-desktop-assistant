"""IT-0: a phrase about to be dropped that sounds like a command. Claude guesses which command
JenJo asked ("Spenditi." in the conversation window, "Audio Bust." after a reply) and
Claudio acts on it, asks "Hai detto: ...?" first, or drops it as before."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from claudio.app import Assistant
from claudio.brain.guess import SCHEMA, Guesser, catalog, describe, parse
from claudio.config import ClaudeSettings, GuessSettings
from claudio.cost.pricing import Pricing, Usage, load_pricing
from claudio.intake import (
    CONVERSATION,
    Decision,
    Doubt,
    Guess,
    Heard,
    Kind,
    Recovery,
    Verdict,
    verdict,
)
from claudio.metrics import Interaction
from claudio.router.actions import MEDIA_KEYS, VK_F4, VK_MENU, ActionRunner
from claudio.router.matcher import Command, CommandMatcher, Near, load_commands
from claudio.router.responder import LocalResponder
from claudio.wakeword.verifier import after_clipped_name
from tests.conftest import COMMANDS_FILE
from tests.unit.test_actions import Recorder
from tests.unit.test_conversation import build as build_conversation
from tests.unit.test_conversation import talk
from tests.unit.test_cost import PRICES, tracker


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


# --- how close a phrase is to the commands --------------------------------------------


def test_nearest_lists_the_closest_commands_best_first(matcher: CommandMatcher) -> None:
    near = matcher.nearest(["Prossima."])
    assert near[0].command.id == "media_next"
    assert [n.score for n in near] == sorted((n.score for n in near), reverse=True)
    assert len({n.command.id for n in near}) == len(near)  # one entry per command


def test_nearest_counts_exact_only_commands(matcher: CommandMatcher) -> None:
    # "spegniti" never matches fuzzily (exact = true), but it is what was said.
    assert matcher.match("Spenditi.") is None
    near = matcher.nearest(["Spenditi."])
    assert near[0].command.id == "off" and near[0].score >= 85
    assert near[0].phrase == "spegniti"  # as written in commands.toml, for Claude


def test_nearest_compares_only_the_heard_language(matcher: CommandMatcher) -> None:
    # "Come stai?" is 74 from the English "come again", and means nothing like it.
    near = matcher.nearest(["Come stai?"], "it")
    assert all(n.phrase in n.command.phrases_in("it") for n in near)


def test_nearest_keeps_the_best_of_all_hearings(matcher: CommandMatcher) -> None:
    alone = matcher.nearest(["Pasta."])
    both = matcher.nearest(["Pasta.", "Basta."])
    assert both[0].command.id == "stop" and both[0].score >= alone[0].score


def test_nearest_limit_and_nothing_heard(matcher: CommandMatcher) -> None:
    assert len(matcher.nearest(["apri"], limit=2)) == 2
    assert matcher.nearest([""]) == []


def test_command_by_id(matcher: CommandMatcher) -> None:
    cmd = matcher.command("volume_up")
    assert cmd is not None and cmd.id == "volume_up"
    assert matcher.command("no_such_command") is None


@pytest.mark.parametrize(
    ("text", "rest"),
    [
        ("Audio, alsa il volu", "alsa il volu"),
        ("Ehi l'audio basta", "basta"),
        ("Alza il volume", ""),
    ],
)
def test_words_after_a_clipped_name(text: str, rest: str) -> None:
    assert after_clipped_name(text) == rest


# --- the verdict ------------------------------------------------------------------------


DOUBT = Doubt("Spenditi.", "it", "not_a_request", close=True)


@pytest.mark.parametrize(
    ("guess", "doubt", "asks_first", "expected"),
    [
        (None, DOUBT, False, Verdict.DROP),
        (Guess("", 0.95), DOUBT, False, Verdict.DROP),  # Claude: not for Claudio
        (Guess("basta", 0.4), DOUBT, False, Verdict.DROP),
        (Guess("basta", 0.6), DOUBT, False, Verdict.ASK),
        (Guess("basta", 0.8), DOUBT, False, Verdict.ACT),
        (Guess("basta", 0.95), Doubt("x", "it", "unsure", close=False), False, Verdict.DROP),
        (Guess("spegniti", 0.99), DOUBT, True, Verdict.ASK),  # never off on a guess alone
    ],
)
def test_verdict(guess: Guess | None, doubt: Doubt, asks_first: bool, expected: Verdict) -> None:
    assert verdict(guess, doubt, 0.8, 0.5, asks_first) is expected


def test_ask_cannot_be_above_act() -> None:
    with pytest.raises(ValueError):
        GuessSettings(act_at=0.5, ask_at=0.7)


# --- what Claude is told and what it answers ---------------------------------------------


def test_catalog_has_every_command_with_a_few_phrases(matcher: CommandMatcher) -> None:
    commands = _commands(matcher)
    text = catalog(commands)
    assert text.count("\n") == len(commands) - 1
    assert "- volume_up: alza il volume / " in text
    assert "- off: spegniti / spengiti / spegniti pure" in text
    assert "puoi spegnerti" not in text  # three phrases each: the prompt stays short


def test_describe_tells_how_the_phrase_was_heard(matcher: CommandMatcher) -> None:
    doubt = Doubt(
        "Audio Bust.",
        "it",
        "not_a_request",
        alternatives=("Audio basta.",),
        wake_word=CONVERSATION,
        name_heard=True,
        close=True,
        logprob=-0.9,
        said_before="C'era una volta un robot " * 20,
    )
    text = describe(doubt, matcher.nearest(["Audio Bust.", "basta"]))
    assert text.startswith('Heard: "Audio Bust."')
    assert 'Other hearings: "Audio basta."' in text
    assert "did not look addressed to Claudio" in text
    assert "conversation window" in text and "close to the microphone" in text
    assert "very unsure" in text
    assert "Name at the start: yes." in text
    assert "Claudio's reply just before:" in text and text.count("robot") < 20  # cut
    assert "Closest commands: stop" in text


def test_describe_after_a_wake_word(matcher: CommandMatcher) -> None:
    doubt = Doubt("Prossima.", "it", "name_not_heard", wake_word="hey_claudio", wake_score=0.93)
    text = describe(doubt, matcher.nearest(["Prossima."]))
    assert 'after the wake word "hey_claudio" (score 0.93)' in text
    assert "the name was not in the phrase" in text
    assert "reply just before" not in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            '{"why": "x", "command": "stop", "said": "basta", "probability": 0.9}',
            Guess("basta", 0.9, "stop", "x"),
        ),
        (
            '{"why": "", "command": "stop", "said": " basta ", "probability": 3}',
            Guess("basta", 1.0, "stop", ""),
        ),
        (
            '{"why": "", "command": "spegni_tutto", "said": "x", "probability": 0.7}',
            Guess("x", 0.7, "", ""),  # not one of Claudio's commands
        ),
        ('{"why": "cut', None),  # cut by max_tokens
        ('{"command": "stop"}', None),
    ],
)
def test_parse(raw: str, expected: Guess | None, matcher: CommandMatcher) -> None:
    assert parse(raw, {c.id for c in _commands(matcher)}) == expected


# --- the call ---------------------------------------------------------------------------


class GuessClient:
    """`client.messages.create`, as anthropic.AsyncAnthropic."""

    def __init__(
        self,
        answer: dict[str, Any] | None = None,
        stop: str = "end_turn",
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.answer = answer or {"why": "", "command": "stop", "said": "basta", "probability": 0.9}
        self.stop = stop
        self.error = error
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.answer))],
            usage={"input_tokens": 1000, "output_tokens": 40},
            stop_reason=self.stop,
        )


def guesser(
    pricing: Pricing, matcher: CommandMatcher, client: Any, budget: float = 1.0, **settings: Any
) -> Guesser:
    return Guesser(
        client,
        tracker(pricing, budget),
        ClaudeSettings(),
        GuessSettings(**settings),
        _commands(matcher),
    )


async def test_one_short_call_counted_by_the_spending_counter(
    pricing: Pricing, matcher: CommandMatcher
) -> None:
    client = GuessClient()
    g = guesser(pricing, matcher, client)
    guess = await g(DOUBT, matcher.nearest(["Spenditi."]))
    assert guess == Guess("basta", 0.9, "stop", "")
    params = client.calls[0]
    assert params["model"] == ClaudeSettings().model
    assert params["max_tokens"] == GuessSettings().max_tokens
    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert "- off: spegniti" in params["system"]
    assert params["messages"][0]["content"].startswith('Heard: "Spenditi."')
    assert g.last_call["guess_in"] == 1000 and g.last_call["guess_out"] == 40
    assert g.last_call["guess_cost"] == pytest.approx(0.0001 + 0.00002)
    assert g.tracker.calls_today() == 1


async def test_the_system_prompt_is_the_same_at_every_call(
    pricing: Pricing, matcher: CommandMatcher
) -> None:
    client = GuessClient()
    g = guesser(pricing, matcher, client)
    await g(DOUBT, matcher.nearest(["Spenditi."]))
    await g(Doubt("Prossima.", "it", "unsure"), matcher.nearest(["Prossima."]))
    assert client.calls[0]["system"] == client.calls[1]["system"]


async def test_no_key_and_spent_budget_cost_nothing(
    pricing: Pricing, matcher: CommandMatcher
) -> None:
    assert await guesser(pricing, matcher, None)(DOUBT, []) is None
    client = GuessClient()
    g = guesser(pricing, matcher, client, budget=0.001)
    g.tracker.record(g.model, Usage(input_tokens=100_000), "5m")  # 0.01 $: all spent
    assert await g(DOUBT, []) is None
    assert client.calls == []


@pytest.mark.parametrize(
    ("client", "error"),
    [
        (GuessClient(delay=1.0), "timeout"),
        (GuessClient(error=ConnectionError("down")), "ConnectionError"),
    ],
)
async def test_no_answer_drops_the_phrase(
    pricing: Pricing, matcher: CommandMatcher, client: GuessClient, error: str
) -> None:
    g = guesser(pricing, matcher, client, timeout_s=0.05)
    assert await g(DOUBT, []) is None
    assert g.last_call == {"guess_error": error}


async def test_refusal_is_no_guess(pricing: Pricing, matcher: CommandMatcher) -> None:
    g = guesser(pricing, matcher, GuessClient(stop="refusal"))
    assert await g(DOUBT, []) is None
    assert g.tracker.calls_today() == 1  # paid all the same


# --- the responder: act, ask or drop -------------------------------------------------------


class FakeGuesser:
    def __init__(self, guess: Guess | None) -> None:
        self.guess = guess
        self.calls: list[tuple[Doubt, list[Near]]] = []

    async def __call__(self, doubt: Doubt, near: Sequence[Near]) -> Guess | None:
        self.calls.append((doubt, list(near)))
        return self.guess


def responder(
    matcher: CommandMatcher, guess: Guess | None, rec: Recorder | None = None, **settings: Any
) -> tuple[LocalResponder, FakeGuesser, Recorder]:
    rec = rec or Recorder()
    fake = FakeGuesser(guess)
    r = LocalResponder(
        matcher,
        ActionRunner(rec.ctx()),
        fallback_guess=fake,
        fallback_guess_info=lambda: {"guess_ms": 400},
        guess=GuessSettings(**settings),
    )
    return r, fake, rec


async def say(r: LocalResponder, text: str, language: str = "it") -> str:
    return "".join([c async for c in r.respond(text, language)])


async def test_nothing_close_to_a_command_costs_nothing(matcher: CommandMatcher) -> None:
    r, fake, _ = responder(matcher, Guess("basta", 0.9, "stop"))
    assert await r.guess(Doubt("Grazie a tutti.", "it", "unsure")) is None
    assert fake.calls == []


async def test_sure_guess_is_answered(matcher: CommandMatcher) -> None:
    r, fake, rec = responder(matcher, Guess("canzone successiva", 0.9, "media_next", "song"))
    recovery = await r.guess(Doubt("Prossima.", "it", "not_a_request", close=True))
    assert recovery is not None and recovery.verdict is Verdict.ACT
    assert recovery.said == "canzone successiva" and not r.awaiting
    assert recovery.info["guess_p"] == 0.9 and recovery.info["guess_ms"] == 400
    assert recovery.info["guess_near"].startswith("media_next ")
    assert fake.calls[0][1][0].command.id == "media_next"
    await say(r, recovery.said)
    assert rec.keys == [MEDIA_KEYS["next_track"]]


async def test_the_words_after_a_clipped_name_are_compared(matcher: CommandMatcher) -> None:
    r, fake, _ = responder(matcher, Guess("alza il volume", 0.9, "volume_up"))
    recovery = await r.guess(Doubt("Audio, alsa il volu", "it", "name_not_heard"))
    assert recovery is not None and recovery.verdict is Verdict.ACT
    assert any(n.command.id == "volume_up" for n in fake.calls[0][1])


async def test_switching_off_is_always_asked(matcher: CommandMatcher) -> None:
    r, _, _ = responder(matcher, Guess("spegniti", 0.99, "off"))
    recovery = await r.guess(Doubt("Spenditi.", "it", "not_a_request", close=True))
    assert recovery is not None and recovery.verdict is Verdict.ASK
    assert recovery.question == "Hai detto: spegniti?"
    assert r.awaiting and not r.actions.system.off
    assert await say(r, "sì") == "Mi spengo. Per riaccendermi di' Claudio, accenditi."
    assert r.actions.system.off and not r.awaiting


async def test_no_drops_the_guess(matcher: CommandMatcher) -> None:
    r, _, _ = responder(matcher, Guess("spegniti", 0.99, "off"))
    await r.guess(Doubt("Spenditi.", "it", "not_a_request"))
    assert await say(r, "no") == "Va bene, lascio stare."
    assert not r.actions.system.off and not r.awaiting
    assert r.last_route["route"] == "guess_refused"


async def test_no_and_another_request_answers_that(matcher: CommandMatcher) -> None:
    r, _, rec = responder(matcher, Guess("spegniti", 0.99, "off"))
    await r.guess(Doubt("Spenditi.", "it", "not_a_request"))
    await say(r, "no, alza il volume")
    assert not r.actions.system.off
    assert rec.keys and set(rec.keys) == {MEDIA_KEYS["volume_up"]}


async def test_the_questions_own_echo_is_no_answer(matcher: CommandMatcher) -> None:
    r, _, _ = responder(matcher, Guess("spegniti", 0.99, "off"))
    await r.guess(Doubt("Spenditi.", "it", "not_a_request"))
    assert await say(r, "detto spegniti") == ""
    assert not r.actions.system.off and not r.awaiting
    assert r.last_route["route"] == "guess_echo"


async def test_unsure_guess_asks_and_a_yes_needs_no_second_question(
    matcher: CommandMatcher,
) -> None:
    r, _, rec = responder(matcher, Guess("chiudi la finestra", 0.6, "close_window"))
    recovery = await r.guess(Doubt("Chiudi la fenestra.", "it", "not_a_request"))
    assert recovery is not None and recovery.verdict is Verdict.ASK
    assert await say(r, "sì, grazie") == "Fatto."  # no "Confermi?" after "Hai detto"
    assert rec.combos == [(VK_MENU, VK_F4)]


async def test_sure_level_2_command_still_asks_its_own_question(matcher: CommandMatcher) -> None:
    r, _, rec = responder(matcher, Guess("chiudi la finestra", 0.9, "close_window"))
    recovery = await r.guess(Doubt("Chiudi la fenestra.", "it", "not_a_request"))
    assert recovery is not None and recovery.verdict is Verdict.ACT
    assert (await say(r, recovery.said)).endswith("Confermi?")
    assert rec.combos == []


async def test_silence_after_the_question_does_nothing(matcher: CommandMatcher) -> None:
    r, _, _ = responder(matcher, Guess("spegniti", 0.99, "off"))
    await r.guess(Doubt("Spenditi.", "it", "not_a_request"))
    r.cancel_pending(silence=True)
    assert not r.awaiting and not r.actions.system.off


@pytest.mark.parametrize(
    "guess",
    [
        None,  # no answer from Claude
        Guess("", 0.0),  # not for Claudio
        Guess("che tempo fa", 0.9),  # a request, but not one of the commands
        Guess("basta", 0.3, "stop"),
    ],
)
async def test_dropped_as_before(matcher: CommandMatcher, guess: Guess | None) -> None:
    r, fake, _ = responder(matcher, guess)
    recovery = await r.guess(Doubt("Pasta.", "it", "unsure"))
    assert recovery is not None and recovery.verdict is Verdict.DROP
    assert len(fake.calls) == 1 and not r.awaiting


async def test_claudes_words_become_the_commands_own(matcher: CommandMatcher) -> None:
    # Claude named off but wrote words no command matches: its first phrase is used.
    r, _, _ = responder(matcher, Guess("spegni il programma", 0.95, "off"))
    recovery = await r.guess(Doubt("Spenditi.", "it", "not_a_request"))
    assert recovery is not None and recovery.said == "spegniti"


async def test_the_number_said_is_kept(matcher: CommandMatcher) -> None:
    r, _, rec = responder(matcher, Guess("alza volume di 5", 0.9, "volume_up"))
    recovery = await r.guess(Doubt("Alsa il volu di 5", "it", "unsure"))
    assert recovery is not None
    await say(r, recovery.said)
    assert rec.keys == [MEDIA_KEYS["volume_up"]] * 5


@pytest.mark.parametrize(
    ("doubt", "settings", "paused"),
    [
        (Doubt("Spenditi.", "it", "not_a_request", close=False), {}, False),  # far: the TV
        (Doubt("Spenditi.", "it", "not_a_request"), {"enabled": False}, False),
        (Doubt("Spenditi.", "it", "not_a_request"), {}, True),
        (Doubt("Spenditi.", "it", "not_a_request"), {"near_floor": 95}, False),
    ],
)
async def test_no_call(
    matcher: CommandMatcher, doubt: Doubt, settings: dict[str, Any], paused: bool
) -> None:
    r, fake, _ = responder(matcher, Guess("spegniti", 0.99, "off"), **settings)
    r.actions.system.paused = paused
    assert await r.guess(doubt) is None
    assert fake.calls == []


async def test_no_guess_while_a_question_waits(matcher: CommandMatcher) -> None:
    r, fake, _ = responder(matcher, Guess("spegniti", 0.99, "off"))
    await say(r, "chiudi la finestra")  # "Confermi?"
    assert await r.guess(Doubt("Spenditi.", "it", "not_a_request")) is None
    assert fake.calls == []


async def test_no_guess_while_off(matcher: CommandMatcher) -> None:
    # Off ("spegniti") nothing goes to Claude: a misheard "accenditi" is not guessed either.
    r, fake, _ = responder(matcher, Guess("accenditi", 0.99, "on"))
    r.actions.system.off = True
    assert await r.guess(Doubt("A cendi ti.", "it", "not_a_request")) is None
    assert fake.calls == []


# --- the assistant ----------------------------------------------------------------------


def guessing(r: LocalResponder, assistant: Assistant, guess: Guess | None) -> FakeGuesser:
    fake = FakeGuesser(guess)
    r.fallback_guess = fake
    assistant.guess = r.guess
    return fake


async def test_a_misheard_command_in_the_window_is_done(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, rec, r = build_conversation(
        fake_vad, vad_settings, matcher, ["che ore sono", "Prossima."]
    )
    fake = guessing(r, assistant, Guess("canzone successiva", 0.9, "media_next"))
    await talk(assistant, make_frames, speaker, requests=2)
    assert rec.keys == [MEDIA_KEYS["next_track"]]
    assert [row["outcome"] for row in assistant.history][:2] == ["ok", "ok"]
    row = assistant.history[1]
    assert row["guessed_from"] == "Prossima." and row["guess_verdict"] == "act"
    doubt = fake.calls[0][0]
    assert doubt.wake_word == CONVERSATION and doubt.reason == "not_a_request"
    assert doubt.said_before.startswith("Sono le")  # the reply before


async def test_an_unsure_guess_is_asked_then_done_on_yes(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["che ore sono", "Spenditi.", "sì"]
    )
    guessing(r, assistant, Guess("spegniti", 0.95, "off"))
    await talk(assistant, make_frames, speaker, requests=3)
    assert speaker.spoken[1:] == [
        "Hai detto: spegniti?",
        "Mi spengo. Per riaccendermi di' Claudio, accenditi.",
    ]
    assert [row["outcome"] for row in assistant.history][:3] == ["ok", "ask", "ok"]
    assert r.actions.system.off


async def test_a_wrong_guess_is_dropped_as_before(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, rec, r = build_conversation(
        fake_vad, vad_settings, matcher, ["che ore sono", "Prossima."]
    )
    guessing(r, assistant, Guess("", 0.1))
    await talk(assistant, make_frames, speaker, requests=2)
    assert rec.keys == [] and len(speaker.spoken) == 1
    assert [row["outcome"] for row in assistant.history] == ["ok", "conversation_end"]
    assert assistant.history[1]["guess_verdict"] == "drop"


async def test_a_failing_guess_drops_the_phrase(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, _ = build_conversation(
        fake_vad, vad_settings, matcher, ["che ore sono", "Prossima."]
    )

    async def broken(doubt: Doubt) -> Recovery | None:
        raise RuntimeError("boom")

    assistant.guess = broken
    await talk(assistant, make_frames, speaker, requests=2)
    assert [row["outcome"] for row in assistant.history] == ["ok", "conversation_end"]


@pytest.mark.parametrize(
    "decision",
    [
        Decision(Kind.NOT_REQUEST, "Basta.", "echo"),  # Claudio's own voice
        Decision(Kind.NOISE, "Basta.", "quiet"),  # nobody at the microphone
        Decision(Kind.EMPTY, ""),
    ],
)
async def test_own_voice_and_silence_are_never_guessed(
    fake_vad, vad_settings, matcher, decision: Decision
) -> None:
    assistant, _, _, r = build_conversation(fake_vad, vad_settings, matcher, [])
    fake = guessing(r, assistant, Guess("basta", 0.99, "stop"))
    inter = Interaction(clock=assistant.clock)
    assert await assistant._recover(decision, "it", Heard(CONVERSATION), inter) is None
    assert fake.calls == []


def _commands(matcher: CommandMatcher) -> list[Command]:
    return load_commands(COMMANDS_FILE)
