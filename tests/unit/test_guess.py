"""IT-0: a phrase about to be dropped that sounds like a command. Claude guesses which command
JenJo asked ("Spenditi." in the conversation window, "Audio Bust." after a reply) and
Claudio acts on it, asks "Hai detto: ...?" first, or drops it as before."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from claudio.app import Assistant
from claudio.brain.guess import SCHEMA, Guesser, catalog, describe, parse
from claudio.config import ClaudeSettings, GuessSettings
from claudio.cost.pricing import Pricing, Usage, load_pricing
from claudio.intake import (
    CONVERSATION,
    PRECALL,
    Decision,
    Doubt,
    Guess,
    Heard,
    Hearing,
    Kind,
    Recovery,
    Segment,
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
from tests.unit.test_conversation import claude_says, talk
from tests.unit.test_cost import PRICES, tracker
from tests.unit.test_safety import NOT_A_YES


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
    ("guess", "doubt", "action", "expected"),
    [
        (None, DOUBT, "", Verdict.DROP),
        (Guess("", 0.95), DOUBT, "", Verdict.DROP),  # Claude: not for Claudio
        (Guess("alza", 0.4), DOUBT, "media_key", Verdict.DROP),
        (Guess("alza", 0.6), DOUBT, "media_key", Verdict.ASK),
        (Guess("alza", 0.8), DOUBT, "media_key", Verdict.ACT),
        (Guess("alza", 0.95), Doubt("x", "it", "unsure", close=False), "media_key", Verdict.DROP),
        # JenJo 30/10: "basta" from 50 % stops at once, never "Hai detto: basta?"...
        (Guess("basta", 0.4), DOUBT, "stop", Verdict.DROP),
        (Guess("basta", 0.6), DOUBT, "stop", Verdict.ACT),
        (Guess("basta", 0.95), Doubt("x", "it", "unsure", close=False), "stop", Verdict.DROP),
        # ...and "spegniti" is always asked, however sure: never off on a guess alone.
        (Guess("spegniti", 0.99), DOUBT, "system", Verdict.ASK),
        (Guess("chiuditi", 0.99), DOUBT, "system", Verdict.ASK),
    ],
)
def test_verdict(guess: Guess | None, doubt: Doubt, action: str, expected: Verdict) -> None:
    assert verdict(guess, doubt, 0.8, 0.5, action) is expected


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


@pytest.mark.parametrize(("said", "command"), [("spegniti", "off"), ("chiuditi", "shutdown")])
async def test_switching_off_or_closing_is_asked_however_sure(
    matcher: CommandMatcher, said: str, command: str
) -> None:
    r, _, _ = responder(matcher, Guess(said, 0.99, command))
    recovery = await r.guess(Doubt("Spenditi.", "it", "not_a_request", close=True))
    assert recovery is not None and recovery.verdict is Verdict.ASK
    assert recovery.question == f"Hai detto: {said}?" and r.awaiting


async def test_a_guessed_basta_stops_without_asking(matcher: CommandMatcher) -> None:
    # JenJo 30/10: between ask_at and act_at a "basta" is done, never "Hai detto: basta?".
    r, _, _ = responder(matcher, Guess("basta", 0.6, "stop"))
    recovery = await r.guess(Doubt("Pasta.", "it", "not_a_request", close=True))
    assert recovery is not None and recovery.verdict is Verdict.ACT
    assert recovery.said == "basta" and not r.awaiting


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


@pytest.mark.parametrize(
    "said",
    [
        "non spegnerti adesso",
        "non spegnerti",
        "no non spegnerti",
        "va bene ma non adesso",
        "no basta",
    ],
)
async def test_a_refusal_does_not_run_what_it_refuses(matcher: CommandMatcher, said: str) -> None:
    """ "non spegnerti adesso" with the "non" taken out was the request "spegnerti adesso"."""
    r, _, _ = responder(matcher, Guess("spegniti", 0.99, "off"))
    await r.guess(Doubt("Spenditi.", "it", "not_a_request"))
    assert await say(r, said) == "Va bene, lascio stare."
    assert not r.actions.system.off and not r.awaiting
    assert r.last_route["route"] == "guess_refused"


async def test_no_and_lower_the_volume_is_still_a_new_request(matcher: CommandMatcher) -> None:
    r, _, rec = responder(matcher, Guess("spegniti", 0.99, "off"))
    await r.guess(Doubt("Spenditi.", "it", "not_a_request"))
    await say(r, "no, abbassa il volume")
    assert not r.actions.system.off
    assert rec.keys and set(rec.keys) == {MEDIA_KEYS["volume_down"]}


async def test_a_refusal_of_closing_the_window_presses_nothing(matcher: CommandMatcher) -> None:
    r, _, rec = responder(matcher, Guess("chiudi la finestra", 0.6, "close_window"))
    await r.guess(Doubt("Chiudi la fenestra.", "it", "not_a_request"))
    assert await say(r, "non chiudere la finestra") == "Va bene, lascio stare."
    assert rec.combos == []


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


async def test_a_yes_with_more_words_after_it_answers_the_guess(matcher: CommandMatcher) -> None:
    r, _, rec = responder(matcher, None)
    r.propose("chiudi la finestra", "it")
    assert await say(r, "sì, grazie mille davvero") == "Fatto."
    assert rec.combos == [(VK_MENU, VK_F4)]


@pytest.mark.parametrize("guessed", ["spegniti", "chiudi la finestra"])
@pytest.mark.parametrize("said", NOT_A_YES)
async def test_a_phrase_that_is_not_only_a_yes_does_not_answer_the_guess(
    matcher: CommandMatcher, guessed: str, said: str
) -> None:
    r, _, rec = responder(matcher, None)
    r.propose(guessed, "it")
    await say(r, said)
    assert not r.actions.system.off and (VK_MENU, VK_F4) not in rec.combos


@pytest.mark.parametrize("said", ["si chiama Anna", "sì ma dopo", "ok ci penso", "certo ci penso"])
async def test_a_phrase_that_is_not_only_a_yes_is_a_new_request(
    matcher: CommandMatcher, said: str
) -> None:
    """The same path as any request: Claude hears it, word for word."""
    asked: list[str] = []

    async def claude(text: str, language: str) -> AsyncIterator[str]:
        asked.append(text)
        yield "Ok."

    r = LocalResponder(matcher, ActionRunner(Recorder().ctx()), fallback=claude)
    r.propose("spegniti", "it")
    assert await say(r, said) == "Ok."
    assert not r.actions.system.off
    assert asked == [said] and r.last_route["route"] == "claude"


async def test_a_yes_that_repeats_the_guess_answers_it(matcher: CommandMatcher) -> None:
    """Not the question's echo: "sì, chiudi la finestra" at "Hai detto: chiudi la finestra?"."""
    r, _, rec = responder(matcher, None)
    r.propose("chiudi la finestra", "it")
    assert await say(r, "sì, chiudi la finestra") == "Fatto."
    assert rec.combos == [(VK_MENU, VK_F4)]


@pytest.mark.parametrize("said", ["No, Claudio.", "Non Claudio.", "Mai Claudio."])
async def test_a_refusal_with_the_name_refuses_the_guess(
    matcher: CommandMatcher, said: str
) -> None:
    r, _, rec = responder(matcher, None)
    r.propose("spegniti", "it")
    assert await say(r, said) != ""
    assert not r.actions.system.off and rec.combos == []
    assert r.last_route["route"] == "guess_refused"


async def test_a_question_mark_after_the_yes_does_not_run_the_guess(
    matcher: CommandMatcher,
) -> None:
    r, _, rec = responder(matcher, None)
    r.propose("chiudi la finestra", "it")
    await say(r, "Sì? Chiudi la finestra.")
    assert rec.combos == []


async def test_a_yes_that_repeats_switching_off_switches_off(matcher: CommandMatcher) -> None:
    r, _, _ = responder(matcher, None)
    r.propose("spegniti", "it")
    await say(r, "sì, spegniti")
    assert r.actions.system.off


async def test_the_guess_repeated_without_a_yes_is_still_the_echo(matcher: CommandMatcher) -> None:
    r, _, rec = responder(matcher, None)
    r.propose("chiudi la finestra", "it")
    assert await say(r, "chiudi la finestra") == ""
    assert rec.combos == [] and r.last_route["route"] == "guess_echo"


async def test_a_yes_and_another_request_is_a_new_request(matcher: CommandMatcher) -> None:
    asked: list[str] = []

    async def claude(text: str, language: str) -> AsyncIterator[str]:
        asked.append(text)
        yield "Ok."

    rec = Recorder()
    r = LocalResponder(matcher, ActionRunner(rec.ctx()), fallback=claude)
    r.propose("chiudi la finestra", "it")
    assert await say(r, "sì, chiudi la porta") == "Ok."
    assert rec.combos == [] and asked == ["sì, chiudi la porta"]


async def test_a_yes_to_a_guessed_level_3_command_still_needs_its_confermo() -> None:
    level_3 = Command(id="delete", phrases=("cancella tutto",), action="close_window", level=3)
    r, _, rec = responder(CommandMatcher([level_3]), None)
    r.propose("cancella tutto", "it")
    assert "confermo" in await say(r, "sì")
    assert rec.combos == [] and r.awaiting and r.last_route["route"] == "confirm"
    await say(r, "confermo")
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


async def test_an_ask_row_keeps_the_route_of_its_question(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    # propose() writes last_route while the request is decided: the row must carry it.
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["che ore sono", "Spenditi."]
    )
    guessing(r, assistant, Guess("spegniti", 0.95, "off"))
    await talk(assistant, make_frames, speaker, requests=2)
    row = assistant.history[1]
    assert row["outcome"] == "ask" and row["route"] == "guess" and row["guess"] == "spegniti"


async def test_a_guessed_basta_ends_the_conversation_at_once(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["che ore sono", "Pasta."]
    )
    guessing(r, assistant, Guess("basta", 0.6, "stop"))
    await talk(assistant, make_frames, speaker, requests=2)
    assert speaker.spoken[1:] == [""]  # "basta" says nothing: no "Hai detto: basta?"
    assert [row["outcome"] for row in assistant.history] == ["ok", "stopped"]
    assert assistant.history[1]["guess_verdict"] == "act"
    assert speaker.beeps[-1] == "cancel"


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


# --- G3c Task 4: Haiku's pre-call on a request Whisper was unsure of ---------------------


UNSURE = Segment(0.05, -0.65)
PRECALL_DOUBT = Doubt(
    "Claudio, chi udite?",
    "it",
    PRECALL,
    alternatives=("Claudio, chiuditi.",),
    close=True,
    nearby="Cartelle: Tesi. Finestre aperte: Blocco note.",
    last_command="volume_up",
)


def test_describe_precall_gives_the_names_and_the_last_command(matcher: CommandMatcher) -> None:
    text = describe(PRECALL_DOUBT, matcher.nearest(["Claudio, chiuditi."]))
    assert "no local command understood it" in text and "ignored" not in text
    assert 'Other hearings: "Claudio, chiuditi."' in text
    assert "Names on the PC: Cartelle: Tesi." in text
    assert "Last local command: volume_up" in text


async def test_the_precall_uses_the_guess_model_and_its_own_prompt(
    pricing: Pricing, matcher: CommandMatcher
) -> None:
    client = GuessClient({"why": "x", "command": "", "said": "apri Tesi", "probability": 0.7})
    g = guesser(pricing, matcher, client, model="claude-haiku-5-5")
    assert await g(PRECALL_DOUBT, []) == Guess("apri Tesi", 0.7, "", "x")
    params = client.calls[0]
    assert params["model"] == "claude-haiku-5-5"
    assert params["thinking"] == {"type": "disabled"}
    assert "Claudio was about to ignore" not in params["system"]
    assert "- off: spegniti" in params["system"]
    await g(DOUBT, [])
    assert client.calls[1]["system"] != params["system"]  # a dropped phrase keeps its prompt


@pytest.mark.parametrize(
    ("settings", "paused", "off", "doubt"),
    [
        ({"precall": False}, False, False, PRECALL_DOUBT),
        ({"precall": True}, True, False, PRECALL_DOUBT),
        ({"precall": True}, False, True, PRECALL_DOUBT),
        ({"precall": True}, False, False, replace(PRECALL_DOUBT, close=False)),
        ({"precall": True, "enabled": False}, False, False, PRECALL_DOUBT),
    ],
)
async def test_no_precall(
    matcher: CommandMatcher, settings: dict[str, Any], paused: bool, off: bool, doubt: Doubt
) -> None:
    r, fake, _ = responder(matcher, Guess("chiuditi", 0.99, "shutdown"), **settings)
    r.actions.system.paused, r.actions.system.off = paused, off
    assert await r.guess(doubt) is None
    assert fake.calls == []


async def test_no_precall_while_a_question_waits(matcher: CommandMatcher) -> None:
    r, fake, _ = responder(matcher, Guess("chiuditi", 0.99, "shutdown"), precall=True)
    await say(r, "chiudi la finestra")  # "Confermi?"
    assert await r.guess(PRECALL_DOUBT) is None
    assert fake.calls == []


async def test_the_precall_asks_without_a_closeness_floor(matcher: CommandMatcher) -> None:
    r, fake, _ = responder(matcher, Guess("chiuditi", 0.99, "shutdown"), precall=True)
    recovery = await r.guess(Doubt("Qualcosa di strano.", "it", PRECALL, close=True))
    assert recovery is not None and recovery.verdict is Verdict.ASK
    assert recovery.question == "Hai detto: chiuditi?"
    assert len(fake.calls) == 1


async def test_a_reading_no_command_matches_goes_to_claude_not_to_the_router(
    matcher: CommandMatcher,
) -> None:
    r, _, rec = responder(matcher, Guess("apri la cartella Tesi", 0.7), precall=True)
    recovery = await r.guess(PRECALL_DOUBT)
    assert recovery is not None and recovery.verdict is Verdict.DROP
    assert recovery.said == "apri la cartella Tesi"
    assert not r.awaiting and rec.keys == []


@pytest.mark.parametrize(
    "guess", [Guess("", 0.0), Guess("apri Tesi", 0.3), Guess("alza", 0.3, "volume_up")]
)
async def test_a_precall_that_is_not_clear_leaves_no_reading(
    matcher: CommandMatcher, guess: Guess
) -> None:
    r, _, _ = responder(matcher, guess, precall=True)
    recovery = await r.guess(PRECALL_DOUBT)
    assert recovery is not None and recovery.verdict is Verdict.DROP and recovery.said == ""


def precalling(
    fake_vad, vad_settings, matcher, guess: Guess | None, texts=("Claudio, chi udite?",), **settings
):
    """A conversation whose first request Whisper is unsure of; Claude answers what no command
    understands. Returns the pieces plus the fake Haiku and the hearings Claude is told."""
    assistant, speaker, rec, r = build_conversation(fake_vad, vad_settings, matcher, list(texts))
    claude_says(r, "Risposta di Claude.")
    r.guess_settings = GuessSettings(**{"precall": True, **settings})
    fake = guessing(r, assistant, guess)
    told: list[Hearing] = []
    assistant.on_heard = lambda h: (told.append(h), r.heard(h))[0]
    assistant.unsure_logprob = -0.2
    assistant.stt_segments = lambda: [UNSURE]
    assistant.alternatives = 2
    assistant.closed_list = lambda audio, language: ["Claudio, chi udite?", "Claudio, chiuditi."]
    return assistant, speaker, rec, r, fake, told


async def test_chi_udite_asks_hai_detto_chiuditi_and_closes_nothing_without_a_yes(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, rec, r, fake, _ = precalling(
        fake_vad, vad_settings, matcher, Guess("chiuditi", 0.99, "shutdown")
    )
    await talk(assistant, make_frames, speaker, requests=1)
    assert speaker.spoken == ["Hai detto: chiuditi?"]
    assert not r.actions.system.stopping and rec.keys == []
    doubt = fake.calls[0][0]
    assert doubt.reason == PRECALL and doubt.alternatives == ("Claudio, chiuditi.",)
    row = assistant.history[0]
    assert row["guess_verdict"] == "ask" and row["guessed_from"] == "Claudio, chi udite?"


async def test_a_no_to_the_precalls_question_closes_nothing(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, r, *_ = precalling(
        fake_vad,
        vad_settings,
        matcher,
        Guess("chiuditi", 0.99, "shutdown"),
        texts=("Claudio, chi udite?", "no"),
    )
    await talk(assistant, make_frames, speaker, requests=2)
    assert not r.actions.system.stopping and not r.awaiting


async def test_a_sure_level_1_reading_is_done_at_once(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, rec, *_ = precalling(
        fake_vad,
        vad_settings,
        matcher,
        Guess("canzone successiva", 0.9, "media_next"),
        texts=("Claudio, prosima?",),
    )
    await talk(assistant, make_frames, speaker, requests=1)
    assert rec.keys == [MEDIA_KEYS["next_track"]]
    assert assistant.history[0]["guess_verdict"] == "act"


async def test_a_level_2_reading_never_becomes_a_yes(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    # A phrase the router does not understand: only the pre-call can read it as a command.
    assert matcher.match("Audio, chiuda fenestra?") is None
    assistant, speaker, rec, r, fake, _ = precalling(
        fake_vad,
        vad_settings,
        matcher,
        Guess("chiudi la finestra", 0.95, "close_window"),
        texts=("Audio, chiuda fenestra?",),
    )
    await talk(assistant, make_frames, speaker, requests=1)
    assert len(fake.calls) == 1 and assistant.history[0]["guess_verdict"] == "act"
    assert rec.combos == [] and r.awaiting  # the router's own "Confermi?", nothing pressed
    assert any("Confermi" in s for s in speaker.spoken)


async def test_a_reading_no_command_matches_is_told_to_claude_first(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, _, _, told = precalling(
        fake_vad, vad_settings, matcher, Guess("apri la cartella Tesi", 0.7)
    )
    await talk(assistant, make_frames, speaker, requests=1)
    assert told[0].unsure and told[0].reading == "apri la cartella Tesi"
    assert told[0].alternatives == ("Claudio, chiuditi.",)
    assert speaker.spoken == ["Risposta di Claude."]


@pytest.mark.parametrize("guess", [None, Guess("", 0.0)])
async def test_a_failed_or_unclear_precall_leaves_todays_path(
    fake_vad, vad_settings, make_frames, matcher, guess: Guess | None
) -> None:
    assistant, speaker, _, _, fake, told = precalling(fake_vad, vad_settings, matcher, guess)
    await talk(assistant, make_frames, speaker, requests=1)
    assert len(fake.calls) == 1
    assert told[0] == Hearing(True, ("Claudio, chiuditi.",))
    assert speaker.spoken == ["Risposta di Claude."]


async def test_a_slow_haiku_times_out_and_the_request_goes_to_claude(
    fake_vad, vad_settings, make_frames, matcher, pricing: Pricing
) -> None:
    assistant, speaker, _, r, _, told = precalling(fake_vad, vad_settings, matcher, None)
    r.fallback_guess = guesser(pricing, matcher, GuessClient(delay=1.0), timeout_s=0.05)
    await talk(assistant, make_frames, speaker, requests=1)
    assert told[0] == Hearing(True, ("Claudio, chiuditi.",))
    assert speaker.spoken == ["Risposta di Claude."]


async def test_precall_off_makes_no_call(fake_vad, vad_settings, make_frames, matcher) -> None:
    assistant, speaker, _, _, fake, _ = precalling(
        fake_vad, vad_settings, matcher, Guess("chiuditi", 0.99, "shutdown"), precall=False
    )
    await talk(assistant, make_frames, speaker, requests=1)
    assert fake.calls == [] and speaker.spoken == ["Risposta di Claude."]


async def test_a_phrase_a_command_understands_makes_no_call(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, _, fake, _ = precalling(
        fake_vad,
        vad_settings,
        matcher,
        Guess("chiuditi", 0.99, "shutdown"),
        texts=("che ore sono",),
    )
    await talk(assistant, make_frames, speaker, requests=1)
    assert fake.calls == []


def _heard(**more) -> Heard:
    return Heard(CONVERSATION, segments=(UNSURE,), **more)


async def _precall_doubt(
    fake_vad, vad_settings, matcher, heard: Heard, text: str = "chi udite?", vocabulary=None
) -> Doubt:
    assistant, _, _, _, fake, _ = precalling(
        fake_vad, vad_settings, matcher, Guess("", 0.0), texts=(text,)
    )
    if vocabulary is not None:
        assistant.transcriber.vocabulary = vocabulary
    inter = Interaction(clock=assistant.clock)
    decision = Decision(Kind.ANSWER, text)
    await assistant._precall(np.zeros(1600, dtype=np.int16), decision, "it", heard, inter)
    return fake.calls[0][0]


async def test_the_name_heard_in_the_wake_audio_counts_for_the_precall(
    fake_vad, vad_settings, matcher
) -> None:
    doubt = await _precall_doubt(fake_vad, vad_settings, matcher, _heard(name_heard=True))
    assert doubt.name_heard
    doubt = await _precall_doubt(fake_vad, vad_settings, matcher, _heard())
    assert not doubt.name_heard


async def test_the_window_names_survive_the_cut(fake_vad, vad_settings, matcher) -> None:
    # The live list is ~460 characters with the windows last (the first cut at 400 dropped them).
    names = (
        "Cartelle: "
        + ", ".join(f"Cartella{n}" for n in range(40))
        + ". Finestre aperte: Blocco note."
    )
    doubt = await _precall_doubt(
        fake_vad, vad_settings, matcher, _heard(), vocabulary=lambda language: names
    )
    assert len(names) > 400 and doubt.nearby.endswith("Finestre aperte: Blocco note.")


async def test_a_failing_guess_leaves_the_precall_path(fake_vad, vad_settings, matcher) -> None:
    assistant, *_ = precalling(fake_vad, vad_settings, matcher, Guess("", 0.0))

    async def broken(doubt: Doubt) -> Recovery | None:
        raise RuntimeError("boom")

    assistant.guess = broken
    inter = Interaction(clock=assistant.clock)
    decision = Decision(Kind.ANSWER, "chi udite?")
    audio = np.zeros(1600, dtype=np.int16)
    assert await assistant._precall(audio, decision, "it", _heard(), inter) is None


async def test_sure_whisper_makes_no_precall(fake_vad, vad_settings, make_frames, matcher) -> None:
    assistant, speaker, _, _, fake, _ = precalling(
        fake_vad, vad_settings, matcher, Guess("chiuditi", 0.99, "shutdown")
    )
    assistant.stt_segments = lambda: [Segment(0.05, -0.05)]
    await talk(assistant, make_frames, speaker, requests=1)
    assert fake.calls == [] and speaker.spoken == ["Risposta di Claude."]


async def test_an_unsure_answer_to_claudes_question_goes_to_claude(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, r, fake, _ = precalling(
        fake_vad,
        vad_settings,
        matcher,
        Guess("chiuditi", 0.99, "shutdown"),
        texts=("Claudio, che tempo fa?", "Claudio, sono a Roma vicino al Colosseo"),
    )
    claude_says(r, "Dove ti trovi?")  # a plain question: its answer is the next phrase
    assistant.intake.claude_min_words = 3  # the window lets longer phrases through to Claude
    assistant.closed_list = lambda audio, language: ["Claudio, sono a Roma"]
    heard_so_far = []  # the first request is sure, the answer to the question is not
    assistant.stt_segments = lambda: [Segment(0.05, -0.05 if not heard_so_far else -0.65)]
    assistant.on_heard = lambda h: (heard_so_far.append(h), r.heard(h))[0]
    await talk(assistant, make_frames, speaker, requests=2)
    assert [c[0].reason for c in fake.calls if c[0].reason == PRECALL] == []
    assert len(heard_so_far) == 2 and heard_so_far[1].unsure
    assert speaker.spoken == ["Dove ti trovi?"] * 2
