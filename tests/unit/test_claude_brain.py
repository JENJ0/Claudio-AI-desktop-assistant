"""Claude as the fallback of the local router (v0.2.0): streaming, budget, errors, history."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from claudio.app import addressed_to_claudio
from claudio.brain.claude import CUT_NOTE, ClaudeBrain
from claudio.brain.prompts import ASK_SONNET_TOOL, SYSTEM_PROMPT
from claudio.config import ClaudeSettings
from claudio.cost.pricing import Pricing, Usage, load_pricing
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import LocalResponder
from tests.unit.test_actions import Recorder
from tests.unit.test_app import run_frames
from tests.unit.test_app_verify_closed_list import request
from tests.unit.test_app_verify_closed_list import setup as setup_closed_list
from tests.unit.test_conversation import build as build_conversation
from tests.unit.test_conversation import talk
from tests.unit.test_cost import PRICES, tracker


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


USAGE = {"input_tokens": 400, "output_tokens": 30, "cache_read_input_tokens": 0}


class FakeStream:
    def __init__(
        self, chunks: list[str], usage: dict[str, int], stop: str, error: Exception | None
    ):
        self.chunks = chunks
        self.usage = usage
        self.stop = stop
        self.error = error
        self.sent = 0

    async def __aenter__(self) -> FakeStream:
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            for chunk in self.chunks:
                self.sent += 1
                yield chunk
                await asyncio.sleep(0)

        return gen()

    @property
    def current_message_snapshot(self) -> Any:
        # As the API: input counted at message_start, output only at the end.
        return SimpleNamespace(usage={"input_tokens": self.usage["input_tokens"]})

    async def get_final_message(self) -> Any:
        return SimpleNamespace(usage=self.usage, stop_reason=self.stop)


class FakeClient:
    def __init__(
        self,
        chunks: list[str] | None = None,
        usage: dict[str, int] | None = None,
        stop: str = "end_turn",
        error: Exception | None = None,
    ):
        self.chunks = chunks if chunks is not None else ["Roma è la ", "capitale d'Italia."]
        self.usage = usage or USAGE
        self.stop = stop
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def stream(self, **params: Any) -> FakeStream:
        self.calls.append(params)
        return FakeStream(self.chunks, self.usage, self.stop, self.error)


class Monotonic:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def brain(pricing: Pricing, client: Any, budget: float = 1.0, **settings: Any) -> ClaudeBrain:
    settings.setdefault("tell_time", False)  # the time tag has its own test
    return ClaudeBrain(
        client, tracker(pricing, budget), ClaudeSettings(**settings), clock=Monotonic()
    )


async def ask(b: ClaudeBrain, text: str, language: str = "it") -> str:
    return "".join([c async for c in b.ask(text, language)])


async def test_streams_the_reply_and_records_the_cost(pricing: Pricing) -> None:
    client = FakeClient()
    b = brain(pricing, client)
    assert await ask(b, "qual è la capitale d'Italia") == "Roma è la capitale d'Italia."
    params = client.calls[0]
    assert params["model"] == "claude-haiku-5-5" and params["max_tokens"] == 300
    assert params["system"] == SYSTEM_PROMPT
    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"] == {"effort": "low"}
    assert params["messages"] == [{"role": "user", "content": "[it] qual è la capitale d'Italia"}]
    assert b.tracker.calls_today() == 1
    # 400 x 0.10 + 30 x 0.50 per million
    assert b.tracker.spent_today() == pytest.approx(0.000055)
    assert b.last_call["model"] == "claude-haiku-5-5" and b.last_call["output_tokens"] == 30


async def test_markdown_never_reaches_the_voice(pricing: Pricing) -> None:
    b = brain(pricing, FakeClient(["**Roma**", " è `bella`"]))
    assert await ask(b, "dimmi una città") == "Roma è bella"


async def test_thinking_on_when_configured(pricing: Pricing) -> None:
    client = FakeClient()
    await ask(brain(pricing, client, thinking=True, effort="medium"), "ciao come va")
    assert "thinking" not in client.calls[0]
    assert client.calls[0]["output_config"] == {"effort": "medium"}


async def test_history_is_sent_back_and_trimmed(pricing: Pricing) -> None:
    client = FakeClient(["Risposta."])
    b = brain(pricing, client, history_turns=2)
    for q in ("uno", "due", "tre"):
        await ask(b, q)
    sent = [m["content"] for m in client.calls[-1]["messages"]]
    assert sent == ["[it] uno", "Risposta.", "[it] due", "Risposta.", "[it] tre"]
    await ask(b, "quattro")
    assert client.calls[-1]["messages"][0]["content"] == "[it] due"


async def test_history_starts_over_after_a_long_silence(pricing: Pricing) -> None:
    client = FakeClient(["Ok."])
    b = brain(pricing, client, history_reset_s=300)
    await ask(b, "prima domanda")
    b.clock.t += 301  # type: ignore[attr-defined]
    await ask(b, "seconda domanda")
    assert client.calls[-1]["messages"] == [{"role": "user", "content": "[it] seconda domanda"}]


async def test_no_history_when_turned_off(pricing: Pricing) -> None:
    client = FakeClient(["Ok."])
    b = brain(pricing, client, history_turns=0)
    await ask(b, "uno")
    await ask(b, "due")
    assert len(client.calls[-1]["messages"]) == 1


async def test_reply_language_is_tagged(pricing: Pricing) -> None:
    client = FakeClient(["Rome."])
    await ask(brain(pricing, client), "what is the capital of Italy", "en")
    assert client.calls[0]["messages"][-1]["content"].startswith("[en] ")


async def test_budget_blocked_sends_nothing(pricing: Pricing) -> None:
    client = FakeClient()
    b = brain(pricing, client, budget=0.0001)
    b.tracker.record("claude-haiku-5-5", Usage(1000, 0))
    assert "budget di oggi" in await ask(b, "chi era Dante")
    assert client.calls == []
    assert b.last_call == {"claude_error": "budget"}


async def test_budget_warning_is_said_once_before_the_reply(pricing: Pricing) -> None:
    client = FakeClient(["Ok."])
    b = brain(pricing, client, budget=0.001)
    b.tracker.record("claude-haiku-5-5", Usage(8500, 0))  # 85 % of the budget
    first = await ask(b, "domanda uno")
    assert first.startswith("Attenzione, hai usato il 85 per cento") and first.endswith("Ok.")
    assert await ask(b, "domanda due") == "Ok."


async def test_missing_key_says_where_it_goes(pricing: Pricing) -> None:
    b = brain(pricing, None)
    assert "punto env" in await ask(b, "chi era Dante")
    assert b.tracker.calls_today() == 0


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic.APIConnectionError(request=REQUEST), "connessione"),
        (anthropic.APITimeoutError(request=REQUEST), "connessione"),
        (
            anthropic.AuthenticationError(
                "bad key", response=httpx2.Response(401, request=REQUEST), body=None
            ),
            "non è valida",
        ),
        (
            anthropic.InternalServerError(
                "overloaded", response=httpx2.Response(529, request=REQUEST), body=None
            ),
            "Riprova",
        ),
    ],
)
async def test_api_errors_become_one_sentence(
    pricing: Pricing, error: Exception, expected: str
) -> None:
    b = brain(pricing, FakeClient(error=error))
    assert expected in await ask(b, "chi era Dante")
    assert b.tracker.calls_today() == 0  # nothing reached the model


async def test_refusal_without_text_is_said(pricing: Pricing) -> None:
    b = brain(pricing, FakeClient([], stop="refusal"))
    assert await ask(b, "domanda") == "Su questo non posso aiutarti."
    assert b.tracker.calls_today() == 1  # the input was billed


async def test_cut_short_still_counts_the_tokens(pricing: Pricing) -> None:
    """ "basta" while Claude speaks: the speaker closes the stream after the first sentence."""
    client = FakeClient(["Prima frase lunga. ", "Seconda frase. ", "Terza frase."])
    b = brain(pricing, client)
    stream = b.ask("raccontami una storia", "it")
    first = await anext(stream)
    await stream.aclose()
    assert first == "Prima frase lunga. "
    assert b.tracker.calls_today() == 1
    assert b.last_call["input_tokens"] == 400
    assert b.last_call["output_tokens"] == len(first) // 3  # estimate from the text
    # A reply cut short stays as context, marked: "continua la storia" must know the story.
    assert list(b._history) == [
        ("[it] raccontami una storia", f"Prima frase lunga. {CUT_NOTE}"),
    ]


# --- the responder: local first, Claude for the rest ---------------------------------


def hybrid(matcher: CommandMatcher, b: ClaudeBrain) -> LocalResponder:
    return LocalResponder(
        matcher,
        ActionRunner(Recorder().ctx()),
        fallback=b.ask,
        fallback_info=lambda: b.last_call,
    )


async def say(r: LocalResponder, text: str, language: str = "it") -> str:
    return "".join([c async for c in r.respond(text, language)])


async def test_local_commands_never_call_claude(matcher: CommandMatcher, pricing: Pricing) -> None:
    client = FakeClient()
    r = hybrid(matcher, brain(pricing, client))
    await say(r, "che ore sono")
    assert client.calls == [] and r.last_route["route"] == "local"


async def test_unknown_goes_to_claude_and_can_be_repeated(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    client = FakeClient(["Dante Alighieri."])
    r = hybrid(matcher, brain(pricing, client))
    assert await say(r, "chi ha scritto la Divina Commedia") == "Dante Alighieri."
    route = r.last_route
    assert route["route"] == "claude" and route["model"] == "claude-haiku-5-5"
    assert route["cost"] > 0
    assert await say(r, "ripeti") == "Dante Alighieri."
    assert len(client.calls) == 1


async def test_claude_reply_language_follows_the_voices(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    client = FakeClient(["Ciao."])
    r = hybrid(matcher, brain(pricing, client))
    r.voiced = {"it"}  # no English voice: the reply must be Italian
    await say(r, "who wrote the Divine Comedy", "en")
    assert r.language == "it"
    assert client.calls[0]["messages"][-1]["content"].startswith("[it] ")


async def test_paused_never_calls_claude(matcher: CommandMatcher, pricing: Pricing) -> None:
    client = FakeClient()
    r = hybrid(matcher, brain(pricing, client))
    r.actions.system.paused = True
    assert await say(r, "chi era Dante") == ""
    assert client.calls == []


# --- the assistant: conversation window and closed list with Claude --------------------


@pytest.mark.parametrize(
    ("text", "addressed"),
    [
        ("E quanto è alto?", True),
        ("Dimmi di più su di lui", True),
        ("Claudio, e la Francia", True),
        ("How tall is it?", True),
        ("e poi gli ho detto di no", False),
        ("ma che bella giornata oggi", False),
    ],
)
def test_addressed_to_claudio(text: str, addressed: bool) -> None:
    assert addressed_to_claudio(text) is addressed


def with_claude(assistant: Any, r: LocalResponder, b: ClaudeBrain) -> None:
    r.fallback = b.ask
    r.fallback_info = lambda: b.last_call
    assistant.claude_min_words = 3


async def test_conversation_follow_up_goes_to_claude(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    client = FakeClient(["Risposta."])
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["chi era Dante", "e dove è nato?", "boh"]
    )
    with_claude(assistant, r, brain(pricing, client))
    await talk(assistant, make_frames, speaker, requests=3)
    assert speaker.spoken == ["Risposta.", "Risposta."]  # "boh": too short, no call
    assert len(client.calls) == 2
    assert [row["outcome"] for row in assistant.history] == ["ok", "ok", "conversation_end"]
    assert assistant.history[0]["route"] == "claude"


async def test_tv_in_the_window_costs_nothing(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    client = FakeClient(["Sono le 14."])
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["che ore sono", "e poi gli ho detto di no"]
    )
    with_claude(assistant, r, brain(pricing, client))
    await talk(assistant, make_frames, speaker, requests=2)
    assert client.calls == []
    assert [row["outcome"] for row in assistant.history] == ["ok", "conversation_end"]


async def test_window_answers_everything_when_asked_to(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    """JenJo's choice (23/10): "La chat non funziona." is answered too, until "basta"."""
    client = FakeClient(["Vediamo."])
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["che ore sono", "la chat non funziona", "boh"]
    )
    with_claude(assistant, r, brain(pricing, client))
    assistant.only_addressed = False
    await talk(assistant, make_frames, speaker, requests=3)
    assert len(client.calls) == 1  # "boh" is still too short
    assert [row["outcome"] for row in assistant.history] == ["ok", "ok", "conversation_end"]


async def test_long_questions_skip_the_closed_list(fake_vad, vad_settings, make_frames) -> None:
    long = "Claudio quanto è distante la Luna dalla Terra"
    assistant, _, _, calls = setup_closed_list(fake_vad, vad_settings, long, ["apri chrome"])
    assistant.closed_list_max_words = 5
    await run_frames(assistant, request(make_frames))
    assert calls == []
    short = "Claudio a prick roam"
    assistant, speaker, _, calls = setup_closed_list(fake_vad, vad_settings, short, ["apri chrome"])
    assistant.closed_list_max_words = 5
    await run_frames(assistant, request(make_frames))
    assert calls == ["it"] and speaker.spoken == ["Apro Chrome."]


async def test_answer_to_claudes_question_needs_no_wake_word(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    """ "Da dove parti?" ... "Milano.": one word, but it is the answer."""
    client = FakeClient(["Da dove parti?"])
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["quanto dista Napoli", "Milano."]
    )
    with_claude(assistant, r, brain(pricing, client))
    await talk(assistant, make_frames, speaker, requests=2)
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["content"] == "[it] Milano."
    assert assistant.history[0]["reply"] == "Da dove parti?"


# --- v0.2.2: Sonnet when needed, caching, the time, the early connection ----------------


class ScriptedStream(FakeStream):
    def __init__(self, chunks: list[str], stop: str, content: list[Any], usage: dict[str, int]):
        super().__init__(chunks, usage, stop, None)
        self.content = content

    async def get_final_message(self) -> Any:
        return SimpleNamespace(usage=self.usage, stop_reason=self.stop, content=self.content)


HANDOFF = [SimpleNamespace(type="tool_use", name="ask_sonnet", input={})]


class ScriptedClient:
    """One reply per call, in order: (chunks, stop reason, content blocks)."""

    def __init__(self, *replies: tuple[list[str], str, list[Any]]):
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.retrieved: list[str] = []
        self.messages = self
        self.models = SimpleNamespace(retrieve=self._retrieve)

    async def _retrieve(self, model: str) -> Any:
        self.retrieved.append(model)
        return SimpleNamespace(id=model)

    def stream(self, **params: Any) -> ScriptedStream:
        self.calls.append(params)
        chunks, stop, content = self.replies.pop(0)
        return ScriptedStream(chunks, stop, content, dict(USAGE))


async def test_haiku_gets_the_tool_and_the_cache_marker(pricing: Pricing) -> None:
    client = ScriptedClient((["Ciao."], "end_turn", []))
    await ask(brain(pricing, client), "ciao come va")
    params = client.calls[0]
    assert params["tools"][0]["name"] == "ask_sonnet"
    assert params["cache_control"] == {"type": "ephemeral"}


async def test_no_tool_without_sonnet(pricing: Pricing) -> None:
    client = ScriptedClient((["Ciao."], "end_turn", []))
    await ask(brain(pricing, client, smart_model="", cache=False), "ciao come va")
    assert "tools" not in client.calls[0] and "cache_control" not in client.calls[0]


async def test_sonnet_only_after_a_yes(pricing: Pricing) -> None:
    """JenJo, 23/10: never Sonnet without asking. Haiku proposes, Claudio asks, "sì" runs it."""
    client = ScriptedClient(
        ([], "tool_use", HANDOFF),
        (["Conviene il treno: ", "costa meno."], "end_turn", []),
    )
    b = brain(pricing, client)
    question = await ask(b, "meglio il treno o l'auto per Roma, considerando i costi")
    assert question.endswith("Lo uso?") and len(client.calls) == 1
    assert b.last_call["escalated"] == "offered"
    assert await ask(b, "sì") == "Conviene il treno: costa meno."
    haiku, sonnet = client.calls
    assert sonnet["model"] == "claude-sonnet-5-5" and sonnet["max_tokens"] == 1024
    assert "thinking" not in sonnet and "tools" not in sonnet  # Sonnet 5.5 refuses "disabled"
    assert sonnet["output_config"] == {"effort": "low"}
    assert sonnet["messages"] == haiku["messages"]  # the question, not "sì"
    assert b.last_call["escalated"] == "approved" and b.last_call["model"] == "claude-sonnet-5-5"
    assert b.tracker.calls_today() == 2
    assert b._history[-1] == (haiku["messages"][-1]["content"], "Conviene il treno: costa meno.")


async def test_no_means_haiku_answers(pricing: Pricing) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Il treno."], "end_turn", []))
    b = brain(pricing, client)
    await ask(b, "fammi un confronto tra treno e auto")
    assert await ask(b, "no grazie") == "Il treno."
    first, second = client.calls
    assert second["model"] == "claude-haiku-5-5" and "tools" not in second
    assert second["messages"] == first["messages"]


async def test_a_new_question_drops_the_offer(pricing: Pricing) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Roma."], "end_turn", []))
    b = brain(pricing, client)
    await ask(b, "fammi un piano di viaggio")
    assert await ask(b, "qual è la capitale d'Italia") == "Roma."
    assert client.calls[1]["messages"][-1]["content"].endswith("capitale d'Italia")
    assert b._offer is None


async def test_the_offer_expires(pricing: Pricing) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Ok."], "end_turn", []))
    b = brain(pricing, client)
    await ask(b, "fammi un piano di viaggio")
    b.clock.t += 61  # type: ignore[attr-defined]
    await ask(b, "sì")
    assert client.calls[1]["model"] == "claude-haiku-5-5"  # "sì" is a new request now


async def test_haiku_words_before_the_offer_get_a_space(pricing: Pricing) -> None:
    client = ScriptedClient((["Domanda complessa."], "tool_use", HANDOFF))
    reply = await ask(brain(pricing, client), "spiegami il mutuo variabile")
    assert reply.startswith("Domanda complessa. Per questa serve Sonnet")


async def test_no_offer_for_a_story(pricing: Pricing) -> None:
    """JenJo, 25/10: Sonnet is the exception, never for stories. Haiku tells it, no question."""
    client = ScriptedClient(([], "tool_use", HANDOFF), (["C'era una volta."], "end_turn", []))
    b = brain(pricing, client)
    assert await ask(b, "raccontami una storia lunga sui pirati") == "C'era una volta."
    first, second = client.calls
    assert [c["model"] for c in client.calls] == ["claude-haiku-5-5"] * 2
    assert "tools" not in second and second["messages"] == first["messages"]
    assert b._offer is None and b.last_call["escalated"] == "skipped"


async def test_detail_words_do_not_offer_sonnet_for_a_joke(pricing: Pricing) -> None:
    client = ScriptedClient((["Allora."], "end_turn", []))
    b = brain(pricing, client)
    assert not b.wants_offer("dimmi una barzelletta nel dettaglio", "it")
    assert b.wants_offer("spiegami nel dettaglio il mutuo variabile", "it")
    assert await ask(b, "tell me a joke in detail", "en") == "Allora."
    assert b._offer is None


def test_ask_sonnet_is_a_rare_exception() -> None:
    """The tool text keeps stories and chat out, and stays short (it is in the cached prefix)."""
    text = ASK_SONNET_TOOL["description"]
    for word in ("stories", "jokes", "chat", "code", "maths", "rare"):
        assert word in text
    assert len(text) < 500


async def test_asking_to_think_goes_straight_to_sonnet(pricing: Pricing) -> None:
    client = ScriptedClient((["Allora."], "end_turn", []))
    b = brain(pricing, client)
    assert await ask(b, "Pensaci bene: quanto fa 17 per 23?") == "Allora."
    assert [c["model"] for c in client.calls] == ["claude-sonnet-5-5"]
    assert b.last_call["escalated"] == "asked" and "smart_first_ms" in b.last_call


async def test_think_phrase_in_another_language(pricing: Pricing) -> None:
    b = brain(pricing, ScriptedClient())
    assert b.wants_smart("think hard about this", "it")
    assert not b.wants_smart("pensa che bello", "it")
    b.smart_model = None
    assert not b.wants_smart("pensaci bene", "it")


async def test_no_sonnet_when_the_budget_runs_out_before_the_yes(pricing: Pricing) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF))
    b = brain(pricing, client, budget=0.00005)  # Haiku's 0.000055 uses it all
    await ask(b, "fammi un confronto dettagliato")
    assert "budget di oggi" in await ask(b, "sì") and len(client.calls) == 1


async def test_the_time_goes_after_the_cached_prefix(pricing: Pricing) -> None:
    from datetime import datetime

    client = ScriptedClient((["Sono le 17 e 15."], "end_turn", []))
    b = ClaudeBrain(
        client,
        tracker(pricing, 1.0),
        ClaudeSettings(),
        clock=Monotonic(),
        wall=lambda: datetime(2026, 10, 23, 17, 15),
    )
    await ask(b, "che ore sono in Thailandia")
    assert client.calls[0]["system"] == SYSTEM_PROMPT  # frozen: cacheable
    content = client.calls[0]["messages"][-1]["content"]
    assert content == "[it | Fri 2026-10-23 17:15] che ore sono in Thailandia"


async def test_warm_up_opens_the_connection_once(pricing: Pricing) -> None:
    client = ScriptedClient()
    b = brain(pricing, client)
    b.warm_up()
    b.warm_up()  # right after: the connection is fresh
    await asyncio.gather(*b._warming)
    assert client.retrieved == ["claude-haiku-5-5"]
    b.clock.t += 16  # type: ignore[attr-defined]
    b.warm_up()
    await asyncio.gather(*b._warming)
    assert len(client.retrieved) == 2


async def test_warm_up_failures_are_silent(pricing: Pricing) -> None:
    client = ScriptedClient()

    async def broken(model: str) -> Any:
        raise anthropic.APIConnectionError(request=REQUEST)

    client.models = SimpleNamespace(retrieve=broken)
    b = brain(pricing, client)
    b.warm_up()
    await asyncio.gather(*b._warming)  # no exception
    brain(pricing, None).warm_up()  # no key: nothing to open


async def test_responder_prepares_only_when_awake(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    calls: list[int] = []
    r = LocalResponder(matcher, ActionRunner(Recorder().ctx()), on_prepare=lambda: calls.append(1))
    r.prepare()
    r.actions.system.paused = True
    r.prepare()  # paused: no Claude call will follow
    assert calls == [1]


def test_sonnet_without_prices_is_turned_off(tmp_path: Any, monkeypatch: Any) -> None:
    from claudio.__main__ import build_brain
    from claudio.config import load_settings

    prices = tmp_path / "pricing.toml"
    prices.write_text(
        'currency = "USD"\ncache_write_5m_multiplier = 1.25\ncache_write_1h_multiplier = 2.0\n'
        "cache_read_multiplier = 0.1\n[models.claude-haiku-5-5]\ninput = 0.1\noutput = 0.5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDIO_COST__PRICING_FILE", str(prices))
    monkeypatch.setenv("CLAUDIO_COST__DB_FILE", str(tmp_path / "spend.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from claudio.__main__ import build_tracker

    settings = load_settings()
    b = build_brain(settings, build_tracker(settings))
    assert b is not None and b.smart_model is None


async def test_the_yes_reaches_claude_through_the_router(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Ecco il piano."], "end_turn", []))
    r = hybrid(matcher, brain(pricing, client))
    await say(r, "fammi un piano per la settimana")
    assert r.asked_question  # the conversation window takes the one-word answer
    assert await say(r, "sì") == "Ecco il piano."
    assert client.calls[1]["model"] == "claude-sonnet-5-5"
