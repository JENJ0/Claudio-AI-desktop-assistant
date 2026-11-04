"""Claude as the fallback of the local router (v0.2.0): streaming, budget, errors, history."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from claudio.app import Assistant, addressed_to_claudio
from claudio.brain.claude import CUT_NOTE, ClaudeBrain
from claudio.brain.prompts import ASK_SONNET_TOOL, OFFER, SYSTEM_PROMPT
from claudio.config import ClaudeSettings
from claudio.cost.pricing import Pricing, Usage, load_pricing
from claudio.protocols import CUT_SHORT
from claudio.router.actions import ActionRunner
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import LocalResponder
from claudio.tts.sentence_splitter import split_stream
from tests.unit.test_actions import Recorder
from tests.unit.test_app import FakeSpeaker, run_frames
from tests.unit.test_app_verify_closed_list import request
from tests.unit.test_app_verify_closed_list import setup as setup_closed_list
from tests.unit.test_conversation import LongSpeaker, barge, speak_over, talk
from tests.unit.test_conversation import build as build_conversation
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


def cached(system: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


async def test_streams_the_reply_and_records_the_cost(pricing: Pricing) -> None:
    client = FakeClient()
    b = brain(pricing, client)
    assert await ask(b, "qual è la capitale d'Italia") == "Roma è la capitale d'Italia."
    params = client.calls[0]
    assert params["model"] == "claude-haiku-5-5" and params["max_tokens"] == 300
    assert params["system"] == cached(SYSTEM_PROMPT)  # tools + system: always read back
    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"] == {"effort": "low"}
    assert params["messages"] == [{"role": "user", "content": "[it] qual è la capitale d'Italia"}]
    assert b.tracker.calls_today() == 1
    # 400 x 0.10 + 30 x 0.50 per million
    assert b.tracker.spent_today() == pytest.approx(0.000055)
    assert b.last_call["model"] == "claude-haiku-5-5" and b.last_call["output_tokens"] == 30


class TimedStream(FakeStream):
    """The API's timeline on a fake clock: headers after `open_s`, first word `first_s`
    later, the end `rest_s` after that."""

    def __init__(self, clock: Monotonic, open_s: float, first_s: float, rest_s: float):
        super().__init__(["Roma.", " Capitale."], USAGE, "end_turn", None)
        self.clock, self.open_s, self.first_s, self.rest_s = clock, open_s, first_s, rest_s

    async def __aenter__(self) -> TimedStream:
        self.clock.t += self.open_s
        return self

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            self.clock.t += self.first_s
            for chunk in self.chunks:
                yield chunk
            self.clock.t += self.rest_s

        return gen()


async def test_each_call_logs_headers_first_word_and_end(
    pricing: Pricing, caplog: pytest.LogCaptureFixture
) -> None:
    """IT-1 5.2: first token, first word and end of every call, and the network alone (the
    warm-up's round trip, no model) for the request after it."""
    client = ScriptedClient()
    b = brain(pricing, client)
    clock = b.clock
    assert isinstance(clock, Monotonic)

    async def retrieve(model: str) -> Any:
        clock.t += 0.22
        return SimpleNamespace(id=model)

    client.models = SimpleNamespace(retrieve=retrieve)
    client.stream = lambda **params: TimedStream(clock, 0.5, 0.07, 0.2)  # type: ignore[method-assign]
    b.warm_up()
    await asyncio.gather(*b._warming)
    with caplog.at_level("INFO", logger="claudio.brain.claude"):
        assert await ask(b, "capitale d'Italia") == "Roma. Capitale."
    assert b.last_call["claude_calls"] == [
        {"model": "claude-haiku-5-5", "ttft_ms": 500, "word_ms": 570, "total_ms": 770}
    ]
    assert b.last_call["api_rtt_ms"] == 220 and b.last_call["claude_first_ms"] == 570
    assert b.last_call["api_rtt_new_connection"]  # the first one opens the connection
    assert "primo token 500 ms, prima parola 570 ms, fine 770 ms" in caplog.text
    await ask(b, "e della Francia?")  # no new warm-up: no network figure
    assert "api_rtt_ms" not in b.last_call


async def test_a_round_trip_is_only_given_to_its_own_request(pricing: Pricing) -> None:
    """Review of 5.2: a warm-up still running when the request goes out is not reported
    with the next request, and one from a wake word long ago (a local command) is not
    reported with a later Claude request."""
    client = ScriptedClient(*[(["Ok."], "end_turn", [])] * 3)
    b = brain(pricing, client)
    clock = b.clock
    assert isinstance(clock, Monotonic)
    gate = asyncio.Event()

    async def retrieve(model: str) -> Any:
        await gate.wait()
        clock.t += 0.9
        return SimpleNamespace(id=model)

    client.models = SimpleNamespace(retrieve=retrieve)
    b.warm_up()
    await asyncio.sleep(0)
    clock.t += 0.5
    await ask(b, "ciao")  # asked while the warm-up is still on its way
    gate.set()
    await asyncio.gather(*b._warming)
    await ask(b, "e poi?")
    assert "api_rtt_ms" not in b.last_call
    clock.t += 100  # a wake word answered by the router: warm-up, no Claude request
    b.warm_up()
    await asyncio.gather(*b._warming)
    clock.t += 60
    await ask(b, "che tempo fa?")
    assert "api_rtt_ms" not in b.last_call


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


CUT_STORY = ["C'era una volta un re. ", "Attraverso il bosco ", "arrivò al castello. Attra"]
CLOSING = " Mi fermo qui: se vuoi, continuo."


async def test_a_reply_cut_at_the_token_limit_ends_with_a_closing_sentence(
    pricing: Pricing,
) -> None:
    """30/10: Haiku stopped at the token limit inside "Attraverso" and Jack said "Attra". The
    marker tells the voice to drop the fragment; the closing sentence after it stays in the
    history, so "continua" makes sense to Claude."""
    b = brain(pricing, FakeClient(CUT_STORY, stop="max_tokens"))
    chunks = [c async for c in b.ask("raccontami una storia", "it")]
    assert chunks[: len(CUT_STORY)] == CUT_STORY
    marker, closing = chunks[len(CUT_STORY) :]
    assert marker is CUT_SHORT and marker == ""  # found by identity, nothing when joined
    assert closing == CLOSING
    assert b.last_call["cut"] == "max_tokens"  # in metrics.jsonl
    assert b.tracker.calls_today() == 1
    assert b._history[-1] == ("[it] raccontami una storia", "".join(CUT_STORY) + CLOSING)


async def test_the_closing_sentence_follows_the_reply_language(pricing: Pricing) -> None:
    b = brain(pricing, FakeClient(["Once upon a time, a ki"], stop="max_tokens"))
    assert (
        await ask(b, "tell me a story", "en")
        == "Once upon a time, a ki I'll stop here: ask and I'll go on."
    )


async def test_a_finished_reply_has_no_marker(pricing: Pricing) -> None:
    b = brain(pricing, FakeClient(["Fine."]))
    assert [c async for c in b.ask("dimmi una cosa", "it")] == ["Fine."]
    assert "cut" not in b.last_call


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


class SentenceSpeaker(FakeSpeaker):
    """Splits the reply as StreamingSpeaker does: the sentences the voice would say."""

    def __init__(self) -> None:
        super().__init__()
        self.sentences: list[str] = []

    async def speak(
        self, text_stream: AsyncIterator[str], on_first_audio: Any = None, language: Any = "it"
    ) -> None:
        said = [s async for s in split_stream(text_stream)]
        if on_first_audio:
            on_first_audio()
        self.sentences += said
        self.spoken.append(" ".join(said))


async def test_the_fragment_of_a_cut_reply_never_reaches_the_voice(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    """The marker crosses the responder, the drained stream and the app as it is (30/10)."""
    client = FakeClient(CUT_STORY, stop="max_tokens")
    speaker = SentenceSpeaker()
    assistant, _, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["raccontami una storia"], speaker=speaker
    )
    with_claude(assistant, r, brain(pricing, client))
    await talk(assistant, make_frames, speaker, requests=1)
    assert speaker.sentences == [
        "C'era una volta un re.",
        "Attraverso il bosco arrivò al castello.",
        CLOSING.strip(),
    ]
    row = assistant.history[0]
    assert row["cut"] == "max_tokens" and row["reply"] == ("".join(CUT_STORY) + CLOSING).strip()


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


async def test_tools_and_system_stay_cached_when_the_history_slides(pricing: Pricing) -> None:
    """26/10: with 6 turns of history, the 7th request dropped the oldest one, the prefix
    changed and every request wrote ~5,000 tokens again. The marker on the system prompt
    keeps tools + system byte-identical and marked on every request."""
    client = ScriptedClient(*[(["Ok."], "end_turn", [])] * 8)
    b = brain(pricing, client, history_turns=2)
    for n in range(4):
        await ask(b, f"domanda {n}")
    systems = [c["system"] for c in client.calls]
    assert all(s == systems[0] for s in systems)
    assert systems[0][-1]["cache_control"] == {"type": "ephemeral"}
    assert client.calls[3]["messages"][0]["content"] != client.calls[2]["messages"][0]["content"]


async def test_no_tool_without_sonnet(pricing: Pricing) -> None:
    client = ScriptedClient((["Ciao."], "end_turn", []))
    await ask(brain(pricing, client, smart_model="", cache=False), "ciao come va")
    assert "tools" not in client.calls[0] and "cache_control" not in client.calls[0]
    assert client.calls[0]["system"] == SYSTEM_PROMPT  # no marker without caching


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


@pytest.mark.parametrize("said", ["sì ma dopo", "ok ci penso", "certo che ci penso"])
async def test_a_phrase_that_is_not_only_a_yes_is_no_answer_to_the_offer(
    pricing: Pricing, said: str
) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Ok."], "end_turn", []))
    b = brain(pricing, client)
    await ask(b, "fammi un piano di viaggio")
    assert await ask(b, said) == "Ok."
    assert client.calls[1]["model"] == "claude-haiku-5-5"
    assert client.calls[1]["messages"][-1]["content"].endswith(said)
    assert b._offer is None


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


@pytest.mark.parametrize(
    ("replies", "text"),
    [
        ([(["Fa 391. Il cal"], "max_tokens", [])], "Pensaci bene: quanto fa 17 per 23?"),
        (  # the story Haiku tells itself after asking for Sonnet: _haiku()
            [([], "tool_use", HANDOFF), (["C'era una volta. Un pir"], "max_tokens", [])],
            "raccontami una storia lunga sui pirati",
        ),
    ],
)
async def test_sonnet_and_the_second_haiku_pass_the_cut_marker_on(
    pricing: Pricing, replies: list[tuple[list[str], str, list[Any]]], text: str
) -> None:
    b = brain(pricing, ScriptedClient(*replies))
    chunks = [c async for c in b.ask(text, "it")]
    assert chunks[-2] is CUT_SHORT and chunks[-1] == CLOSING
    assert b.last_call["cut"] == "max_tokens"
    assert b._history[-1][1].endswith(CLOSING)


async def settled() -> None:
    """A generator dropped without aclose() is closed by the event loop a few ticks later."""
    for _ in range(5):
        await asyncio.sleep(0)


TWO_SENTENCES = (["Prima frase. ", "Seconda frase."], "end_turn", [])


@pytest.mark.parametrize(
    ("replies", "before", "text", "question"),
    [
        ([TWO_SENTENCES], [], "Pensaci bene: 17 per 23?", "[it] Pensaci bene: 17 per 23?"),
        (  # the "no" to the offer: Haiku answers without its tools
            [([], "tool_use", HANDOFF), TWO_SENTENCES],
            ["fammi un confronto tra treno e auto"],
            "no",
            "[it] fammi un confronto tra treno e auto",
        ),
    ],
)
async def test_a_cut_through_the_model_that_answers_is_recorded_at_once(
    pricing: Pricing,
    replies: list[tuple[list[str], str, list[Any]]],
    before: list[str],
    text: str,
    question: str,
) -> None:
    """Its words and tokens are in by the time aclose() returns, not a few ticks later."""
    b = brain(pricing, ScriptedClient(*replies))
    for said in before:
        await ask(b, said)
    stream = b.ask(text, "it")
    assert await anext(stream) == "Prima frase. "
    await stream.aclose()
    assert list(b._history) == [(question, f"Prima frase. {CUT_NOTE}")]
    assert b.last_call["input_tokens"] == 400
    assert b.last_call["output_tokens"] == 4  # the 13 characters said, at 3 per token


async def test_a_cut_in_the_story_haiku_tells_itself_keeps_its_words_and_tokens(
    pricing: Pricing,
) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), TWO_SENTENCES)
    b = brain(pricing, client)
    stream = b.ask("raccontami una storia lunga sui pirati", "it")
    assert await anext(stream) == "Prima frase. "
    await stream.aclose()
    await settled()
    # What "continua la storia" reads. The "no answer" entry before it is the doubled cut
    # bookkeeping: a known bug, not pinned here.
    assert b._history[-1] == (
        "[it] raccontami una storia lunga sui pirati",
        f"Prima frase. {CUT_NOTE}",
    )
    assert b.tracker.calls_today() == 2


async def test_a_cut_at_the_offer_keeps_the_words(pricing: Pricing) -> None:
    b = brain(pricing, ScriptedClient((["Domanda complessa."], "tool_use", HANDOFF)))
    stream = b.ask("spiegami il mutuo variabile", "it")
    assert await anext(stream) == "Domanda complessa."
    assert (await anext(stream)).endswith("Lo uso?")
    # Ready by the time the voice says it; the cut itself reaches the brain through
    # cancel_pending(cut=True), not through closing the stream.
    assert b._offer is not None
    await stream.aclose()
    assert b._offer is not None
    assert list(b._history) == [
        ("[it] spiegami il mutuo variabile", f"Domanda complessa. {CUT_NOTE}")
    ]


async def test_a_handoff_with_no_sonnet_to_offer_ends_the_reply_as_it_is(pricing: Pricing) -> None:
    client = ScriptedClient((["Il mutuo."], "tool_use", HANDOFF))
    b = brain(pricing, client, smart_model="")
    assert await ask(b, "spiegami il mutuo variabile") == "Il mutuo."
    assert b._offer is None and len(client.calls) == 1
    assert list(b._history) == [("[it] spiegami il mutuo variabile", "Il mutuo.")]


@pytest.mark.parametrize("answer", ["sì", "no"])
async def test_the_answer_to_the_offer_keeps_the_language_of_the_question(
    pricing: Pricing, answer: str
) -> None:
    """A short answer may be heard as another language: the reply still follows the question."""
    client = ScriptedClient(
        ([], "tool_use", HANDOFF),
        (["It depends. A b"], "max_tokens", []),
    )
    b = brain(pricing, client)
    await ask(b, "compare trains and cars for me", "en")
    reply = await ask(b, answer, "it")
    assert reply == "It depends. A b I'll stop here: ask and I'll go on."


NETWORK_SAID = "Non riesco a collegarmi a Claude. Controlla la connessione."


class BreakingStream(ScriptedStream):
    """The connection dies after its chunks. The SDK wraps errors only at request time: here
    the transport's own exception comes out of `text_stream` (tried on anthropic 1.15)."""

    def __init__(self, chunks: list[str], error: BaseException):
        super().__init__(chunks, "end_turn", [], dict(USAGE))
        self.broken = error

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            for chunk in self.chunks:
                yield chunk
            raise self.broken

        return gen()


class BreakingClient(ScriptedClient):
    """Its last reply is cut by `error`."""

    def __init__(self, error: BaseException, *replies: tuple[list[str], str, list[Any]]):
        super().__init__(*replies)
        self.error = error

    def stream(self, **params: Any) -> ScriptedStream:
        stream = super().stream(**params)
        return stream if self.replies else BreakingStream(stream.chunks, self.error)


@pytest.mark.parametrize(
    "error",
    [
        httpx2.RemoteProtocolError("peer closed"),
        httpx2.ReadError("reset"),
        httpx2.ReadTimeout("stalled"),
        httpx2.DecodingError("bad gzip"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_a_connection_lost_during_a_reply_is_said_once_and_keeps_the_words(
    pricing: Pricing, error: Exception
) -> None:
    """ask() raised: an error beep, no sentence, and the turn was lost."""
    b = brain(pricing, BreakingClient(error, (["Prima frase. ", "Seconda fra"], "end_turn", [])))
    chunks = [c async for c in b.ask("raccontami una storia", "it")]
    assert chunks == ["Prima frase. ", "Seconda fra", "", f" {NETWORK_SAID}"]
    assert chunks[2] is CUT_SHORT
    assert b.last_call["claude_error"] == "network"
    assert list(b._history) == [("[it] raccontami una storia", "Prima frase. Seconda fra")]
    # What was generated is billed: 400 in, the 24 characters said at 3 per token.
    assert b.tracker.calls_today() == 1
    assert b.last_call["input_tokens"] == 400 and b.last_call["output_tokens"] == 8


async def test_the_voice_says_the_network_error_without_the_unfinished_sentence(
    pricing: Pricing,
) -> None:
    client = BreakingClient(httpx2.ReadError("reset"), (["Prima frase. Attra"], "end_turn", []))
    b = brain(pricing, client)
    assert [s async for s in split_stream(b.ask("raccontami una storia", "it"))] == [
        "Prima frase.",
        "Non riesco a collegarmi a Claude.",
        "Controlla la connessione.",
    ]


@pytest.mark.parametrize(
    ("replies", "before", "text", "question"),
    [
        ([(["Prima fra"], "end_turn", [])], [], "Pensaci bene: 17?", "[it] Pensaci bene: 17?"),
        (  # the "no" to the offer: Haiku answers without its tools
            [([], "tool_use", HANDOFF), (["Prima fra"], "end_turn", [])],
            ["fammi un confronto tra treno e auto"],
            "no",
            "[it] fammi un confronto tra treno e auto",
        ),
    ],
)
async def test_a_connection_lost_keeps_the_words_of_the_model_that_answers(
    pricing: Pricing,
    replies: list[tuple[list[str], str, list[Any]]],
    before: list[str],
    text: str,
    question: str,
) -> None:
    b = brain(pricing, BreakingClient(httpx2.ReadError("reset"), *replies))
    for said in before:
        await ask(b, said)
    assert await ask(b, text) == f"Prima fra {NETWORK_SAID}"
    assert list(b._history) == [(question, "Prima fra")]


async def test_a_connection_refused_at_request_time_keeps_nothing(pricing: Pricing) -> None:
    b = brain(pricing, FakeClient(error=anthropic.APIConnectionError(request=REQUEST)))
    assert await ask(b, "chi era Dante") == NETWORK_SAID
    assert b.last_call["claude_error"] == "network"
    assert not b._history and b.tracker.calls_today() == 0


API_SAID = "Claude non ha risposto. Riprova tra poco."
# As the SDK raises an SSE error event: the HTTP response itself was a 200, hence the class.
OVERLOADED_MID_REPLY = anthropic.APIStatusError(
    "{'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}",
    response=httpx2.Response(200, request=REQUEST),
    body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (OVERLOADED_MID_REPLY, 200),
        (
            anthropic.InternalServerError(
                "overloaded", response=httpx2.Response(529, request=REQUEST), body=None
            ),
            529,
        ),
    ],
    ids=["sse_event", "status_class"],
)
async def test_a_status_error_during_a_reply_is_said_once_and_keeps_the_words(
    pricing: Pricing, error: Exception, status: int
) -> None:
    """ "Seconda fraClaude non ha risposto": glued to the fragment, and the turn was lost."""
    b = brain(pricing, BreakingClient(error, (["Prima frase. ", "Seconda fra"], "end_turn", [])))
    chunks = [c async for c in b.ask("raccontami una storia", "it")]
    assert chunks == ["Prima frase. ", "Seconda fra", "", f" {API_SAID}"]
    assert chunks[2] is CUT_SHORT
    assert b.last_call["claude_error"] == status
    assert list(b._history) == [("[it] raccontami una storia", "Prima frase. Seconda fra")]


async def test_the_voice_says_the_status_error_without_the_unfinished_sentence(
    pricing: Pricing,
) -> None:
    client = BreakingClient(OVERLOADED_MID_REPLY, (["Prima frase. Attra"], "end_turn", []))
    b = brain(pricing, client)
    assert [s async for s in split_stream(b.ask("raccontami una storia", "it"))] == [
        "Prima frase.",
        "Claude non ha risposto.",
        "Riprova tra poco.",
    ]


async def test_a_status_error_at_request_time_keeps_nothing(pricing: Pricing) -> None:
    error = anthropic.InternalServerError(
        "overloaded", response=httpx2.Response(529, request=REQUEST), body=None
    )
    b = brain(pricing, FakeClient(error=error))
    assert await ask(b, "chi era Dante") == API_SAID
    assert b.last_call["claude_error"] == 529
    assert not b._history and b.tracker.calls_today() == 0


def sse(event: str, **data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps({'type': event, **data})}\n\n".encode()


async def test_an_error_event_through_the_real_sdk_is_a_status_error_of_200(
    pricing: Pricing,
) -> None:
    """The SDK's own way of raising it (anthropic 1.15): the response status, not the overload."""
    usage = {"input_tokens": 400, "output_tokens": 1}
    message = {"id": "m", "type": "message", "role": "assistant", "content": [], "usage": usage}
    body = b"".join(
        [
            sse("message_start", message={**message, "model": "claude-haiku-5-5"}),
            sse("content_block_start", index=0, content_block={"type": "text", "text": ""}),
            sse(
                "content_block_delta",
                index=0,
                delta={"type": "text_delta", "text": "Prima frase. Seconda fra"},
            ),
            sse("error", error={"type": "overloaded_error", "message": "Overloaded"}),
        ]
    )

    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(answer))
    client = anthropic.AsyncAnthropic(api_key="not-a-key", max_retries=0, http_client=http)
    b = brain(pricing, client)
    assert await ask(b, "raccontami una storia") == f"Prima frase. Seconda fra {API_SAID}"
    assert b.last_call["claude_error"] == 200
    assert list(b._history) == [("[it] raccontami una storia", "Prima frase. Seconda fra")]


async def test_a_cancel_during_a_reply_is_not_a_network_error(pricing: Pricing) -> None:
    client = BreakingClient(asyncio.CancelledError(), (["Prima frase. "], "end_turn", []))
    b = brain(pricing, client)
    chunks = []
    with pytest.raises(asyncio.CancelledError):
        async for chunk in b.ask("raccontami una storia", "it"):
            chunks.append(chunk)
    assert chunks == ["Prima frase. "]  # nothing is said on the way out of a cut
    assert "claude_error" not in b.last_call
    assert list(b._history) == [("[it] raccontami una storia", f"Prima frase. {CUT_NOTE}")]


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
    assert client.calls[0]["system"] == cached(SYSTEM_PROMPT)  # frozen: cacheable
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


# --- an offer survives the end of its own request, not a cut ----------------------------

OFFER_CASES = [
    ("it", "fammi un confronto tra treno e auto", "sì"),
    ("en", "compare trains and cars for me", "yes"),
]


def wired(assistant: Assistant, r: LocalResponder, b: ClaudeBrain) -> None:
    """The lines of __main__.build_responder that matter to an offer."""
    r.fallback = b.ask
    r.fallback_info = lambda: b.last_call
    r.fallback_awaiting = lambda: b.awaiting
    r.fallback_cancel = b.cancel_pending
    assistant.claude_min_words = 3


def sonnet_calls(client: ScriptedClient) -> int:
    return sum(c["model"] == "claude-sonnet-5-5" for c in client.calls)


@pytest.mark.parametrize(("language", "question", "yes"), OFFER_CASES)
async def test_a_completed_offer_survives_its_request_and_the_yes_reaches_sonnet(
    fake_vad, vad_settings, make_frames, matcher, pricing, language, question, yes
) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Ok."], "end_turn", []))
    assistant, speaker, _, r = build_conversation(fake_vad, vad_settings, matcher, [question, yes])
    assistant.transcriber.language = language
    b = brain(pricing, client)
    wired(assistant, r, b)
    await talk(assistant, make_frames, speaker, requests=2)
    assert speaker.spoken[0] == OFFER[language]
    assert sonnet_calls(client) == 1


async def test_the_yes_after_the_window_timed_out_still_reaches_sonnet(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Ok."], "end_turn", []))
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["fammi un confronto tra treno e auto"]
    )
    b = brain(pricing, client)
    wired(assistant, r, b)
    await talk(assistant, make_frames, speaker, requests=1, tail_frames=80)  # the window times out
    assert await say(r, "Hey Claudio, sì") == "Ok."
    assert sonnet_calls(client) == 1


async def test_basta_as_its_own_request_then_yes_does_not_start_sonnet(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Ok."], "end_turn", []))
    assistant, speaker, _, r = build_conversation(
        fake_vad, vad_settings, matcher, ["fammi un confronto tra treno e auto", "basta"]
    )
    b = brain(pricing, client)
    wired(assistant, r, b)
    await talk(assistant, make_frames, speaker, requests=2)
    assert [h["outcome"] for h in assistant.history] == ["ok", "stopped"]
    await say(r, "sì")
    assert sonnet_calls(client) == 0


async def test_basta_over_the_offer_then_yes_does_not_start_sonnet(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Ok."], "end_turn", []))
    speaker = LongSpeaker()
    assistant, _, _, r = build_conversation(
        fake_vad,
        vad_settings,
        matcher,
        ["fammi un confronto tra treno e auto"],
        speaker=speaker,
        barge_in=barge(vad_settings, ["basta"]),
    )
    b = brain(pricing, client)
    wired(assistant, r, b)
    await speak_over(assistant, make_frames, speaker)
    assert speaker.cut
    await say(r, "sì")
    assert sonnet_calls(client) == 0


async def test_a_yes_said_over_the_offer_is_a_takeover_and_reaches_sonnet(
    fake_vad, vad_settings, make_frames, matcher, pricing
) -> None:
    client = ScriptedClient(([], "tool_use", HANDOFF), (["Ok."], "end_turn", []))
    speaker = LongSpeaker()
    assistant, _, _, r = build_conversation(
        fake_vad,
        vad_settings,
        matcher,
        ["fammi un confronto tra treno e auto", "Claudio, sì"],
        speaker=speaker,
        # a bare "sì" is not heard over a reply: the name makes it a request
        barge_in=barge(vad_settings, ["Claudio, sì"], resume_min_chars=0),
    )
    b = brain(pricing, client)
    wired(assistant, r, b)
    await speak_over(assistant, make_frames, speaker)
    first, second = list(assistant.history)[:2]
    assert first["outcome"] == "interrupted" and second["wake_word"] == "barge_in"
    assert sonnet_calls(client) == 1
