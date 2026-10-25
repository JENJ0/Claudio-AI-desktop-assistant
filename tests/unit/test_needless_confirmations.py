"""26/10 00:25 (JenJo's log): "Apri Chrome e cerca Dainese Moto" asked "Confermi?" because the
search was typed (type_text, level 2), and "Confermo." said after Haiku's own spoken
"Confermi?" was thrown away as a false wake. Web searches are now one level 1 tool, and a
short answer to Claude's recent question is not lost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from claudio.brain.prompts import SYSTEM_PROMPT_TOOLS
from claudio.config import PcSettings
from claudio.cost.pricing import load_pricing
from claudio.intake import Kind
from claudio.pc.tools import DEFINITIONS, LEVELS, PcTools
from claudio.router.matcher import CommandMatcher
from claudio.router.responder import ANSWER_WINDOW_S, LocalResponder
from tests.unit.test_claude_brain import FakeClient, Monotonic, brain, hybrid, say
from tests.unit.test_conversation import build, talk
from tests.unit.test_cost import PRICES
from tests.unit.test_intake import Commands, decide
from tests.unit.test_pc_tools import FakeDesktop, call


@pytest.fixture
def desk() -> FakeDesktop:
    return FakeDesktop()


@pytest.fixture
def tools(desk: FakeDesktop, tmp_path: Path) -> Any:
    t = PcTools(desk, PcSettings(), home=tmp_path)
    yield t
    t.close()


# --- web_search: level 1, only a results page -------------------------------------------------


def test_web_search_is_offered_and_needs_no_yes(tools: PcTools) -> None:
    assert "web_search" in {d["name"] for d in tools.definitions()}
    assert LEVELS["web_search"] == 1
    assert tools.level(call("web_search", query="Dainese moto", browser="Chrome")) == 1
    assert {d["name"] for d in DEFINITIONS} == set(LEVELS)


async def test_web_search_opens_the_results_in_the_named_browser(
    tools: PcTools, desk: FakeDesktop
) -> None:
    out = await tools.run(call("web_search", query="Dainese  Moto", browser="Chrome"))
    assert not out.error
    assert desk.done[-1] == (
        "url",
        "https://www.google.com/search?q=Dainese+Moto",  # the whole query, encoded
        "chrome.exe",
    )
    assert not any(d[0] in ("keys", "paste") for d in desk.done)  # nothing is typed


async def test_web_search_without_a_browser_uses_the_default(
    tools: PcTools, desk: FakeDesktop
) -> None:
    await tools.run(call("web_search", query="meteo Milano & dintorni?"))
    assert desk.done[-1] == (
        "url",
        "https://www.google.com/search?q=meteo+Milano+%26+dintorni%3F",
        None,
    )


async def test_a_query_cannot_change_the_address(tools: PcTools, desk: FakeDesktop) -> None:
    await tools.run(call("web_search", query="https://evil.example/x?a=1#frag"))
    url = desk.done[-1][1]
    assert url.startswith("https://www.google.com/search?q=")
    assert "/" not in url.removeprefix("https://www.google.com/search?q=")


async def test_unknown_or_missing_browser_falls_back(tools: PcTools, desk: FakeDesktop) -> None:
    await tools.run(call("web_search", query="moto", browser="Netscape"))
    assert desk.done[-1][2] is None  # not a browser we know: the default one
    out = await tools.run(call("web_search", query="moto", browser="Firefox"))
    assert "default browser" in out.text and not out.error  # not installed on this PC


async def test_empty_query_is_an_error(tools: PcTools, desk: FakeDesktop) -> None:
    out = await tools.run(call("web_search", query="   "))
    assert out.error and not any(d[0] == "url" for d in desk.done)


def test_search_engine_is_configurable_but_https_only() -> None:
    bing = PcSettings(web_search_url="https://www.bing.com/search?q={query}")
    assert bing.web_search_url.startswith("https://www.bing.com")
    for bad in ("http://www.google.com/search?q={query}", "https://www.google.com/"):
        with pytest.raises(ValidationError):
            PcSettings(web_search_url=bad)


def test_prompt_sends_web_searches_to_the_tool_and_forbids_spoken_confirmations() -> None:
    assert "web_search" in SYSTEM_PROMPT_TOOLS
    assert 'write "Confermi?" or another yes/no question about an action yourself' in (
        SYSTEM_PROMPT_TOOLS.replace("\n", " ")
    )


# --- "Confermo." after Claude's question is not a false wake ----------------------------------


def verifying(**kw: Any) -> Any:
    from claudio.intake import RequestFilter

    return RequestFilter(verify_wake=True, **kw)


def test_short_answer_to_a_recent_question_is_not_a_false_wake() -> None:
    r = Commands()
    assert decide("Confermo.", r, f=verifying()).kind is Kind.FALSE_WAKE  # no question asked
    r.answer_expected = True
    assert decide("Confermo.", r, f=verifying()).kind is Kind.ANSWER
    assert decide("No, va bene così.", r, f=verifying()).kind is Kind.ANSWER
    # A whole sentence without the name is still the TV, question or not.
    long = "e poi gli ho detto che non era il caso"
    assert decide(long, r, f=verifying()).kind is Kind.FALSE_WAKE


async def test_answer_expected_lasts_only_a_while(matcher: CommandMatcher) -> None:
    client = FakeClient(["Lo cerco su Chrome. Confermi?"])
    r = hybrid(matcher, brain(load_pricing(PRICES), client))
    clock = Monotonic()
    r.clock = clock
    await say(r, "cerca dainese moto")
    assert r.asked_question and r.answer_expected
    clock.t += ANSWER_WINDOW_S + 1
    assert r.asked_question and not r.answer_expected


# --- the conversation window outlives one empty phrase while a question waits -----------------


async def test_window_survives_one_empty_phrase_when_an_answer_is_expected(
    fake_vad, vad_settings, make_frames, matcher, monkeypatch
) -> None:
    monkeypatch.setattr(LocalResponder, "answer_expected", property(lambda self: True))
    assistant, speaker, _, _ = build(
        fake_vad, vad_settings, matcher, ["che ore sono", "", "ripeti"]
    )
    await talk(assistant, make_frames, speaker, requests=3)
    outcomes = [row["outcome"] for row in assistant.history]
    assert outcomes == ["ok", "empty", "ok", "timeout"]


async def test_only_one_empty_phrase_is_forgiven(
    fake_vad, vad_settings, make_frames, matcher, monkeypatch
) -> None:
    monkeypatch.setattr(LocalResponder, "answer_expected", property(lambda self: True))
    assistant, speaker, _, _ = build(fake_vad, vad_settings, matcher, ["che ore sono", "", ""])
    await talk(assistant, make_frames, speaker, requests=3)
    assert [row["outcome"] for row in assistant.history] == ["ok", "empty", "empty"]


async def test_without_a_question_an_empty_phrase_ends_the_window(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    assistant, speaker, _, _ = build(fake_vad, vad_settings, matcher, ["che ore sono", ""])
    await talk(assistant, make_frames, speaker, requests=2)
    assert [row["outcome"] for row in assistant.history] == ["ok", "empty"]
