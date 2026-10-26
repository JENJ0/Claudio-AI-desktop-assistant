"""JenJo changes how Claudio speaks by voice: set_style (words + Jack's voice), kept on disk."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from claudio.brain.claude import ClaudeBrain
from claudio.brain.prompts import SET_STYLE_TOOL, SYSTEM_PROMPT
from claudio.config import ClaudeSettings, PcSettings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.style import SpeakingStyle, Style
from tests.unit.test_claude_brain import Monotonic, ScriptedClient
from tests.unit.test_cost import PRICES, tracker
from tests.unit.test_jack import make
from tests.unit.test_jack import synths as synths

saved = sys.stdout
try:
    from claudio.tts import jack_worker as w
finally:
    sys.stdout = saved  # the worker sends its prints to stderr when imported


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


def use(tool: str, **args: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=f"tu_{tool}", name=tool, input=args)


def styled_brain(pricing: Pricing, client: Any, style: SpeakingStyle) -> ClaudeBrain:
    return ClaudeBrain(
        client,
        tracker(pricing, 1.0),
        ClaudeSettings(tell_time=False),
        clock=Monotonic(),
        pc=PcSettings(),
        style=style,
    )


async def ask(b: ClaudeBrain, said: str) -> str:
    return "".join([c async for c in b.ask(said, "it")])


# --- the style itself -----------------------------------------------------------------------


def test_only_the_named_fields_change_and_reset_clears(tmp_path: Path) -> None:
    style = SpeakingStyle(tmp_path / "style.json")
    style.update({"tone": "drammatico e teatrale"})
    style.update({"pace": 0.85})
    assert style.current == Style(tone="drammatico e teatrale", pace=0.85)
    assert style.tag() == 'style: tone "drammatico e teatrale", pace 0.85'
    style.update({"reset": True})
    assert style.current.is_default and style.tag() == ""
    style.update({"reset": True, "pitch": -2})  # reset first, then the change
    assert style.current == Style(pitch=-2)


def test_out_of_range_numbers_are_clamped_not_refused(tmp_path: Path) -> None:
    style = SpeakingStyle()
    assert style.update({"pace": 0.3, "pitch": 9}) == Style(pace=0.7, pitch=4)


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"pace": "lento"},
        {"pace": True},
        {"pace": float("nan")},
        {"energy": "furious"},
        {"tone": 3},
        {"tone": "x" * 81},
    ],
)
def test_nonsense_is_refused_and_nothing_changes(args: dict[str, Any]) -> None:
    style = SpeakingStyle()
    style.update({"tone": "calmo"})
    with pytest.raises(ValueError):
        style.update(args)
    assert style.current == Style(tone="calmo")


def test_the_tone_stays_on_one_line_in_the_tag() -> None:
    style = SpeakingStyle()
    style.update({"tone": 'come un\n"narratore"   epico'})
    assert style.tag() == "style: tone \"come un 'narratore' epico\""


def test_kept_across_restarts(tmp_path: Path) -> None:
    path = tmp_path / "data" / "style.json"
    SpeakingStyle(path).update({"tone": "sarcastico", "energy": "lively"})
    assert SpeakingStyle(path).current == Style(tone="sarcastico", energy="lively")


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"pace": "fast"}'])
def test_a_broken_file_means_the_normal_style(tmp_path: Path, content: str) -> None:
    path = tmp_path / "style.json"
    path.write_text(content, encoding="utf-8")
    assert SpeakingStyle(path).current.is_default


def test_hand_edited_file_is_clamped(tmp_path: Path) -> None:
    path = tmp_path / "style.json"
    path.write_text(json.dumps({"pace": 3, "unknown": 1}), encoding="utf-8")
    assert SpeakingStyle(path).current == Style(pace=1.3)


def test_voice_values_for_jack() -> None:
    style = SpeakingStyle()
    assert style.voice(0.9, 0.7) == {"speed": 0.9, "pitch": 0, "temperature": 0.7}
    style.update({"pace": 0.85, "pitch": -2, "energy": "lively"})
    assert style.voice(0.9, 0.7) == {"speed": 0.765, "pitch": -2, "temperature": 0.8}
    style.update({"pace": 0.7, "energy": "calm"})
    assert style.voice(0.7, 0.45) == {"speed": 0.6, "pitch": -2, "temperature": 0.4}


# --- Haiku's tool -----------------------------------------------------------------------------


async def test_haiku_changes_the_style_and_answers_in_it(pricing: Pricing) -> None:
    """Log 26/10 19:50: "tono più drammatico" had no command, Haiku only played along."""
    style = SpeakingStyle()
    client = ScriptedClient(
        ([], "tool_use", [use("set_style", tone="drammatico e teatrale")]),
        (["Il sipario si alza, socio."], "end_turn", []),
        (["Le nove, tragicamente."], "end_turn", []),
    )
    b = styled_brain(pricing, client, style)
    assert await ask(b, "usa un tono più drammatico") == "Il sipario si alza, socio."
    assert style.current.tone == "drammatico e teatrale"
    first, second = client.calls
    assert [t["name"] for t in first["tools"]] == ["ask_sonnet", "set_style"]
    assert first["system"] == SYSTEM_PROMPT and first["tools"] == second["tools"]  # cached
    assert second["messages"][-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tu_set_style",
        "content": 'style now: tone "drammatico e teatrale"',
    }
    assert b.last_call["tools"] == ["set_style(tone=drammatico e teatrale) ok"]
    await ask(b, "che ore sono")  # every later request carries the style, after the cache
    assert client.calls[2]["messages"][-1]["content"] == (
        '[it | style: tone "drammatico e teatrale"] che ore sono'
    )


async def test_a_bad_value_goes_back_to_haiku(pricing: Pricing) -> None:
    style = SpeakingStyle()
    client = ScriptedClient(
        ([], "tool_use", [use("set_style", energy="furioso")]),
        (["Non so farlo così."], "end_turn", []),
    )
    await ask(styled_brain(pricing, client, style), "parla furioso")
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "energy" in result["content"]
    assert style.current.is_default


async def test_without_a_style_there_is_no_tool(pricing: Pricing) -> None:
    client = ScriptedClient((["Ciao."], "end_turn", []))
    b = ClaudeBrain(client, tracker(pricing, 1.0), ClaudeSettings(tell_time=False))
    await ask(b, "ciao come va")
    assert [t["name"] for t in client.calls[0]["tools"]] == ["ask_sonnet"]
    assert "tool_choice" not in client.calls[0]


async def test_a_pc_tool_without_pc_tools_is_refused(pricing: Pricing) -> None:
    client = ScriptedClient(
        ([], "tool_use", [use("open_app", name="chrome")]),
        (["Non posso."], "end_turn", []),
    )
    await ask(styled_brain(pricing, client, SpeakingStyle()), "apri chrome")
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "unknown tool" in result["content"]


def test_the_tool_says_jack_is_the_only_voice() -> None:
    assert "always Jack's" in SET_STYLE_TOOL["description"]
    assert "style:" in SYSTEM_PROMPT


# --- Jack's voice -----------------------------------------------------------------------------


def test_each_sentence_carries_the_current_voice(tmp_path: Path, synths: list) -> None:
    style = SpeakingStyle()
    synth, _ = make(tmp_path, "ok", startup_timeout_s=10, sentence_timeout_s=2)
    synth.voice = lambda: style.voice(0.9, 0.7)
    synths.append(synth)
    synth.start()
    sent: list[dict] = []
    original = synth._send
    synth._send = lambda message, proc=None: (sent.append(message), original(message, proc))[1]
    synth.synthesize("Ciao.")
    style.update({"pace": 0.85, "pitch": -2})
    synth.synthesize("Ciao.")
    speaks = [m for m in sent if m["op"] == "speak"]
    assert [(m["speed"], m["pitch"]) for m in speaks] == [(0.9, 0), (0.765, -2)]


def test_worker_takes_the_request_voice_within_bounds() -> None:
    assert w.voice_of({}, 0.9, 0.7) == (0.9, 0, 0.7)
    assert w.voice_of({"speed": 0.765, "pitch": -2, "temperature": 0.8}, 0.9, 0.7) == (
        0.765,
        -2,
        0.8,
    )
    assert w.voice_of({"speed": 9, "pitch": -40, "temperature": "x"}, 0.9, 0.7) == (1.5, -6, 0.7)
    assert w.voice_of({"speed": float("nan"), "pitch": True}, 0.9, 0.7) == (0.9, 0, 0.7)


def test_sentence_converter_adds_the_pitch_to_the_pace_correction() -> None:
    rvc = object.__new__(w.RvcSentence)
    rvc.set_voice(0.9, -2)
    up, down, semitones = w.speed_plan(0.9, w.SENTENCE_RATE / w.QWEN_RATE)
    assert (rvc.up, rvc.down, rvc.semitones) == (up, down, semitones - 2)
