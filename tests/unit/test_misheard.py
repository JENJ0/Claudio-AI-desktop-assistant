"""Understanding JenJo when Whisper mishears: Claude is told when Whisper was unsure (with its
other guesses), replies cut short stay in the history, and a misheard folder name still
finds the folder (examples from JenJo's logs of 25/10)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from claudio.brain.claude import CUT_BEFORE_NOTE
from claudio.config import PcSettings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.intake import Hearing, Segment, unsure
from claudio.pc import files
from claudio.pc.tools import PcTools, ToolCall
from tests.unit.test_app import FakeResponder, FakeStt, build, run_frames
from tests.unit.test_claude_brain import PRICES, FakeClient, ask, brain
from tests.unit.test_pc_tools import FakeDesktop

SURE = Segment(0.01, -0.05)
DOUBTED = Segment(0.02, -0.31)  # "Apri la cartella desi." on 25/10


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


# -- intake --


def test_unsure_when_a_segment_is_below_the_threshold() -> None:
    assert unsure([SURE, DOUBTED], -0.2)
    assert not unsure([SURE], -0.2)
    assert not unsure([], -0.2)  # no figures: nothing is judged


# -- the brain's tag --


async def test_unsure_request_carries_the_alternatives(pricing: Pricing) -> None:
    client = FakeClient(["Apro Tesi."])
    b = brain(pricing, client)
    b.heard(Hearing(True, ("Apri la cartella Tesi.", 'Apri la "cartella" tesi')))
    await ask(b, "Apri la cartella desi.")
    content = client.calls[0]["messages"][-1]["content"]
    assert content == (
        '[it | unsure: "Apri la cartella Tesi." / "Apri la \'cartella\' tesi"] '
        "Apri la cartella desi."
    )


async def test_unsure_without_alternatives_and_read_once(pricing: Pricing) -> None:
    client = FakeClient(["Ok."])
    b = brain(pricing, client, history_turns=0)
    b.heard(Hearing(True))
    await ask(b, "pure sono")
    await ask(b, "che ore sono")  # the next request was not doubted: no stale note
    assert client.calls[0]["messages"][-1]["content"] == "[it | unsure] pure sono"
    assert client.calls[1]["messages"][-1]["content"] == "[it] che ore sono"


async def test_sure_request_has_the_plain_tag(pricing: Pricing) -> None:
    client = FakeClient(["Ok."])
    b = brain(pricing, client)
    b.heard(Hearing())
    await ask(b, "che ore sono")
    assert client.calls[0]["messages"][-1]["content"] == "[it] che ore sono"


def test_system_prompt_explains_misheard_requests(pricing: Pricing) -> None:
    system = brain(pricing, FakeClient()).system
    assert "misheard" in system and '"unsure"' in system and "[cut off" in system


# -- replies cut short --


class SlowStream:
    """A stream that never sends its first word: JenJo interrupts while Claude thinks."""

    def __init__(self) -> None:
        self.text_stream = self._chunks()
        self.current_message_snapshot = None

    async def _chunks(self):
        await asyncio.sleep(10)
        yield "never"

    async def __aenter__(self) -> SlowStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class SlowClient(FakeClient):
    def stream(self, **params: Any) -> SlowStream:  # type: ignore[override]
        self.calls.append(params)
        return SlowStream()


async def test_request_cut_before_the_answer_stays_in_the_history(pricing: Pricing) -> None:
    b = brain(pricing, SlowClient())
    task = asyncio.create_task(ask(b, "raccontami una storia lunga"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(b._history) == [("[it] raccontami una storia lunga", CUT_BEFORE_NOTE)]


# -- in the Assistant --


class Claude(FakeResponder):
    """No local command understands anything: everything goes to "Claude"."""

    def __init__(self) -> None:
        super().__init__()
        self.hearings: list[Hearing] = []

    def understands(self, text: str, language: str = "it") -> bool:
        return "chrome" in text


def request(make_frames):
    return make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800)


def setup(fake_vad, vad_settings, text: str, segments: list[Segment], candidates: list[str]):
    assistant, _, _, _ = build(fake_vad, vad_settings, FakeStt(text), {1})
    responder = Claude()
    assistant.responder = responder
    assistant.stt_segments = lambda: segments
    assistant.unsure_logprob = -0.2
    assistant.alternatives = 2
    assistant.on_heard = responder.hearings.append
    passes: list[str] = []

    def closed_list(audio, language):
        passes.append(language)
        return candidates

    assistant.closed_list = closed_list
    return assistant, responder, passes


async def test_doubted_request_gets_whispers_other_guesses(
    fake_vad, vad_settings, make_frames
) -> None:
    assistant, responder, passes = setup(
        fake_vad,
        vad_settings,
        "Apri la cartella desi.",
        [DOUBTED],
        ["Apri la cartella desi.", "apri la cartella desi", "Apri la cartella Tesi.", "x", "y"],
    )
    await run_frames(assistant, request(make_frames))
    # The closed list already ran (short, not a command): its n-best is reused, no new pass.
    assert passes == ["it"]
    assert responder.hearings == [Hearing(True, ("Apri la cartella Tesi.", "x"))]
    row = assistant.history[-1]
    assert row["stt_unsure"] is True and row["stt_alternatives"] == ["Apri la cartella Tesi.", "x"]


async def test_long_doubted_question_pays_one_more_pass(
    fake_vad, vad_settings, make_frames
) -> None:
    question = "Il punto è una storia lunga su un robot che vive in città?"
    assistant, responder, passes = setup(
        fake_vad, vad_settings, question, [DOUBTED], ["Raccontami una storia lunga su un robot"]
    )
    await run_frames(assistant, request(make_frames))
    assert passes == ["it"]  # the closed list skipped it (a question): one pass, for Claude
    assert responder.hearings == [Hearing(True, ("Raccontami una storia lunga su un robot",))]
    assert "alternatives_ms" in assistant.history[-1]


async def test_sure_request_costs_nothing_more(fake_vad, vad_settings, make_frames) -> None:
    assistant, responder, passes = setup(
        fake_vad, vad_settings, "Quanto dista la Luna dalla Terra?", [SURE], ["x"]
    )
    await run_frames(assistant, request(make_frames))
    assert passes == [] and responder.hearings == [Hearing()]


async def test_local_command_is_never_doubted(fake_vad, vad_settings, make_frames) -> None:
    assistant, responder, passes = setup(fake_vad, vad_settings, "apri chrome", [DOUBTED], ["x"])
    await run_frames(assistant, request(make_frames))
    assert passes == [] and responder.hearings == [Hearing()]


async def test_off_without_a_threshold(fake_vad, vad_settings, make_frames) -> None:
    assistant, responder, passes = setup(
        fake_vad, vad_settings, "Il punto è una storia lunga?", [DOUBTED], ["x"]
    )
    assistant.unsure_logprob = None  # Claude off, or switched off
    await run_frames(assistant, request(make_frames))
    assert passes == [] and responder.hearings == [Hearing()]


# -- misheard folder names --


@pytest.fixture
def home(tmp_path: Path) -> Path:
    (tmp_path / "Desktop" / "UNIVERSITA'" / "TESI").mkdir(parents=True)
    (tmp_path / "Desktop" / "PROGETTI").mkdir(parents=True)
    (tmp_path / "Documents").mkdir()
    return tmp_path


def test_walk_finds_names_that_sound_alike(home: Path) -> None:
    walked = files.walk(["desi"], [home], "folder", 5, 2.0, similar=3)
    assert walked.found == [] and [f.path.name for f in walked.similar] == ["TESI"]
    walked = files.walk(["attesi"], [home], "folder", 5, 2.0, similar=3)
    assert [f.path.name for f in walked.similar] == ["TESI"]
    assert files.walk(["desi"], [home], "folder", 5, 2.0).similar == []  # not asked


def test_walk_finds_an_exact_name_with_no_similar(home: Path) -> None:
    walked = files.walk(["tesi"], [home], "folder", 5, 2.0, similar=3)
    assert [f.path.name for f in walked.found] == ["TESI"] and walked.similar == []


async def test_empty_index_falls_back_to_the_walk(home: Path) -> None:
    desk = FakeDesktop()
    desk.index = []  # the index answers, but has not got the folder (25/10)
    tools = PcTools(desk, PcSettings(), home=home)
    try:
        out = await tools.run(
            ToolCall("1", "search_files", {"query": "Progetti", "kind": "folder"})
        )
        assert out.text.startswith("1 found:") and "PROGETTI" in out.text
        out = await tools.run(ToolCall("2", "search_files", {"query": "desi", "kind": "folder"}))
        assert out.text.startswith("No name contains: desi. Similar names")
        assert "TESI" in out.text
        out = await tools.run(ToolCall("3", "search_files", {"query": "zzqx", "kind": "folder"}))
        assert out.text == "No files found for: zzqx."
    finally:
        tools.close()


async def test_index_paths_outside_the_roots_still_walk(home: Path, tmp_path: Path) -> None:
    desk = FakeDesktop()
    desk.index = [files.Found(Path("/elsewhere/Utenti/TESI"), 0.0)]  # refused by the policy
    tools = PcTools(desk, PcSettings(), home=home)
    try:
        out = await tools.run(ToolCall("1", "search_files", {"query": "tesi", "kind": "folder"}))
        assert out.text.startswith("1 found:") and "TESI" in out.text
    finally:
        tools.close()
