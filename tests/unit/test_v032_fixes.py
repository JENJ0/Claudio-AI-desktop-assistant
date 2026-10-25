"""v0.3.2 fixes from JenJo's v0.3.1 test (24/10 logs): short follow-ups in the conversation
window are answered, and a question followed by a short sentence still waits for its answer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from claudio.brain.claude import ClaudeBrain
from claudio.config import ClaudeSettings, PcSettings
from claudio.cost.pricing import Pricing, load_pricing
from claudio.intake import short_request
from claudio.pc.tools import PcTools, ToolCall
from claudio.router.matcher import CommandMatcher
from tests.unit.test_claude_brain import FakeClient, Monotonic, ScriptedClient, hybrid, say
from tests.unit.test_cost import PRICES, tracker
from tests.unit.test_pc_screen_terminal import command
from tests.unit.test_pc_tools import FakeDesktop, acting_brain, ask


@pytest.fixture(scope="module")
def pricing() -> Pricing:
    return load_pricing(PRICES)


@pytest.fixture
def desk() -> FakeDesktop:
    return FakeDesktop()


@pytest.fixture
def tools(desk: FakeDesktop, tmp_path: Path) -> Any:
    t = PcTools(desk, PcSettings(), home=tmp_path)
    yield t
    t.close()


@pytest.mark.parametrize(
    ("text", "wanted"),
    [
        ("Allora scrivi.", True),  # dropped on 24/10 00:25
        ("Scrivila pure.", True),
        ("Apri Chrome.", True),
        ("Felice.", False),  # one word: only as the answer to a question
        ("Vai via.", False),
        ("Ma dai.", False),
        ("Scrivi.", False),
    ],
)
def test_short_requests(text: str, wanted: bool) -> None:
    assert short_request(text) is wanted


async def test_a_question_before_a_last_sentence_still_waits_for_the_answer(
    matcher: CommandMatcher, pricing: Pricing
) -> None:
    """24/10 00:14: "Che tema vuoi? Dimmelo e la scrivo subito." then "Felice." was dropped."""
    client = FakeClient(["Che tema vuoi? Dimmelo e la scrivo subito."])
    b = ClaudeBrain(
        client, tracker(pricing, 1.0), ClaudeSettings(tell_time=False), clock=Monotonic()
    )
    r = hybrid(matcher, b)
    await say(r, "scrivimi una poesia")
    assert r.asked_question
    client.chunks = ["Ecco una lunga risposta senza domande, che finisce con un punto."]
    await say(r, "grazie mille davvero")
    assert not r.asked_question


# --- 24/10 03:09: "confermo" said 30 s later was lost, and Haiku denied a command it ran -----


async def test_confermo_with_the_wake_word_after_the_window_closed(
    matcher: CommandMatcher, pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    client = ScriptedClient(
        ([], "tool_use", [command()]),
        (["Chrome è il più pesante."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    r = hybrid(matcher, b)
    r.fallback_awaiting = lambda: b.awaiting
    r.fallback_cancel = b.cancel_pending
    await say(r, "quali programmi usano più memoria")
    r.cancel_pending(silence=True)  # the follow-up window timed out
    assert r.awaiting
    assert await say(r, "Hey Claudio, confermo") == "Eseguo. Chrome è il più pesante."
    assert [d for d in desk.done if d[0] == "command"]


async def test_a_new_request_replaces_the_pending_action(
    pricing: Pricing, tools: PcTools, desk: FakeDesktop
) -> None:
    client = ScriptedClient(
        ([], "tool_use", [command()]),
        (["Sono le tre."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    await ask(b, "quali programmi usano più memoria")
    assert await ask(b, "che ore sono adesso in Giappone") == "Sono le tre."
    assert not [d for d in desk.done if d[0] == "command"] and not b.awaiting


async def test_history_says_the_command_ran(pricing: Pricing, tools: PcTools) -> None:
    client = ScriptedClient(
        ([], "tool_use", [command()]),
        (["Chrome."], "end_turn", []),
        (["Sì, l'ho eseguito."], "end_turn", []),
    )
    b = acting_brain(pricing, client, tools)
    await ask(b, "quali programmi usano più memoria")
    await ask(b, "confermo")
    await ask(b, "l'hai eseguito davvero?")
    history = client.calls[-1]["messages"]
    assert history[1]["content"].endswith("[Claudio ran: run_command ok]")


async def test_any_folder_is_found_by_name(
    tools: PcTools, desk: FakeDesktop, tmp_path: Path
) -> None:
    """24/10: "Apri la cartella Tesi" failed; it is Desktop\\UNIVERSITA'\\TESI."""
    tesi = tmp_path / "Desktop" / "UNIVERSITA'" / "TESI"
    tesi.mkdir(parents=True)
    (tmp_path / "miniconda3" / "pkgs" / "tesi").mkdir(parents=True)  # skipped: a big tree
    out = await tools.run(ToolCall("t1", "open_path", {"path": "~\\Tesi"}))
    assert not out.error and ("open", tesi.resolve()) in desk.done
    desk.index = []  # the index knows nothing of it: the folders are walked
    out = await tools.run(ToolCall("t2", "search_files", {"query": "tesi", "kind": "folder"}))
    assert "TESI" in out.text and "miniconda3" not in out.text


async def test_index_results_outside_home_still_walk_the_folders(
    tools: PcTools, desk: FakeDesktop, tmp_path: Path
) -> None:
    """25/10: an Italian Windows index gave "C:\\Utenti\\..." paths, all dropped as outside ~,
    and the walk never ran: "Apri la cartella Progetti" found nothing."""
    from claudio.pc.files import Found

    progetti = tmp_path / "Desktop" / "PROGETTI"
    progetti.mkdir(parents=True)
    desk.index = [Found(Path("/Utenti/JenJo/Desktop/PROGETTI"), 0.0)]
    out = await tools.run(ToolCall("t1", "search_files", {"query": "Progetti", "kind": "folder"}))
    assert "PROGETTI" in out.text and "Utenti" not in out.text


@pytest.mark.parametrize(
    ("url", "wanted"),
    [
        ("file:C:/Users/JenJo/Desktop/PROGETTI", Path("C:/Users/JenJo/Desktop/PROGETTI")),
        (
            "FILE:C:/Users/JenJo/Desktop/UNIVERSITA'/TESI",
            Path("C:/Users/JenJo/Desktop/UNIVERSITA'/TESI"),
        ),
        ("mapi://{S-1-5}/Inbox", None),
        ("file:", None),
    ],
)
def test_index_path_reads_the_item_url(url: str, wanted: Path | None) -> None:
    from claudio.pc.files import index_path

    assert index_path(url) == wanted


@pytest.mark.parametrize(
    ("spoken", "name", "wanted"),
    [
        ("desi", "TESI", True),  # 25/10: Whisper heard "desi"
        ("attesi", "TESI", True),  # ... and "attesi"
        ("progett", "PROGETTI", True),
        ("tesi", "Documents", False),
        ("pz", "pc", False),
    ],
)
def test_similar_names(spoken: str, name: str, wanted: bool) -> None:
    from claudio.pc.files import similar

    assert similar(spoken, name) is wanted


async def test_a_misheard_folder_offers_the_similar_name(
    tools: PcTools, desk: FakeDesktop, tmp_path: Path
) -> None:
    (tmp_path / "Desktop" / "UNIVERSITA'" / "TESI").mkdir(parents=True)
    desk.index = []
    out = await tools.run(ToolCall("t1", "search_files", {"query": "desi", "kind": "folder"}))
    assert "Similar names" in out.text and "TESI" in out.text and not out.error
