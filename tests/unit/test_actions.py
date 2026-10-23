from datetime import datetime
from pathlib import Path

import pytest

from claudio.router.actions import ActionContext, ActionRunner, say_date, say_time
from claudio.router.matcher import Command, CommandMatcher, Match
from claudio.system import SystemControl


class Recorder:
    def __init__(self, folder: Path | None = None) -> None:
        self.ran: list[list[str]] = []
        self.opened: list[str] = []
        self.keys: list[int] = []
        self.combos: list[tuple[int, ...]] = []
        self.folder = folder

    def ctx(self) -> ActionContext:
        return ActionContext(
            run=self.ran.append,
            open_path=self.opened.append,
            press_key=self.keys.append,
            press_combo=self.combos.append,
            now=lambda: datetime(2026, 10, 18, 14, 5),
            folder_lookup=lambda _id: self.folder,
            home=Path("/home/jenjo"),
        )


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


def run(matcher: CommandMatcher, rec: Recorder, text: str) -> str:
    match = matcher.match(text)
    assert match is not None
    return ActionRunner(rec.ctx()).execute(match)


def test_open_chrome(matcher, rec) -> None:
    assert run(matcher, rec, "apri chrome") == "Apro Chrome."
    assert rec.ran == [["cmd", "/c", "start", "", "chrome"]]


@pytest.mark.parametrize(
    ("text", "exe"),
    [
        ("apri il blocco note", "notepad.exe"),
        ("apri la calcolatrice", "calc.exe"),
        ("apri esplora file", "explorer.exe"),
    ],
)
def test_open_programs(matcher, rec, text, exe) -> None:
    run(matcher, rec, text)
    assert rec.ran == [[exe]]


def test_known_folder_is_preferred() -> None:
    rec = Recorder(folder=Path("D:/OneDrive/Documenti"))
    matcher = CommandMatcher(
        [Command("d", ("apri documenti",), "open_folder", {"folder": "documents"}, "Ok.")]
    )
    run(matcher, rec, "apri documenti")
    assert rec.opened == [str(Path("D:/OneDrive/Documenti"))]


def test_folder_fallback_to_home(matcher, rec) -> None:
    run(matcher, rec, "apri la cartella download")
    assert rec.opened == [str(Path("/home/jenjo") / "Downloads")]


def test_volume_default_steps_and_number(matcher, rec) -> None:
    run(matcher, rec, "alza il volume")
    assert rec.keys == [0xAF] * 5
    rec.keys.clear()
    run(matcher, rec, "abbassa il volume di 3")
    assert rec.keys == [0xAE] * 3


def test_volume_steps_are_capped(matcher, rec) -> None:
    run(matcher, rec, "alza il volume di 500")
    assert len(rec.keys) == 50


def test_mute_presses_once_and_stays_silent(matcher, rec) -> None:
    assert run(matcher, rec, "muto") == ""
    assert rec.keys == [0xAD]


def test_time_and_date(matcher, rec) -> None:
    assert run(matcher, rec, "che ore sono") == "Sono le 14 e 5."
    assert run(matcher, rec, "che giorno è oggi") == "Oggi è domenica 18 ottobre 2026."


@pytest.mark.parametrize(
    ("h", "m", "text"),
    [(0, 0, "È mezzanotte in punto."), (1, 30, "È l'una e 30."), (13, 0, "Sono le 13 in punto.")],
)
def test_say_time(h, m, text) -> None:
    assert say_time(datetime(2026, 1, 1, h, m)) == text


def test_say_date_first_of_month() -> None:
    assert say_date(datetime(2026, 11, 1)) == "Oggi è domenica primo novembre 2026."


def test_unknown_action_raises(rec) -> None:
    match = Match(Command("x", ("x",), "explode"), 100)
    with pytest.raises(ValueError, match="explode"):
        ActionRunner(rec.ctx()).execute(match)


def test_system_commands_change_state_once(matcher, rec) -> None:
    system = SystemControl()
    runner = ActionRunner(rec.ctx(), system)

    def say(text: str) -> str:
        match = matcher.match(text)
        assert match is not None
        return runner.execute(match)

    assert say("vai a dormire").startswith("Vado in pausa") and system.paused
    assert say("vai in pausa") == "Sono già in pausa."
    assert say("svegliati") == "Eccomi." and not system.paused
    assert say("svegliati") == "Sono già sveglio."
    assert say("spegniti") == "A presto." and system.stopping
    assert rec.ran == [] and rec.keys == []  # nothing touches the PC


def test_unknown_system_command_raises() -> None:
    with pytest.raises(ValueError, match="reboot"):
        SystemControl().apply("reboot")


def test_english_replies(matcher, rec) -> None:
    runner = ActionRunner(rec.ctx(), SystemControl())

    def say(text: str) -> str:
        match = matcher.match(text, "en")
        assert match is not None
        return runner.execute(match, match.language)

    assert say("open chrome") == "Opening Chrome."
    assert say("what time is it") == "It's 2 oh 5 p.m."
    assert say("what day is it") == "Today is Sunday, October 18th, 2026."
    assert say("go to sleep").startswith("Pausing.")
    assert say("pause") == "I'm already paused."
    assert say("wake up") == "I'm here."
    assert say("wake up") == "I'm already awake."
    assert say("shut down") == "See you soon."


@pytest.mark.parametrize(
    ("h", "m", "text"),
    [
        (0, 0, "It's 12 a.m."),
        (12, 0, "It's 12 p.m."),
        (9, 0, "It's 9 o'clock."),
        (9, 30, "It's 9 30 a.m."),
        (23, 7, "It's 11 oh 7 p.m."),
    ],
)
def test_say_time_english(h, m, text) -> None:
    assert say_time(datetime(2026, 1, 1, h, m), "en") == text


@pytest.mark.parametrize(
    ("day", "text"),
    [(1, "Sunday, November 1st"), (2, "Monday, November 2nd"), (11, "Wednesday, November 11th")],
)
def test_say_date_english(day, text) -> None:
    assert say_date(datetime(2026, 11, day), "en") == f"Today is {text}, 2026."


def test_tell_spend_without_and_with_a_counter(matcher, rec) -> None:
    from claudio.config import PROJECT_ROOT
    from claudio.cost.ledger import SpendLedger
    from claudio.cost.pricing import Usage, load_pricing
    from claudio.cost.tracker import CostTracker

    match = matcher.match("quanto ho speso oggi")
    assert match is not None and match.command.id == "tell_spend"
    assert ActionRunner(rec.ctx()).execute(match) == "Oggi non hai speso nulla."

    tracker = CostTracker(
        load_pricing(PROJECT_ROOT / "config" / "pricing.toml"),
        SpendLedger(":memory:"),
        1.0,
        now=lambda: datetime(2026, 10, 18, 14, 5),
    )
    tracker.record("claude-haiku-5-5", Usage(output_tokens=500_000))  # $0.25
    runner = ActionRunner(rec.ctx(), tracker=tracker)
    assert runner.execute(match) == "Oggi hai speso 0,25 dollari, il 25 per cento del budget."
    en = matcher.match("how much did I spend today")
    assert en is not None
    assert runner.execute(en, "en") == "Today you spent 0.25 dollars, 25 percent of the budget."


class FakeChat:
    def __init__(self) -> None:
        self.open_now = False
        self.calls: list[str] = []

    def open(self) -> bool:
        self.calls.append("open")
        changed, self.open_now = not self.open_now, True
        return changed

    def close(self) -> bool:
        self.calls.append("close")
        changed, self.open_now = self.open_now, False
        return changed


def test_chat_opens_once_and_closes(matcher, rec) -> None:
    chat = FakeChat()
    runner = ActionRunner(rec.ctx(), chat=chat)  # type: ignore[arg-type]

    def say(text: str, language: str = "it") -> str:
        match = matcher.match(text, language)
        assert match is not None
        return runner.execute(match, match.language)

    assert say("apri la chat") == "Apro la chat."
    assert say("fammi scrivere") == "La chat è già aperta."
    assert say("chiudi la chat") == "Chiudo la chat."
    assert say("close the chat", "en") == "The chat isn't open."
    assert chat.calls == ["open", "open", "close", "close"]


def test_chat_without_launcher_says_so(matcher, rec) -> None:
    match = matcher.match("apri la chat")
    assert match is not None
    assert ActionRunner(rec.ctx()).execute(match) == "Qui la chat non si può aprire."
