"""The tray icon: its color and tooltip follow CLAUDIO's state (voice commands included), and
its menu changes the state on the event loop, as a voice command would. A fake pystray
stands in for the real one; the real one is tried in the slow test at the end."""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from typing import Any

import pytest

from claudio import __version__
from claudio.system import OFF, ON, PAUSE, SHUTDOWN, SystemControl
from claudio.tray import COLORS, Tray, icon_image, start_tray, tooltip, tray_state


class FakeItem:
    def __init__(
        self,
        text: Any,
        action: Callable[[], None] | None,
        checked: Any = None,
        radio: bool = False,
        default: bool = False,
        visible: bool = True,
        enabled: Any = True,
    ) -> None:
        self._text, self.action, self._checked = text, action, checked
        self.default, self._enabled = default, enabled

    @staticmethod
    def _value(value: Any) -> Any:
        return value(None) if callable(value) else value

    @property
    def text(self) -> str:
        return self._value(self._text)

    @property
    def checked(self) -> bool | None:
        return self._value(self._checked)

    @property
    def enabled(self) -> bool:
        return self._value(self._enabled)


class FakeMenu:
    SEPARATOR = FakeItem("- - - -", None)

    def __init__(self, *items: FakeItem) -> None:
        self.items = items


class FakeIcon:
    def __init__(self, name: str, icon: Any = None, title: str = "", menu: Any = None) -> None:
        self.name, self.icon, self.title, self.menu = name, icon, title, menu
        self.menu_updates = 0
        self.running = threading.Event()
        self.stopped = threading.Event()

    def run(self) -> None:  # the message loop of pystray: until stop()
        self.running.set()
        self.stopped.wait(10)

    def stop(self) -> None:
        self.stopped.set()

    def update_menu(self) -> None:
        self.menu_updates += 1


class FakePystray:
    Icon, Menu, MenuItem = FakeIcon, FakeMenu, FakeItem


class FakeChat:
    def __init__(self) -> None:
        self.opened = 0

    def open(self) -> bool:
        self.opened += 1
        return True


def item(icon: FakeIcon, text: str) -> FakeItem:
    return next(i for i in icon.menu.items if i.text == text)


async def click(menu_item: FakeItem) -> None:
    """A click: pystray calls the action in its own thread; the loop then runs the change."""
    assert menu_item.action is not None
    await asyncio.to_thread(menu_item.action)
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.fixture
async def shown() -> Any:
    system = SystemControl()
    chat = FakeChat()
    icon_tray = Tray(system, asyncio.get_running_loop(), chat, backend=FakePystray)
    assert icon_tray.start()
    icon_tray.set_ready()
    yield icon_tray, system, chat, icon_tray._icon
    icon_tray.stop()


def test_the_state_shown() -> None:
    system = SystemControl()
    assert tray_state(system, ready=False) == "starting"
    assert tray_state(system, ready=True) == "on"
    system.apply(PAUSE)
    assert tray_state(system, True) == "paused"
    system.apply(OFF)
    assert tray_state(system, True) == "off"  # off wins over the pause
    system.apply(SHUTDOWN)
    assert tray_state(system, True) == "closing"


def test_tooltip_and_colors() -> None:
    assert tooltip("on") == f"CLAUDIO {__version__}: acceso"
    assert "accenditi" in tooltip("off")
    assert "svegliati" in tooltip("paused")
    for state, color in COLORS.items():
        image = icon_image(state)
        assert image.size == (64, 64)
        assert image.mode == "RGBA"
        assert image.getpixel((32, 5)) == (*color, 255)  # the disc, above the white "C"
        assert image.getpixel((0, 0))[3] == 0  # transparent corners


async def test_the_menu(shown: Any) -> None:
    _, _, _, icon = shown
    texts = [i.text for i in icon.menu.items]
    assert texts == ["Spegni", "Pausa", "Apri la chat", "- - - -", "Chiudi CLAUDIO"]
    assert item(icon, "Apri la chat").default  # a left click opens the chat
    assert icon.title == f"CLAUDIO {__version__}: acceso"


async def test_off_and_on_from_the_menu(shown: Any) -> None:
    _, system, _, icon = shown
    updates = icon.menu_updates
    await click(item(icon, "Spegni"))
    assert system.off
    assert "spento" in icon.title
    assert icon.icon.getpixel((32, 5))[:3] == COLORS["off"]
    assert icon.menu_updates > updates
    assert item(icon, "Accendi").text == "Accendi"  # the same item, now the other way
    assert item(icon, "Pausa").enabled is False  # off: only "accenditi"
    await click(item(icon, "Accendi"))
    assert not system.off
    assert icon.title.endswith("acceso")


async def test_pause_from_the_menu(shown: Any) -> None:
    _, system, _, icon = shown
    await click(item(icon, "Pausa"))
    assert system.paused
    assert item(icon, "Pausa").checked is True
    assert "in pausa" in icon.title
    await click(item(icon, "Pausa"))
    assert not system.paused
    assert item(icon, "Pausa").checked is False


async def test_chat_and_close_from_the_menu(shown: Any) -> None:
    _, system, chat, icon = shown
    await click(item(icon, "Apri la chat"))
    assert chat.opened == 1
    await click(item(icon, "Chiudi CLAUDIO"))
    assert system.stopping
    assert icon.title.endswith("si chiude...")


async def test_a_voice_command_changes_the_icon(shown: Any) -> None:
    """ "Claudio, spegniti" goes through system.apply on the loop, as the router does."""
    _, system, _, icon = shown
    system.apply(OFF)
    assert "spento" in icon.title
    system.apply(ON)
    assert icon.title.endswith("acceso")


async def test_starting_until_the_models_load() -> None:
    system = SystemControl()
    icon_tray = Tray(system, asyncio.get_running_loop(), None, backend=FakePystray)
    assert icon_tray.start()
    icon, thread = icon_tray._icon, icon_tray._thread
    assert thread is not None
    try:
        assert icon.title.endswith("si avvia...")
        assert "Apri la chat" not in [i.text for i in icon.menu.items]  # no chat window
        assert icon.running.wait(5)  # its loop runs in the "tray" thread
        icon_tray.set_ready()
        assert icon.title.endswith("acceso")
    finally:
        icon_tray.stop()
    assert icon.stopped.is_set()
    assert not thread.is_alive()  # the loop really ended
    icon_tray.stop()  # twice: harmless (atexit calls it again)
    system.apply(OFF)  # after stop the listener does nothing


async def test_without_pystray_claudio_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "pystray", None)  # import pystray -> ImportError
    assert start_tray(True, SystemControl()) is None
    assert "system tray non disponibile" in caplog.text
    assert start_tray(False, SystemControl()) is None  # [startup] tray = false


async def test_a_failing_icon_never_breaks_a_command(caplog: pytest.LogCaptureFixture) -> None:
    system = SystemControl()
    icon_tray = Tray(system, asyncio.get_running_loop(), None, backend=FakePystray)
    assert icon_tray.start()

    def broken() -> None:
        raise OSError("Explorer is restarting")

    try:
        icon_tray.set_ready()
        icon_tray._icon.update_menu = broken
        assert system.apply(OFF) is True  # the command still happened
        assert system.off
        assert "non aggiornata" in caplog.text
    finally:
        icon_tray.stop()


def test_a_click_after_the_loop_has_closed_is_ignored() -> None:
    loop = asyncio.new_event_loop()
    loop.close()
    icon_tray = Tray(SystemControl(), loop, None, backend=FakePystray)
    icon_tray._close()  # call_soon_threadsafe on a closed loop: no exception


@pytest.mark.slow
async def test_the_real_pystray_icon() -> None:
    """Shows the real icon next to the clock for a moment and changes it (needs pystray)."""
    pytest.importorskip("pystray")
    system = SystemControl()
    icon_tray = Tray(system, asyncio.get_running_loop(), FakeChat())
    assert icon_tray.start()
    icon, thread = icon_tray._icon, icon_tray._thread
    assert thread is not None
    try:
        for _ in range(50):  # pystray creates its window in its own thread
            if icon.visible:
                break
            await asyncio.sleep(0.1)
        assert icon.visible
        icon_tray.set_ready()
        system.apply(OFF)
        assert "spento" in icon.title
        system.apply(ON)
        assert icon.title.endswith("acceso")
    finally:
        icon_tray.stop()
    # pystray's message loop ended, and its last step removes the icon (NIM_DELETE; its
    # `visible` flag is not reset, so the thread is what tells).
    assert not thread.is_alive()


class Idle:
    """The assistant as close_when_idle sees it."""

    def __init__(self) -> None:
        from claudio.app import State

        self.state = State.IDLE


async def test_close_from_the_menu_without_a_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No frame comes (microphone unplugged): "Chiudi CLAUDIO" must end the run anyway."""
    from claudio import __main__ as entry
    from claudio.app import State

    monkeypatch.setattr(entry, "CLOSE_POLL_S", 0.01)
    monkeypatch.setattr(entry, "CLOSE_GRACE_S", 0.05)
    system, assistant = SystemControl(), Idle()
    run = asyncio.create_task(asyncio.Event().wait())  # a run waiting for a frame forever
    closer = asyncio.create_task(entry.close_when_idle(assistant, system, run))
    assistant.state = State.SPEAKING
    system.apply(SHUTDOWN)
    await asyncio.sleep(0.2)
    assert not run.done()  # "A presto." is still playing: wait for it
    assistant.state = State.IDLE
    await asyncio.wait_for(closer, 2)
    with pytest.raises(asyncio.CancelledError):
        await run


async def test_close_with_a_microphone_leaves_the_run_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With frames coming the run ends by itself: close_when_idle cancels nothing."""
    from claudio import __main__ as entry

    monkeypatch.setattr(entry, "CLOSE_POLL_S", 0.01)
    monkeypatch.setattr(entry, "CLOSE_GRACE_S", 0.05)
    system = SystemControl()

    async def ends_at_the_next_frame() -> None:
        while not system.stopping:
            await asyncio.sleep(0.01)

    run = asyncio.create_task(ends_at_the_next_frame())
    closer = asyncio.create_task(entry.close_when_idle(Idle(), system, run))
    system.apply(SHUTDOWN)
    await asyncio.wait_for(closer, 2)
    assert run.done() and not run.cancelled()


def test_listeners_hear_every_real_change() -> None:
    system = SystemControl()
    seen: list[str] = []

    def broken(what: str) -> None:
        raise RuntimeError("tray gone")

    system.subscribe(broken)  # an error in one listener...
    system.subscribe(seen.append)  # ...does not stop the next one
    assert system.apply(PAUSE) is True
    assert system.apply(PAUSE) is False  # already paused: nothing to tell
    system.apply(OFF)
    system.apply(ON)
    assert seen == [PAUSE, OFF, ON]
