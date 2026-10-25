"""The real `Desktop`: Windows APIs behind the PC tools. Runs only on Windows, on the tools'
single worker thread (COM is initialised there once, single-threaded apartment).

- windows: the Alt+Tab list (Win32 EnumWindows: fast), text read through UI Automation
  (pywinauto), walked breadth-first with a time and character budget (Chrome can hold
  thousands of elements);
- apps: the Start menu "All apps" folder (shell:AppsFolder), classic and Store apps alike;
- files: the Windows Search index (the same one as the Start menu search), via OLE DB;
- screen: a screenshot of a window or of the primary screen (Pillow), in physical pixels;
- commands: PowerShell without profile, never elevated (pc/terminal.py)."""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from claudio.pc import terminal
from claudio.pc.files import INDEX_KINDS, SHORTCUTS, Found, index_path, shortcuts
from claudio.pc.terminal import CommandResult
from claudio.pc.tools import App, Window

log = logging.getLogger(__name__)

WM_CLOSE = 0x0010
GW_OWNER = 4
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_CLOAKED = 14
PASTE_SETTLE_S = 0.3
DWMWA_EXTENDED_FRAME_BOUNDS = 9  # the visible frame, without the invisible resize border
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
UIA_VALUE_PROPERTY = 30045
TEXT_TYPES = frozenset({"Edit", "Document"})  # their value is the text, not their name


class WindowsDesktop:
    def thread_init(self) -> None:
        try:
            import pythoncom

            pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
        except Exception:
            log.exception("COM non inizializzato per le azioni sul PC")

    # --- apps ---------------------------------------------------------------------------

    def apps(self) -> list[App]:
        import win32com.client

        shell = win32com.client.Dispatch("Shell.Application")
        folder = shell.NameSpace("shell:AppsFolder")
        found = []
        for item in folder.Items():
            name, target = str(item.Name or ""), str(item.Path or "")
            if name and target and not name.lower().startswith(("uninstall", "disinstalla")):
                found.append(App(name, target))
        names = {app.name.casefold() for app in found}
        for name, path in _shortcut_files():
            if name.casefold() not in names:  # games: .url shortcuts the list leaves out
                names.add(name.casefold())
                found.append(App(name, str(path)))
        log.debug("App nel menu Start: %d", len(found))
        return found

    def launch(self, app: App) -> None:
        target = Path(app.target)
        if target.suffix.lower() in SHORTCUTS and target.is_file():
            os.startfile(str(target))  # a Steam/Epic game shortcut (steam://, epic...://)
            return
        # explorer resolves both Store apps (their AUMID) and classic shortcuts.
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app.target}"])

    def open_path(self, path: Path) -> None:
        os.startfile(str(path))

    # --- files --------------------------------------------------------------------------

    def search_index(
        self, terms: list[str], kind: str | None, roots: list[Path], limit: int
    ) -> list[Found] | None:
        try:
            return self._query_index(terms, kind, roots, limit)
        except Exception as exc:  # index off or service stopped: walk the folders instead
            log.warning("Indice di Windows non disponibile (%s): cerco nelle cartelle", exc)
            return None

    def _query_index(
        self, terms: list[str], kind: str | None, roots: list[Path], limit: int
    ) -> list[Found]:
        import win32com.client

        where = [f"System.FileName LIKE '%{t}%'" for t in terms]  # terms: letters/digits only
        if roots and not any(r.anchor == str(r) for r in roots):
            scopes = " OR ".join(
                "SCOPE='file:" + str(r).replace("\\", "/").replace("'", "''") + "'" for r in roots
            )
            where.append(f"({scopes})")
        if kind == "folder":
            where.append("System.Kind = 'folder'")
        elif kind in INDEX_KINDS:
            where.append(f"System.Kind = '{INDEX_KINDS[kind]}'")
        sql = (
            f"SELECT TOP {int(limit)} System.ItemUrl, System.DateModified "
            f"FROM SystemIndex WHERE {' AND '.join(where)} ORDER BY System.DateModified DESC"
        )
        conn = win32com.client.Dispatch("ADODB.Connection")
        conn.Open("Provider=Search.CollatorDSO;Extended Properties='Application=Windows';")
        try:
            rs = win32com.client.Dispatch("ADODB.Recordset")
            rs.Open(sql, conn)
            found = []
            while not rs.EOF:
                path = index_path(str(rs.Fields.Item("System.ItemUrl").Value))
                modified = rs.Fields.Item("System.DateModified").Value
                stamp = modified.timestamp() if isinstance(modified, datetime) else 0.0
                if path is not None:
                    found.append(Found(path, stamp))
                rs.MoveNext()
            rs.Close()
            return found
        finally:
            conn.Close()

    # --- windows ------------------------------------------------------------------------

    def windows(self) -> list[Window]:
        import psutil
        import win32gui
        import win32process

        active = win32gui.GetForegroundWindow()
        found: list[Window] = []

        def visit(hwnd: int, _: Any) -> bool:
            if not _alt_tab_window(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                name = psutil.Process(pid).name()
            except Exception:
                name = "?"
            found.append(Window(hwnd, title, name, hwnd == active))
            return True

        win32gui.EnumWindows(visit, None)
        return found  # front to back (Z order)

    def read_window(self, window: Window, max_chars: int, timeout_s: float) -> str:
        from pywinauto.uia_element_info import UIAElementInfo

        deadline = time.monotonic() + timeout_s
        root = UIAElementInfo(handle_or_elem=window.handle)
        lines: list[str] = []
        total = 0
        queue: deque[Any] = deque([root])
        while queue and total < max_chars and time.monotonic() < deadline:
            info = queue.popleft()
            text = _element_text(info)
            if text and (not lines or lines[-1] != text) and text != window.title:
                lines.append(text)
                total += len(text) + 1
            try:
                queue.extend(info.children())
            except Exception:
                continue
        if queue and time.monotonic() >= deadline:
            log.info("Lettura di %r interrotta dopo %.1f s", window.title, timeout_s)
        return "\n".join(lines)

    def focus(self, window: Window) -> None:
        from pywinauto.controls.hwndwrapper import HwndWrapper

        HwndWrapper(window.handle).set_focus()  # also restores a minimised window

    def close(self, window: Window) -> None:
        import win32gui

        win32gui.PostMessage(window.handle, WM_CLOSE, 0, 0)  # as the X button: may ask to save

    def send_keys(self, sequence: str) -> None:
        from pywinauto.keyboard import send_keys

        send_keys(sequence, pause=0.02, with_spaces=True, with_tabs=True, with_newlines=True)

    def paste(self, text: str) -> None:
        """Ctrl+V of `text`; JenJo's clipboard (text only) is put back afterwards."""
        import win32clipboard
        from pywinauto.keyboard import send_keys

        old = _clipboard_text()
        _set_clipboard(text.replace("\r\n", "\n").replace("\n", "\r\n"))
        try:
            send_keys("^v", pause=0.02)
            time.sleep(PASTE_SETTLE_S)  # the app reads the clipboard after the key press
        finally:
            if old is not None:
                _set_clipboard(old)
            else:
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                finally:
                    win32clipboard.CloseClipboard()

    # --- screen and commands (v0.3.2) -----------------------------------------------------

    def screenshot(self, window: Window | None) -> Any:
        from PIL import ImageGrab

        if window is None:
            return ImageGrab.grab()  # the primary screen
        box = _window_box(window.handle)
        if box is None:  # minimised or off screen: what JenJo sees is the screen
            log.info("Finestra %r non visibile: screenshot dello schermo", window.title)
            return ImageGrab.grab()
        return ImageGrab.grab(bbox=box, all_screens=True)

    def is_admin(self) -> bool:
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return True  # unknown: treated as elevated, so commands stay off

    def run_command(self, command: str, timeout_s: float, cwd: Path) -> CommandResult:
        return terminal.run_process(terminal.powershell_argv(command), timeout_s, cwd)


def _shortcut_files() -> list[tuple[str, Path]]:
    """Shortcuts in both Start menus (all levels) and on the desktops (top level only)."""
    env = os.environ
    menus = [
        Path(env.get(var, "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        for var in ("APPDATA", "ProgramData")
        if env.get(var)
    ]
    home = Path.home()
    desks = [home / "Desktop", home / "OneDrive" / "Desktop"]
    if env.get("PUBLIC"):
        desks.append(Path(env["PUBLIC"]) / "Desktop")
    return shortcuts(menus, deep=True) + shortcuts(desks, deep=False)


def _alt_tab_window(hwnd: int) -> bool:
    """Visible, not owned, not a tool window and not cloaked (hidden Store apps)."""
    import win32gui

    if not win32gui.IsWindowVisible(hwnd) or win32gui.GetWindow(hwnd, GW_OWNER):
        return False
    if win32gui.GetWindowLong(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
        return False
    cloaked = ctypes.c_int(0)
    try:
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
        )
    except Exception:
        return True
    return cloaked.value == 0


def _clipboard_text() -> str | None:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
            return str(win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT))
        return None
    finally:
        win32clipboard.CloseClipboard()


def _set_clipboard(text: str) -> None:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


def _window_box(hwnd: int) -> tuple[int, int, int, int] | None:
    """The window's visible frame in physical pixels (Pillow grabs per-monitor DPI aware, so
    the rectangle must be read the same way), or None when minimised or empty."""
    import ctypes.wintypes

    import win32gui

    if win32gui.IsIconic(hwnd):
        return None
    user32 = ctypes.windll.user32
    previous = None
    try:
        user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p  # a handle, not an int
        previous = user32.SetThreadDpiAwarenessContext(
            ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        )
    except Exception:
        pass  # older Windows: coordinates may be scaled, the crop is then approximate
    try:
        rect = ctypes.wintypes.RECT()
        ok = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
        )
        if ok != 0:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        else:
            left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
    finally:
        if previous:
            user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))
    if right - left < 10 or bottom - top < 10:
        return None
    return left, top, right, bottom


def _element_text(info: Any) -> str:
    try:
        if info.control_type in TEXT_TYPES:
            value = info.element.GetCurrentPropertyValue(UIA_VALUE_PROPERTY)
            if value:
                return str(value).strip()
        return str(info.name or "").strip()
    except Exception:
        return ""
