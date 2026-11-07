"""The real Windows clipboard, through the helpers `WindowsDesktop.paste` uses.

Slow on purpose (`-m slow`, never in the normal suite): it overwrites the clipboard of whoever
runs it, then puts the real content back in a `finally`. No key is ever sent: the failure in
the middle of a paste is simulated by raising after `_set_clipboard`."""

from __future__ import annotations

import struct
import sys
from typing import Any

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "win32", reason="the Windows clipboard"),
]

# A 1 x 1 pixel DIB: BITMAPINFOHEADER (40 bytes) and one 32-bit pixel.
PIXEL_DIB = struct.pack("<IiiHHIIiiII", 40, 1, 1, 1, 32, 0, 4, 0, 0, 0, 0) + b"\x10\x20\x30\x00"
# DROPFILES (20 bytes, wide characters) and one path.
FILE_LIST = struct.pack("<IiiII", 20, 0, 0, 0, 1) + "C:\\Windows\\notepad.exe\0\0".encode(
    "utf-16-le"
)


def _put(content: dict[int, Any]) -> None:
    import win32clipboard

    from claudio.pc import windows

    windows._open_clipboard()  # retries: Win+V may still hold the clipboard after our last write
    try:
        win32clipboard.EmptyClipboard()
        for fmt, data in content.items():
            win32clipboard.SetClipboardData(fmt, data)
    finally:
        win32clipboard.CloseClipboard()


def _same(a: dict[int, bytes], b: dict[int, bytes]) -> bool:
    """Windows may round a block up with zeros: trailing zeros do not count."""
    return a.keys() == b.keys() and all(a[k].rstrip(b"\0") == b[k].rstrip(b"\0") for k in a)


def test_the_clipboard_comes_back_after_an_error_in_the_middle_of_a_paste() -> None:
    import win32clipboard

    from claudio.pc import windows

    real = windows._save_clipboard()
    try:
        for content in (
            {win32clipboard.CF_UNICODETEXT: "ciao\r\nmondo"},
            {win32clipboard.CF_DIB: PIXEL_DIB},
            {win32clipboard.CF_HDROP: FILE_LIST},
        ):
            _put(content)
            before = windows._save_clipboard()
            assert before, f"the test could not put {list(content)} on the clipboard"
            saved = windows._save_clipboard()
            try:
                windows._set_clipboard("testo incollato")
                raise RuntimeError("the paste fails here")
            except RuntimeError:
                pass
            finally:
                windows._restore_clipboard(saved)
            assert _same(windows._save_clipboard(), before)
    finally:
        windows._restore_clipboard(real)
        # Only the formats paste keeps can come back: say so loudly if even those did not.
        after = windows._save_clipboard()
        assert _same(after, real), (
            "JenJo's own clipboard did NOT come back as it was: "
            f"it held formats {sorted(real)}, it holds {sorted(after)} now. "
            "Copy it again, and do not trust paste with this content."
        )
