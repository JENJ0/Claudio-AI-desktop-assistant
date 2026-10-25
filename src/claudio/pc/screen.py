"""The screenshot sent to Haiku, as a last resort when a window cannot be read as text (v0.3.2).

Small on purpose: the long side at most `[pc] screenshot_side` pixels (1024 by default) and a
JPEG, so one look costs about 800-1400 input tokens (~0.0001 USD with Haiku). Nothing is saved
on disk; the picture lives only in the request that needs it."""

from __future__ import annotations

import base64
import io
from typing import Any

MEDIA_TYPE = "image/jpeg"


def to_jpeg(image: Any, max_side: int, quality: int) -> bytes:
    """`image` (a PIL image) shrunk to `max_side` on its long side, as JPEG bytes."""
    from PIL import Image

    img = image.convert("RGB")  # JPEG has no alpha; screenshots on Windows come as RGBA/BGRX
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def image_block(jpeg: bytes) -> dict[str, Any]:
    """The picture as a content block of a tool_result (Messages API)."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": MEDIA_TYPE,
            "data": base64.b64encode(jpeg).decode("ascii"),
        },
    }


def estimated_tokens(width: int, height: int) -> int:
    """Anthropic's rule of thumb for an image: width * height / 750 tokens."""
    return max(1, round(width * height / 750))
