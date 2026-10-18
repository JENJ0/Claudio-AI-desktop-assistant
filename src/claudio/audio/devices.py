"""Resolve configured audio devices by index or (partial) name."""

from __future__ import annotations

from typing import Any


def resolve_device(
    spec: str | int, kind: str, devices: list[dict[str, Any]] | None = None
) -> int | None:
    """Return a device index, or None for the system default.

    kind: "input" or "output". Raises ValueError if nothing matches.
    """
    if spec == "" or spec is None:
        return None
    if isinstance(spec, int) or str(spec).strip().isdigit():
        return int(spec)
    if devices is None:
        import sounddevice as sd

        devices = [dict(d) for d in sd.query_devices()]
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    needle = str(spec).lower()
    for index, dev in enumerate(devices):
        if needle in str(dev["name"]).lower() and dev[channels_key] > 0:
            return index
    raise ValueError(
        f"Nessun dispositivo di {kind} contiene '{spec}'. "
        "Elencali con: uv run python scripts/list_audio_devices.py"
    )
