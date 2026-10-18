"""Resolve configured audio devices by index or (partial) name."""

from __future__ import annotations

from typing import Any

PREFERRED_HOST_API = "WASAPI"  # lowest latency on Windows; same name also appears under MME/DS


def resolve_device(
    spec: str | int,
    kind: str,
    devices: list[dict[str, Any]] | None = None,
    hostapis: list[dict[str, Any]] | None = None,
) -> int | None:
    """Return a device index, or None for the system default.

    kind: "input" or "output". A name may match the same device under several host APIs
    (MME, DirectSound, WASAPI...): WASAPI wins. Raises ValueError if nothing matches.
    """
    if spec == "" or spec is None:
        return None
    if isinstance(spec, int) or str(spec).strip().isdigit():
        return int(spec)
    if devices is None or hostapis is None:
        import sounddevice as sd

        devices = [dict(d) for d in sd.query_devices()]
        hostapis = [dict(h) for h in sd.query_hostapis()]
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    needle = str(spec).lower()
    matches = [
        index
        for index, dev in enumerate(devices)
        if needle in str(dev["name"]).lower() and dev[channels_key] > 0
    ]
    if not matches:
        raise ValueError(
            f"Nessun dispositivo di {kind} contiene '{spec}'. "
            "Elencali con: uv run python scripts/list_audio_devices.py"
        )
    for index in matches:
        if PREFERRED_HOST_API in str(hostapis[devices[index]["hostapi"]]["name"]):
            return index
    return matches[0]


def describe_device(device: int | None, kind: str) -> str:
    """ "name (host API)" of a device, or of the default one when device is None."""
    import sounddevice as sd

    info = sd.query_devices(device, kind)
    return f"{info['name']} ({sd.query_hostapis(info['hostapi'])['name']})"


def stream_settings(device: int | None) -> Any:
    """WASAPI shared mode only accepts the mixer's rate (e.g. 48 kHz): let Windows convert
    our 16 kHz / 22.05 kHz streams. Other host APIs convert on their own."""
    if device is None:
        return None
    import sounddevice as sd

    api = sd.query_hostapis(sd.query_devices(device)["hostapi"])["name"]
    return sd.WasapiSettings(auto_convert=True) if PREFERRED_HOST_API in api else None
