"""Resolve configured audio devices by index or (partial) name, and follow Windows' default."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
from typing import Any

from claudio.i18n import tr

log = logging.getLogger(__name__)

PREFERRED_HOST_API = "WASAPI"  # lowest latency on Windows; same name also appears under MME/DS
MME_NAME_LENGTH = 31  # MME cuts device names to this many characters


def _query(
    devices: list[dict[str, Any]] | None, hostapis: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if devices is None or hostapis is None:
        import sounddevice as sd

        devices = [dict(d) for d in sd.query_devices()]
        hostapis = [dict(h) for h in sd.query_hostapis()]
    return devices, hostapis


def default_device(
    kind: str,
    devices: list[dict[str, Any]] | None = None,
    hostapis: list[dict[str, Any]] | None = None,
) -> int | None:
    """Windows' default device as seen by WASAPI, or None (PortAudio's default, i.e. MME)."""
    _, hostapis = _query(devices, hostapis)
    for api in hostapis:
        if PREFERRED_HOST_API in str(api["name"]):
            index = int(api.get(f"default_{kind}_device", -1))
            return index if index >= 0 else None
    return None


def match_endpoint(
    name: str,
    kind: str,
    devices: list[dict[str, Any]] | None = None,
    hostapis: list[dict[str, Any]] | None = None,
) -> int | None:
    """Index of the device Windows calls `name` (WASAPI first; MME has it cut short)."""
    devices, hostapis = _query(devices, hostapis)
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    usable = [i for i, dev in enumerate(devices) if dev[channels_key] > 0]
    exact = [i for i in usable if devices[i]["name"] == name]
    for index in exact:
        if PREFERRED_HOST_API in str(hostapis[devices[index]["hostapi"]]["name"]):
            return index
    if exact:
        return exact[0]
    for index in usable:
        short = str(devices[index]["name"])
        if len(short) >= MME_NAME_LENGTH and name.startswith(short):
            return index
    return None


def pick_output(spec: str | int, windows_default: str | None = None) -> int | None:
    """The configured output; with none configured, the one Windows is using right now."""
    if spec != "" and spec is not None:
        return resolve_device(spec, "output")
    if windows_default:
        index = match_endpoint(windows_default, "output")
        if index is not None:
            return index
    return resolve_device("", "output")


class WindowsDefaultOutput:
    """Name of the output device Windows uses right now (Core Audio, below a millisecond).

    PortAudio reads the default device once at start-up; asking Windows directly follows a
    change made while CLAUDIO runs. Returns None off Windows or if Core Audio is unavailable."""

    def __init__(self) -> None:
        self._local = threading.local()  # COM objects belong to the thread that made them

    def __call__(self) -> str | None:
        if sys.platform != "win32":
            return None
        try:
            return self._query()
        except Exception as exc:
            log.debug("Uscita predefinita di Windows non leggibile: %s", exc)
            return None

    def _query(self) -> str | None:
        import comtypes
        from pycaw.pycaw import AudioUtilities

        local = self._local
        if not hasattr(local, "enumerator"):
            with contextlib.suppress(OSError):  # already initialised in this thread
                comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
            local.enumerator = AudioUtilities.GetDeviceEnumerator()
            local.id, local.name = None, None
        device = local.enumerator.GetDefaultAudioEndpoint(0, 1)  # render, multimedia
        device_id = device.GetId()
        if device_id != local.id:
            local.id, local.name = device_id, AudioUtilities.CreateDevice(device).FriendlyName
        return local.name


def resolve_device(
    spec: str | int,
    kind: str,
    devices: list[dict[str, Any]] | None = None,
    hostapis: list[dict[str, Any]] | None = None,
) -> int | None:
    """Return a device index; for "" the system default (under WASAPI when there is one).

    kind: "input" or "output". A name may match the same device under several host APIs
    (MME, DirectSound, WASAPI...): WASAPI wins. Raises ValueError if nothing matches.
    """
    if spec == "" or spec is None:
        return default_device(kind, devices, hostapis)
    if isinstance(spec, int) or str(spec).strip().isdigit():
        return int(spec)
    devices, hostapis = _query(devices, hostapis)
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    needle = str(spec).lower()
    matches = [
        index
        for index, dev in enumerate(devices)
        if needle in str(dev["name"]).lower() and dev[channels_key] > 0
    ]
    if not matches:
        raise ValueError(
            tr(
                "Nessun dispositivo di {kind} contiene '{spec}'. Elencali con: "
                "uv run python scripts/list_audio_devices.py",
                kind=kind,
                spec=spec,
            )
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
