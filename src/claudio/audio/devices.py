"""Resolve configured audio devices by index or (partial) name, and follow Windows' default."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from typing import Any

from claudio.i18n import tr

log = logging.getLogger(__name__)

AUTO = "auto"  # input_device: the microphone with the most signal (probe_inputs)
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


def pick_device(spec: str | int, kind: str, windows_default: str | None = None) -> int | None:
    """The configured device; with none configured, the one Windows is using right now."""
    if spec != "" and spec is not None:
        return resolve_device(spec, kind)
    if windows_default:
        index = match_endpoint(windows_default, kind)
        if index is not None:
            return index
    return resolve_device("", kind)


def pick_output(spec: str | int, windows_default: str | None = None) -> int | None:
    return pick_device(spec, "output", windows_default)


class WindowsDefaultOutput:
    """Name of the output device Windows uses right now (Core Audio, below a millisecond).

    PortAudio reads the default device once at start-up; asking Windows directly follows a
    change made while CLAUDIO runs. Returns None off Windows or if Core Audio is unavailable."""

    FLOW = 0  # eRender

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
        device = local.enumerator.GetDefaultAudioEndpoint(self.FLOW, 1)  # multimedia role
        device_id = device.GetId()
        if device_id != local.id:
            local.id, local.name = device_id, AudioUtilities.CreateDevice(device).FriendlyName
        return local.name


class WindowsDefaultInput(WindowsDefaultOutput):
    """Name of the microphone Windows uses right now (Settings > Sound > Input)."""

    FLOW = 1  # eCapture


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
    if spec == "" or spec is None or spec == AUTO:
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


# Inputs that are not a microphone: they hear what the PC plays (Claudio's own voice).
NOT_A_MICROPHONE = ("loopback", "stereo mix", "missaggio stereo", "what u hear", "wave out")
PROBE_S = 0.6  # how long every microphone is listened to when choosing one
PROBE_WINDOW = 512  # 32 ms at 16 kHz: the level is the loudest window
DEFAULT_MARGIN_DB = 10.0  # another microphone must be this much louder than Windows' one
DEAD_DBFS = -90.0  # quieter than this: switched off or muted, not a quiet room


def candidate_inputs(
    devices: list[dict[str, Any]] | None = None, hostapis: list[dict[str, Any]] | None = None
) -> list[int]:
    """Microphones worth listening to: one entry per device (WASAPI when there is one)."""
    devices, hostapis = _query(devices, hostapis)
    usable = [
        i
        for i, dev in enumerate(devices)
        if dev["max_input_channels"] > 0
        and not any(word in str(dev["name"]).lower() for word in NOT_A_MICROPHONE)
    ]
    wasapi = [
        i for i in usable if PREFERRED_HOST_API in str(hostapis[devices[i]["hostapi"]]["name"])
    ]
    return wasapi or usable


def choose_input(levels: dict[int, float], default: int | None) -> int | None:
    """The microphone that hears the most (loudest 32 ms, dBFS). Windows' own keeps the job
    unless another is clearly louder or it hears nothing at all (switched off, muted): room
    noise alone must not make Claudio jump between microphones."""
    alive = {i: db for i, db in levels.items() if db > DEAD_DBFS}
    if not alive:
        return default
    best = max(alive, key=lambda i: alive[i])
    if default in alive and alive[best] - alive[default] < DEFAULT_MARGIN_DB:
        return default
    return best


def probe_inputs(sample_rate: int, seconds: float = PROBE_S) -> dict[int, float]:
    """Listen to every microphone at once for `seconds`: {device index: loudest 32 ms in
    dBFS}. One that cannot be opened is left out. Blocking (run it in a thread)."""
    import sounddevice as sd

    loudest: dict[int, float] = {}
    streams = []

    def callback_for(index: int) -> Any:
        def callback(indata: Any, frames: int, time_info: object, status: object) -> None:
            import numpy as np

            audio = indata[:, 0].astype(np.float32) / 32768.0
            n = len(audio) // PROBE_WINDOW
            if n:
                windows = audio[: n * PROBE_WINDOW].reshape(n, PROBE_WINDOW)
                level = float(np.max(np.mean(np.square(windows), axis=1)))
                loudest[index] = max(loudest.get(index, 0.0), level)

        return callback

    for index in candidate_inputs():
        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                blocksize=PROBE_WINDOW,
                channels=1,
                dtype="int16",
                device=index,
                callback=callback_for(index),
                extra_settings=stream_settings(index),
            )
            stream.start()
            streams.append(stream)
        except Exception as exc:  # busy, unplugged, exclusive mode...
            log.debug("Microfono %d non apribile per la prova: %s", index, exc)
    try:
        time.sleep(seconds)
    finally:
        for stream in streams:
            with contextlib.suppress(Exception):
                stream.close()
    return {i: _dbfs(loudest.get(i, 0.0)) for i in loudest}


def _dbfs(mean_square: float) -> float:
    import math

    return round(10 * math.log10(mean_square), 1) if mean_square > 0 else -120.0
