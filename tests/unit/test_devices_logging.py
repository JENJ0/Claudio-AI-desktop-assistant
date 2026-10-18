import json
import logging

import pytest

from claudio.audio.capture import apply_gain
from claudio.audio.devices import resolve_device
from claudio.logging_setup import JsonFormatter, setup_logging

DEVICES = [
    {
        "name": "Microsoft Sound Mapper - Input",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
    },
    {
        "name": "Cuffie (HyperX Cloud)",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
    },
    {
        "name": "Microfono (HyperX Cloud)",
        "hostapi": 0,
        "max_input_channels": 1,
        "max_output_channels": 0,
    },
]


def test_default_and_index() -> None:
    assert resolve_device("", "input", DEVICES, HOSTAPIS) is None
    assert resolve_device(2, "input", DEVICES, HOSTAPIS) == 2
    assert resolve_device("1", "output", DEVICES, HOSTAPIS) == 1


def test_partial_name_respects_direction() -> None:
    assert resolve_device("hyperx", "input", DEVICES, HOSTAPIS) == 2
    assert resolve_device("HyperX", "output", DEVICES, HOSTAPIS) == 1


def test_unknown_name() -> None:
    with pytest.raises(ValueError, match="list_audio_devices"):
        resolve_device("Blue Yeti", "input", DEVICES, HOSTAPIS)


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.makeLogRecord(
        {
            "msg": "Azione %s",
            "args": ("x",),
            "levelname": "INFO",
            "name": "t",
            "command": "open_chrome",
        }
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "Azione x" and payload["command"] == "open_chrome"


def test_setup_logging_writes_json_file(tmp_path) -> None:
    setup_logging(tmp_path, "INFO")
    logging.getLogger("claudio.test").info("ciao à", extra={"route": "local"})
    for handler in logging.getLogger().handlers:
        handler.flush()
    line = (tmp_path / "claudio.log.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    assert json.loads(line)["route"] == "local"
    for handler in list(logging.getLogger().handlers):
        handler.close()
        logging.getLogger().removeHandler(handler)


HOSTAPIS = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]
SAME_NAME = [
    {
        "name": "Altoparlanti (Wireless Stereo H",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
    },
    {
        "name": "Altoparlanti (Wireless Stereo Headset)",
        "hostapi": 1,
        "max_input_channels": 0,
        "max_output_channels": 2,
    },
    {
        "name": "Altoparlanti (Wireless Stereo Headset)",
        "hostapi": 2,
        "max_input_channels": 0,
        "max_output_channels": 2,
    },
]


def test_wasapi_wins_when_names_tie() -> None:
    assert resolve_device("Wireless Stereo", "output", SAME_NAME, HOSTAPIS) == 2


def test_first_match_without_wasapi() -> None:
    assert resolve_device("Wireless Stereo", "output", SAME_NAME[:2], HOSTAPIS) == 0


def test_input_gain_boosts_and_clips_instead_of_wrapping() -> None:
    import numpy as np

    frame = np.array([0, 1000, -1000, 20000, -20000], dtype=np.int16)
    boosted = apply_gain(frame, 10 ** (12 / 20))
    assert boosted.dtype == np.int16
    assert list(boosted[:3]) == [0, 3981, -3981]
    assert list(boosted[3:]) == [32767, -32768]
