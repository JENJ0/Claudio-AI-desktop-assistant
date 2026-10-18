import json
import logging

import pytest

from claudio.audio.devices import resolve_device
from claudio.logging_setup import JsonFormatter, setup_logging

DEVICES = [
    {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Cuffie (HyperX Cloud)", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Microfono (HyperX Cloud)", "max_input_channels": 1, "max_output_channels": 0},
]


def test_default_and_index() -> None:
    assert resolve_device("", "input", DEVICES) is None
    assert resolve_device(2, "input", DEVICES) == 2
    assert resolve_device("1", "output", DEVICES) == 1


def test_partial_name_respects_direction() -> None:
    assert resolve_device("hyperx", "input", DEVICES) == 2
    assert resolve_device("HyperX", "output", DEVICES) == 1


def test_unknown_name() -> None:
    with pytest.raises(ValueError, match="list_audio_devices"):
        resolve_device("Blue Yeti", "input", DEVICES)


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
