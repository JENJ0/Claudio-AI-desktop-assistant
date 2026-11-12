"""--check on a machine without the models: it reports every missing piece, never crashes."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from claudio.checks import run_checks
from claudio.config import load_settings


def test_missing_models_fail_the_check(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    settings = load_settings().model_copy(update={"models_dir": tmp_path})
    with caplog.at_level(logging.INFO):
        assert run_checks(settings) is False
    text = caplog.text
    assert "commands.toml" in text or "comandi" in text.lower()  # the router was checked
    assert len(caplog.records) >= 5  # one line per check


# --- IT-1: the Whisper model CLAUDIO will load must be downloaded --------------------------


def fake_snapshot(root: Path, repo: str, files: tuple[str, ...]) -> Path:
    """A Hugging Face cache as faster-whisper leaves it in models/whisper (no symlinks)."""
    cache = root / f"models--{repo.replace('/', '--')}"
    revision = "0" * 40
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text(revision, encoding="utf-8")
    folder = cache / "snapshots" / revision
    folder.mkdir(parents=True)
    for name in files:
        (folder / name).write_bytes(b"x" * 10)
    return folder


COMPLETE = ("model.bin", "config.json", "tokenizer.json", "vocabulary.json")


def test_whisper_model_found_without_downloading(tmp_path: Path) -> None:
    from claudio.checks import find_whisper_model

    folder = fake_snapshot(tmp_path, "Systran/faster-whisper-large-v3", COMPLETE)
    assert find_whisper_model("large-v3", tmp_path) == folder


def test_whisper_model_missing_or_half_downloaded(tmp_path: Path) -> None:
    from claudio.checks import find_whisper_model

    with pytest.raises(FileNotFoundError, match="large-v3 non scaricato"):
        find_whisper_model("large-v3", tmp_path)
    fake_snapshot(tmp_path, "Systran/faster-whisper-small", ("config.json", "vocabulary.txt"))
    with pytest.raises(FileNotFoundError, match=r"incompleto.*model\.bin, tokenizer\.json"):
        find_whisper_model("small", tmp_path)
    with pytest.raises(ValueError):  # a typo in [stt] model
        find_whisper_model("large-v9", tmp_path)


def test_whisper_model_in_a_local_folder(tmp_path: Path) -> None:
    from claudio.checks import find_whisper_model

    folder = tmp_path / "my-model"
    folder.mkdir()
    for name in COMPLETE:
        (folder / name).write_bytes(b"x")
    assert find_whisper_model(str(folder), tmp_path / "unused") == folder


@pytest.mark.parametrize(
    ("device", "gpus", "expected"),
    [
        ("auto", 1, ("large-v3", "small")),  # the GPU model, the CPU one in case
        ("auto", 0, ("small", "large-v3")),  # no GPU: Whisper falls back to the CPU model
        ("cuda", 1, ("large-v3", None)),  # never falls back
        ("cpu", 1, ("small", None)),
    ],
)
def test_the_whisper_model_that_will_load(device: str, gpus: int, expected: tuple) -> None:
    from claudio.checks import whisper_models

    stt = load_settings().stt.model_copy(
        update={"device": device, "model": "large-v3", "fallback_model": "small"}
    )
    assert whisper_models(stt, gpus) == expected


def test_check_reports_the_whisper_model(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from claudio.checks import check_whisper

    base = load_settings()
    stt = base.stt.model_copy(
        update={"device": "auto", "model": "large-v3", "fallback_model": "small"}
    )
    settings = base.model_copy(update={"models_dir": tmp_path, "stt": stt})
    results: list[tuple[bool, str]] = []
    with caplog.at_level(logging.WARNING):
        check_whisper(settings, 1, lambda good, what: results.append((good, what)))
    assert results[0][0] is False
    assert "large-v3" in results[0][1] and "download_models.py --whisper large-v3" in results[0][1]
    assert "small" in caplog.text and "senza GPU" in caplog.text  # the fallback: a warning

    fake_snapshot(tmp_path / "whisper", "Systran/faster-whisper-large-v3", COMPLETE)
    results.clear()
    check_whisper(settings, 1, lambda good, what: results.append((good, what)))
    assert results == [(True, "modello Whisper large-v3 (0.0 GB)")]


def test_cuda_without_a_gpu_is_an_error(tmp_path: Path) -> None:
    """device = "cuda" never falls back to the CPU: without a GPU CLAUDIO cannot start."""
    from claudio.checks import check_whisper

    fake_snapshot(tmp_path / "whisper", "Systran/faster-whisper-large-v3", COMPLETE)
    base = load_settings()
    stt = base.stt.model_copy(update={"device": "cuda", "model": "large-v3"})
    settings = base.model_copy(update={"models_dir": tmp_path, "stt": stt})
    results: list[tuple[bool, str]] = []
    check_whisper(settings, 0, lambda good, what: results.append((good, what)))
    assert results[0] == (False, '[stt] device = "cuda" senza GPU CUDA: Whisper non partirà')
    results.clear()
    check_whisper(settings, 1, lambda good, what: results.append((good, what)))
    assert [good for good, _ in results] == [True]


def test_a_broken_faster_whisper_does_not_stop_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from claudio import checks

    def broken(name: str, root: Path) -> Path:
        raise ImportError("DLL load failed while importing _ext")

    monkeypatch.setattr(checks, "find_whisper_model", broken)
    settings = load_settings().model_copy(update={"models_dir": tmp_path})
    results: list[tuple[bool, str]] = []
    checks.check_whisper(settings, 1, lambda good, what: results.append((good, what)))
    assert results[0][0] is False
    assert "DLL load failed" in results[0][1]


# --- IT-2 R1: what run_checks reports, and in which order ----------------------------------

check_log = logging.getLogger("claudio.checks")
HEY = "hey_jarvis_v0.1.onnx"
NO_HARDWARE = RuntimeError("no hardware")


class Machine:
    """What run_checks asks of the computer: a healthy one until a test breaks a piece."""

    def __init__(self) -> None:
        self.gpus: int | Exception = 1
        self.compute_types_error: Exception | None = None
        self.whisper_gpus: list[int] = []  # the `gpus` check_whisper was given
        self.audio_errors: dict[str, Exception] = {}  # by kind: "input" / "output"
        self.output_calls: list[tuple[object, object]] = []  # what pick_output was given


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Machine:
    """Fakes only the hardware: packages, GPU, Whisper's cache and the audio devices."""
    from claudio import checks

    m = Machine()
    monkeypatch.setattr(checks, "PACKAGES", ("keyword",))  # a real import, no version

    def whisper(settings: object, gpus: int, report: object) -> None:
        m.whisper_gpus.append(gpus)
        check_log.info("whisper checked")

    monkeypatch.setattr(checks, "check_whisper", whisper)

    def device_count() -> int:
        if isinstance(m.gpus, Exception):
            raise m.gpus
        return m.gpus

    def compute_types(device: str) -> set[str]:
        if m.compute_types_error:
            raise m.compute_types_error
        return {"int8", "float16"}

    monkeypatch.setattr("ctranslate2.get_cuda_device_count", device_count)
    monkeypatch.setattr("ctranslate2.get_supported_compute_types", compute_types)
    monkeypatch.setattr("claudio.stt.transcriber.register_cuda_dlls", lambda: None)

    def resolve(spec: object, kind: str) -> int:
        if kind in m.audio_errors:
            raise m.audio_errors[kind]
        return 3

    def pick(spec: object, windows_default: object = None) -> int:
        if "output" in m.audio_errors:
            raise m.audio_errors["output"]
        m.output_calls.append((spec, windows_default))
        return 7

    class FakeWindowsDefault:
        def __call__(self) -> str:
            return "Speakers"

    monkeypatch.setattr("claudio.audio.devices.resolve_device", resolve)
    monkeypatch.setattr("claudio.audio.devices.pick_output", pick)
    monkeypatch.setattr("claudio.audio.devices.WindowsDefaultOutput", FakeWindowsDefault)
    monkeypatch.setattr(
        "claudio.audio.devices.describe_device", lambda index, kind: f"{kind} device {index}"
    )
    return m


def settings_for(models: Path, engine: str = "piper"):
    from claudio.config import WakeWordModel

    base = load_settings()
    return base.model_copy(
        update={
            "models_dir": models,
            "tts": base.tts.model_copy(update={"engine": engine}),
            "wakeword": base.wakeword.model_copy(
                update={"models": [WakeWordModel(name="hey_jarvis")]}
            ),
        }
    )


def install(models: Path, *rels: str) -> None:
    for rel in rels:
        path = models / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def install_everything(models: Path) -> None:
    install(
        models,
        "vad/silero_vad.onnx",
        "piper/it_IT-paola-medium.onnx",
        "piper/it_IT-paola-medium.onnx.json",
        "wakeword/melspectrogram.onnx",
        "wakeword/embedding_model.onnx",
        f"wakeword/{HEY}",
    )


def run(settings, caplog: pytest.LogCaptureFixture) -> tuple[bool, list[str]]:
    """run_checks and what it logged, as "LEVEL message" (OK lines read "INFO OK   ...")."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="claudio.checks"):
        ok = run_checks(settings)
    lines = [
        f"{r.levelname} {r.getMessage()}" for r in caplog.records if r.name == "claudio.checks"
    ]
    return ok, lines


def test_a_healthy_machine_passes_and_reports_in_a_fixed_order(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    """The log is the report the user reads: packages, commands, models, wake words, GPU,
    Whisper, then the audio devices; no closing error when everything is fine."""
    install_everything(tmp_path)
    ok, lines = run(settings_for(tmp_path), caplog)
    assert ok is True
    assert "comandi in" in lines[1]
    del lines[1]  # the count of commands changes with commands.toml
    sep = os.sep
    assert lines == [
        "INFO OK   keyword",
        f"INFO OK   modello vad{sep}silero_vad.onnx",
        f"INFO OK   modello piper{sep}it_IT-paola-medium.onnx",
        f"INFO OK   modello piper{sep}it_IT-paola-medium.onnx.json",
        f"INFO OK   modello wakeword{sep}melspectrogram.onnx",
        f"INFO OK   modello wakeword{sep}embedding_model.onnx",
        f"INFO OK   wake word {HEY} (soglia 0.5)",
        "INFO OK   GPU CUDA: 1 (tipi: float16, int8)",
        "INFO whisper checked",
        "INFO OK   microfono: input device 3",
        "INFO OK   uscita audio: output device 7 (segue l'uscita predefinita di Windows)",
    ]
    assert machine.whisper_gpus == [1]


def test_a_failure_closes_the_report_with_a_hint(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    ok, lines = run(settings_for(tmp_path), caplog)  # nothing installed
    assert ok is False
    assert lines[-1] == (
        "ERROR Qualcosa manca: vedi le righe ERR. Modelli: uv run python scripts/download_models.py"
    )


def test_a_package_that_does_not_import_is_an_error(
    tmp_path: Path,
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from claudio import checks

    install_everything(tmp_path)
    monkeypatch.setattr(checks, "PACKAGES", ("keyword", "no_such_package_xyz"))
    ok, lines = run(settings_for(tmp_path), caplog)
    assert ok is False
    assert lines[0] == "INFO OK   keyword"
    assert lines[1] == "ERROR ERR  no_such_package_xyz: No module named 'no_such_package_xyz'"


def test_unreadable_commands_are_an_error_and_the_report_goes_on(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    install_everything(tmp_path)
    base = settings_for(tmp_path)
    router = base.router.model_copy(update={"commands_file": tmp_path / "missing.toml"})
    ok, lines = run(base.model_copy(update={"router": router}), caplog)
    assert ok is False
    assert lines[1].startswith("ERROR ERR  comandi: ")
    assert "INFO whisper checked" in lines  # the later checks still ran


def test_kokoro_files_are_checked_only_for_the_engines_that_speak_with_kokoro(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    from claudio.tts.kokoro import KOKORO_FILES

    install_everything(tmp_path)
    _, piper = run(settings_for(tmp_path, engine="piper"), caplog)
    assert not any("kokoro" in line for line in piper)
    first_feature = f"INFO OK   modello wakeword{os.sep}melspectrogram.onnx"
    for engine in ("kokoro", "jack"):
        ok, lines = run(settings_for(tmp_path, engine=engine), caplog)
        assert ok is False  # nothing of Kokoro is installed
        errors = [line for line in lines if line.startswith("ERROR ERR  modello kokoro")]
        assert errors == [f"ERROR ERR  modello kokoro{os.sep}{name}" for name in KOKORO_FILES]
        assert lines.index(errors[0]) < lines.index(first_feature)  # after Piper's, before the rest


def test_jack_files_are_optional_and_a_missing_one_is_only_a_warning(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    from claudio.tts.kokoro import KOKORO_FILES

    install_everything(tmp_path)
    install(tmp_path, *(f"kokoro/{name}" for name in KOKORO_FILES))
    settings = settings_for(tmp_path, engine="jack")
    tts = settings.tts
    install(tmp_path, str(tts.jack_python))  # the only Jack file that is there
    ok, lines = run(settings, caplog)
    assert ok is True  # warnings never fail the check
    assert f"INFO OK   modello {tts.jack_python}" in lines
    for missing in (tts.jack_checkpoint, tts.jack_rvc_model, tts.jack_applio):
        assert f"WARNING WARN voce di Jack non installata ({missing}): parlerà Kokoro" in lines
    assert sum("Jack non installata" in line for line in lines) == 3
    _, others = run(settings_for(tmp_path, engine="kokoro"), caplog)
    assert not any("Jack" in line for line in others)


def test_each_wake_word_is_reported_apart(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    from claudio.config import WakeWordModel

    install_everything(tmp_path)
    base = settings_for(tmp_path)
    models = [WakeWordModel(name="hey_jarvis", threshold=0.7), WakeWordModel(name="not_there")]
    settings = base.model_copy(
        update={"wakeword": base.wakeword.model_copy(update={"models": models})}
    )
    ok, lines = run(settings, caplog)
    assert ok is False
    assert f"INFO OK   wake word {HEY} (soglia 0.7)" in lines
    missing = "ERROR ERR  Modello wake word 'not_there' non trovato"
    assert any(line.startswith(missing) for line in lines)


def test_without_a_gpu_the_check_warns_and_whisper_is_told_so(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    install_everything(tmp_path)
    machine.gpus = 0
    settings = settings_for(tmp_path)
    ok, lines = run(settings, caplog)
    assert ok is True
    fallback = settings.stt.fallback_model
    assert f"WARNING WARN nessuna GPU CUDA: lo STT userà la CPU ({fallback})" in lines
    assert not any(line.startswith("INFO OK   GPU") for line in lines)
    assert machine.whisper_gpus == [0]


def test_a_gpu_check_that_fails_halfway_keeps_the_count_it_found(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    """The count worked and the next call did not: Whisper is still told there are 2 GPUs."""
    install_everything(tmp_path)
    machine.gpus = 2
    machine.compute_types_error = RuntimeError("driver")
    ok, lines = run(settings_for(tmp_path), caplog)
    assert ok is True
    assert "WARNING WARN controllo GPU fallito: driver" in lines
    assert machine.whisper_gpus == [2]

    machine.gpus = RuntimeError("no cuda")  # the count itself fails
    machine.whisper_gpus.clear()
    ok, lines = run(settings_for(tmp_path), caplog)
    assert ok is True
    assert "WARNING WARN controllo GPU fallito: no cuda" in lines
    assert machine.whisper_gpus == [0]


def test_an_unusable_audio_device_is_an_error_but_no_hardware_is_a_warning(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    install_everything(tmp_path)
    machine.audio_errors = {"input": ValueError("nessun microfono")}
    ok, lines = run(settings_for(tmp_path), caplog)
    assert ok is False
    assert "ERROR ERR  microfono: nessun microfono" in lines

    machine.audio_errors = {"input": NO_HARDWARE, "output": NO_HARDWARE}  # e.g. CI
    ok, lines = run(settings_for(tmp_path), caplog)
    assert ok is True
    assert "WARNING WARN microfono: no hardware" in lines
    assert "WARNING WARN uscita audio: no hardware" in lines


def test_an_output_by_name_is_resolved_without_asking_windows(
    tmp_path: Path, machine: Machine, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the default ("") output follows Windows' default and says so."""
    install_everything(tmp_path)
    base = settings_for(tmp_path)
    audio = base.audio.model_copy(update={"output_device": "Cuffie"})
    _, lines = run(base.model_copy(update={"audio": audio}), caplog)
    assert machine.output_calls == [("Cuffie", None)]
    assert "INFO OK   uscita audio: output device 7" in lines

    machine.output_calls.clear()
    _, lines = run(base, caplog)
    assert machine.output_calls == [("", "Speakers")]
