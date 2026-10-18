"""`python -m claudio --check`: fast diagnostics without starting the assistant."""

from __future__ import annotations

import importlib
import logging

from claudio.config import Settings

log = logging.getLogger(__name__)

PACKAGES = (
    "numpy",
    "sounddevice",
    "onnxruntime",
    "openwakeword",
    "faster_whisper",
    "ctranslate2",
    "piper",
    "rapidfuzz",
)


def run_checks(settings: Settings) -> bool:
    ok = True

    def report(good: bool, what: str) -> None:
        nonlocal ok
        ok &= good
        (log.info if good else log.error)("%s %s", "OK  " if good else "ERR ", what)

    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            report(True, f"{name} {getattr(mod, '__version__', '')}".strip())
        except Exception as exc:
            report(False, f"{name}: {exc}")

    try:
        from claudio.router.matcher import CommandMatcher, load_commands

        commands = load_commands(settings.resolve(settings.router.commands_file))
        CommandMatcher(commands, settings.router.fuzzy_threshold)
        report(True, f"{len(commands)} comandi in {settings.router.commands_file}")
    except Exception as exc:
        report(False, f"comandi: {exc}")

    models = settings.resolve(settings.models_dir)
    from claudio.wakeword.detector import FEATURE_MODELS, find_wakeword_model

    wanted = [
        models / "vad" / "silero_vad.onnx",
        models / "piper" / f"{settings.tts.voice}.onnx",
        models / "piper" / f"{settings.tts.voice}.onnx.json",
    ]
    wanted += [models / "wakeword" / f for f in FEATURE_MODELS]
    for path in wanted:
        report(path.is_file(), f"modello {path.relative_to(models)}")
    for wake in settings.wakeword.models:
        try:
            path = find_wakeword_model(wake.name, models / "wakeword")
            report(True, f"wake word {path.name} (soglia {wake.threshold})")
        except FileNotFoundError as exc:
            report(False, str(exc))

    try:
        from claudio.stt.transcriber import register_cuda_dlls

        register_cuda_dlls()
        import ctranslate2

        gpus = ctranslate2.get_cuda_device_count()
        if gpus:
            types = sorted(ctranslate2.get_supported_compute_types("cuda"))
            report(True, f"GPU CUDA: {gpus} (tipi: {', '.join(types)})")
        else:
            log.warning(
                "WARN nessuna GPU CUDA: lo STT userà la CPU (%s)", settings.stt.fallback_model
            )
    except Exception as exc:
        log.warning("WARN controllo GPU fallito: %s", exc)

    from claudio.audio.devices import describe_device, resolve_device

    for kind, spec, label in (
        ("input", settings.audio.input_device, "microfono"),
        ("output", settings.audio.output_device, "uscita audio"),
    ):
        try:
            index = resolve_device(spec, kind)
            report(True, f"{label}: {describe_device(index, kind)}")
        except ValueError as exc:
            report(False, f"{label}: {exc}")
        except Exception as exc:  # no audio hardware at all (e.g. CI)
            log.warning("WARN %s: %s", label, exc)

    if not ok:
        log.error(
            "Qualcosa manca: vedi le righe ERR. Modelli: uv run python scripts/download_models.py"
        )
    return ok
