"""`python -m claudio --check`: fast diagnostics without starting the assistant."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from claudio.config import Settings
from claudio.i18n import tr

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


def kokoro_voices(kokoro_dir: Path) -> set[str] | None:
    """The voices in Kokoro's voices file; None when it cannot be read (reported apart)."""
    import numpy as np

    try:
        with np.load(kokoro_dir / "voices-v1.0.bin") as data:
            return set(data.files)
    except (OSError, ValueError):
        return None


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
        matcher = CommandMatcher(commands, settings.router.fuzzy_threshold)
        languages = ", ".join(matcher.languages)
        what = tr(
            "{count} comandi in {file} ({languages})",
            count=len(commands),
            file=settings.router.commands_file,
            languages=languages,
        )
        report(True, what)
    except Exception as exc:
        report(False, tr("comandi: {error}", error=exc))

    models = settings.resolve(settings.models_dir)
    from claudio.wakeword.detector import FEATURE_MODELS, find_wakeword_model

    wanted = [
        models / "vad" / "silero_vad.onnx",
        models / "piper" / f"{settings.tts.voice}.onnx",
        models / "piper" / f"{settings.tts.voice}.onnx.json",
    ]
    if settings.tts.engine in ("kokoro", "jack"):  # Jack's fallback is Kokoro
        from claudio.tts.kokoro import KOKORO_FILES

        wanted += [models / "kokoro" / name for name in KOKORO_FILES]
    if settings.tts.engine == "jack":
        tts = settings.tts
        wanted += [
            models / path
            for path in (tts.jack_python, tts.jack_checkpoint, tts.jack_rvc_model, tts.jack_applio)
        ]
    wanted += [models / "wakeword" / f for f in FEATURE_MODELS]
    for path in wanted:
        report(path.exists(), tr("modello {path}", path=path.relative_to(models)))
    for language in settings.stt.languages:
        if settings.tts.engine in ("kokoro", "jack"):  # one model, a voice per language inside it
            voice = settings.tts.kokoro_voice_for(language)
            found = kokoro_voices(models / "kokoro")
            if found is not None and voice not in found and voice != settings.tts.kokoro_voice:
                log.warning(
                    "WARN voce Kokoro %s (%s) sconosciuta: risponderò con quella principale",
                    voice,
                    language,
                )
            continue
        voice = settings.tts.piper_voice_for(language)
        if voice != settings.tts.voice and not (models / "piper" / f"{voice}.onnx").is_file():
            log.warning(
                "WARN voce %s (%s) mancante: risponderò con quella principale. "
                "Scaricala con: uv run python scripts/download_models.py --skip-whisper",
                voice,
                language,
            )
    for wake in settings.wakeword.models:
        try:
            path = find_wakeword_model(wake.name, models / "wakeword")
            report(
                True,
                tr(
                    "wake word {name} (soglia {threshold})",
                    name=path.name,
                    threshold=wake.threshold,
                ),
            )
        except FileNotFoundError as exc:
            report(False, str(exc))

    try:
        from claudio.stt.transcriber import register_cuda_dlls

        register_cuda_dlls()
        import ctranslate2

        gpus = ctranslate2.get_cuda_device_count()
        if gpus:
            types = sorted(ctranslate2.get_supported_compute_types("cuda"))
            report(
                True, tr("GPU CUDA: {count} (tipi: {types})", count=gpus, types=", ".join(types))
            )
        else:
            log.warning(
                "WARN nessuna GPU CUDA: lo STT userà la CPU (%s)", settings.stt.fallback_model
            )
    except Exception as exc:
        log.warning("WARN controllo GPU fallito: %s", exc)

    from claudio.audio.devices import (
        WindowsDefaultOutput,
        describe_device,
        pick_output,
        resolve_device,
    )

    for kind, spec, label in (
        ("input", settings.audio.input_device, tr("microfono")),
        ("output", settings.audio.output_device, tr("uscita audio")),
    ):
        try:
            if kind == "output":
                index = pick_output(spec, WindowsDefaultOutput()() if spec == "" else None)
            else:
                index = resolve_device(spec, kind)
            follows = (
                tr(" (segue l'uscita predefinita di Windows)")
                if kind == "output" and spec == ""
                else ""
            )
            report(True, f"{label}: {describe_device(index, kind)}{follows}")
        except ValueError as exc:
            report(False, f"{label}: {exc}")
        except Exception as exc:  # no audio hardware at all (e.g. CI)
            log.warning("WARN %s: %s", label, exc)

    if not ok:
        log.error(
            "Qualcosa manca: vedi le righe ERR. Modelli: uv run python scripts/download_models.py"
        )
    return ok
