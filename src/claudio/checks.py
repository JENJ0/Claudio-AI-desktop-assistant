"""`python -m claudio --check`: fast diagnostics without starting the assistant."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path

from claudio.config import Settings, SttSettings

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


# What faster-whisper loads from the model folder; without tokenizer.json it would go online.
WHISPER_FILES = ("model.bin", "config.json", "tokenizer.json")
WHISPER_VOCABULARIES = ("vocabulary.json", "vocabulary.txt")


def whisper_models(stt: SttSettings, gpus: int) -> tuple[str, str | None]:
    """(the Whisper model CLAUDIO will load, the one it falls back to or None), as
    WhisperTranscriber chooses them: "auto" takes `model` on the GPU and `fallback_model` on
    the CPU, "cuda" never falls back."""
    if stt.device == "cpu":
        return stt.fallback_model, None
    if stt.device == "cuda":
        return stt.model, None
    first, other = (stt.model, stt.fallback_model) if gpus else (stt.fallback_model, stt.model)
    return first, (other if other != first else None)


def find_whisper_model(name: str, root: Path) -> Path:
    """The folder of the downloaded Whisper model `name` (a size such as "large-v3", a
    Hugging Face id or a local folder), found as WhisperModel does but without downloading.
    FileNotFoundError when it is missing or half downloaded, ValueError for an unknown name."""
    folder = Path(name)
    if not folder.is_dir():
        from faster_whisper.utils import download_model

        try:
            folder = Path(download_model(name, local_files_only=True, cache_dir=str(root)))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"modello Whisper {name} non scaricato in {root}") from exc
    missing = [f for f in WHISPER_FILES if not (folder / f).is_file()]
    if not any((folder / f).is_file() for f in WHISPER_VOCABULARIES):
        missing.append(WHISPER_VOCABULARIES[0])
    if missing:
        raise FileNotFoundError(
            f"modello Whisper {name} incompleto in {folder}: mancano {', '.join(missing)}"
        )
    return folder


def check_whisper(settings: Settings, gpus: int, report: Callable[[bool, str], None]) -> None:
    """The model Whisper will load must be on disk (an error otherwise: CLAUDIO would try to
    download ~3 GB at startup); the fallback one only gets a warning."""
    root = settings.resolve(settings.models_dir) / "whisper"
    if settings.stt.device == "cuda" and not gpus:  # "cuda" never falls back to the CPU
        report(False, '[stt] device = "cuda" senza GPU CUDA: Whisper non partirà')
    needed, other = whisper_models(settings.stt, gpus)
    try:
        folder = find_whisper_model(needed, root)
        size = (folder / "model.bin").stat().st_size / 1e9
        report(True, f"modello Whisper {needed} ({size:.1f} GB)")
    # OSError: missing, or a cache it cannot read; ImportError: faster_whisper is broken (its
    # own ERR line is above), and the rest of the report must still be printed.
    except (OSError, ValueError, ImportError) as exc:
        report(False, f"{exc}. Scaricalo con: {download_hint(needed)}")
    if other is None:
        return
    try:
        find_whisper_model(other, root)
    except (OSError, ValueError, ImportError) as exc:
        when = "senza GPU" if other == settings.stt.fallback_model else "con la GPU"
        log.warning("WARN %s: serve solo %s (%s)", exc, when, download_hint(other))


def download_hint(name: str) -> str:
    return f"uv run python scripts/download_models.py --whisper {name}"


def check_packages(report: Callable[[bool, str], None]) -> None:
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            report(True, f"{name} {getattr(mod, '__version__', '')}".strip())
        except Exception as exc:
            report(False, f"{name}: {exc}")


def check_commands(settings: Settings, report: Callable[[bool, str], None]) -> None:
    try:
        from claudio.router.matcher import CommandMatcher, load_commands

        commands = load_commands(settings.resolve(settings.router.commands_file))
        matcher = CommandMatcher(commands, settings.router.fuzzy_threshold)
        languages = ", ".join(matcher.languages)
        what = f"{len(commands)} comandi in {settings.router.commands_file} ({languages})"
        report(True, what)
    except Exception as exc:
        report(False, f"comandi: {exc}")


def check_model_files(
    settings: Settings, models: Path, report: Callable[[bool, str], None]
) -> None:
    from claudio.wakeword.detector import FEATURE_MODELS

    wanted = [
        models / "vad" / "silero_vad.onnx",
        models / "piper" / f"{settings.tts.voice}.onnx",
        models / "piper" / f"{settings.tts.voice}.onnx.json",
    ]
    if settings.tts.engine in ("kokoro", "jack"):  # Jack's fallback is Kokoro
        from claudio.tts.kokoro import KOKORO_FILES

        wanted += [models / "kokoro" / name for name in KOKORO_FILES]
    wanted += [models / "wakeword" / f for f in FEATURE_MODELS]
    for path in wanted:
        report(path.exists(), f"modello {path.relative_to(models)}")


def check_jack(settings: Settings, models: Path, report: Callable[[bool, str], None]) -> None:
    """Optional: without it Kokoro speaks (a warning, not an error)."""
    if settings.tts.engine != "jack":
        return
    tts = settings.tts
    for rel in (tts.jack_python, tts.jack_checkpoint, tts.jack_rvc_model, tts.jack_applio):
        if (models / rel).exists():
            report(True, f"modello {rel}")
        else:
            log.warning("WARN voce di Jack non installata (%s): parlerà Kokoro", rel)


def check_voices(settings: Settings, models: Path) -> None:
    """The voice of each language Whisper may hear: a missing one only gets a warning."""
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


def check_wake_words(settings: Settings, models: Path, report: Callable[[bool, str], None]) -> None:
    from claudio.wakeword.detector import find_wakeword_model

    for wake in settings.wakeword.models:
        try:
            path = find_wakeword_model(wake.name, models / "wakeword")
            report(
                True,
                f"wake word {path.name} (soglia {wake.threshold})",
            )
        except FileNotFoundError as exc:
            report(False, str(exc))


def check_gpu(settings: Settings, report: Callable[[bool, str], None]) -> int:
    """The number of CUDA GPUs found, 0 when there is none or counting fails (a warning)."""
    gpus = 0
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
    return gpus


def check_audio(settings: Settings, report: Callable[[bool, str], None]) -> None:
    from claudio.audio.devices import (
        WindowsDefaultOutput,
        describe_device,
        pick_output,
        resolve_device,
    )

    for kind, spec, label in (
        ("input", settings.audio.input_device, "microfono"),
        ("output", settings.audio.output_device, "uscita audio"),
    ):
        try:
            if kind == "output":
                index = pick_output(spec, WindowsDefaultOutput()() if spec == "" else None)
            else:
                index = resolve_device(spec, kind)
            follows = (
                " (segue l'uscita predefinita di Windows)"
                if kind == "output" and spec == ""
                else ""
            )
            report(True, f"{label}: {describe_device(index, kind)}{follows}")
        except ValueError as exc:
            report(False, f"{label}: {exc}")
        except Exception as exc:  # no audio hardware at all (e.g. CI)
            log.warning("WARN %s: %s", label, exc)


def run_checks(settings: Settings) -> bool:
    ok = True

    def report(good: bool, what: str) -> None:
        nonlocal ok
        ok &= good
        (log.info if good else log.error)("%s %s", "OK  " if good else "ERR ", what)

    check_packages(report)
    check_commands(settings, report)
    models = settings.resolve(settings.models_dir)
    check_model_files(settings, models, report)
    check_jack(settings, models, report)
    check_voices(settings, models)
    check_wake_words(settings, models, report)
    gpus = check_gpu(settings, report)
    check_whisper(settings, gpus, report)
    check_audio(settings, report)

    if not ok:
        log.error(
            "Qualcosa manca: vedi le righe ERR. Modelli: uv run python scripts/download_models.py"
        )
    return ok
