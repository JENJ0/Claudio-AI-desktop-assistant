"""Entry point: python -m claudio  (or: uv run claudio)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claudio import __version__
from claudio.config import ConfigError, Settings, load_settings
from claudio.logging_setup import setup_logging

if TYPE_CHECKING:
    from claudio.router.responder import LocalResponder
    from claudio.tts.speaker import StreamingSpeaker

log = logging.getLogger("claudio")


def build_responder(settings: Settings) -> LocalResponder:
    from claudio.router.actions import ActionRunner
    from claudio.router.matcher import CommandMatcher, load_commands
    from claudio.router.responder import LocalResponder

    commands = load_commands(settings.resolve(settings.router.commands_file))
    matcher = CommandMatcher(commands, settings.router.fuzzy_threshold)
    log.info("Router: %d comandi locali", len(commands))
    return LocalResponder(matcher, ActionRunner(), settings.router.unknown_reply)


def build_synth(settings: Settings) -> tuple[Any, str]:
    """The configured voice; Piper if Kokoro cannot start, so Claudio always speaks."""
    from claudio.tts.speaker import PiperSynth, voice_path

    models = settings.resolve(settings.models_dir)
    if settings.tts.engine == "kokoro":
        try:
            from claudio.tts.kokoro import KokoroSynth

            return KokoroSynth(
                models / "kokoro", settings.tts
            ), f"Kokoro {settings.tts.kokoro_voice}"
        except Exception as exc:
            log.warning("Kokoro non disponibile (%s): uso Piper %s", exc, settings.tts.voice)
    synth = PiperSynth(voice_path(models / "piper", settings.tts.voice), settings.tts)
    return synth, f"Piper {settings.tts.voice}"


def build_speaker(settings: Settings) -> StreamingSpeaker:
    from claudio.audio.player import AudioPlayer
    from claudio.tts.speaker import StreamingSpeaker

    synth, name = build_synth(settings)
    player = AudioPlayer(settings.audio.output_sample_rate, settings.audio.output_device)
    player.start()
    speaker = StreamingSpeaker(synth, player, settings.audio.beep_volume)
    speaker.warm_up()
    log.info("TTS: %s (%d Hz)", name, synth.sample_rate)
    return speaker


async def run_text(settings: Settings, text: str) -> None:
    """Router + TTS on typed text: tests the reply path without microphone or STT."""
    responder = build_responder(settings)
    speaker = build_speaker(settings)
    start = time.perf_counter()
    first: list[float] = []
    await speaker.speak(
        responder.respond(text), on_first_audio=lambda: first.append(time.perf_counter())
    )
    if first:
        log.info("Prima parola dopo %.0f ms (router + TTS)", (first[0] - start) * 1000)
    speaker.player.close()


async def run_assistant(settings: Settings) -> None:
    from claudio.app import Assistant
    from claudio.audio.capture import AudioCapture
    from claudio.audio.devices import resolve_device
    from claudio.audio.recorder import UtteranceRecorder
    from claudio.audio.vad import Endpointer, SileroVad
    from claudio.metrics import MetricsWriter
    from claudio.stt.transcriber import WhisperTranscriber
    from claudio.wakeword.detector import OpenWakeWordDetector

    models = settings.resolve(settings.models_dir)
    t0 = time.perf_counter()
    vad = SileroVad(models / "vad" / "silero_vad.onnx", settings.audio.sample_rate)
    wake = OpenWakeWordDetector(settings.wakeword, models / "wakeword", settings.audio.block_size)
    transcriber = WhisperTranscriber(settings.stt, models / "whisper")
    log.info("STT: %s su %s", transcriber.model_name, transcriber.device.upper())
    responder = build_responder(settings)
    speaker = build_speaker(settings)
    resolve_device(settings.audio.input_device, "input")  # fail fast on a wrong name
    capture = AudioCapture(
        settings.audio.sample_rate,
        settings.audio.block_size,
        settings.audio.input_device,
        gain_db=settings.audio.input_gain_db,
    )

    logs = settings.resolve(settings.logs_dir)
    recorder = (
        UtteranceRecorder(logs / "audio", settings.logging.keep_audio, settings.audio.sample_rate)
        if settings.logging.save_audio
        else None
    )
    assistant = Assistant(
        wake=wake,
        endpointer=Endpointer(vad, settings.vad, settings.audio.sample_rate),
        transcriber=transcriber,
        responder=responder,
        speaker=speaker,
        metrics=MetricsWriter(logs / "metrics.jsonl"),
        audio_sink=recorder,
    )
    log.info(
        'Modelli caricati in %.1f s. Di\' "%s" per iniziare (Ctrl+C per uscire).',
        time.perf_counter() - t0,
        '" o "'.join(wake.names),
    )
    try:
        await assistant.run(capture.frames())
    finally:
        speaker.player.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claudio", description="Assistente vocale CLAUDIO")
    parser.add_argument("--config", type=Path, help="file TOML (default config/claudio.toml)")
    parser.add_argument("--text", help="salta microfono e STT: rispondi a questa frase")
    parser.add_argument(
        "--check", action="store_true", help="verifica configurazione, comandi e modelli, poi esci"
    )
    parser.add_argument("--version", action="version", version=f"CLAUDIO {__version__}")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"Errore di configurazione: {exc}", file=sys.stderr)
        return 2
    setup_logging(settings.resolve(settings.logs_dir), settings.logging.level)
    log.info("CLAUDIO %s", __version__)

    try:
        if args.check:
            from claudio.checks import run_checks

            return 0 if run_checks(settings) else 1
        if args.text:
            asyncio.run(run_text(settings, args.text))
        else:
            asyncio.run(run_assistant(settings))
    except KeyboardInterrupt:
        log.info("Chiusura richiesta: arrivederci")
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
