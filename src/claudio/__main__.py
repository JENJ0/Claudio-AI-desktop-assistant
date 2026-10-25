"""Entry point: python -m claudio  (or: uv run claudio)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claudio import __version__
from claudio.config import (
    DEFAULT_LANGUAGE,
    ConfigError,
    Settings,
    configured_language,
    load_settings,
)
from claudio.i18n import set_language, tr
from claudio.logging_setup import setup_logging

if TYPE_CHECKING:
    from claudio.app import BargeIn
    from claudio.brain.claude import ClaudeBrain
    from claudio.chat.launcher import ChatLauncher
    from claudio.cost.tracker import CostTracker
    from claudio.pc.tools import PcTools
    from claudio.router.responder import LocalResponder
    from claudio.system import SystemControl
    from claudio.tts.jack import JackSynth
    from claudio.tts.speaker import StreamingSpeaker

log = logging.getLogger("claudio")

CHAT_DB = "claudio.db"  # in data_dir: every conversation, shown by `claudio --ui`


def build_tracker(settings: Settings) -> CostTracker:
    from claudio.cost.pricing import load_pricing
    from claudio.cost.tracker import open_tracker

    cost = settings.cost
    return open_tracker(
        load_pricing(settings.resolve(cost.pricing_file)),
        settings.resolve(cost.db_file),
        cost.daily_budget,
        cost.warn_at,
    )


def build_responder(
    settings: Settings,
    system: SystemControl | None = None,
    chat: ChatLauncher | None = None,
    tracker: CostTracker | None = None,
) -> LocalResponder:
    """`tracker`: open it on the thread that runs the requests (its SQLite connection only
    works there) when this runs in a loader thread."""
    from claudio.router.actions import ActionRunner
    from claudio.router.matcher import CommandMatcher, load_commands
    from claudio.router.responder import LocalResponder

    commands = load_commands(settings.resolve(settings.router.commands_file))
    matcher = CommandMatcher(commands, settings.router.fuzzy_threshold)
    log.info("Router: %d comandi locali (%s)", len(commands), ", ".join(matcher.languages))
    if tracker is None:
        tracker = build_tracker(settings)
    brain = build_brain(settings, tracker)
    return LocalResponder(
        matcher,
        ActionRunner(system=system, tracker=tracker, chat=chat),
        settings.router.unknown_reply,
        settings.router.error_reply,
        fallback=brain.ask if brain else None,
        fallback_info=(lambda: brain.last_call) if brain else None,
        on_prepare=brain.warm_up if brain else None,
        fallback_awaiting=(lambda: brain.awaiting) if brain else None,
        fallback_cancel=brain.cancel_pending if brain else None,
        fallback_heard=brain.heard if brain else None,
    )


def build_pc_tools(settings: Settings) -> PcTools | None:
    """The tools Claude acts on the PC with: Windows only, and only with [pc] enabled."""
    if not settings.pc.enabled:
        log.info("Azioni sul PC spente ([pc] enabled = false)")
        return None
    if sys.platform != "win32":
        log.info("Azioni sul PC: solo su Windows")
        return None
    from claudio.pc.tools import PcTools
    from claudio.pc.windows import WindowsDesktop

    tools = PcTools(WindowsDesktop(), settings.pc)
    log.info(
        "Azioni sul PC: %d strumenti, cartelle %s, %d tasti senza conferma",
        len(tools.definitions()),
        ", ".join(settings.pc.roots),
        len(tools.safe_keys),
    )
    return tools


def build_brain(settings: Settings, tracker: CostTracker) -> ClaudeBrain | None:
    """Claude for what no command understands; None when it is switched off. A missing key
    still builds it: Claudio then says where the key goes instead of "I can't do that"."""
    from claudio.brain.claude import ClaudeBrain, make_client
    from claudio.brain.personality import load_personality
    from claudio.cost.pricing import PricingError

    claude = settings.claude
    if not claude.enabled:
        log.info("Claude spento ([claude] enabled = false): solo comandi locali")
        return None
    try:
        tracker.pricing.price(claude.model)  # no prices, no calls: the spend must be counted
    except PricingError as exc:
        log.error("Claude spento: %s", exc)
        return None
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    if not key.strip():
        log.warning("ANTHROPIC_API_KEY mancante nel file .env: Claude non risponderà")
    client = make_client(key.strip(), claude.timeout_s, claude.keepalive_s) if key.strip() else None
    personality = load_personality(
        settings.resolve(claude.personality_file), claude.personality_max_chars
    )
    brain = ClaudeBrain(
        client,
        tracker,
        claude,
        tools=build_pc_tools(settings),
        pc=settings.pc,
        personality=personality,
    )
    if brain.smart_model is not None:
        try:
            tracker.pricing.price(brain.smart_model)
        except PricingError as exc:
            log.error("Sonnet spento, solo Haiku: %s", exc)
            brain.smart_model = None
    log.info(
        "Claude: %s (ragionamenti: %s), max %d token, cache %s, budget %.2f al giorno, "
        "personalità %d caratteri",
        claude.model,
        brain.smart_model or "-",
        claude.max_tokens,
        "on" if claude.cache else "off",
        tracker.daily_budget,
        len(personality),
    )
    return brain


def make_jack(settings: Settings) -> JackSynth:
    """Jack's worker, not started yet; its fallback voice is set by build_synth()."""
    from claudio.tts.jack import JackSynth, worker_command

    tts = settings.tts
    models = settings.resolve(settings.models_dir)
    return JackSynth(
        worker_command(models, tts),
        None,
        tts.jack_startup_timeout_s,
        tts.jack_sentence_timeout_s,
        rvc=tts.jack_rvc,
        words=tts.jack_words,
        first_wait_s=tts.jack_first_wait_s,
    )


def build_synth(settings: Settings, jack: JackSynth | None = None) -> tuple[Any, str]:
    """The configured voice; Piper if Kokoro cannot start, so Claudio always speaks.
    With engine "jack" that voice is the fallback of Jack's worker. `jack` is a worker
    already starting in the background (run_assistant); without it one is started here and
    waited for (--text)."""
    if settings.tts.engine != "jack":
        return build_base_synth(settings)
    from claudio.tts.jack import WorkerError

    fallback, fallback_name = build_base_synth(settings)
    if jack is None:
        jack = make_jack(settings)
        try:
            jack.start()
        except WorkerError as exc:
            log.warning("Jack non disponibile (%s): uso %s", exc, fallback_name)
            return fallback, fallback_name
    jack.fallback = fallback
    return jack, f"Jack (fallback {fallback_name})"


def build_base_synth(settings: Settings) -> tuple[Any, str]:
    from claudio.tts.speaker import PiperSynth, voice_path

    models = settings.resolve(settings.models_dir)
    if settings.tts.engine in ("kokoro", "jack"):
        try:
            from claudio.tts.kokoro import KokoroSynth

            return KokoroSynth(
                models / "kokoro", settings.tts
            ), f"Kokoro {settings.tts.kokoro_voice}"
        except Exception as exc:
            log.warning("Kokoro non disponibile (%s): uso Piper %s", exc, settings.tts.voice)
    synth = PiperSynth(voice_path(models / "piper", settings.tts.voice), settings.tts)
    return synth, f"Piper {settings.tts.voice}"


def build_voices(settings: Settings, synth: Any) -> dict[str, tuple[Any, str]]:
    """A voice for every other language Whisper listens for, same engine as `synth`. A voice
    that cannot load is skipped: that language is read by the main voice."""
    from claudio.tts.kokoro import KokoroSynth
    from claudio.tts.speaker import PiperSynth, voice_path

    piper_dir = settings.resolve(settings.models_dir) / "piper"
    voices: dict[str, tuple[Any, str]] = {}
    for language in settings.stt.languages:
        if language == DEFAULT_LANGUAGE:
            continue
        try:
            base = getattr(synth, "fallback", synth)  # Jack speaks Italian only
            if isinstance(base, KokoroSynth):
                other = base.for_language(language)
                voices[language] = other, f"Kokoro {other.voice}"
            else:
                name = settings.tts.piper_voice_for(language)
                voices[language] = PiperSynth(voice_path(piper_dir, name), settings.tts), name
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Voce per '%s' non disponibile (%s): uso quella principale", language, exc)
    return voices


def build_speaker(
    settings: Settings, jack: JackSynth | None = None, start_player: bool = True
) -> StreamingSpeaker:
    """`start_player` False: the caller opens the audio output (run_assistant does it on the
    main thread, as this runs in a loader thread there)."""
    from claudio.audio.player import AudioPlayer
    from claudio.tts.effects import VoiceEffect
    from claudio.tts.speaker import StreamingSpeaker

    synth, name = build_synth(settings, jack)
    voices = build_voices(settings, synth)
    player = AudioPlayer(settings.audio.output_sample_rate, settings.audio.output_device)
    if start_player:
        player.start()
    effect = VoiceEffect(settings.tts.effect) if settings.tts.effect != "none" else None
    speaker = StreamingSpeaker(
        synth,
        player,
        settings.audio.beep_volume,
        effect,
        {language: voice for language, (voice, _) in voices.items()},
        first_clause_chars=settings.tts.first_clause_chars,
    )
    speaker.warm_up()
    others = "".join(f", {lang}: {label}" for lang, (_, label) in voices.items())
    log.info(
        "TTS: %s (%d Hz)%s%s",
        name,
        synth.sample_rate,
        others,
        tr(", effetto {effect}", effect=settings.tts.effect) if effect else "",
    )
    return speaker


def startup_notices(settings: Settings, jack_running: bool, stt_device: str) -> list[str]:
    """What Claudio says out loud at startup when it runs degraded: a warning in the console
    alone went unnoticed (on 23/10 Jack's worker failed and Kokoro answered all evening)."""
    notices = []
    if settings.tts.engine == "jack" and not jack_running:
        notices.append(tr("La voce di Jack non è partita: uso quella di riserva."))
    if settings.stt.device == "auto" and stt_device == "cpu":
        notices.append(tr("La scheda video non è disponibile: ti capisco più lentamente."))
    if notices:
        notices.append(tr("Il motivo è nel log."))
    return notices


def load_all(tasks: dict[str, Callable[[], Any]], parallel: bool = True) -> dict[str, Any]:
    """Run the loaders, at the same time when `parallel` (threads: the models load in C code
    that releases the GIL, so they overlap), and log how long each took. An error is raised
    once every loader has ended, so none is left half way."""
    took: dict[str, float] = {}

    def timed(name: str, load: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        try:
            return load()
        finally:
            took[name] = time.perf_counter() - start

    if parallel:
        with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="load") as pool:
            futures = {name: pool.submit(timed, name, load) for name, load in tasks.items()}
            wait(futures.values())
        results = {name: future.result() for name, future in futures.items()}
    else:
        results = {name: timed(name, load) for name, load in tasks.items()}
    log.info(
        "Avvio: %s (in parallelo: %s)",
        ", ".join(f"{name} {took[name]:.1f} s" for name in tasks),
        "sì" if parallel else "no",
    )
    return results


async def run_text(settings: Settings, text: str) -> None:
    """Router + TTS on typed text: tests the reply path without microphone or STT."""
    responder = build_responder(settings)
    speaker = build_speaker(settings)
    responder.voiced = {DEFAULT_LANGUAGE, *speaker.voices}
    start = time.perf_counter()
    first: list[float] = []
    await speaker.speak(
        responder.respond(text, settings.stt.languages[0]),  # as if Whisper heard it
        on_first_audio=lambda: first.append(time.perf_counter()),
        language=lambda: responder.language,
    )
    if first:
        log.info("Prima parola dopo %.0f ms (router + TTS)", (first[0] - start) * 1000)
    speaker.player.close()


def build_vocabulary(settings: Settings) -> Any:
    """The folder and program names for Whisper's prompt, read now in a thread."""
    from claudio.stt.vocabulary import Vocabulary, default_folders, default_shortcut_folders

    folders = default_folders(Path.home())
    vocabulary = Vocabulary(
        lambda: folders,
        lambda: default_shortcut_folders(folders[0]),
        words=settings.stt.vocabulary_words,
        max_folders=settings.stt.vocabulary_folders,
        max_programs=settings.stt.vocabulary_programs,
    )
    vocabulary.refresh()
    return vocabulary


def build_barge_in(settings: Settings, transcriber: Any) -> BargeIn:
    """The listener used while CLAUDIO speaks: a second Silero VAD (its own state) that
    closes a phrase after a short pause, or every barge_in_max_s of continuous speech."""
    from claudio.app import BargeIn
    from claudio.audio.vad import Endpointer, SileroVad

    vad = SileroVad(
        settings.resolve(settings.models_dir) / "vad" / "silero_vad.onnx",
        settings.audio.sample_rate,
    )
    c = settings.conversation
    vad_settings = settings.vad.model_copy(
        update={
            "threshold": c.barge_in_vad,  # stricter than for a request: the reply is playing
            "min_speech_ms": c.barge_in_min_speech_ms,
            "end_silence_ms": 300,
            "long_end_silence_ms": 0,
            "max_utterance_s": c.barge_in_max_s,
        }
    )
    return BargeIn(
        Endpointer(vad, vad_settings, settings.audio.sample_rate),
        lambda audio: transcriber.transcribe(audio, quick=True),
        mode=c.barge_in_mode,
        min_dbfs=c.barge_in_min_dbfs,
        duck_volume=c.duck_volume,
        resume_s=c.resume_s,
    )


async def run_assistant(
    settings: Settings, session: Path, chat_window: ChatLauncher | None = None
) -> None:
    from claudio.app import Assistant
    from claudio.audio.capture import AudioCapture
    from claudio.audio.devices import resolve_device
    from claudio.audio.level import AutoGain
    from claudio.audio.recorder import UtteranceRecorder
    from claudio.audio.vad import Endpointer, SileroVad
    from claudio.chat.store import ChatStore
    from claudio.metrics import MetricsWriter
    from claudio.session_log import ConversationLog, purge_old_logs
    from claudio.stt.transcriber import WhisperTranscriber
    from claudio.system import SystemControl
    from claudio.wakeword.detector import OpenWakeWordDetector

    models = settings.resolve(settings.models_dir)
    t0 = time.perf_counter()
    # Jack's worker is a process of its own: it loads (8-30 s) while everything else does,
    # and Claudio listens without waiting for it (a reply meanwhile waits jack_first_wait_s).
    jack = make_jack(settings) if settings.tts.engine == "jack" else None
    jack_error: list[Exception | None] = []
    if jack is not None:
        jack.start_in_background(jack_error.append)
    system = SystemControl()
    tracker = build_tracker(settings)  # SQLite: opened on the thread that uses it
    loaded = load_all(
        {
            "whisper": lambda: WhisperTranscriber(settings.stt, models / "whisper"),
            "wake word": lambda: (
                SileroVad(models / "vad" / "silero_vad.onnx", settings.audio.sample_rate),
                OpenWakeWordDetector(
                    settings.wakeword, models / "wakeword", settings.audio.block_size
                ),
            ),
            "voce": lambda: build_speaker(settings, jack, start_player=False),
            "claude": lambda: build_responder(settings, system, chat_window, tracker),
        },
        parallel=settings.startup.parallel,
    )
    transcriber: WhisperTranscriber = loaded["whisper"]
    if settings.stt.vocabulary:
        transcriber.vocabulary = build_vocabulary(settings).text
    vad, wake = loaded["wake word"]
    speaker: StreamingSpeaker = loaded["voce"]
    responder: LocalResponder = loaded["claude"]
    speaker.player.start()
    log.info(
        "STT: %s su %s, lingue %s",
        transcriber.model_name,
        transcriber.device.upper(),
        ", ".join(settings.stt.languages),
    )
    responder.voiced = {DEFAULT_LANGUAGE, *speaker.voices}  # no English text in Italian voice
    resolve_device(settings.audio.input_device, "input")  # fail fast on a wrong name
    capture = AudioCapture(
        settings.audio.sample_rate,
        settings.audio.block_size,
        settings.audio.input_device,
        gain_db=settings.audio.input_gain_db,
    )
    level = (
        AutoGain(
            settings.audio.input_gain_db,
            capture.set_gain_db,
            settings.resolve(settings.data_dir) / "audio_level.json",
            settings.audio.input_device,
        )
        if settings.audio.auto_gain
        else None
    )

    logs = settings.resolve(settings.logs_dir)
    chat = ChatStore(settings.resolve(settings.data_dir) / CHAT_DB, session.name)
    if purged := chat.purge_older_than(settings.logging.retention_days):
        log.info(
            "Chat: %d messaggi più vecchi di %d giorni eliminati",
            purged,
            settings.logging.retention_days,
        )
    if purged := purge_old_logs(logs, settings.logging.retention_days, current=session):
        log.info(
            "Log: %d file più vecchi di %d giorni eliminati",
            purged,
            settings.logging.retention_days,
        )
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
        metrics=MetricsWriter(logs / "metrics.jsonl", ConversationLog(session), chat),
        audio_sink=recorder,
        level=level,
        system=system,
        second_chance=(
            (lambda audio: transcriber.transcribe(audio, quick=True))
            if settings.wakeword.second_chance
            else None
        ),
        retry_language=(
            settings.stt.languages[0]
            if settings.stt.retry_in_first_language and len(settings.stt.languages) > 1
            else None
        ),
        conversation_s=settings.conversation.window_s,
        barge_in=build_barge_in(settings, transcriber) if settings.conversation.barge_in else None,
        verify_wake=settings.wakeword.verify,
        closed_list=(
            (lambda audio, language: transcriber.candidates(audio, language=language))
            if settings.stt.closed_list
            else None
        ),
        closed_list_max_words=settings.stt.closed_list_max_words,
        claude_min_words=settings.claude.conversation_min_words if responder.fallback else None,
        only_addressed=settings.conversation.only_addressed,
        noise=settings.noise,
        confirm_below=settings.wakeword.confirm_below,
        always_confirm=wake.confirmed,
        stt_segments=lambda: transcriber.last_segments,
        unsure_logprob=settings.stt.unsure_logprob if responder.fallback else None,
        alternatives=settings.stt.alternatives if settings.stt.closed_list else 0,
        on_heard=responder.heard,
        continue_s=settings.vad.continue_s,
    )
    log.info(
        'Modelli caricati in %.1f s. Di\' "%s" per iniziare (Ctrl+C per uscire).',
        time.perf_counter() - t0,
        tr('" o "').join(wake.names),
    )
    inbox = (
        asyncio.create_task(serve_inbox(assistant, chat, settings)) if settings.chat.typed else None
    )
    ready = asyncio.create_task(
        announce_ready(assistant, settings, jack, jack_error, transcriber.device, t0)
    )
    try:
        await assistant.run(capture.frames())
    finally:
        ready.cancel()
        if inbox is not None:
            inbox.cancel()
        speaker.player.close()
        if chat_window is not None:
            chat_window.close()  # "spegniti" or Ctrl+C: the window goes with Claudio


async def announce_ready(
    assistant: Any,
    settings: Settings,
    jack: JackSynth | None,
    jack_error: list[Exception | None],
    stt_device: str,
    t0: float,
) -> None:
    """Once Jack's worker has loaded too (or failed), say so: the notices of a degraded start
    and a short "ready", in Jack's voice when it is there. Claudio already listens by then."""
    if jack is not None:
        await asyncio.to_thread(jack.wait_started)
    jack_running = jack is not None and not (jack_error and jack_error[0])
    if jack is not None and not jack_running:
        log.warning(
            "Jack non disponibile (%s): uso %s", jack_error[0], type(jack.fallback).__name__
        )
    log.info("Avvio completo in %.1f s", time.perf_counter() - t0)
    notices = startup_notices(settings, jack_running, stt_device)
    for notice in notices:
        log.warning("Avviso a voce: %s", notice)
    if settings.startup.announce_ready:
        notices.append(tr("Sono pronto."))
    if notices:
        await assistant.announce(" ".join(notices))


async def serve_inbox(assistant: Any, chat: Any, settings: Settings) -> None:
    """Answer what is written in the chat window (`claudio --ui`), one message at a time.
    Typed text has no detected language: the first configured one is used."""
    language = settings.stt.languages[0] if settings.stt.languages else DEFAULT_LANGUAGE
    while True:
        try:
            pending = await asyncio.to_thread(chat.take_pending, settings.chat.max_age_s)
            for _, text in pending:
                await assistant.handle_typed(text, language, speak=settings.chat.speak_typed)
        except Exception:  # a locked or broken database: try again at the next poll
            log.exception("Messaggi della finestra non letti")
        await asyncio.sleep(settings.chat.poll_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claudio", description="Assistente vocale CLAUDIO")
    parser.add_argument("--config", type=Path, help="file TOML (default config/claudio.toml)")
    parser.add_argument("--text", help="salta microfono e STT: rispondi a questa frase")
    parser.add_argument(
        "--check", action="store_true", help="verifica configurazione, comandi e modelli, poi esci"
    )
    parser.add_argument("--ui", action="store_true", help="apri la finestra con le conversazioni")
    parser.add_argument("--version", action="version", version=f"CLAUDIO {__version__}")
    args = parser.parse_args(argv)

    with contextlib.suppress(ValueError):  # so that a configuration error is in it too
        set_language(configured_language(args.config))
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(tr("Errore di configurazione: {error}", error=exc), file=sys.stderr)
        return 2
    set_language(settings.language)
    setup_logging(
        settings.resolve(settings.logs_dir),
        settings.logging.level,
        "chat-window.log.jsonl" if args.ui else "claudio.log.jsonl",
    )
    if args.ui:
        from claudio.chat.window import open_window

        open_window(settings.resolve(settings.data_dir) / CHAT_DB)
        return 0
    session = None
    if not args.check and not args.text:
        from claudio.session_log import start_session

        session = start_session(settings.resolve(settings.logs_dir), settings.logging.keep_sessions)
    log.info("CLAUDIO %s", __version__)
    if session is not None:
        log.info("Sessione salvata in %s", session)

    try:
        if args.check:
            from claudio.checks import run_checks

            return 0 if run_checks(settings) else 1
        if args.text:
            asyncio.run(run_text(settings, args.text))
        else:
            from claudio.chat.launcher import ChatLauncher, window_command

            assert session is not None
            chat_window = ChatLauncher(window_command(args.config))
            asyncio.run(run_assistant(settings, session, chat_window))
    except KeyboardInterrupt:
        log.info("Chiusura richiesta: arrivederci")
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
