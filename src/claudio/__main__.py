"""Entry point: python -m claudio  (or: uv run claudio)."""

from __future__ import annotations

import argparse
import asyncio
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
    load_settings,
)
from claudio.gpu import GpuMemory, GpuSampler, has_room, read_gpu
from claudio.logging_setup import setup_logging
from claudio.session_log import drop_old_lines

if TYPE_CHECKING:
    from claudio.app import Assistant, BargeIn
    from claudio.audio.level import AutoGain
    from claudio.audio.recorder import UtteranceRecorder
    from claudio.audio.vad import SileroVad
    from claudio.brain.claude import ClaudeBrain
    from claudio.chat.launcher import ChatLauncher
    from claudio.cost.tracker import CostTracker
    from claudio.memory.store import MemoryStore
    from claudio.memory.tools import MemoryTools
    from claudio.metrics import MetricsWriter
    from claudio.pc.tools import Desktop, PcTools
    from claudio.protocols import Speaker
    from claudio.router.actions import ActionContext
    from claudio.router.command_tool import CommandTool
    from claudio.router.matcher import CommandMatcher
    from claudio.router.responder import LocalResponder
    from claudio.stt.recovery import CommandRecovery, Rescorer
    from claudio.stt.transcriber import WhisperTranscriber
    from claudio.system import SystemControl
    from claudio.tts.jack import JackSynth
    from claudio.tts.speaker import StreamingSpeaker
    from claudio.wakeword.detector import OpenWakeWordDetector

log = logging.getLogger("claudio")

CHAT_DB = "claudio.db"  # in data_dir: every conversation, shown by `claudio --ui`
BARGE_LEAD_MS = 250  # kept before a voice heard over the reply: its soft first sound
CLOSE_POLL_S = 0.5  # how often close_when_idle looks for "chiuditi"
CLOSE_GRACE_S = 1.0  # a microphone sends a frame every 80 ms: this long means none is coming


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


def build_memory(
    settings: Settings, clock: Callable[[], float] | None = None
) -> MemoryStore | None:
    """The long-term memory; None when it is off or its database cannot be opened (Claudio
    then works as before, without memories). `clock`: the store's wall clock (the replay gate
    fixes it: a memory's date reaches Claude's prompt)."""
    import sqlite3

    from claudio.memory.store import MemoryStore

    memory = settings.memory
    if not memory.enabled:
        log.info("Memoria spenta ([memory] enabled = false)")
        return None
    path = settings.resolve(memory.db_file)
    try:
        store = MemoryStore(
            path, memory.max_items, memory.max_chars, **({} if clock is None else {"clock": clock})
        )
    except (OSError, sqlite3.Error) as exc:
        log.error("Memoria non disponibile (%s): %s", path, exc)
        return None
    log.info("Memoria: %d ricordi in %s", len(store), path)
    return store


def build_responder(
    settings: Settings,
    ctx: ActionContext,
    system: SystemControl | None = None,
    chat: ChatLauncher | None = None,
    tracker: CostTracker | None = None,
    client: Any | None = None,
    desktop: Desktop | None = None,
    home: Path | None = None,
    memory_clock: Callable[[], float] | None = None,
    clock: Callable[[], float] | None = None,
) -> LocalResponder:
    """`tracker`: open it on the thread that runs the requests (its SQLite connection only
    works there) when this runs in a loader thread. `client`, `desktop`, `home` and
    `memory_clock` are the replay gate's edges (a cassette client, a fake desktop, a fake home,
    a fixed clock); `clock` is the brain's own clock; None: as in production."""
    from claudio.memory.voice import MemoryVoice
    from claudio.router.actions import ActionRunner
    from claudio.router.command_tool import CommandTool
    from claudio.router.matcher import CommandMatcher, load_commands
    from claudio.router.responder import LocalResponder

    commands = load_commands(settings.resolve(settings.router.commands_file))
    matcher = CommandMatcher(commands, settings.router.fuzzy_threshold)
    log.info("Router: %d comandi locali (%s)", len(commands), ", ".join(matcher.languages))
    if tracker is None:
        tracker = build_tracker(settings)
    store = build_memory(settings, memory_clock)
    memory = MemoryVoice(store) if store is not None else None
    actions = ActionRunner(ctx, system=system, tracker=tracker, chat=chat, memory=memory)
    # Claude runs the commands it recognises in garbled words through the same actions.
    own_commands = CommandTool(commands, actions, matcher.threshold)
    brain = build_brain(
        settings,
        tracker,
        chat,
        own_commands,
        store,
        client=client,
        desktop=desktop,
        home=home,
        clock=clock,
    )
    if brain is not None:  # off or paused, Claude asks no question
        brain.silenced = lambda: actions.system.off or actions.system.paused
    guesser = None
    if brain is not None and settings.guess.enabled:
        from claudio.brain.guess import Guesser
        from claudio.cost.pricing import PricingError

        try:
            tracker.pricing.price(settings.guess.model)  # no price: the spend would not count
        except PricingError as exc:
            log.error("Ipotesi di Claude spente: %s", exc)
        else:
            # Same client as the replies, its own cheap model ([guess] model): ~0.0001 $ a call.
            guesser = Guesser(brain.client, tracker, settings.claude, settings.guess, commands)
    return LocalResponder(
        matcher,
        actions,
        settings.router.unknown_reply,
        settings.router.error_reply,
        fallback=brain.ask if brain else None,
        fallback_info=(lambda: brain.last_call) if brain else None,
        on_prepare=brain.warm_up if brain else None,
        fallback_awaiting=(lambda: brain.awaiting) if brain else None,
        fallback_cancel=brain.cancel_pending if brain else None,
        fallback_repeats=brain.repeats_waiting if brain else None,
        fallback_stop=brain.stop_acting if brain else None,
        fallback_note=brain.note_local if brain else None,
        fallback_cut=brain.reply_cut if brain else None,
        fallback_dropped=brain.dropped_action if brain else None,
        fallback_heard=brain.heard if brain else None,
        fallback_guess=guesser,
        fallback_guess_info=(lambda: guesser.last_call) if guesser else None,
        guess=settings.guess,
        memory=memory,
    )


def build_pc_tools(
    settings: Settings,
    commands: CommandTool | None = None,
    memory: MemoryTools | None = None,
    desktop: Desktop | None = None,
    home: Path | None = None,
) -> PcTools | None:
    """The tools Claude acts on the PC with: Windows only, and only with [pc] enabled.
    `commands`: Claudio's own commands (volume, chat, pause...) as one more tool; `memory`:
    "remember" and "forget". `desktop` and `home`: a fake desktop and home (the replay gate);
    None: Windows and the user's folder."""
    if not settings.pc.enabled:
        log.info("Azioni sul PC spente ([pc] enabled = false)")
        return None
    if sys.platform != "win32":
        log.info("Azioni sul PC: solo su Windows")
        return None
    from claudio.pc.tools import PcTools

    if desktop is None:
        from claudio.pc.windows import WindowsDesktop

        desktop = WindowsDesktop()
    tools = PcTools(desktop, settings.pc, home=home, commands=commands, memory=memory)
    log.info(
        "Azioni sul PC: %d strumenti, cartelle %s, %d tasti senza conferma",
        len(tools.definitions()),
        ", ".join(settings.pc.roots),
        len(tools.safe_keys),
    )
    return tools


def build_brain(
    settings: Settings,
    tracker: CostTracker,
    chat: ChatLauncher | None = None,
    commands: CommandTool | None = None,
    memory: MemoryStore | None = None,
    client: Any | None = None,
    desktop: Desktop | None = None,
    home: Path | None = None,
    clock: Callable[[], float] | None = None,
) -> ClaudeBrain | None:
    """Claude for what no command understands; None when it is switched off. A missing key
    still builds it: Claudio then says where the key goes instead of "I can't do that".
    With a `memory`, Claude sees the memories and may save and forget them. `client`,
    `desktop`, `home`, `clock`: the replay gate's edges (None: the real ones)."""
    from claudio.brain.claude import ClaudeBrain, make_client, reply_cap
    from claudio.brain.personality import load_personality
    from claudio.cost.pricing import PricingError
    from claudio.memory.prompt import MemoryPrompt
    from claudio.memory.tools import MemoryTools

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
    if not key.strip() and client is None:
        log.warning("ANTHROPIC_API_KEY mancante nel file .env: Claude non risponderà")
    if client is None and key.strip():
        client = make_client(key.strip(), claude.timeout_s, claude.keepalive_s)
    personality = load_personality(
        settings.resolve(claude.personality_file), claude.personality_max_chars
    )
    remember = settings.memory
    brain = ClaudeBrain(
        client,
        tracker,
        claude,
        tools=build_pc_tools(
            settings,
            commands,
            MemoryTools(memory, remember.learn) if memory is not None and remember.tools else None,
            desktop=desktop,
            home=home,
        ),
        pc=settings.pc,
        clock=clock if clock is not None else time.monotonic,
        personality=personality,
        on_show=chat.open if chat is not None else None,  # a command is read in the chat
        memory=(
            MemoryPrompt(memory, remember.block_chars, remember.relevant)
            if memory is not None
            else None
        ),
    )
    if brain.smart_model is not None:
        try:
            tracker.pricing.price(brain.smart_model)
        except PricingError as exc:
            log.error("Sonnet spento, solo Haiku: %s", exc)
            brain.smart_model = None
    second = brain.smart_model if brain.smart_model not in (None, claude.model) else None
    log.info(
        "Claude: %s (ragionamenti: %s), max %d token (con strumenti %d), thinking %s, "
        "cache %s, budget %.2f al giorno, personalità %d caratteri",
        claude.model,
        second or "-",
        reply_cap(claude, settings.pc, with_tools=False),
        reply_cap(claude, settings.pc, with_tools=True),
        "on" if claude.thinking else "off",
        "on" if claude.cache else "off",
        tracker.daily_budget,
        len(personality),
    )
    return brain


def make_jack(settings: Settings, session: Path | None = None) -> JackSynth:
    """Jack's worker, not started yet; its fallback voice is set by build_synth(). With a
    `session` folder (and logging.save_voice) every piece Jack says is saved and checked
    there, in jack/. Takes are checked before they are played (jack_take_check) and each
    text keeps the seed of its good take in data/jack_takes.json."""
    from claudio.tts.jack import JackSynth, worker_command
    from claudio.tts.takes import TakeStore, voice_profile
    from claudio.tts.voice_check import TakeJudge, VoiceMonitor

    tts = settings.tts
    models = settings.resolve(settings.models_dir)
    target = tts.jack_lufs or -24.0  # with the level guard off, still the reference
    judge = TakeJudge(target_lufs=target, usual_hz=tts.jack_usual_hz)
    takes = TakeStore(settings.resolve(settings.data_dir) / "jack_takes.json", voice_profile(tts))
    return JackSynth(
        worker_command(models, tts),
        None,
        tts.jack_startup_timeout_s,
        tts.jack_sentence_timeout_s,
        rvc=tts.jack_rvc,
        words=tts.jack_words,
        first_wait_s=tts.jack_first_wait_s,
        monitor=VoiceMonitor(session / "jack", judge=judge)
        if session is not None and settings.logging.save_voice
        else None,
        judge=judge if tts.jack_take_check else None,
        takes=takes,
        seed=tts.jack_seed,
        retries=tts.jack_take_retries,
        first_retries=tts.jack_first_retries,
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


def build_speaker(
    settings: Settings, jack: JackSynth | None = None, start_player: bool = True
) -> StreamingSpeaker:
    """`start_player` False: the caller opens the audio output (run_assistant does it on the
    main thread, as this runs in a loader thread there)."""
    from claudio.audio.player import AudioPlayer
    from claudio.tts.speaker import StreamingSpeaker

    synth, name = build_synth(settings, jack)
    player = AudioPlayer(settings.audio.output_sample_rate, settings.audio.output_device)
    if start_player:
        player.start()
    speaker = StreamingSpeaker(
        synth,
        player,
        settings.audio.beep_volume,
        first_clause_chars=settings.tts.first_clause_chars,
        group_chars=settings.tts.jack_group_chars if settings.tts.engine == "jack" else 0,
        group_first_chars=settings.tts.jack_group_first_chars,
        group_pause_s=settings.tts.jack_group_pause_ms / 1000,
    )
    speaker.warm_up()
    log.info("TTS: %s (%d Hz)", name, synth.sample_rate)
    return speaker


def jack_wait_s(settings: Settings, gpu: GpuMemory | None) -> float | None:
    """How long replies may wait for Jack's worker while it loads: not at all when the
    graphics card is already full (a game): it loaded in 21 s, or never (30/10), and every
    reply meanwhile waited up to jack_first_wait_s. None = the configured wait."""
    need = settings.tts.jack_min_free_vram_mb
    if gpu is None or not need or gpu.free_mb >= need:
        return None
    log.warning(
        "Scheda video occupata (libera %d MB su %d%s): finché Jack non è pronto risponde la "
        "voce di riserva",
        gpu.free_mb,
        gpu.total_mb,
        f", calcolo al {gpu.busy_pct}%" if gpu.busy_pct is not None else "",
    )
    return 0.0


def startup_notices(
    settings: Settings, jack_running: bool, stt_device: str, gpu_busy: bool = False
) -> list[str]:
    """What Claudio says out loud at startup when it runs degraded: a warning in the console
    alone went unnoticed (on 23/10 Jack's worker failed and Kokoro answered all evening)."""
    notices = []
    if settings.tts.engine == "jack" and not jack_running:
        notices.append("La voce di Jack non è partita: uso quella di riserva.")
        if gpu_busy:  # a fact, maybe not the cause: the cause is in the log
            notices.append("All'avvio la scheda video era quasi piena.")
    if settings.stt.device == "auto" and stt_device == "cpu":
        notices.append("La scheda video non è disponibile: ti capisco più lentamente.")
    if notices:
        notices.append("Il motivo è nel log.")
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
    from claudio.router.actions import ActionContext

    responder = build_responder(settings, ActionContext.real())
    speaker = build_speaker(settings)
    start = time.perf_counter()
    first: list[float] = []
    await speaker.speak(
        responder.respond(text, DEFAULT_LANGUAGE),  # as if Whisper heard it
        on_first_audio=lambda: first.append(time.perf_counter()),
        language=lambda: responder.language,
    )
    if first:
        log.info("Prima parola dopo %.0f ms (router + TTS)", (first[0] - start) * 1000)
    speaker.player.close()


def build_vocabulary(settings: Settings) -> Any:
    """The folder, program and open window names for Whisper's prompt (folders and programs
    read now in a thread)."""
    from claudio.stt.vocabulary import Vocabulary, default_folders, default_shortcut_folders

    folders = default_folders(Path.home())
    windows = None
    if sys.platform == "win32":
        from claudio.pc.windows import window_titles

        windows = window_titles
    vocabulary = Vocabulary(
        lambda: folders,
        lambda: default_shortcut_folders(folders[0]),
        words=settings.stt.vocabulary_words,
        max_folders=settings.stt.vocabulary_folders,
        max_programs=settings.stt.vocabulary_programs,
        windows=windows,
        max_windows=settings.stt.vocabulary_windows,
    )
    vocabulary.refresh()
    return vocabulary


def build_recovery(
    settings: Settings, whisper: Rescorer, matcher: CommandMatcher
) -> CommandRecovery | None:
    """Whisper's second opinion on a short phrase no command understands (stt/recovery.py);
    None when [recovery] is off, and when Whisper fell back to the CPU: there the small model
    adds 1-3 s to every short request and the thresholds were measured on large-v3."""
    from claudio.router.matcher import load_commands
    from claudio.stt.recovery import CommandRecovery

    if not settings.recovery.enabled:
        return None
    device, model = getattr(whisper, "device", "cuda"), getattr(whisper, "model_name", "")
    if device != "cuda" or model != settings.stt.model:
        log.info("Seconda opinione di Whisper spenta: Whisper %s su %s", model, device)
        return None
    commands = load_commands(settings.resolve(settings.router.commands_file))
    return CommandRecovery(whisper, matcher, commands, settings.recovery)


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
            # The voice that started the phrase and BARGE_LEAD_MS before it: the request's
            # 800 ms were mostly the reply's echo, and filled 2/3 of every slice (30/10).
            "pre_roll_ms": c.barge_in_min_speech_ms + BARGE_LEAD_MS,
            # barge_in_max_s of voice in a slice: what came before the voice does not count.
            "max_utterance_s": c.barge_in_max_s + BARGE_LEAD_MS / 1000,
        }
    )
    return BargeIn(
        Endpointer(vad, vad_settings, settings.audio.sample_rate),
        lambda audio: transcriber.transcribe(audio, quick=True),
        mode=c.barge_in_mode,
        min_dbfs=c.barge_in_min_dbfs,
        duck_volume=c.duck_volume,
        needs_name=c.barge_in_needs_name,
        echo_similarity=c.echo_similarity,
        resume_s=c.resume_s,
        rest_s=settings.claude.history_reset_s,
        resume_min_chars=c.resume_min_chars,
    )


def build_assistant(
    settings: Settings,
    *,
    wake: OpenWakeWordDetector,
    vad: SileroVad,
    wake_vad: SileroVad,
    transcriber: WhisperTranscriber,
    responder: LocalResponder,
    speaker: Speaker,
    system: SystemControl,
    level: AutoGain | None = None,
    metrics: MetricsWriter | None = None,
    audio_sink: UtteranceRecorder | None = None,
    rejected_sink: UtteranceRecorder | None = None,
    clock: Callable[[], float] | None = None,
) -> Assistant:
    """The Assistant exactly as production wires it from the settings (run_assistant) and as
    the replay gate does (scripts/replay_gate.py): only what the caller owns is passed in.
    `clock=None` leaves the Assistant its own."""
    from claudio.app import Assistant
    from claudio.audio.vad import Endpointer, VoiceMeter

    return Assistant(
        wake=wake,
        endpointer=Endpointer(vad, settings.vad, settings.audio.sample_rate),
        transcriber=transcriber,
        responder=responder,
        speaker=speaker,
        metrics=metrics,
        audio_sink=audio_sink,
        level=level,
        system=system,
        second_chance=(
            (lambda audio: transcriber.transcribe(audio, quick=True))
            if settings.wakeword.second_chance
            else None
        ),
        conversation_s=settings.conversation.window_s,
        barge_in=build_barge_in(settings, transcriber) if settings.conversation.barge_in else None,
        verify_wake=settings.wakeword.verify,
        closed_list=(
            (lambda audio, _language: transcriber.candidates(audio))
            if settings.stt.closed_list
            else None
        ),
        closed_list_max_words=settings.stt.closed_list_max_words,
        claude_min_words=settings.claude.conversation_min_words if responder.fallback else None,
        only_addressed=settings.conversation.only_addressed,
        noise=settings.noise,
        confirm_below=settings.wakeword.confirm_below,
        always_confirm=wake.confirmed,
        unclear_min=settings.wakeword.unclear_min,
        stt_segments=lambda: transcriber.last_segments,
        unsure_logprob=settings.stt.unsure_logprob if responder.fallback else None,
        alternatives=settings.stt.alternatives if settings.stt.closed_list else 0,
        on_heard=responder.heard,
        continue_s=settings.vad.continue_s,
        echo_tail_ms=settings.conversation.echo_tail_ms,
        guess=responder.guess if responder.fallback_guess else None,
        command_recovery=build_recovery(settings, transcriber, responder.matcher),
        # At the endpointer's "not silence" level: a name clipped by the headset's gate
        # stays above [vad] threshold for 64 ms only (30/10: 128 ms or more at this level,
        # clicks and the gate 0).
        voice_meter=VoiceMeter(
            wake_vad, max(settings.vad.threshold - 0.15, 0.01), settings.audio.sample_rate
        ),
        wake_min_voice_ms=settings.wakeword.min_voice_ms,
        rejected_sink=rejected_sink,
        input_gain_db=settings.audio.input_gain_db,
        **({} if clock is None else {"clock": clock}),
    )


async def run_assistant(
    settings: Settings, session: Path, chat_window: ChatLauncher | None = None
) -> None:
    from claudio.audio.capture import AudioCapture
    from claudio.audio.devices import resolve_device
    from claudio.audio.level import AutoGain
    from claudio.audio.recorder import UtteranceRecorder
    from claudio.audio.vad import SileroVad
    from claudio.chat.store import ChatStore
    from claudio.metrics import MetricsWriter
    from claudio.router.actions import ActionContext
    from claudio.session_log import ConversationLog, purge_old_logs
    from claudio.stt.transcriber import WhisperTranscriber
    from claudio.system import SystemControl
    from claudio.tray import start_tray
    from claudio.wakeword.detector import OpenWakeWordDetector

    models = settings.resolve(settings.models_dir)
    t0 = time.perf_counter()
    # Jack's worker is a process of its own: it loads (8-30 s) while everything else does,
    # and Claudio listens without waiting for it (a reply meanwhile waits jack_first_wait_s,
    # or not at all when a game fills the graphics card).
    jack = make_jack(settings, session) if settings.tts.engine == "jack" else None
    jack_error: list[Exception | None] = []
    gpu_busy = False
    if jack is not None:
        need = settings.tts.jack_min_free_vram_mb
        wait_s = jack_wait_s(settings, read_gpu() if need else None)
        gpu_busy = wait_s is not None
        if need:  # a worker that timed out on a full card is tried again once it has room
            jack.can_start = lambda: has_room(need)
        jack.start_in_background(jack_error.append, wait_s)
    system = SystemControl()
    tray = start_tray(settings.startup.tray, system, chat_window)  # blue while loading
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
            "claude": lambda: build_responder(
                settings, ActionContext.real(), system, chat_window, tracker
            ),
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
        "STT: %s su %s",
        transcriber.model_name,
        transcriber.device.upper(),
    )
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
    if level is not None:
        capture.on_device = level.use  # every microphone keeps its own gain

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
    rejected = (
        UtteranceRecorder(
            logs / "audio" / "rejected", settings.logging.keep_rejected, settings.audio.sample_rate
        )
        if settings.logging.save_audio and settings.logging.keep_rejected
        else None
    )
    wake_vad = SileroVad(models / "vad" / "silero_vad.onnx", settings.audio.sample_rate)
    assistant = build_assistant(
        settings,
        wake=wake,
        vad=vad,
        wake_vad=wake_vad,
        transcriber=transcriber,
        responder=responder,
        speaker=speaker,
        system=system,
        level=level,
        metrics=MetricsWriter(
            logs / "metrics.jsonl", ConversationLog(session), chat, gpu=GpuSampler()
        ),
        audio_sink=recorder,
        rejected_sink=rejected,
    )
    if tray is not None:
        tray.set_ready()
    log.info(
        'Modelli caricati in %.1f s. Di\' "%s" per iniziare (Ctrl+C per uscire).',
        time.perf_counter() - t0,
        '" o "'.join(wake.names),
    )
    inbox = (
        asyncio.create_task(serve_inbox(assistant, chat, settings)) if settings.chat.typed else None
    )
    ready = asyncio.create_task(
        announce_ready(assistant, settings, jack, jack_error, transcriber.device, t0, gpu_busy)
    )
    run = asyncio.create_task(assistant.run(capture.frames()))
    closer = asyncio.create_task(close_when_idle(assistant, system, run))
    try:
        await run
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():  # Ctrl+C: not close_when_idle's doing
            raise
    finally:
        closer.cancel()
        ready.cancel()
        if inbox is not None:
            inbox.cancel()
        speaker.player.close()
        if chat_window is not None:
            chat_window.close()  # "chiuditi" or Ctrl+C: the window goes with Claudio


async def close_when_idle(assistant: Any, system: SystemControl, run: asyncio.Task[None]) -> None:
    """ "Chiuditi" or the tray menu end the run at the next microphone frame. With no
    microphone (unplugged, switched off) no frame comes: the run is cancelled instead, once
    CLAUDIO is idle and a frame had CLOSE_GRACE_S to end it."""
    from claudio.app import State

    while not (system.stopping and assistant.state is State.IDLE):
        await asyncio.sleep(CLOSE_POLL_S)
    await asyncio.sleep(CLOSE_GRACE_S)
    if not run.done():
        log.info("Nessun audio dal microfono: chiudo senza aspettarlo")
        run.cancel()


async def announce_ready(
    assistant: Any,
    settings: Settings,
    jack: JackSynth | None,
    jack_error: list[Exception | None],
    stt_device: str,
    t0: float,
    gpu_busy: bool = False,
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
    notices = startup_notices(settings, jack_running, stt_device, gpu_busy)
    for notice in notices:
        log.warning("Avviso a voce: %s", notice)
    if settings.startup.announce_ready:
        notices.append("Sono pronto.")
    if notices:
        await assistant.announce(" ".join(notices))


async def serve_inbox(assistant: Any, chat: Any, settings: Settings) -> None:
    """Answer what is written in the chat window (`claudio --ui`), one message at a time.
    Typed text has no detected language: Italian is used."""
    language = DEFAULT_LANGUAGE
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

    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"Errore di configurazione: {exc}", file=sys.stderr)
        return 2
    logs_dir = settings.resolve(settings.logs_dir)
    log_name = "chat-window.log.jsonl" if args.ui else "claudio.log.jsonl"
    # Before setup_logging: Windows will not replace a file our own handler holds open.
    trimmed = drop_old_lines(
        logs_dir / log_name, time.time() - settings.logging.retention_days * 86400
    )
    setup_logging(logs_dir, settings.logging.level, log_name)
    if trimmed:
        log.info(
            "Log: %d righe più vecchie di %d giorni tolte da %s",
            trimmed,
            settings.logging.retention_days,
            log_name,
        )
    if args.ui:
        from claudio.chat.window import open_window

        open_window(settings.resolve(settings.data_dir) / CHAT_DB)
        return 0
    session = None
    instance = None
    if not args.check and not args.text:
        from claudio.session_log import start_session
        from claudio.single_instance import EXIT_ALREADY_RUNNING, SingleInstance

        instance = SingleInstance()  # one CLAUDIO per microphone: login task, a terminal...
        if not instance.acquire():
            log.warning("CLAUDIO è già acceso: questa copia si chiude")
            return EXIT_ALREADY_RUNNING
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
    except Exception:  # pythonw has no console: without this the reason would be lost
        log.exception("CLAUDIO si è fermato per un errore")
        return 1
    finally:
        if instance is not None:
            instance.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
