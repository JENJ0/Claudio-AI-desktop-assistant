"""CLAUDIO's own messages (console, log files, --check) in Italian or English, chosen by the
`language` setting. The code writes them in Italian; EN maps each one, %-placeholders and
{names} included, to its English version. Spoken replies are not here: they follow the
language of the request (config/commands.toml)."""

from __future__ import annotations

import logging
from typing import Any

LANGUAGES = ("it", "en")
_current = "it"

EN: dict[str, str] = {
    # __main__
    "Router: %d comandi locali (%s)": "Router: %d local commands (%s)",
    "TTS: %s (%d Hz)%s%s": "TTS: %s (%d Hz)%s%s",
    ", effetto {effect}": ", effect {effect}",
    "STT: %s su %s, lingue %s": "STT: %s on %s, languages %s",
    'Modelli caricati in %.1f s. Di\' "%s" per iniziare (Ctrl+C per uscire).': (
        'Models loaded in %.1f s. Say "%s" to start (Ctrl+C to quit).'
    ),
    '" o "': '" or "',
    "Prima parola dopo %.0f ms (router + TTS)": "First word after %.0f ms (router + TTS)",
    "Errore di configurazione: {error}": "Configuration error: {error}",
    "Chiusura richiesta: arrivederci": "Shutdown requested: goodbye",
    "Kokoro non disponibile (%s): uso Piper %s": "Kokoro not available (%s): using Piper %s",
    "Voce per '%s' non disponibile (%s): uso quella principale": (
        "Voice for '%s' not available (%s): using the main one"
    ),
    # app
    "Pronto: in ascolto della wake word": "Ready: listening for the wake word",
    "Wake word rilevata: %s (%.2f)": "Wake word detected: %s (%.2f)",
    "Solo la wake word: ascolto la richiesta": "Only the wake word: listening for the request",
    "Stato: %s -> %s": "State: %s -> %s",
    "Trascritto (%s): %r": "Transcribed (%s): %r",
    "Wake word non confermata da Whisper, ignoro: %r": (
        "Wake word not confirmed by Whisper, ignoring: %r"
    ),
    "Lista chiusa: %r -> %r": "Closed list: %r -> %r",
    "Microfono basso: la voce arriva al massimo a %.0f dBFS (meglio tra -20 e -10). Aumenta il "
    "guadagno, per esempio CLAUDIO_AUDIO__INPUT_GAIN_DB=12 in .env": (
        "Microphone too quiet: the voice peaks at %.0f dBFS (best between -20 and -10). Raise "
        "the gain, for example CLAUDIO_AUDIO__INPUT_GAIN_DB=12 in .env"
    ),
    "Nessuna richiesta dopo la wake word: torno in attesa": (
        "No request after the wake word: back to waiting"
    ),
    "Errore durante la richiesta": "Error while handling the request",
    "Seconda possibilità non riuscita": "The second chance failed",
    "Seconda possibilità: Whisper ha sentito %r (punteggio %.2f): attivo": (
        "Second chance: Whisper heard %r (score %.2f): waking up"
    ),
    "Seconda possibilità: Whisper ha sentito %r: niente da fare": (
        "Second chance: Whisper heard %r: nothing to do"
    ),
    "Spegnimento richiesto a voce: chiudo": "Shutdown asked by voice: closing",
    "Impossibile salvare l'audio della wake word mancata": (
        "Could not save the audio of the missed wake word"
    ),
    "Impossibile riprodurre il segnale di errore": "Could not play the error sound",
    "Impossibile salvare l'audio della richiesta": "Could not save the audio of the request",
    # audio
    "Stato audio in ingresso: %s": "Audio input status: %s",
    "Microfono: %s%s": "Microphone: %s%s",
    "Microfono di Windows cambiato: %s": "Windows microphone changed: %s",
    "Microfoni sentiti: %s": "Microphones heard: %s",
    "WARN voce di Jack non installata (%s): parlerà Kokoro": (
        "WARN Jack's voice is not installed (%s): Kokoro will speak"
    ),
    "Il microfono non sente niente: ne scelgo un altro": (
        "The microphone hears nothing: choosing another one"
    ),
    "Microfono %d non apribile per la prova: %s": (
        "Microphone %d could not be opened for the probe: %s"
    ),
    "Microfono non disponibile (%s): riprovo tra 1 s": (
        "Microphone not available (%s): retrying in 1 s"
    ),
    "Microfono ancora non disponibile da %.0f s (%s): ricollegalo": (
        "Microphone still not available after %.0f s (%s): plug it back in"
    ),
    "Microfono di nuovo disponibile dopo %.0f s": "Microphone available again after %.0f s",
    "Nessun dispositivo di {kind} contiene '{spec}'. Elencali con: "
    "uv run python scripts/list_audio_devices.py": (
        "No {kind} device contains '{spec}'. List them with: "
        "uv run python scripts/list_audio_devices.py"
    ),
    "Uscita predefinita di Windows non leggibile: %s": "Windows default output not readable: %s",
    "Microfono: guadagno da +%.0f a +%.0f dB (%s)": "Microphone: gain from +%.0f to +%.0f dB (%s)",
    "Microfono: guadagno +%.0f dB (salvato dall'ultima volta)": (
        "Microphone: gain +%.0f dB (saved last time)"
    ),
    "Microfono basso anche con +%.0f dB (voce a %.0f dBFS): alza il livello del microfono nelle "
    "impostazioni audio di Windows o avvicinati": (
        "Microphone still too quiet with +%.0f dB (voice at %.0f dBFS): raise the microphone "
        "level in Windows sound settings or move closer"
    ),
    "File del livello microfono illeggibile: riparto da zero": (
        "Microphone level file unreadable: starting from zero"
    ),
    "Impossibile salvare il livello del microfono: %s": "Could not save the microphone level: %s",
    "Uscita audio: %s": "Audio output: %s",
    "Uscita predefinita di Windows cambiata in %s, che non c'era all'avvio: riavvia CLAUDIO per "
    "usarla": (
        "Windows default output changed to %s, which was not there at start-up: restart CLAUDIO "
        "to use it"
    ),
    "Uscita predefinita di Windows cambiata: %s": "Windows default output changed: %s",
    "Uscita %s non apribile (%s): uso quella di base": (
        "Output %s cannot be opened (%s): using the default"
    ),
    "Uscita audio non disponibile (%s): riapro il dispositivo": (
        "Audio output not available (%s): reopening the device"
    ),
    "Modello Silero VAD mancante: {path}": "Silero VAD model missing: {path}",
    # checks
    "{count} comandi in {file} ({languages})": "{count} commands in {file} ({languages})",
    "comandi: {error}": "commands: {error}",
    "modello {path}": "model {path}",
    "wake word {name} (soglia {threshold})": "wake word {name} (threshold {threshold})",
    "GPU CUDA: {count} (tipi: {types})": "CUDA GPU: {count} (types: {types})",
    "microfono": "microphone",
    "uscita audio": "audio output",
    " (segue l'uscita predefinita di Windows)": " (follows the Windows default output)",
    "Qualcosa manca: vedi le righe ERR. Modelli: uv run python scripts/download_models.py": (
        "Something is missing: see the ERR lines. Models: uv run python scripts/download_models.py"
    ),
    "WARN voce %s (%s) mancante: risponderò con quella principale. Scaricala con: "
    "uv run python scripts/download_models.py --skip-whisper": (
        "WARN voice %s (%s) missing: I will answer with the main one. Download it with: "
        "uv run python scripts/download_models.py --skip-whisper"
    ),
    "WARN voce Kokoro %s (%s) sconosciuta: risponderò con quella principale": (
        "WARN Kokoro voice %s (%s) unknown: I will answer with the main one"
    ),
    "WARN nessuna GPU CUDA: lo STT userà la CPU (%s)": (
        "WARN no CUDA GPU: STT will use the CPU (%s)"
    ),
    "WARN controllo GPU fallito: %s": "WARN GPU check failed: %s",
    # config
    "File di configurazione non trovato: {path}": "Configuration file not found: {path}",
    "{path}: TOML non valido ({error})": "{path}: invalid TOML ({error})",
    "{path}: valori non validi -> {problems}": "{path}: invalid values -> {problems}",
    # metrics
    "Latenza M1=%s ms M2=%s ms (STT %s ms, TTS prima frase %s ms)": (
        "Latency M1=%s ms M2=%s ms (STT %s ms, TTS first sentence %s ms)"
    ),
    "Metriche: %s": "Metrics: %s",
    # router
    "Azione eseguita: %s": "Action done: %s",
    "Azione sconosciuta '{action}' nel comando '{command}'": (
        "Unknown action '{action}' in command '{command}'"
    ),
    "{path}: il comando '{command}' ha un livello non valido ({level})": (
        "{path}: command '{command}' has an invalid level ({level})"
    ),
    "{path}: comando duplicato '{command}'": "{path}: duplicate command '{command}'",
    "{path}: il comando '{command}' non ha frasi": "{path}: command '{command}' has no phrases",
    "Frase '{phrase}' ambigua: {first} e {second}": (
        "Ambiguous phrase '{phrase}': {first} and {second}"
    ),
    "In pausa, ignoro: %r": "Paused, ignoring: %r",
    "Richiesta non riconosciuta: %r": "Request not recognised: %r",
    "Errore eseguendo %s": "Error running %s",
    # stt
    "STT riscaldato in %.0f ms": "STT warmed up in %.0f ms",
    "Lingua: %s (%.2f)": "Language: %s (%.2f)",
    "Trascrizione scartata (allucinazione nota): %r": (
        "Transcript dropped (known hallucination): %r"
    ),
    "Trascrizione scartata (ripetizione): %r": "Transcript dropped (repetition): %r",
    "nessuna GPU CUDA trovata": "no CUDA GPU found",
    "GPU non utilizzabile (%s): passo alla CPU con '%s' %s": (
        "GPU not usable (%s): switching to the CPU with '%s' %s"
    ),
    # system
    "Sistema: %s": "System: %s",
    "Comando di sistema sconosciuto '{command}'": "Unknown system command '{command}'",
    "Riprovato in %s: %r -> %r": "Retried in %s: %r -> %r",
    "Riprovato in %s, non capito neanche così: %r": "Retried in %s, still not understood: %r",
    # safety
    "Conferma richiesta (livello %d): %s": "Confirmation asked (level %d): %s",
    "Conferma data (livello %d): eseguo %s": "Confirmation given (level %d): running %s",
    "Conferma rifiutata (livello %d): %s non eseguito": (
        "Confirmation refused (level %d): %s not run"
    ),
    "Conferma non chiara (livello %d): %s non eseguito": (
        "Confirmation unclear (level %d): %s not run"
    ),
    "Conferma scaduta: %s non eseguito": "Confirmation expired: %s not run",
    "Basta: fine della conversazione": "Stop: end of the conversation",
    "Ripeto: %r": "Repeating: %r",
    "Fine della conversazione: torno in attesa della wake word": (
        "End of the conversation: waiting for the wake word again"
    ),
    "Log: %d file più vecchi di %d giorni eliminati": "Logs: %d files older than %d days deleted",
    "Impossibile eliminare il vecchio log %s": "Could not delete the old log %s",
    "Risposta troppo lontana dal microfono, non la conto: %r": (
        "Answer too far from the microphone, not counted: %r"
    ),
    "Rumore scambiato per parole (%s), ignoro: %r": "Noise heard as words (%s), ignored: %r",
    "Non è una richiesta, fine della conversazione: %r": (
        "Not a request, end of the conversation: %r"
    ),
    "Ascolto durante la risposta non riuscito": "Listening while speaking failed",
    "Sentito mentre parlo: %r": "Heard while speaking: %r",
    "Interrotto a voce: %r": "Interrupted by voice: %r",
    "Risposta interrotta": "Reply interrupted",
    # cost
    "Nessun prezzo per il modello '{model}' in config/pricing.toml": (
        "No price for model '{model}' in config/pricing.toml"
    ),
    "File dei prezzi non trovato: {path}": "Prices file not found: {path}",
    "Budget giornaliero: %d%% speso": "Daily budget: %d%% spent",
    "Costo chiamata %s: %.5f %s": "Call cost %s: %.5f %s",
    # tts
    "Effetto voce '{name}' sconosciuto (disponibili: {known})": (
        "Unknown voice effect '{name}' (available: {known})"
    ),
    "Kokoro: mancano {files} in {folder}. Esegui: uv run python scripts/download_models.py": (
        "Kokoro: {files} missing in {folder}. Run: uv run python scripts/download_models.py"
    ),
    "Frase troppo lunga per Kokoro (%d fonemi): la taglio": (
        "Sentence too long for Kokoro (%d phonemes): cutting it"
    ),
    "Voce Kokoro '{voice}' sconosciuta (disponibili: {known})": (
        "Unknown Kokoro voice '{voice}' (available: {known})"
    ),
    "Voce Piper '{voice}' mancante in {folder}. Esegui: uv run python scripts/download_models.py": (
        "Piper voice '{voice}' missing in {folder}. Run: uv run python scripts/download_models.py"
    ),
    "TTS (%s): %s": "TTS (%s): %s",
    "Jack non disponibile (%s): uso %s": "Jack not available (%s): using %s",
    "worker Jack non avviabile: {error}": "Jack worker cannot start: {error}",
    "worker Jack: {error}": "Jack worker: {error}",
    "worker Jack: nessuna risposta in {seconds} s": "Jack worker: no answer in {seconds} s",
    "worker Jack terminato": "Jack worker ended",
    "Worker Jack pronto in %.1f s (VRAM GPU %d MB)": (
        "Jack worker ready in %.1f s (GPU VRAM %d MB)"
    ),
    "Worker Jack: protocollo rotto (%s)": "Jack worker: broken protocol (%s)",
    "Jack non ha risposto (%s): uso la voce di ripiego": (
        "Jack did not answer (%s): using the fallback voice"
    ),
    "Riavvio del worker Jack fallito: %s": "Jack worker restart failed: %s",
    # wake word
    "Modello wake word '{model}' non trovato in {folder}. Se lo hai addestrato tu, "
    "copialo lì (scripts/wakeword/README.md); altrimenti esegui: "
    "uv run python scripts/download_models.py": (
        "Wake word model '{model}' not found in {folder}. If you trained it yourself, "
        "copy it there (scripts/wakeword/README.md); otherwise run: "
        "uv run python scripts/download_models.py"
    ),
    "Modello {name} mancante in {folder}": "Model {name} missing in {folder}",
    "Wake word: %s": "Wake word: %s",
    "Wake word sotto soglia: %s %.2f (soglia %.2f)": (
        "Wake word below threshold: %s %.2f (threshold %.2f)"
    ),
    "Verifica della wake word non riuscita": "Wake word check failed",
    "Nome sentito prima della richiesta: %r": "Name heard before the request: %r",
    "Conversazione non salvata": "Conversation not saved",
    "Rotazione di %s non riuscita": "Rotation of %s failed",
    "Sessione salvata in %s": "Session saved in %s",
    "Chat: %d messaggi più vecchi di %d giorni eliminati": (
        "Chat: %d messages older than %d days deleted"
    ),
    "Chat non salvata": "Chat not saved",
    "Chat aperta": "Chat opened",
    "Chat chiusa": "Chat closed",
    "Comando chat sconosciuto '{command}'": "Unknown chat command '{command}'",
    "Avviso a voce: %s": "Spoken notice: %s",
    "La voce di Jack non è partita: uso quella di riserva.": (
        "Jack's voice did not start: using the backup voice."
    ),
    "La scheda video non è disponibile: ti capisco più lentamente.": (
        "The graphics card is not available: I understand you more slowly."
    ),
    "Il motivo è nel log.": "The reason is in the log.",
    # Claude (brain/, Phase 2)
    "Claude: %s (ragionamenti: %s), max %d token, cache %s, budget %.2f al giorno": (
        "Claude: %s (reasoning: %s), max %d tokens, cache %s, budget %.2f a day"
    ),
    "Sonnet spento, solo Haiku: %s": "Sonnet off, Haiku only: %s",
    "Haiku propone Sonnet: chiedo conferma": "Haiku proposes Sonnet: asking first",
    "Sonnet approvato": "Sonnet approved",
    "Sonnet rifiutato: risponde Haiku": "Sonnet declined: Haiku answers",
    "Richiesta dettagliata: propongo Sonnet": "Detailed request: proposing Sonnet",
    "Wake word debole (%.2f): Whisper controlla il nome": (
        "Weak wake word (%.2f): Whisper checks the name"
    ),
    "Richiesta di ragionare: chiedo direttamente a Sonnet": (
        "Asked to reason: asking Sonnet directly"
    ),
    "Claude: connessione anticipata non riuscita: %s": "Claude: early connection failed: %s",
    "Preparazione della risposta non riuscita": "Getting the reply ready failed",
    "Scritto nella finestra: %r": "Written in the window: %r",
    "Errore rispondendo al messaggio scritto": "Error answering the written message",
    "Messaggi della finestra non letti": "Window messages not read",
    "Messaggio dalla finestra rifiutato (origine %s)": "Window message refused (origin %s)",
    "Messaggio dalla finestra in coda: %d": "Window message queued: %d",
    "Finestra della chat non portata in primo piano": "Chat window not brought to the front",
    "Claude spento ([claude] enabled = false): solo comandi locali": (
        "Claude off ([claude] enabled = false): local commands only"
    ),
    "ANTHROPIC_API_KEY mancante nel file .env: Claude non risponderà": (
        "ANTHROPIC_API_KEY missing from the .env file: Claude will not answer"
    ),
    "Claude spento: %s": "Claude off: %s",
    "Claude: chiave API mancante (ANTHROPIC_API_KEY nel file .env)": (
        "Claude: API key missing (ANTHROPIC_API_KEY in the .env file)"
    ),
    "Claude bloccato: budget di oggi finito (%d%%)": (
        "Claude blocked: today's budget is spent (%d%%)"
    ),
    "Claude ha rifiutato la richiesta": "Claude refused the request",
    "Claude %s: %d in, %d out, cache %d letti / %d scritti (TTL %s)%s": (
        "Claude %s: %d in, %d out, cache %d read / %d written (TTL %s)%s"
    ),
    ", interrotta (stima)": ", cut short (estimate)",
    "Claude: chiave API non valida (401)": "Claude: invalid API key (401)",
    "Claude non raggiungibile: %s": "Claude unreachable: %s",
    "Claude errore %s: %s": "Claude error %s: %s",
    "Claude: spesa non registrata": "Claude: spending not recorded",
    "Non è un comando, chiedo a Claude: %r": "Not a command, asking Claude: %r",
    # Phase 3: actions on the PC
    "Azioni sul PC: %d strumenti, cartelle %s, %d tasti senza conferma": (
        "PC actions: %d tools, folders %s, %d keys without confirmation"
    ),
    "Azioni sul PC spente ([pc] enabled = false)": "PC actions off ([pc] enabled = false)",
    "Azioni sul PC: solo su Windows": "PC actions: Windows only",
    "Conferma data (livello %d): eseguo": "Confirmed (level %d): running",
    "Conferma %s: azione non eseguita": "Confirmation %s: action not run",
    "Azione PC %s livello %d: %s (%d ms)": "PC action %s level %d: %s (%d ms)",
    "[pc] safe_keys: tasto non valido ignorato: %r": "[pc] safe_keys: invalid key ignored: %r",
    "Azione PC fallita: %s": "PC action failed: %s",
    "Risposta alla conferma di Claude: %r": "Answer to Claude's confirmation: %r",
    "App nel menu Start: %d": "Apps in the Start menu: %d",
    "Lettura di %r interrotta dopo %.1f s": "Reading %r stopped after %.1f s",
    "COM non inizializzato per le azioni sul PC": "COM not initialised for the PC actions",
    "Indice di Windows non disponibile (%s): cerco nelle cartelle": (
        "Windows Search index unavailable (%s): walking the folders"
    ),
}


def set_language(code: str) -> None:
    global _current
    if code not in LANGUAGES:
        raise ValueError(f"language must be one of {', '.join(LANGUAGES)}, not '{code}'")
    _current = code


def current() -> str:
    return _current


def tr(text: str, **values: Any) -> str:
    """`text` in the program's language, with {names} filled in from `values`."""
    if _current == "en":
        text = EN.get(text, text)
    return text.format(**values) if values else text


class TranslateFilter(logging.Filter):
    """Translates every log message on its way to a handler (the %-arguments stay as they are)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _current != "it" and isinstance(record.msg, str):
            record.msg = EN.get(record.msg, record.msg)
        return True
