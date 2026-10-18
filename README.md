# CLAUDIO

Assistente vocale per Windows, in italiano, che gira **tutto in locale**: wake word, trascrizione,
comandi e voce. Nella Fase 1 non fa nessuna chiamata API, quindi il costo è zero.

```
Microfono ─► openWakeWord ("Hey Claudio" / "Claudio") ─► bip
         └─► Silero VAD (fine frase) ─► faster-whisper (GPU) ─► router locale ─► Piper (frase per frase)
```

Stato: **v0.1.0**, scheletro funzionante della Fase 1. La wake word è ancora il segnaposto
`hey jarvis` finché non addestriamo "Hey Claudio" e "Claudio" (v0.1.1, vedi
[scripts/wakeword/README.md](scripts/wakeword/README.md)). Specifica completa: [CLAUDE.md](CLAUDE.md).

## Requisiti

- Windows 10/11, Python 3.12 (3.11-3.13 vanno bene), [uv](https://docs.astral.sh/uv/)
- Per la GPU (consigliata): driver NVIDIA recente (serie 570 o successiva per la RTX 50xx).
  Le librerie CUDA 12.8+ e cuDNN 9 si installano con pip, non serve il CUDA Toolkit.
- ~2,5 GB di disco per i modelli.

## Installazione

```powershell
# una volta sola: installa uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/JENJ0/Claudio-AI-desktop-assistant.git
cd Claudio-AI-desktop-assistant
uv sync --extra gpu                       # senza GPU: uv sync
uv run python scripts/download_models.py  # Silero, openWakeWord, Piper, Whisper turbo + small
uv run claudio --check                    # tutte le righe devono essere OK
uv run python scripts/check_gpu.py        # verifica che Whisper giri davvero sulla RTX
```

Se PowerShell dice che `uv` non è riconosciuto, chiudi e riapri il terminale (il PATH si
aggiorna solo nelle finestre nuove). Se non basta, aggiungi la cartella di uv al PATH una volta:

```powershell
$p=[Environment]::GetEnvironmentVariable('Path','User'); if ($p -notlike "*\.local\bin*") { [Environment]::SetEnvironmentVariable('Path', "$p;$env:USERPROFILE\.local\bin", 'User') }
```

Chi preferisce pip: `python -m venv .venv`, attivalo, poi `pip install -r requirements.txt`
e `pip install -e . --no-deps`.

## Uso

```powershell
uv run claudio                         # assistente completo (Ctrl+C per uscire)
uv run claudio --text "che ore sono"   # prova router + voce senza microfono
uv run python scripts/list_audio_devices.py
```

Di' la wake word, aspetta il bip, poi il comando. Se non parli entro 4 s senti un bip basso e
Claudio torna in attesa. Il segnaposto `hey jarvis` è un modello inglese: detto in fretta o
attaccato al comando spesso non scatta, quindi in v0.1.0 dillo con calma e fai una pausa.

| Comando (esempi) | Cosa fa |
|---|---|
| "apri Chrome", "apri il browser" | avvia Chrome |
| "apri la cartella Download" | Esplora file su Download |
| "apri Documenti" | Esplora file su Documenti |
| "apri Esplora file" | Esplora file |
| "apri il Blocco note" | Blocco note |
| "apri la calcolatrice" | Calcolatrice |
| "alza il volume" / "alza il volume di 10" | volume su (5 scatti, o quelli detti) |
| "abbassa il volume" | volume giù |
| "muto", "togli l'audio" | muto / riattiva |
| "che ore sono", "che giorno è oggi" | risponde a voce |

Le frasi stanno in [config/commands.toml](config/commands.toml): aggiungerne non richiede codice.
Tutte le altre richieste ricevono "Questo non lo so ancora fare" (in Fase 2 andranno a Claude Haiku).

## Configurazione

Tutto è in [config/claudio.toml](config/claudio.toml): microfono e uscita, soglie della wake word,
VAD, modello STT, voce. Ogni valore si può sovrascrivere con una variabile d'ambiente o in `.env`,
per esempio `CLAUDIO_STT__DEVICE=cpu`.

Microfono e cuffie si scelgono per nome (basta una parte, la vedi con
`scripts/list_audio_devices.py`); a parità di nome Claudio usa WASAPI:

```
CLAUDIO_AUDIO__INPUT_DEVICE=Wireless Stereo Head
CLAUDIO_AUDIO__OUTPUT_DEVICE=Wireless Stereo Head
CLAUDIO_AUDIO__INPUT_GAIN_DB=12
```

`INPUT_GAIN_DB` alza un microfono che resta basso anche al 100% in Windows: se serve, Claudio lo
scrive nel log ("Microfono basso"). La voce dovrebbe arrivare tra -20 e -10 dBFS (colonna `voce`
di `benchmark_latency.py --log`).

Se la GPU non è utilizzabile, Claudio passa da solo a Whisper `small` su CPU e lo scrive nel log.

## Latenza

Ogni richiesta finisce in `logs/metrics.jsonl` con:

- **M1** = fine del parlato → primo suono della risposta (obiettivo ≤ 1 s);
- **M2** = wake word → primo suono.

Per tarare l'ascolto ogni riga ha anche `input_dbfs` (volume medio del microfono dopo il bip) e
`vad_max` (quanto Silero era sicuro di sentire voce, 0-1). In `logs/audio/` restano gli ultimi
50 WAV di quello che Claudio ha sentito (1,5 s prima della wake word, poi tutto il resto), con il
nome dell'esito (`ok`, `timeout`, `empty`, `error`). Restano sul PC; si spengono con
`[logging] save_audio = false`.

Anche i tentativi in cui la wake word non è scattata finiscono nel log come `near_miss`, con il
punteggio massimo raggiunto: `benchmark_latency.py --log` li elenca per capire quanto abbassare
la soglia.

```powershell
uv run python scripts/benchmark_latency.py --log                      # dal vivo
uv run python scripts/record_samples.py commands --repeat 2           # registra i comandi
uv run python scripts/benchmark_latency.py --wav-dir samples/commands  # turbo vs small vs base
uv run python scripts/benchmark_tts.py                                 # voci Piper
```

## Sviluppo

```powershell
uv run pytest                 # test veloci (senza microfono né modelli)
uv run pytest -m slow -s      # test sui modelli reali (dopo download_models.py)
uv run ruff check . ; uv run ruff format . ; uv run mypy src
```

La CI (GitHub Actions, Windows) esegue lint, tipi, test veloci e un giro completo
Piper → Silero → Whisper `small` (il ripiego su CPU) → router sui modelli veri.

Struttura: `src/claudio/` (audio, wakeword, stt, tts, router, `app.py` con la macchina a stati),
`config/`, `scripts/`, `tests/`. Modelli in `models/` e log in `logs/`, entrambi fuori da Git.
