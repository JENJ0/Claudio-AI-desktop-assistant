# CLAUDIO

[English](README.md) | **Italiano**

Assistente vocale per Windows che gira **tutto in locale**: wake word, trascrizione, comandi e
voce. Nella Fase 1 non fa nessuna chiamata API, quindi il costo è zero. Capisce italiano e inglese
e risponde nella lingua in cui gli parli.

```
Microfono ─► openWakeWord ("Hey Claudio") ─► bip
         └─► Silero VAD (fine frase) ─► faster-whisper (GPU) ─► router locale ─► Kokoro, voce Nicola (frase per frase)
```

Stato: **v0.1.1**, Fase 1, seconda iterazione. La wake word è il nostro "Hey Claudio", addestrato
sulla voce del proprietario (vedi [scripts/wakeword/README.md](scripts/wakeword/README.md), in
inglese; il file del modello non è su Git). Anche "Claudio" da solo è addestrato ma non è ancora
abbastanza affidabile, quindi è spento. Specifica completa: [CLAUDE.md](CLAUDE.md).

## Requisiti

- Windows 10/11, Python 3.12 (3.11-3.13 vanno bene), [uv](https://docs.astral.sh/uv/)
- Per la GPU (consigliata): driver NVIDIA recente (serie 570 o successiva per la RTX 50xx).
  Le librerie CUDA 12.8+ e cuDNN 9 si installano con pip, non serve il CUDA Toolkit.
- ~3 GB di disco per i modelli.

## Installazione

```powershell
# una volta sola: installa uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/JENJ0/Claudio-AI-desktop-assistant.git
cd Claudio-AI-desktop-assistant
uv sync --extra gpu                       # senza GPU: uv sync
uv run python scripts/download_models.py  # Silero, openWakeWord, Kokoro, Piper, Whisper turbo + small
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
Claudio torna in attesa. Puoi anche dire tutto d'un fiato ("hey claudio che ore sono"): le parole
dette prima del bip vengono recuperate. Se il modello manca di poco, Whisper ha una seconda
possibilità di sentire il nome (vedi `wakeword.second_chance`).

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
| "vai in pausa", "vai a dormire" | pausa: finché non dici "svegliati" non risponde ad altro |
| "svegliati" | esce dalla pausa |
| "spegniti" | chiude CLAUDIO |

Ogni comando funziona anche in inglese: "open Chrome", "open my downloads", "turn up the volume by
10", "mute", "what time is it", "what day is it", "go to sleep", "wake up", "shut down". Per ogni
richiesta Whisper decide se è italiano o inglese, e Claudio risponde in quella lingua con una voce
inglese.

Le frasi stanno in [config/commands.toml](config/commands.toml): aggiungerne non richiede codice
(`phrases_en` e `reply_en` sono quelle inglesi). Tutte le altre richieste ricevono "Questo non lo
so ancora fare" o "I can't do that yet" (in Fase 2 andranno a Claude Haiku).

## Configurazione

Tutto è in [config/claudio.toml](config/claudio.toml): microfono e uscita, soglie della wake word,
VAD, modello STT, voce. Ogni valore si può sovrascrivere con una variabile d'ambiente o in `.env`,
per esempio `CLAUDIO_STT__DEVICE=cpu`.

Il microfono si sceglie per nome (basta una parte, la vedi con `scripts/list_audio_devices.py`);
a parità di nome Claudio usa WASAPI:

```
CLAUDIO_AUDIO__INPUT_DEVICE=C270
```

Senza `OUTPUT_DEVICE` Claudio parla dall'uscita che Windows sta usando in quel momento: se la cambi
mentre è acceso (cuffie, schermo, casse), la frase dopo esce già lì. Un dispositivo collegato dopo
l'avvio richiede un riavvio di Claudio (lo scrive nel log). Con `CLAUDIO_AUDIO__OUTPUT_DEVICE=...`
resta invece sempre su quello.

Il livello del microfono lo regola Claudio da solo: dopo ogni richiesta guarda quanto forte è
arrivata la voce e corregge il guadagno digitale (sale di 3 dB alla volta, scende di 6 se la voce
satura), puntando a picchi intorno a -20 dBFS. Il valore viene salvato per ogni microfono in
`data/audio_level.json` e riusato all'avvio successivo; `INPUT_GAIN_DB` è solo il punto di
partenza. Il livello del microfono in Windows non viene toccato. Per regolarlo a mano:
`CLAUDIO_AUDIO__AUTO_GAIN=false` e `CLAUDIO_AUDIO__INPUT_GAIN_DB=12`. Le colonne `voce` e `guad`
di `benchmark_latency.py --log` mostrano livello e guadagno di ogni richiesta.

Se la GPU non è utilizzabile, Claudio passa da solo a Whisper `small` su CPU e lo scrive nel log.

Lingue: `[stt] languages = ["it", "en"]` sono quelle tra cui Whisper sceglie. Il riconoscimento
riusa il passaggio dell'encoder della trascrizione, quindi non costa quasi nulla; quando Whisper è
incerto (sotto `language_confidence`, 0.6) Claudio tiene la lingua della richiesta precedente. Con
una lingua sola (`CLAUDIO_STT__LANGUAGES=it` in `.env`) il riconoscimento non c'è proprio. Il prompt di
Whisper è per lingua, in `[stt.prompts]`. I messaggi del programma (console, log, `--check`) sono
in italiano; con `language = "en"` (o `CLAUDIO_LANGUAGE=en`) passano all'inglese.

La voce è Nicola (Kokoro, maschile); `CLAUDIO_TTS__ENGINE=piper` torna a Paola (Piper), che resta
anche come ripiego automatico (`CLAUDIO_TTS__VOICE=it_IT-riccardo-x_low` per Riccardo, maschile e
la più veloce). Le risposte in inglese usano Michael con Kokoro e Ryan (`en_US-ryan-low`) con
Piper, impostate in `[tts.voices.en]`, con lo stesso effetto. La velocità si regola per entrambe con `CLAUDIO_TTS__LENGTH_SCALE`
(più basso = più veloce, 0.77 = +30%, il valore predefinito). Il tono robotico è
`CLAUDIO_TTS__EFFECT`: `ultron_estremo` (predefinito), `estremo`, `ultron`, `jarvis` o `none` per
la voce pulita. L'effetto costa circa 5 ms a frase.

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

Un tentativo mancato ha anche una seconda possibilità (`wakeword.second_chance`): Whisper ascolta
l'ultimo secondo e mezzo, circa 130 ms sulla GPU e solo in quei momenti, e se sente il nome
CLAUDIO si attiva lo stesso. La riga dice cosa ha sentito (`second_chance`, `second_chance_ms`),
così si vedono anche i tentativi mancati che non erano la wake word.

```powershell
uv run python scripts/benchmark_latency.py --log                      # dal vivo
uv run python scripts/record_samples.py commands --repeat 2           # registra i comandi
uv run python scripts/benchmark_latency.py --wav-dir samples/commands  # turbo vs small vs base
uv run python scripts/benchmark_tts.py                                 # voci Kokoro e Piper
uv run python scripts/benchmark_latency.py --replay                    # rifà le richieste salvate
```

`--replay` ripassa i WAV di `logs/audio/` nella stessa pipeline dal vivo (senza eseguire azioni)
e mostra per ognuno il testo di allora, quello di adesso e il comando riconosciuto, poi il tempo
mediano di Whisper. Con `--lead-in 0.4` (secondi) o `--lead-in none` cambia quanto audio detto
prima del bip arriva a Whisper (di base tutto, wake word compresa); con `--beam 1` Whisper prova
una sola ipotesi invece di 5, con `--model large-v3` un altro modello (va prima scaricato con
`download_models.py --whisper large-v3`).

## Sviluppo

```powershell
uv run pytest                 # test veloci (senza microfono né modelli)
uv run pytest -m slow -s      # test sui modelli reali (dopo download_models.py)
uv run ruff check . ; uv run ruff format . ; uv run mypy src
```

La CI (GitHub Actions, Windows) esegue lint, tipi, test veloci e un giro completo
Kokoro e Piper → Silero → Whisper `small` (il ripiego su CPU) → router sui modelli veri.

Struttura: `src/claudio/` (audio, wakeword, stt, tts, router, `app.py` con la macchina a stati),
`config/`, `scripts/`, `tests/`. Modelli in `models/` e log in `logs/`, entrambi fuori da Git.
