# CLAUDIO

[English](README.md) | **Italiano**

Assistente vocale per Windows: wake word, trascrizione, comandi e voce girano **sul PC**, gratis.
A quello che nessun comando locale capisce risponde Claude (API Anthropic, Haiku), con un budget
giornaliero. Capisce italiano e inglese e risponde nella lingua in cui gli parli.

```
Microfono ─► openWakeWord ("Hey Claudio") ─► bip
         └─► Silero VAD (fine frase) ─► faster-whisper (GPU) ─► router locale ─► Kokoro, voce Nicola (frase per frase)
```

Stato: **v0.2.0**, Fase 2 (Claude), prima iterazione. La wake word è il nostro "Hey Claudio", addestrato
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
uv sync                                   # senza GPU: uv sync --no-group gpu
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
| "quanto ho speso oggi" | la spesa di oggi per Claude e la quota del budget giornaliero |
| "chiudi la finestra" | chiude la finestra in primo piano, dopo "Confermi?" e un "sì" |
| "metti in pausa la musica", "canzone successiva", "canzone precedente" | tasti media: play/pausa, successivo, precedente (Spotify, YouTube...) |
| "ripeti", "non ho capito" | ripete l'ultima risposta |
| "basta", "stop" | chiude la conversazione; detto mentre Claudio parla, lo interrompe |

Ogni comando funziona anche in inglese: "open Chrome", "open my downloads", "turn up the volume by
10", "mute", "what time is it", "what day is it", "go to sleep", "wake up", "shut down". Per ogni
richiesta Whisper decide se è italiano o inglese, e Claudio risponde in quella lingua con una voce
inglese.

Le frasi stanno in [config/commands.toml](config/commands.toml): aggiungerne non richiede codice
(`phrases_en` e `reply_en` sono quelle inglesi). Tutte le altre richieste ricevono "Questo non lo
so ancora fare" o "I can't do that yet" (in Fase 2 andranno a Claude Haiku).

Ogni azione ha un livello di sicurezza (`level` in `commands.toml`). Il livello 1 parte subito.
Il livello 2 chiede "Confermi?" e Claudio ascolta subito la risposta, senza wake word: "sì",
"ok", "va bene" (o "yes") la eseguono, tutto il resto (un "no", il silenzio, una parola che non
capisce) la annulla. Il livello 3 legge un riepilogo di cosa succederà e solo "confermo" / "I
confirm" la esegue. Ogni domanda e risposta finisce nel log.

Dopo una risposta Claudio resta in ascolto per 6 secondi senza wake word (lo dice un bip): un
altro comando non richiede "Hey Claudio". La finestra si chiude con il silenzio, con "basta" o
con qualcosa che non è un comando (la TV, qualcun altro che parla), con un bip basso e nessuna
risposta. Mentre parla, Claudio continua ad
ascoltare. Una voce vicina al microfono (`barge_in_min_dbfs`, almeno 0,3 s) abbassa subito la
risposta; quando Whisper sente parole la risposta si ferma e quello che hai detto diventa la
richiesta successiva, senza "Hey Claudio" ("basta" / "stop" ferma e basta). Se era rumore il
volume torna normale. La sua voce che rientra dalle casse viene riconosciuta e ignorata, ma con
le cuffie va meglio. Tutto sta in `[conversation]` (`window_s = 0` spegne la finestra,
`barge_in = false` l'ascolto mentre parla, `barge_in_mode` come reagisce).

## Claude (domande)

1. Crea una chiave API su <https://console.anthropic.com> e aggiungi un po' di credito
   prepagato (l'abbonamento Claude Pro non include l'API).
2. Copia `.env.example` in `.env` nella cartella del progetto (accanto a `pyproject.toml`) e
   scrivi la chiave dopo `ANTHROPIC_API_KEY=`. `.env` è in `.gitignore`: non arriva mai su GitHub.
3. Provalo senza microfono: `uv run claudio --text "chi era Leonardo da Vinci"`.

I comandi locali vengono sempre prima e non costano nulla; Claude risponde al resto, frase per
frase. Ogni chiamata è registrata con token (cache inclusa) e costo in `data/spend.db`;
`[cost] daily_budget` (1 USD) avvisa all'80 % e blocca Claude al 100 %. Nella finestra di
conversazione (senza wake word) una frase va a Claude solo se sembra rivolta a lui (una domanda,
"dimmi...", "spiegami...", o il nome): la TV non costa nulla. "basta" ferma la risposta e chiude
lo stream; "ripeti" la ripete. Impostazioni: `[claude]` in `config/claudio.toml`
(`enabled = false` per solo comandi locali). Una domanda a Haiku costa circa 0,0002 USD.

## Personalità (CLAUDIO.md)

Il carattere di Claudio sta in `data/CLAUDIO.md`, creato al primo avvio da `config/CLAUDIO.md`
(Jack il Bello di Borderlands: spaccone, ironico, utile). Modificalo a mano e riavvia; gli
aggiornamenti non lo toccano e i `<!-- commenti -->` non vengono inviati. Viene letto una volta
e va dopo il prompt di sistema, quindi in cache: tienilo corto (`personality_max_chars = 3000`).
Claudio risponde in italiano, in inglese solo se la frase è chiaramente inglese o glielo chiedi.

## Azioni sul PC

Con Claude acceso, Claudio agisce anche su Windows (`[pc]` in `config/claudio.toml`): apre app
dal menu Start, file e cartelle, cerca file con l'indice di Windows, elenca e legge le finestre
(UI Automation, solo testo, ~1500 caratteri), porta una finestra davanti, preme tasti, digita
testo, chiude finestre e sposta file. Più azioni in una risposta, fino a 4 giri per richiesta.

I livelli di sicurezza sono fissi nel codice (`src/claudio/pc/tools.py`), mai scelti da Claude:
aprire, cercare, leggere e i tasti innocui (frecce, Pagina giù, Ctrl+C, Ctrl+T...) partono
subito; digitare, chiudere una finestra, spostare un file e ogni altro tasto fanno chiedere a
Claudio solo "Confermi?" (`confirm_brief = false` gli fa prima dire cosa farà). Solo un sì breve detto a voce lo esegue (mai dalla chat).
File: solo dentro la tua cartella utente (`roots = ["~"]`), mai sovrascritti; programmi e
script non si aprono come file. I gestori di password non vengono mai letti. Ogni azione
finisce nei log (`tools` in `metrics.jsonl`, "Azioni" in `conversazione.md`). Un'azione con un
tool costa circa 0,0003-0,0007 USD.

## Finestra con la chat

Con Claudio acceso (`uv run claudio`) di' **"apri la chat"**, "chat" o "fammi scrivere": la
finestra si apre e Claudio continua ad ascoltare. "chiudi la chat" la chiude, e si chiude anche
quando si spegne Claudio. Le frasi sono in `config/commands.toml` (`open_chat`, `close_chat`),
riconosciute in locale a costo zero. `uv run claudio --ui` apre la stessa finestra da sola (solo
lettura, senza voce).

La finestra mostra le conversazioni: scegli una sessione a sinistra,
cerca in tutte dalla barra in alto, vedi quanto è costata ogni risposta. Legge `data/claudio.db`,
quindi funziona con Claudio acceso (si aggiorna da sola) o spento. I messaggi più vecchi di
`logging.retention_days` (30) vengono cancellati all'avvio di Claudio, insieme ai log delle
sessioni più vecchi (`logs/sessions/`), alle registrazioni in `logs/audio/` (anche le wake word
mancate) e ai vecchi `logs/metrics.jsonl.*`.

## Log delle sessioni

Ogni avvio scrive `logs/sessions/<data>_<ora>/console.log` e `conversazione.md` (richieste,
risposte, latenza, costo): basta mandare quella cartella, o farla leggere a Claude, invece di
copiare la console. Restano le ultime 30 sessioni.

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

Spesa (pronta per la Fase 2, quando risponderà Claude): `[cost]` imposta il budget giornaliero
(`daily_budget`, in dollari, 1,0 di base). Ogni chiamata a Claude verrà scritta in `data/spend.db`
(SQLite) con modello, token in ingresso e in uscita, scritture e letture della cache
(`cache_creation_input_tokens`, `cache_read_input_tokens`) e TTL, e il costo calcolato da
[config/pricing.toml](config/pricing.toml), dove stanno prezzi e moltiplicatori della cache (mai
nel codice). All'80 % Claudio lo dice una volta al giorno; al 100 % smette di chiamare Claude fino
al giorno dopo, mentre i comandi locali continuano a funzionare.

Se la GPU non è utilizzabile, Claudio passa da solo a Whisper `small` su CPU e lo scrive nel log.

Lingue: `[stt] languages = ["it", "en"]` sono quelle tra cui Whisper sceglie. Il riconoscimento
riusa il passaggio dell'encoder della trascrizione, quindi non costa quasi nulla; quando Whisper è
incerto (sotto `language_confidence`, 0.6; per l'inglese `other_confidence`, 0.8, perché l'italiano
veloce a Whisper sembra spesso inglese) Claudio tiene la lingua della richiesta precedente. Una
richiesta sentita come inglese che nessun comando capisce viene ritrascritta in italiano (circa
120 ms, solo in quel caso): "cue the calculator" diventa "apri la calcolatrice".
`retry_in_first_language = false` lo disattiva. Con
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

`compare_stt.py` confronta i motori di trascrizione sulle stesse richieste salvate e conta quante
ognuno ne trasforma in un comando conosciuto, con tempo mediano e p90:

```powershell
uv run python scripts/compare_stt.py                                   # turbo contro lista chiusa
uv run python scripts/compare_stt.py --engines turbo large-v3 lista parakeet --details
```

`lista` è la lista chiusa che CLAUDIO usa dal vivo (`stt.closed_list`): se nessun comando capisce
la trascrizione migliore, Whisper ne propone 5 e vince il primo comando conosciuto. `parakeet`
richiede `uv pip install onnx-asr` (NVIDIA Parakeet TDT 0.6B v3, scaricato al primo uso).
L'ultima riga dice quante richieste scarterebbe il verificatore della wake word
(`wakeword.verify`): l'attivazione resta solo se Whisper ha sentito il nome all'inizio o la
richiesta è un comando locale, così "Hey, ciao, come stai?" finisce nel silenzio e, da v0.2.0,
non costa mai una chiamata a Claude.

Quando Whisper è insicuro di una richiesta che va a Claude (un segmento con `avg_logprob` sotto
`stt.unsure_logprob`, -0.2), Haiku lo sa dall'etichetta della richiesta, con fino a
`stt.alternatives` (2) trascrizioni alternative dall'n-best di Whisper: `[it | unsure: "Apri la
cartella Tesi."] Apri la cartella desi.` Il prompt di sistema chiede a Haiku di dedurre cosa è
stato detto più probabilmente. Si riusa l'n-best della lista chiusa; altrimenti parte un'altra
passata di Whisper (~120 ms), solo su queste richieste. `alternatives = 0` lascia solo la nota.

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
