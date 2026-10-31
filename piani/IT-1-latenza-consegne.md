# IT-1 – Consegne: latenza (sezioni 5.1, 5.2, 5.3 di `IT-1-consegne.md`)

Ramo locale `it1/latenza` (worktree `C:\Users\JenJo\Desktop\PROGETTI\claudio-it1-latenza`),
partito da `main` v0.4.0, mai pushato. Tre commit: la funzione, le correzioni della revisione,
test a tempo più robusti. **Non unito a `main`**: l'unione la fa il coordinatore con gli altri
rami IT-1 (vedi "Unione con gli altri rami" sotto).

## Fatto

| Cosa | Dove |
|---|---|
| **5.1** Con gli strumenti la prima frase non aspetta più tutte le chiamate: il brain chiude la frase di Claude (uno spazio) prima di eseguire gli strumenti, e il primo pezzo di Jack parte quando il testo di Claude si ferma per 200 ms (con la frase che lo splitter teneva da parte) invece di aspettare 40 caratteri. Il testo dopo gli strumenti torna a essere raggruppato | `src/claudio/brain/claude.py` (`_loop`), `src/claudio/tts/sentence_splitter.py` (`group_stream`, `SentenceSplitter.ended`), `src/claudio/tts/speaker.py`, `[tts] jack_group_pause_ms` |
| **5.1** "Basta" mentre girano gli strumenti ferma subito la risposta (prima aspettava il pezzo dopo) e un pezzo tagliato prima di suonare non conta come detto | `src/claudio/tts/speaker.py` (`interrupt`, `speak`) |
| **5.2** Tempi di ogni chiamata a Claude: primo token, prima parola, fine; la rete a parte (andata e ritorno della connessione anticipata, nessun modello), solo sulla richiesta a cui appartiene e segnata quando apre una connessione nuova | `brain/claude.py` (`_stream`, `_record`, `_warm`, `ask`), `claude_calls`, `api_rtt_ms` in `logs\metrics.jsonl`, riga "TTFT" in `conversazione.md` |
| **5.3** `claudio_command` ha il campo `last`: quando Claude dice che il comando è tutta la richiesta, Claudio dice subito la risposta del comando (la stessa del router, nella lingua della richiesta) e non richiama Claude per dire "Fatto". Senza `last`, dopo altri strumenti o con una richiesta di Sonnet, Claude continua come prima: nulla va perso | `brain/claude.py` (`_run_calls`, `_end_with`, `_confirm`), `pc/tools.py` (`ToolOutcome.spoken`), `router/command_tool.py` |
| **5.3** Jack con la scheda video piena: all'avvio `nvidia-smi` legge la memoria libera; sotto 6000 MB (un gioco) le risposte non aspettano Jack (parla subito la voce di riserva), la console dice perché, l'avviso a voce dice che la scheda era quasi piena. Un worker andato in timeout per la scheda piena riparte quando c'è spazio (prima: Kokoro fino al riavvio di CLAUDIO). Una risposta iniziata con la voce di riserva finisce con quella | `src/claudio/gpu.py` (nuovo, nessuna dipendenza), `__main__.py` (`jack_wait_s`, `startup_notices`), `tts/jack.py` (`WorkerTimeout`, `can_start`, `begin_reply`), `[tts] jack_min_free_vram_mb` |
| **5.3** Bug trovato strada facendo: ogni pezzo di una risposta aspettava di nuovo `jack_first_wait_s` (30 s): con il worker bloccato una risposta in tre pezzi poteva aspettare 90 s. Ora un'unica scadenza dall'avvio del worker | `src/claudio/tts/jack.py` (`start_in_background`, `_wait_for_start`) |
| Test: 36 nuovi (pausa del testo, frase trattenuta, raggruppamento dopo gli strumenti, "basta" durante gli strumenti, tempi per chiamata, rete sulla richiesta giusta, comando detto subito solo con `last`, nessun lavoro perso, scheda piena, nvidia-smi bloccato, attesa unica, una voce per risposta, Jack che riparte) | `tests/unit/test_sentence_splitter.py`, `test_speaker.py`, `test_pc_tools.py`, `test_claude_brain.py`, `test_command_tool.py`, `test_session_log.py`, `test_gpu_busy.py` (nuovo) |
| CHANGELOG (sezione IT-1), README (attesa di Jack), `config\claudio.toml` | |

Risultati: `pytest` 1789 passed, 4 skipped (prima 1753), anche con il PC carico (due suite
insieme, GPU al 95 %); ruff e mypy puliti; `claudio --check` ok. Sotto carico, 1 fallimento su
circa 20 giri dei test a tempo, non riprodotto né identificato: i test a tempo già esistenti
in `test_sentence_splitter.py` e `test_speaker.py` hanno finestre di 20-50 ms.

## Revisione multi-agente (ultracode)

Tre revisori (flusso, brain, Jack/GPU) hanno trovato 15 punti; lo scettico finale si è fermato
per il limite d'uso, quindi li ho verificati io sul codice e con test: 15 veri o sensati,
tutti corretti tranne uno documentato (più schede video: si legge solo la prima). I più
importanti: la frase trattenuta dallo splitter spezzava il primo pezzo in due generazioni;
"basta" durante gli strumenti non fermava nulla fino al pezzo dopo; la scorciatoia di
`claudio_command` poteva perdere una seconda domanda ("abbassa il volume e dimmi che ore
sono"), la risposta di un giro precedente o la proposta di Sonnet.

## Unione con gli altri rami (prova fatta, 30/10 sera)

Conflitti di testo (`git merge-tree`): `it1/voce` (sentence_splitter, speaker, jack, i18n,
claudio.toml, test_speaker), `it1/memoria` (pc/tools.py, brain/claude.py import, CHANGELOG),
`it1/sempreattivo` (CHANGELOG). Nessuno con ascolto, comandi, pronuncia. Quasi tutti "tieni
entrambe le parti". Da fare con attenzione:

1. **Conflitto di significato con `it1/voce`** (senza conflitto di testo!): `voce` mette nel
   flusso il segnale `CUT_SHORT` (risposta tagliata a max_tokens, il "Attra" del 30/10) e lo
   gestisce in `split_stream`; il mio `group_stream` legge il testo da sé, quindi con Jack il
   frammento tornerebbe. In `group_stream.read()`, prima di `if chunk.strip():`, aggiungere:
   ```python
   if chunk is CUT_SHORT:  # cut at max_tokens: never the fragment ("Attra")
       for sentence in splitter.drop_unfinished():
           await ready.put(sentence)
       continue
   ```
   (con `from claudio.protocols import CUT_SHORT`, già importato da `voce`). Tenere sia
   `ended()` (mio) sia `drop_unfinished()` (voce) in `SentenceSplitter`.
2. `tts/jack.py` `_next_event`: tenere il `_poll()` di `voce` ma sollevare il mio
   `WorkerTimeout` (non `WorkerError`), altrimenti Jack non riparte quando la scheda si libera.
3. Test di `voce` `test_a_cut_reply_drops_its_unfinished_sentence`: usa `_sentences(...)`, che
   ora aggiunge uno spazio a ogni pezzo e perde il segnale: passarlo con `_raw((0, c) ...)`.
4. `pc/tools.py` con `memoria`: la mia riga `self._own_command(call, language)` più il loro
   `elif call.name in memory_tools.NAMES`.

Prova completa sul ramo `it1/prova-latenza-voce` (unione già risolta, da cancellare dopo):
1873 test ok, ruff e mypy puliti; senza il punto 1 due test falliscono ("Attra" detto con
Jack). Con `memoria` sopra: 2045 ok, più 2 test instabili sotto carico che passano da soli
(il mio sul raggruppamento, ora con margini più larghi, e
`test_voice_check.py::test_the_log_says_n_d_when_the_pitch_is_unknown` di `voce`, che
dipende dall'ordine dei test). I rami degli altri si sono mossi dopo la prova.

## Misure

| Misura | Prima | Dopo |
|---|---|---|
| Replay della richiesta delle 14:07:15 (tre chiamate, due strumenti) nel vero brain + speaker, tempi API finti presi dal log: "TTS prima frase" | 1972-2026 ms (dal vivo 2104) | **293-311 ms** ("Ci guardo, socio." mentre gli strumenti girano) |
| Chiamate API per un comando storpiato da solo (Haiku vero, 30/10) | 2 | **1** (−0,5-1 s, circa −0,0001 $) |
| Comando + altra domanda (Haiku vero) | 2 | 2 (entrambe le risposte dette) |
| Attesa di una risposta con il worker di Jack bloccato (GPU piena), risposta in 3 pezzi | fino a 90 s | **0 s** con la scheda piena, al massimo 30 s dall'avvio altrimenti |

TTFT vero di Haiku (prompt vero di CLAUDIO, ~5.500-6.000 token in cache): primo token
**469-609 ms** a connessione calda, 687 ms a freddo, **~1030 ms quando la cache va riscritta**
(per esempio dopo un cambio delle definizioni degli strumenti). L'andata e ritorno senza
modello (`models.retrieve`, gratis) è **213-260 ms** (305 a freddo): circa 0,22 s sono rete e
ingresso dell'API, 0,25-0,35 s il modello. Scoperta: l'API manda gli header HTTP **insieme al
primo token**, quindi gli header non separano rete e modello.

Spesa API di questo thread per le misure: **0,0020 $** (4 + 3 chiamate a Haiku).

Dal vivo (30/10 16:30 e 21:40): la scheda aveva 2-4 GB liberi su 16, calcolo al 95-98 %:
riconosciuta come occupata.

## Decisioni prese (5.2: "idee solo se misurate")

- **Interiezione locale "Mh."** mentre Claude pensa: non fatta. Non accorcia l'attesa vera,
  con Jack costa una generazione in più sulla GPU proprio quando serve alla risposta (e un
  cambio di voce), e su 0,5 s di TTFT il guadagno percepito è piccolo.
- **Richiesta partita sulla trascrizione parziale**: non fatta. Risparmierebbe al massimo lo
  STT (144-460 ms) ma con richieste sbagliate e chiamate doppie (costo).
- La rete (~0,22 s) non si accorcia senza cambiare endpoint; la connessione anticipata già
  toglie l'handshake.

## NON fatto

1. **Whisper turbo per i controlli veloci** (primo punto di 5.3): non fatto, fuori dal
   perimetro dato dal coordinatore; oggi STT 144-460 ms, la seconda possibilità ~325 ms.
2. Haiku a volte scrive una frase prima di `claudio_command` anche con `last` ("Ci guardo."):
   Claudio la dice e poi aggiunge la risposta del comando. Innocuo, un po' ridondante.
3. La scheda video piena **dopo** l'avvio (un gioco aperto più tardi) non cambia l'attesa
   delle frasi: vale il timeout per frase di 15 s; però un riavvio del worker ora aspetta
   che la scheda abbia spazio.
4. Più schede video: si legge solo la prima (il tuo PC ne ha una).
5. Non verificato dal vivo: il primo pezzo con gli strumenti e i tempi nuovi alla prossima
   prova; il caricamento vero di Jack con la scheda piena (non l'ho avviato per non
   disturbare chi usava la GPU al 95-98 %).

## Test manuale passo passo

1. Avvia CLAUDIO come al solito.
2. Di' "Claudio, apri la cartella Progetti". Atteso: "Ci guardo, socio." (o simile) subito,
   poi il resto. Nella console la riga `Latenza ... TTS prima frase` sotto ~0,8 s (prima
   1,3-2,1 s); in `logs\sessions\<data_ora>\conversazione.md` la riga `TTFT 594 · 610 ms
   (rete 220 ms)` con un numero per chiamata.
3. Ripeti la richiesta del punto 2 e, appena senti "Ci guardo", di' "basta": la risposta si
   ferma subito (prima finiva il giro degli strumenti).
4. Apri la chat (`uv run claudio --ui`) e scrivi una frase storpiata sulla spesa che il
   router non riconosce, per esempio "quan to o spe zo og gi di intelligenza". Atteso: la
   risposta con la spesa di oggi e nel log **una sola** riga `Claude claude-haiku-5-5: ...
   primo token ... ms` con `claudio_command(command=tell_spend, last=True)`.
5. Facoltativo, GPU occupata: avvia un gioco, poi CLAUDIO. Atteso nella console:
   `Scheda video occupata (libera ... MB ...)`; la prima risposta arriva subito con la voce
   di riserva; Jack parla dalla risposta dopo quella in cui è pronto.
