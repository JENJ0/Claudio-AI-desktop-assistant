# IT-1 – Consegne: latenza (sezioni 5.1, 5.2, 5.3 di `IT-1-consegne.md`)

Ramo locale `it1/latenza` (worktree `C:\Users\JenJo\Desktop\PROGETTI\claudio-it1-latenza`),
partito da `main` v0.4.0, mai pushato.

## Fatto

| Cosa | Dove |
|---|---|
| **5.1** Con gli strumenti la prima frase non aspetta più tutte le chiamate: il brain chiude la frase di Claude (uno spazio) prima di eseguire gli strumenti, e il primo pezzo di Jack parte quando il testo di Claude si ferma per 200 ms invece di aspettare 40 caratteri | `src/claudio/brain/claude.py` (`_loop`), `src/claudio/tts/sentence_splitter.py` (`group_stream`), `src/claudio/tts/speaker.py`, `[tts] jack_group_pause_ms` |
| **5.2** Tempi di ogni chiamata a Claude: primo token, prima parola, fine; la parte di rete a parte (andata e ritorno della connessione anticipata, nessun modello) | `brain/claude.py` (`_stream`, `_record`, `_warm`), `claude_calls` e `api_rtt_ms` in `logs\metrics.jsonl`, riga "TTFT" in `conversazione.md` |
| **5.3** Dopo `claudio_command` nessuna seconda chiamata: Claudio dice subito la risposta del comando (la stessa del router, nella lingua della richiesta). Vale anche per "sì" dopo "spegniti/chiuditi/pausa". I tasti multimediali restano muti come col router | `brain/claude.py` (`_run_calls`, `_end_with`, `_confirm`), `pc/tools.py` (`ToolOutcome.spoken`), `router/command_tool.py` |
| **5.3** Jack con la scheda video piena: all'avvio `nvidia-smi` legge la memoria libera; sotto 6000 MB (un gioco) le risposte non aspettano Jack (parla subito la voce di riserva finché lui non è pronto), la console dice perché e, se Jack non parte, l'avviso a voce nomina la scheda occupata | `src/claudio/gpu.py` (nuovo, nessuna dipendenza), `__main__.py` (`jack_wait_s`, `startup_notices`), `[tts] jack_min_free_vram_mb` |
| **5.3** Bug trovato strada facendo: ogni pezzo di una risposta aspettava di nuovo `jack_first_wait_s` (30 s): con il worker bloccato una risposta in tre pezzi poteva aspettare 90 s. Ora un'unica scadenza dall'avvio del worker | `src/claudio/tts/jack.py` (`start_in_background`, `_wait_for_start`) |
| Test: 23 nuovi (pausa del testo, frase lunga che non è una pausa, spazio che non conta, tempi per chiamata, comando detto subito, giri misti o falliti che tornano a Claude, scheda piena, attesa unica) | `tests/unit/test_sentence_splitter.py`, `test_pc_tools.py`, `test_claude_brain.py`, `test_command_tool.py`, `test_session_log.py`, `test_gpu_busy.py` (nuovo) |
| CHANGELOG (sezione IT-1) | `CHANGELOG.md` |

Risultati: `pytest` 1776 passed, 4 skipped (prima 1753); ruff e mypy puliti; `claudio --check` ok.

## Misure

| Misura | Prima | Dopo |
|---|---|---|
| Replay della richiesta delle 14:07:15 (tre chiamate, due strumenti) nel vero brain + speaker, tempi API finti presi dal log: "TTS prima frase" | 2026 ms (dal vivo 2104) | **311 ms** ("Ci guardo, socio." mentre gli strumenti girano) |
| Chiamate API per una richiesta che finisce in `claudio_command` | 2 | **1** (−0,6-1 s, circa −0,0001 $) |
| Attesa di una risposta con il worker di Jack bloccato (GPU piena), risposta in 3 pezzi | fino a 90 s | **0 s** con la scheda piena, al massimo 30 s dall'avvio altrimenti |

TTFT vero di Haiku (4 chiamate col prompt vero di CLAUDIO, ~5.500 token in cache, costo
**0,00099 $**): primo token 687 ms a connessione fredda, **469-609 ms** a connessione calda.
L'andata e ritorno senza modello (`models.retrieve`, gratis) è **213-260 ms** (305 a freddo):
circa 0,22 s sono rete e ingresso dell'API, 0,25-0,35 s il modello.
Scoperta: l'API manda gli header HTTP **insieme al primo token** (uguali entro 16 ms), quindi
gli header non separano rete e modello; per questo la rete si misura con la connessione
anticipata alla wake word.

Dal vivo, ora (30/10 16:30): la scheda aveva 2074 MB liberi su 16303, in uso al 98 %:
`jack_wait_s` l'ha riconosciuta come occupata.

## Decisioni prese (5.2: "idee solo se misurate")

- **Interiezione locale "Mh."** mentre Claude pensa: non fatta. Non accorcia l'attesa vera,
  con Jack costa una generazione in più sulla GPU proprio quando serve alla risposta (e un
  cambio di voce), e su 0,5 s di TTFT il guadagno percepito è piccolo.
- **Richiesta partita sulla trascrizione parziale**: non fatta. Risparmierebbe al massimo lo
  STT (144-460 ms) ma con richieste sbagliate e chiamate doppie (costo).
- La rete (~0,22 s) non si accorcia senza cambiare endpoint; la connessione anticipata già
  toglie l'handshake (~0,08-0,2 s).

## NON fatto

1. **Whisper turbo per i controlli veloci** (primo punto di 5.3): non fatto, fuori dal
   perimetro dato dal coordinatore; oggi STT 144-460 ms, la seconda possibilità ~325 ms.
2. Se Haiku scrive una frase prima di `claudio_command` ("Alzo il volume, socio."), Claudio
   aggiunge la risposta del comando ("Volume alzato."): un po' ridondante. Si toglie con una
   riga nel prompt di sistema, che però cambia il prefisso in cache; lasciato com'era.
3. La scheda video piena **dopo** l'avvio (un gioco aperto più tardi) non è controllata: lì
   vale ancora il timeout per frase di 15 s e poi la voce di riserva.
4. Non verificato dal vivo: il primo pezzo con gli strumenti e i tempi nuovi nei log alla
   prossima prova; il caricamento vero di Jack con la scheda piena (non l'ho avviato per non
   disturbare chi usava la GPU al 98 %).

## Test manuale passo passo

1. Avvia CLAUDIO come al solito.
2. Di' "Claudio, apri la cartella Progetti". Atteso: "Ci guardo, socio." (o simile) subito,
   poi il resto. Nella console la riga `Latenza ... TTS prima frase` sotto ~0,8 s (prima
   1,3-2,1 s); in `logs\sessions\<data_ora>\conversazione.md` la riga `TTFT 594 · 610 ms
   (rete 220 ms)` con un numero per chiamata.
3. Apri la chat (`uv run claudio --ui`) e scrivi una frase storpiata per il volume che il
   router non riconosce, per esempio "Audio alsa il volu" (26/10). Atteso: risposta "Volume
   alzato." e nel log **una sola** riga `Claude claude-haiku-5-5: ... primo token ... ms`
   (se invece compare `Azione eseguita: volume_up` senza Claude, l'ha capita il router:
   prova un'altra frase storpiata).
4. Facoltativo, GPU occupata: avvia un gioco, poi CLAUDIO. Atteso nella console:
   `Scheda video occupata (libera ... MB ...)`; la prima risposta arriva subito con la voce
   di riserva; Jack parla da quando è pronto.
