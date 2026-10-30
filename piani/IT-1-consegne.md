# IT-1 – Consegne (unione delle cinque consegne IT-0)

Stato al 30/10: ramo locale `it-0` = `main` 9e23d36 + `it0/avvio`, `it0/microfono`,
`it0/autoattivazione`, `it0/incerte`, `it0/fluidita`. Nessun push. 1753 test passano
(+8 lenti), ruff e mypy puliti, `claudio --check` ok, `--text "che ore sono"` ok.

Ogni punto è verificato sul codice unito (grep o test), non solo copiato dalle consegne.
Ordine: le priorità di CLAUDE.md (affidabilità, costi, latenza, sicurezza, codice).

## Cambiato nell'unione (da sapere prima delle prove)

- **"spegniti" ora spegne, non chiude** (thread avvio): resta in ascolto solo di "Claudio,
  accenditi". Chiude il programma "chiuditi". Il thread incerte era scritto sul vecchio
  significato: "Hai detto: spegniti?" + "sì" ora **spegne** (non dice più "A presto.").
- **Da spento Claude non stima più le frasi incerte** (correzione d'integrazione, con test):
  da spento nulla va a Claude, nemmeno la stima.
- Tutti i suoni passano dal filtro "non sentirsi" (autoattivazione) e da spento restano muti
  (avvio): `_beep_now()` usa `_sound()`.

## A. Serve JenJo al PC (bloccano la chiusura di IT-1)

1. **Prova dal vivo unica sulla versione unita** (sotto, "Test manuale"). Le soglie nuove sono
   tarate solo sui log: guess 0,8 / 0,5 / 65, `off_threshold_boost` 0,1, `echo_tail_ms` 400,
   `ECHO_VOICE_MS` 1000.
2. **Registrare l'avvio all'accesso** (`scripts\install_autostart.ps1`): oggi non c'è nessun
   task "Claudio" (verificato con `schtasks`). Da verificare dopo: che "Riavvia in caso di
   errore" scatti con codice d'uscita ≠ 0 e che il `.venv` parta fuori dalla app Claude.
3. **Ascolto di Jack R contro K** e scelta del seme (`jack_variants.py --only R K`, poi
   `--seeds 1 7 42 99`). Le misure dicono K (seme 7 + top_p 0,9).
4. **Decisioni rimaste**:
   - "basta" dedotto tra 50 e 80 %: chiedere "Hai detto: basta?" (oggi) o fermarsi subito?
   - "spegniti" dedotto sopra il 90 %: chiedere sempre (oggi) o spegnersi subito?
   - CLAUDE.md dice ancora "spegniti: chiude il programma": va aggiornato (è il tuo documento).
   - Tag di versione dopo la revisione (CLAUDE.md: ogni iterazione chiude con un tag).

## B. Affidabilità

1. **Istanza singola** (nessun lock nel codice, verificato): con l'avvio automatico un
   `uv run claudio` aperto a mano usa lo stesso microfono e risponde due volte. Mutex di
   Windows con nome. Piccolo, da fare prima di registrare il task.
2. **Cancellazione d'eco vera (AEC, es. WebRTC APM)** al posto di "ignora per tempo e per
   testo". Risolverebbe insieme quattro problemi aperti: la frase iniziata sopra la fine della
   risposta perde l'inizio (con le casse); barge-in con le casse fermato da un'eco storpiata che
   contiene il nome; le frasi dette mentre Claudio parla non stimate; la risposta molto rapida
   a "Confermi?" (entro ~0,4 s dal bip) persa. Costo zero, lavoro medio: prima una prova.
3. **Tarare la coda dell'eco sulle casse Odyssey (HDMI)**: PortAudio non vede il ritardo del
   monitor. Dati già nei log nuovi: `listened_ms`, `voice_ms`, "Uscita audio: ... (latenza N ms)".
4. **Richieste vere scartate come falsa wake word** ("Come ti chiami?" 0,995, "Raccontami una
   storia lunga su un robot." 0,935): punteggio alto + voce vicina dovrebbe bastare a tenerle.
   Regola della verifica wake word, non della stima.
5. **Icona nella system tray** (fase 5, mancante): oggi da avvio automatico non c'è nulla di
   visibile. Serve `pystray` (dipendenza nuova da verificare e fissare).
6. `claudio --check` non controlla che il modello Whisper scelto (ora large-v3, ~3 GB) sia
   scaricato (verificato in `checks.py`).

## C. Latenza

1. **Claude alla prima parola (~1,2-1,4 s)**, ora la parte più grande di M1: misurare il TTFT
   di Haiku nel log (oggi non c'è), poi provare un'interiezione locale ("Mh.") mentre Claude
   pensa e la richiesta partita sulla trascrizione parziale.
2. **large-v3 dal vivo**: misurare M1/M2. Seconda possibilità e conferma della wake word ora
   costano ~200 ms invece di ~100; se pesa, turbo solo per i controlli veloci (+1,6 GB VRAM).
3. **Frasi recuperate da Claude**: +1-1,5 s (quasi tutta rete); dopo `claudio_command` Claude fa
   una seconda chiamata solo per dire "Fatto" (+0,6-1 s): dire subito la risposta del comando.
4. **Worker di Jack con la GPU occupata**: oggi, con un gioco aperto (GPU al 90 %), non ha
   risposto in 90 s e Claudio è passato a Kokoro (comportamento previsto, ma lento). Da
   rivedere: avviso a voce o attesa più corta se la GPU è piena.

## D. Qualità di voce e ascolto

1. **Timbro di Jack tra primo e secondo pezzo della stessa risposta**: il seme li avvicina ma
   non li lega. Idea: modo ICL di Qwen3 con un clip di Jack calmo come riferimento.
2. **Microfono migliore**: il limite è l'hardware (cuffie a banda stretta con gate, webcam
   rumorosa). Consiglio: microfono USB a filo vicino alla bocca.
3. **Banco STT**: Parakeet TDT 0.6B v3 (serve `uv pip install onnx-asr`), rumore vero invece di
   zeri dopo la frase, lista chiusa simulata.
4. Bassa priorità: Smart Turn v3 (fine turno), volume del microfono via pycaw, "Chrome"
   pronunciato male a volte (da controllare con il monitor della voce), "Claudio, grazie.
   Claudio, grazie." non filtrato senza Silero.

## E. Pulizia dopo la tua revisione (azioni da approvare)

- Unire `it-0` in `main` (locale) ed eventualmente pushare: decisione tua.
- Cancellare i worktree `claudio-it0-*` e i rami `it0/*` (già tutti dentro `it-0`), le patch
  `it0-*.patch` in PROGETTI e la cartella `it0-microfono-lavoro`.
- Le registrazioni del banco (`claudio-it0-microfono\data\stt_testset*`) vanno spostate in
  `Claudio-AI-desktop-assistant\data\` prima di cancellare quel worktree, se servono ancora.

## Test manuale unico (versione unita, cartella principale, nessun `$env:` necessario)

`uv run claudio` dalla cartella `Claudio-AI-desktop-assistant` sul ramo `it-0`.

1. All'avvio "Sono pronto": nessuna attivazione da sola dopo. Nel log: "Uscita audio: ...
   (latenza N ms)".
2. "Hey Claudio, che ore sono?" poi silenzio: bip basso, 30 s senza attivarsi da solo.
3. "Claudio, chiudi la finestra" → "Confermi?": non dire niente → nessuna azione. Riprova e di'
   "sì" subito dopo il bip: deve partire.
4. "Claudio, apri la chat", "chiudi la chat", poi nella finestra "apri la chat": la apre.
5. "Claudio, che ore sono?", poi senza nome "spegni ti" → "Hai detto: spegniti?" → "no" →
   "Va bene, lascio stare.".
6. "Claudio, spegniti" → nessun suono né risposta a "che ore sono" detto con il nome;
   "Claudio, accenditi" → "Eccomi, sono acceso.". "Claudio, chiuditi" → "A presto." e chiude.
7. Con musica: "Claudio, che ore sono?" poi "prossima" → canzone dopo (o "Hai detto: ...?").
8. 5-6 domande a Claude, una lunga ("raccontami una storia lunga"): nel log `Latenza M1`
   (atteso sotto 2 s) e `Voce Jack` (gli WARNING sono le risposte da ascoltare).
9. TV o una persona vicino al microfono per un paio di minuti: mai "Hai detto".
