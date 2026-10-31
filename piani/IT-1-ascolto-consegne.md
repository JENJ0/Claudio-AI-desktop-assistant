# IT-1 Ascolto – Consegna (sezioni 1.1-1.4, 2.2-2.4 di IT-1-consegne.md)

Stato al 31/10 notte: ramo `it1/ascolto` (worktree `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\claudio-it1-ascolto`),
ribasato su `main` v0.4.0, **solo locale, nessun push**. 1790 test passano (+11 lenti con i modelli
veri), ruff e mypy puliti, `claudio --check` ok, revisione con più agenti fatta e corretta (sotto).
Decisioni usate: 8 (tic leggero da spento) e 9 (WAV scartati salvati con un limite) = sì.

Tutte le misure sono sui file della prova del 30/10 (`logs\audio\20261030-14*.wav`,
`logs\metrics.jsonl`, `logs\sessions\2026-10-30_14-00\`), con i modelli veri (wake word, Silero,
Whisper large-v3 su GPU). Gli script di misura sono nello scratchpad della sessione, non nel repo.

---

## Cosa cambia, punto per punto

### 1.1 La wake word scattava da sola ~0,5 s dopo ogni bip (bug)
- **Causa confermata** in openWakeWord 0.6.0: `AudioFeatures.reset()` riempie la finestra del modello
  con le feature di 4 s di rumore casuale; il detector passava il silenzio *prima* del reset.
- **Correzione** (`wakeword/detector.py`): prima `Model.reset()`, poi 25 frame di silenzio.
- **Misura**: 100 tratti tranquilli della sessione subito dopo un reset: false wake **19 → 0**.
  Le wake vere dette almeno ~0,5 s dopo il reset hanno lo stesso punteggio di prima.
- **Test**: unitario con un modello finto che si comporta come openWakeWord, e di integrazione
  (`tests/integration/test_wake_reset.py`, lento) con i modelli veri: falliscono col codice vecchio.

### 1.2 "Claudio" detto da solo: la seconda possibilità sentiva "Grazie."
- **Prompt** nella seconda possibilità (solo le frasi di `[stt.prompts]`, senza i nomi delle
  cartelle). Sulle 21 wake della sessione il nome è sentito **12 volte su 12 quando c'era voce**
  (senza prompt 7). Nessun nome falso su 92 finestre senza nome (voce di Jack, frasi nella finestra).
- **Controllo Silero prima di Whisper** (`[wakeword] min_voice_ms = 96`): senza voce Whisper non
  parte. Sui dati: ogni nome vero aveva ≥ 128 ms di voce, le 7 wake su silenzio 0 ms.
- **Mai una wake su silenzio**: la regola "unclear" vale solo se c'era voce, e in
  `config\claudio.toml` è spenta (`unclear_min = 1.0`): con il prompt non serviva più, l'unico caso
  della sessione era una wake falsa.
- **WAV scartati** in `logs\audio\rejected\` (limite `[logging] keep_rejected = 200`, ~80 KB l'uno,
  cancellati anche loro dopo `retention_days`), con il motivo nel nome: `no_voice`, `no_name`. Sono
  i negativi per riaddestrare "claudio" (1.5).
- **Comando già detto**: se la seconda possibilità ha sentito un comando intero (es. "apri Chrome",
  da spento solo "accenditi") e il lead-in non lo trova più, la richiesta è quell'audio.

### 1.3 Nome e richiesta d'un fiato: si svegliava quando avevi già finito
- **Correzione** (`audio/vad.py`): la frase parte dal lead-in anche se era già finita; senza altre
  parole si chiude 0,9 s dopo l'ultima parola (`LEAD_IN_END_S`). "Claudio" da solo seguito da una
  pausa passa dal RELISTEN già esistente.
- **Misura**: replay delle 21 wake: 14:06:40 ("Claudio, spegniti." perso in timeout) ora arriva a
  Whisper come "Claudio, spegni." e la lista chiusa lo corregge in "Claudio, spegniti." → spento
  (verificato end-to-end con i modelli veri). Le altre wake finiscono come prima (±0,1 s).

### 1.4 Da spento "accenditi" andava ripetuto molte volte
- Ogni rilevazione e ogni near miss passa dalla seconda possibilità: si sveglia **solo** se Whisper
  sente il nome o "accenditi"; allora suona un **tic leggero** (corto, 40 % del volume dei bip).
  Nessun altro suono da spento.
- Da spento il router capisce solo "accenditi" (`LocalResponder.understands`): retry di lingua e
  lista chiusa lo cercano anche nelle frasi capite male (14:02:24 "Grazie a te." → "Dai, accenditi.").
- Il log mostra la soglia davvero in uso e quando cambia ("Wake word, soglie da spento: ...").
- Replay end-to-end (14:06:40 e 14:07:01): spento, poi tic e "Eccomi, sono acceso.".

### 2.2 Mentre Claudio parla quello che dici diventava frammenti
- Le fette di `barge_in_max_s` contano solo la voce (il pre-roll è 250 ms prima della voce, prima
  800 ms che riempivano 2/3 della fetta); ogni fetta è controllata solo per "basta"; la frase intera
  è giudicata quando finisce, con il prompt.
- **Misura** su 18 frasi vere della sessione fatte passare dall'ascolto del barge-in: **13 giuste e
  intere contro 6** (es. "Claudio, spiegami in breve che cos'è un mutuo." prima era "Grazie a tutti."
  + "in breve che cos'è un").
- Quello che senti mentre parla e non è per lui va in `logs\audio\rejected\` (`barge_ignored`), con
  testo e dBFS nel log.

### 2.3 Soglia del barge-in prima del guadagno
- `barge_in_min_dbfs` ora si confronta con il livello **prima** del guadagno (AutoGain si è mosso tra
  4,8 e 12,5 dB in 8 minuti). Nuovo valore **-42** = il vecchio -30 a +12 dB. Dati: tu sopra la
  risposta -22..-31 dBFS, l'eco al massimo -47. Anche il livello dell'eco imparato è "prima del
  guadagno".

### 2.4 Sordità dopo il bip o la risposta
- La finestra della conversazione tiene l'ultimo secondo che non ha ascoltato; se la tua voce c'era
  già quando ricomincia ad ascoltare (entro 300 ms), la parte finale **più forte dell'eco** diventa
  l'inizio della frase. **Mai** nella risposta a "Confermi?": lì un "sì" perso lascia solo l'azione
  non fatta, mentre l'eco della domanda non deve mai confermare.
- In `metrics.jsonl`: `deaf_speech_ms` (voce sentita durante la sordità) e `deaf_joined_ms`.
- Non fatto: la misura automatica del ritorno all'avvio (punto 2 della proposta). Con la cuffia
  attuale (uscita "Altoparlanti (Wireless Stereo Headset)") l'eco è minima; il buffer copre già
  la parte persa. Da rivalutare con `deaf_speech_ms` dopo qualche giorno.

---

## Revisione con più agenti (31/10)

Tre revisori (macchina a stati, logica audio, sicurezza e configurazione) e un verificatore che
ha provato a smontare ogni segnalazione con simulazioni sul codice vero: 15 segnalazioni, **14
confermate** (molte uguali tra loro), tutte corrette con un test che fallisce sul codice di prima:

- **Barge-in a fette**: dopo un taglio la frase si chiudeva al primo calo del VAD e tornava a
  pezzi; ora si chiude solo su silenzio vero, con quello che hai detto dopo il taglio e senza buchi
  tra le fette.
- **Frase ancora in corso quando la risposta finisce**: andava persa; ora è giudicata a fine
  risposta e, se è per Claudio, prosegue come richiesta (senza bip).
- **"e basta" dentro una richiesta**: una fetta ferma la risposta solo se è proprio "basta".
- **Seconda possibilità superata da una wake vera**: veniva salvata come "rifiutata" (negativo
  sbagliato per il riaddestramento); ora solo una riga nei log.
- **`min_voice_ms = 0`**: un testo vuoto su 0 ms di voce non sveglia mai Claudio.
- Minori: modalità "stop" con le fette, `speech_start` nelle metriche.

Non corretto, di proposito: la frase intera sopra la risposta è trascritta con il limite corto di
token della seconda possibilità. Serve solo a giudicare (nome, comando, "basta"); se è una
richiesta, viene ritrascritta intera dopo. Cambiarlo toccherebbe `stt`, che è di un altro thread.

---

## Configurazione cambiata (`config\claudio.toml`)

| Chiave | Prima | Ora | Perché |
|---|---|---|---|
| `[wakeword] min_voice_ms` | – | 96 | Silero prima di Whisper |
| `[wakeword] unclear_min` | 0.6 | 1.0 (spenta) | con il prompt il nome si sente |
| `[conversation] barge_in_min_dbfs` | -30 dopo il guadagno | -42 prima del guadagno | indipendente da AutoGain |
| `[logging] keep_rejected` | – | 200 | WAV scartati, solo sul PC |

Il tuo `.env` non tocca nessuna di queste chiavi.

## Prova manuale (dopo l'unione in main)

1. Apri PowerShell e vai nel progetto:
   `cd C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant`
2. Avvia: `uv run claudio` e aspetta "Sono pronto".
3. **1.1**: di' "Claudio, che ore sono?", aspetta la risposta e resta zitto 10 s: non deve partire
   nessun bip da solo.
4. **1.2**: di' solo "Claudio", aspetta il bip, poi "apri la chat".
5. **1.3**: di' d'un fiato e veloce "Claudio spegniti" senza aspettare il bip: deve spegnersi.
6. **1.4**: da spento di' 5 volte "Claudio, accenditi" (la prima volta deve bastare; senti un tic
   leggero e poi "Eccomi, sono acceso"). Poi da spento parla normalmente 1 minuto: nessun tic.
7. **2.2**: chiedi "Claudio, raccontami una storia" e mentre parla di' "Claudio, apri la chat":
   deve fermarsi e aprire la chat; poi riprova con "basta".
8. **2.4**: dopo una risposta, comincia a parlare subito sopra il bip della conversazione ("apri la
   chat"): la frase deve arrivare intera.
9. Chiudi con "Claudio, chiuditi". Guarda `logs\audio\rejected\` (i WAV scartati) e manda il log
   della sessione (`logs\sessions\<data_ora>\console.log`).

## Limiti e cose aperte

- Dopo un reset il modello riparte da silenzio digitale: un "Claudio" iniziato entro ~0,4 s dalla
  fine della sordità ha un punteggio più basso (sui dati, 7-10 su 26 sopra soglia contro 14). Dopo
  0,5 s nessuna differenza. Era molto peggio il rumore del reset vecchio, che inventava wake.
- Nei replay il punteggio del modello dipende da cosa ha sentito prima: per le prove end-to-end la
  wake è stata messa al momento in cui era scattata dal vivo.

- 2.2(c) (passare la frase sentita sopra la risposta dal recupero acustico di 2.1) dipende dal
  thread dei comandi: non fatto qui.
- La coda di 100 ms dopo la frase (endpointer) taglia a volte una fine molto debole ("spegni-ti");
  la lista chiusa la recupera. Tenerne 400 ms aiuterebbe solo quel caso su 33: lasciata così.
- Il modello "claudio" su rumori non vocali (1.5) ora non costa più Whisper né wake; il
  riaddestramento resta da fare con i WAV di `logs\audio\rejected\`.
- Conflitti previsti all'unione (git merge-tree): `app.py` e `__main__.py` con `it1/comandi`,
  `router/responder.py` con `it1/memoria`, `i18n.py` con `it1/voce`. Gli altri rami si uniscono
  senza conflitti.
