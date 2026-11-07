# Passo 0b: le prove di JenJo

Tutte le prove a mano del Passo 0b, passo passo, in un solo posto. Si aggiorna a ogni gruppo: quelle da fare ora stanno sotto **Adesso**, le altre sotto **Dopo** e vengono completate quando il gruppo è pronto.

Finita una prova, scrivi in chat la lettera e cosa è successo. Se qualcosa non torna, fermati e dimmelo: un esito strano è un risultato, non un tuo errore.

## Prima di ogni prova

1. Apri PowerShell nella cartella del progetto:
   ```
   cd C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant
   ```
2. **Claudio parte da solo all'accesso.** Le prove **B** e **C** vogliono Claudio chiuso, perché usano il microfono e i WAV: di' «Claudio, chiuditi» o chiudilo dall'icona nel tray. Controlla che sia chiuso: questo comando non deve stampare nulla.
   ```
   Get-Process python, pythonw -ErrorAction SilentlyContinue
   ```

---

## Adesso: G0

### A. Claudio vero parte e agisce (3 minuti)

**Fatta il 06/11/2026:** Claudio parte, apre il Blocco note, dice l'ora, risponde, e `audible_ms` è nelle righe (locale 915 ms di mediana, n=2). **Il passo 4 è fallito** (3 tentativi su 3): è un guasto da correggere in G2, non di G0. Da rifare dopo G2.

**Perché:** il codice di avvio è stato riscritto e nessun test lo esegue per intero. **Va fatta per prima:** all'accesso Claudio parte da solo, quindi l'avvio lo stai già provando. Costo: meno di 1 centesimo (stima).

1. Accedi a Windows. L'icona compare nel tray, blu mentre carica, verde quando è pronto. **Se non parte o resta blu oltre 2 minuti, fermati e dimmelo.**
2. «Claudio, apri il blocco note»: **il Blocco note si apre davvero.** Se Claudio risponde con un errore, è il guasto che cerchiamo.
3. «Claudio, che ore sono?»: dice l'ora giusta.
4. Con il Blocco note vuoto in primo piano, «Claudio, chiudi la finestra». A «Confermi?» di' «sì»: **si chiude.**
5. «Claudio, quanto dista la Luna?»: risponde a voce.
6. Controlla la metrica nuova:
   ```
   $env:PYTHONIOENCODING='utf-8'; uv run python scripts/benchmark_latency.py --log
   ```
   Cerca la tabella per percorso (locale, conferma, Claude) con la riga `audible_ms`.
7. Controlla la nota accanto ai WAV:
   ```
   Get-ChildItem logs/audio -Filter *.json | Sort-Object Name | Select-Object -Last 1 | Get-Content
   ```
   Deve mostrare `whisper_prompt`, `language`, `gain_db`. Il prompt contiene nomi delle tue cartelle: non incollarlo in chat, dimmi solo se c'è.
8. «Claudio, chiuditi».

**Scrivimi:** A, ok o il passo che non torna, e il numero di `audible_ms` per Claude.

### B. Etichette della sessione del 05/11 (circa 40 minuti, anche a pezzi)

**Fatta il 06/11/2026:** 105 etichette, nessuna riga rotta. Su 42 frasi per Claudio con testo proposto, Whisper sbaglia l'11,1% delle parole.

**Perché:** senza etichette non si può misurare se Claudio capisce quali frasi sono per lui. Servono a G3 e G5. Claudio deve essere chiuso.

```
uv run python scripts/label_wavs.py --session 2026-11-05_17-00 --split dev
```

Per ogni WAV senti l'audio, poi vedi testo proposto e dati della riga.

| Tasto | Cosa fa |
|---|---|
| `Invio` | conferma il testo: la frase era per Claudio |
| `n` | la frase **non** era per Claudio |
| `c` | correggi il testo, poi scegli `Invio` o `n` |
| `r` | riascolta |
| `s` | salta, te lo ripropone la prossima volta |
| `u` | annulla l'ultima etichetta di questa sessione di lavoro e ripropone quel WAV |
| `q` o Ctrl+C | esci: ciò che hai fatto resta |

- Sono 106 WAV. Per riprendere, ripeti il comando anche senza `--split dev`.
- **64 non hanno testo proposto.** `Invio` è rifiutato: usa `n`, oppure `c` per scrivere cosa hai detto.
- Sul testo di una riga `n` puoi sorvolare: conta solo quando la frase è per Claudio.
- I tasti premuti mentre il WAV suona vengono ignorati, così non confermi niente senza leggerlo.
- Premendo `à` si può perdere il tasto successivo. Non scrive etichette sbagliate.
- `u` torna indietro di un'etichetta alla volta, solo in questa sessione di lavoro. **Fanno eccezione le etichette di sessioni precedenti e l'ultima del lavoro:** dopo l'ultimo WAV lo script finisce, quindi quelle si correggono a mano in `samples/corpus/labels.jsonl`. I WAV saltati con `s` nel frattempo vengono riproposti.

Etichette salvate finora:
```
(Get-Content samples/corpus/labels.jsonl | Measure-Object -Line).Lines
```

**Scrivimi:** B, quante ne hai fatte. Vanno raccolte altre 2 sessioni in almeno 2 giorni: vedi **F**.

### C. Frasi di sicurezza (circa 12 minuti, una volta sola)

**Fatta il 06/11/2026, e il test fallisce:** 6 negativi su 64 diventano un sì, 24 sì su 44 capiti. È il guasto che la prova doveva trovare, e lo corregge G2. **Non rifarla adesso:** in G2 il registratore cambia, e registrerai 145 riprese nuove, circa 20-25 minuti.

**Riascolto fatto il 06/11/2026:** voce distorta in tutte le riprese controllate.

| File in `samples/safety/` | Cosa hai sentito |
|---|---|
| `unclear/s__0.wav` | distorta e quasi tagliata |
| `no/non_va_bene__1.wav` | come la prima, anche peggio |
| `unclear/si_fa_tardi__1.wav` | distorta, ma la «r» si sente chiaramente |
| `no/certo_che_no__1.wav` | chiara, ma distorta |
| `no/va_bene_ma_non_adesso__0.wav` e `unclear/si_fa_tardi__0.wav` | chiare, ma distorte |

**Per G2, prima di registrare (2 minuti):** ascolta 3 WAV dal vivo e dimmi se suonano distorti come queste riprese. Per aprirne uno, in PowerShell:
```
Get-ChildItem samples\corpus\2026-11-06_12-48\audio -Filter *_ok.wav | Select-Object -First 3 | ForEach-Object { Start-Process $_.FullName }
```

Quello che segue è la prova come l'hai fatta.

**Perché:** finora i «sì» sono stati provati solo su testo scritto. Qui si prova come Whisper li sente, e che nessun «non va bene» diventi un «va bene». Claudio deve essere chiuso. Usa il microfono che usi di solito con Claudio.

```
uv run python scripts/record_samples.py safety --repeat 2
```

Sono 54 frasi × 2, cioè 108 riprese.

- A ogni riga: premi `Invio`, poi **di' la frase**. Parte a registrare subito dopo `Invio`.
- `r` + `Invio`: rifà la frase **precedente**, quella scritta sulla riga `rifai:`. Di' quella, non la frase che avevi appena letto.
- `q` + `Invio`: esci. Ripeti lo stesso comando e registra solo le riprese mancanti.
- **«sì?» e «confermo?» vanno detti come domande vere, con il tono che sale.**
- Le frasi più importanti sono in fondo alla lista: arriva fino alla fine.

Poi lancia il test su Whisper vero:
```
uv run python -m pytest tests/integration/test_safety_audio.py -m slow -q -s
```

Esito atteso: `negativi ascoltati: 64`, e il test passa. Stampa anche quanti «sì» ha capito su 44.

**Se fallisce** su un negativo diventato un «sì», non è un tuo errore: è un guasto di sicurezza che voglio vedere. **Scrivimi:** C, l'ultima riga del test, e il nome dei file che falliscono.

### D. Clip del gate di replay

**Fatta il 06/11/2026:** tabella di 42 clip approvata (`samples/gate/clips-proposal.md`, fuori da git). Registrazione: 0,0035 $, 21 cassette. Esito: 38 superate, 4 guasti noti, 0 fallite; due esecuzioni senza chiave identiche.

Per rifarla, a Claudio chiuso e senza spesa:
```
uv run python scripts/replay_gate.py --report
```
Esito atteso: la riga che inizia con `Gate di replay:` dice `0 fallite`, `0 cassette mancanti`, e il comando esce senza errori. Una modifica a `data/CLAUDIO.md` o ai tool fa mancare tutte le cassette: lo dice in chiaro, e serve rilanciare con `--record`, che chiede conferma e costa pochi centesimi.

### E. Diff e commit

**Fatta il 06/11/2026:** diff e test mostrati, commit locali di G0 con il tuo ok. Niente push senza un ok a parte.

### F. Altre due sessioni per il corpus (2 volte, in giorni diversi)

**Perché:** servono circa 300 frasi su 3 sessioni in 2 giorni, con dev e test separati per sessione. Il test finale va fatto su sessioni mai usate per sviluppare.

1. Usa Claudio normalmente per 15-20 minuti, parlando anche con altri o da solo, come fai di solito.
2. Chiudilo: «Claudio, chiuditi».
3. Trova il nome della sessione, la cartella più recente:
   ```
   Get-ChildItem logs/sessions -Directory | Sort-Object Name | Select-Object -Last 1 -ExpandProperty Name
   ```
4. Importala ed etichettala. **La seconda sessione usa `--split dev`, la terza `--split test`.**
   ```
   uv run python scripts/label_wavs.py --session NOME --split dev --import
   ```

Se due avvii cadono nello stesso minuto, lo script rifiuta l'importazione: dimmelo.

---

## Adesso: G1

### G. Azioni vere (circa 10 minuti, meno di 1 centesimo)

**Perché:** i test usano una scrivania finta. Qui si prova il Windows vero: finestre, Blocco note, appunti, tempi. **Claudio deve essere acceso**, con la versione nuova: chiudilo se era già partito («Claudio, chiuditi») e riavvialo. Se un passo non torna, fermati e scrivimi quale: è un risultato.

Dopo ogni passo puoi guardare l'ultima riga delle metriche:
```
Get-Content logs\metrics.jsonl -Tail 1 | ConvertFrom-Json | Select-Object cut, cut_retry, promised, tool_ms, audible_ms, m1_ms, cost
```

**Il testo lungo, il cuore di G1**
1. Blocco note chiuso. «Claudio, apri il blocco note e scrivici un piano di dieta dettagliato per tutta la settimana». Aspetta anche 15-20 s in silenzio: è il nuovo tentativo.
   - Atteso: **il testo intero compare**, non si ferma a metà, e Claudio dice che è scritto.
   - Nella riga: `cut` è `tool_use` e `cut_retry` è `ok`, se il testo era davvero lungo. Vuoti vanno bene se Haiku è stato breve: in quel caso ripeti chiedendo «molto dettagliato, ogni pasto».
   - **Se Claudio dice che il testo non si trova nella finestra, o che non è verificato, ma lo vedi: è il guasto che cerco.** Dimmelo: vuol dire che il Blocco note non si fa rileggere.
2. Con quel Blocco note davanti: «Claudio, scrivici ciao». Scrive. Può dire che non è verificato, perché è un testo corto: è previsto.

**La finestra sbagliata**
3. Blocco note davanti. «Claudio, apri YouTube e scrivi ciao». Atteso: **chiede «Confermi?»** e non scrive nel Blocco note senza un sì. Rispondi «no».
4. «Claudio, apri la calcolatrice e scrivici ciao». Atteso: non deve finire testo in un'altra finestra. Anche un «non ho scritto» va bene.
5. Riduci Chrome a icona. «Claudio, porta davanti Chrome». Atteso: viene davanti.
6. Metti Chrome dietro un'altra finestra e ripeti. Atteso: viene davanti.
7. Nel Blocco note, con del testo non salvato, apri Salva con nome senza chiuderlo. «Claudio, porta davanti il blocco note». Scrivimi cosa dice: non so ancora se una finestra di dialogo aperta si legge come «non davanti». Poi annulla Salva con nome.

**L'interruzione, il guasto più serio trovato**
8. «Claudio, scrivi un piano lungo nel blocco note». Mentre aspetti, prima che compaia il testo: «Claudio, lascia stare». Atteso: **non scrive nulla.**
9. Subito dopo, «ok, grazie». Atteso: nessuna scrittura: prima questa frase poteva valere come un sì a una domanda mai sentita.
   - Noto, non è un tuo errore: Claudio non dice che l'azione si è fermata. Se la richiesta nuova è un comando locale, il silenzio è totale. Lo sistema G2.

**Gli appunti**
10. Copia un'immagine con Win+Maiusc+S. Ripeti il passo 1 con un testo lungo. Poi apri Paint e fai Ctrl+V. Atteso: **la tua immagine è ancora lì.**

**I segreti**
11. Crea un file **finto**, senza segreti veri:
    ```
    "CHIAVE_DI_PROVA=non-e-una-chiave-vera" | Set-Content $HOME\prova.env
    ```
    «Claudio, apri il file prova punto env». Atteso: si apre, e Claudio non ripete la riga. Poi chiudilo e cancellalo:
    ```
    Remove-Item $HOME\prova.env
    ```
    Controlla che non sia nell'ultima risposta in chat.

**I tempi**
12. «Claudio, apri la cartella Download». Dimmi se la pausa prima di «aperta» ti sembra troppa: dopo un'apertura ora aspetta fino a circa 1 s la finestra. Il nome italiano «Download» può non coincidere con il titolo, e in quel caso l'attesa è piena.
13. Se hai un documento Word molto grande, «Claudio, apri» quel documento. Dimmi se la pausa ti sembra lunga: la lettura di un documento enorme non ha un tetto. Se non ne hai uno, salta questo passo.
14. A fine sessione:
    ```
    $env:PYTHONIOENCODING='utf-8'; uv run python scripts/benchmark_latency.py --log
    ```
    Copiami la tabella per percorso.

**Scrivimi:** G, i passi che non tornano, cosa hai sentito ai passi 1 e 7, e la tabella del passo 14.

---

## Dopo: si preparano quando il gruppo è pronto

| Gruppo | Prova dal vivo |
|---|---|
| **G2** | Storia, poi «che ore sono?»: risponde e riparte con «Dicevo», anche se la storia finiva con una domanda. Storia, poi una richiesta a Claude: niente «Dicevo». «Continua»: ridice il testo mancante esatto. «Basta» mentre Claude sta ancora pensando: si ferma. «Chiudi la finestra», poi «sì» subito dopo «Confermi?»: si chiude. Di nuovo, senza rispondere, poi «Claudio, confermo»: si chiude. «Che ore sono?», poi una domanda che ci si riferisce: Claudio sa di cosa parli |
| **G3** | «Scrivi.» da solo è accettato. Il filtro si attiva solo se passa il test su almeno 40 frasi per tipo |
| **G4** | Sigle: AMD, RAM, NVIDIA. Pagina di ascolto alla cieca: scegli tu la voce |
| **G5** | Frasi nuove per il test finale, mai usate per sviluppare |
| **G6** | Aprire per nome una cartella fuori da `~` e quella di Steam. Riassunto del file aperto in VS Code. Tasto di stop globale |
| **G7** | Almeno 20 richieste vere per la latenza prima e dopo |
| **G8** | Accensione da freddo: Jack arriva da solo senza riavviare Claudio |

---

## Gate di rilascio di v0.4.2 (dopo G6)

Sessione di 10-15 minuti. Riprende i punti ancora aperti del Passo 0, `IT-2-passo-0-report.md` sezione 7. La numerazione è la stessa di quella sezione.

**Falsi sì.** Di' «chiudi la finestra» su un Blocco note vuoto.
- [ ] 1. A «Confermi?» rispondi «si chiama Anna»: non si chiude.
- [ ] 2. Ripeti e rispondi «sì, grazie mille»: si chiude.
- [ ] 3. Con un ricordo di prova, «dimentica mio padre», poi «sì ma dopo»: il ricordo resta.

**Stop.** Durante una storia lunga:
- [ ] 4. «No, Claudio, non parlare. Claudio, stop.»: si ferma subito, senza risposta né spesa.
- [ ] 5. «Claudio, apri la calcolatrice e poi basta»: non deve fermarsi come uno stop.

**Spento.**
- [ ] 6. «Claudio, spegniti», poi di' una parola insolita vicino al microfono, poi «Claudio, accenditi». Cerca la parola in `logs/`: non deve esserci, e non devono esserci `*_off.wav` nuovi.
- [ ] 7. Con «Confermi?» in attesa spegni dall'icona del tray, poi «sì»: non deve succedere nulla.
- [ ] 8. «raccontami una storia», poi «Claudio, spegniti»: niente «Dicevo:» dopo «Mi spengo».
- [ ] 9. «Claudio, spenitri», poi a «Confermi?» «Claudio, spegni»: si spegne senza altre domande.

**Microfono.**
- [ ] 10. Stacca l'unico microfono da fermo e riattaccalo: riprende da solo.
- [ ] 11. Con due microfoni, resta in silenzio 60 s, poi parla nell'altro.
- [ ] 12. Con `input_device = ""`, cambia il microfono predefinito di Windows.
- [ ] 13. Stacca un microfono USB durante una risposta.
- [ ] 14. Disattiva il microfono da Gestione dispositivi: dopo 2 s deve riprovare.
- [ ] 15. Con `input_device = "auto"`, fai avvenire il cambio di microfono mentre Claudio parla: nessun errore -9988.

**Voce.**
- [ ] 16. «raccontami una storia», poi «basta» dopo 3 s: si ferma in circa 100 ms, senza bip d'errore.
- [ ] 17. Storia, poi «che ore sono?»: risponde e riprende con «Dicevo:». Oggi può fallire se la storia finisce con una domanda: lo corregge G2.
- [ ] 18. Fai 10 richieste fisse e guarda `tts_first_ms` e «Take di Jack scartata».
- [ ] 19. Chiudi il worker di Jack durante una risposta: la voce di riserva la dice una volta sola.
- [ ] 20. «basta» mentre un pezzo è trattenuto dal controllo del take, poi una nuova richiesta: si deve sentire.
- [ ] 21. Una risposta di un solo pezzo oltre 15 s: nessuna voce di riserva si intromette.
- [ ] 22. `scripts/jack_takes.py` una volta sul worker vero.

**GPU.**
- [ ] 23. Dopo qualche richiesta, controlla che in `logs/metrics.jsonl` ci siano `gpu_free_mb` e `gpu_busy_pct`.

**Fatto il 05/11:** la latenza del Passo 0 non è peggiorata (M1 su Claude da 4947 a 3062 ms, locale da 1326 a 1004 ms, 44 richieste). Il confronto non è alla pari: il periodo precedente aveva sessioni in gioco.
