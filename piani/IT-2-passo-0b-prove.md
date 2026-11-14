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

**Per G2, prima di registrare:** la prova H0 più sotto.

Quello che segue è la prova come l'hai fatta. **Superata dalla H1:** il comando di test qui sotto ora legge `samples/safety_v2/` e parte solo con `CLAUDIO_SAFETY_V2_RUN=1`, che imposta il coordinatore.

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

### G bis. Una frase per azione (3 minuti, meno di 1 centesimo)

**Perché:** Claudio ora trattiene le prime due frasi finché non sa se segue un'azione. Va sentito dal vivo, con il costo di latenza. **Claudio acceso, versione nuova**, Chrome aperto.

1. «Claudio, porta avanti Chrome». Atteso: Chrome davanti e **una frase sola**, dopo l'azione.
2. Con Chrome ancora davanti: «Claudio, chiudi la finestra». Atteso: **senti quale finestra** e poi «Confermi?». Rispondi «no»: non si chiude.
3. «Claudio, quanto fa 15 per 18?». Atteso: risponde. Dimmi se la prima parola ti sembra lenta.
4. «Claudio, preparami un piano di dieta per sette giorni». Atteso: propone il modello più potente chiedendo «Lo uso?», oppure risponde. **Se dice «dimmi solo se procedo», è il caso che cerco:** è una promessa a parole, senza un'offerta vera. Rispondi «no» in ogni caso, così Sonnet non parte.
5. A fine prova:
   ```
   $env:PYTHONIOENCODING='utf-8'; uv run python scripts/benchmark_latency.py --log
   ```
   Il riferimento è la sessione `2026-11-07_16-48`: Claude 2681 ms di mediana, locale 824. Trattenere le frasi costa **circa 250-400 ms** sulle risposte brevi.

**Scrivimi:** G bis, i passi che non tornano, la frase del passo 4 e la tabella del passo 5.

**Esito G bis (08/11):** passo 1 una frase sola; passo 2 la domanda locale dice «in primo piano» senza il titolo (da G6); passo 4 offerta vera «Lo uso?». Da sistemare: con una domanda aperta, ripetere lo stesso comando col nome fa dire «Va bene, non lo faccio.» e poi la stessa domanda (G3).

---

## Adesso: G2

### H0. La distorsione, prima di ogni soglia (5 minuti)

**Perché:** nel riascolto del 06/11 le riprese suonavano distorte. Se la distorsione nasce prima di Claudio, la soglia di confidenza scelta su quelle riprese vale poco. **Claudio chiuso.**

1. Ascolta 3 WAV dal vivo:
   ```
   Get-ChildItem samples\corpus\2026-11-06_12-48\audio -Filter *_ok.wav | Select-Object -First 3 | ForEach-Object { Start-Process $_.FullName }
   ```
   Dimmi: suonano distorti come le riprese di `samples\safety\`, meno, o puliti?
2. Ascolta la risposta persa nella prova del 06/11:
   ```
   Get-ChildItem samples,logs -Recurse -Filter 20261106-125111-875_ok.wav | Select-Object -First 1 | ForEach-Object { Start-Process $_.FullName }
   ```
   Dimmi cosa avevi detto e se si sente anche l'inizio. Se non si apre nulla, il file è già ruotato via: dimmelo.
3. Guarda il microfono che usi con Claudio: Impostazioni, Sistema, Audio, il dispositivo di ingresso. Se c'è **Miglioramenti audio**, dimmi se è attivo, senza cambiarlo. Il nome può variare con il driver.

**Scrivimi:** H0 e le tre risposte. I passi dopo (effetti spenti, altro microfono sulle stesse 10 frasi) li preparo solo se servono.

**Esito H0 (08/11):** WAV dal vivo puliti, un po' bassi; si sente il bip di Claudio. Alle 12:51:11 avevi detto quasi certamente «Confermo»: il «Con-» è caduto nel tratto sordo. Microfono con Voice Clarity e messa a fuoco della voce attivi; «Volume di ingresso» segnava 0: ricontrollalo con la cuffia accesa e dimmi il numero, senza cambiarlo.

### H0b. Effetti audio accesi e spenti (circa 5 minuti)

**Perché:** Voice Clarity e la messa a fuoco della voce cambiano l'audio prima di Claudio. Registri le stesse 10 frasi due volte, poi misuro quale dei due fa sbagliare meno Whisper. **Claudio chiuso.** Usa la cuffia di sempre, alla distanza di sempre.

1. Effetti come sono adesso:
   ```
   uv run python "research_notes/Conferma vocale sicura con Whisper/prove/chain_ab.py" on
   ```
   A ogni frase: `Invio`, aspetta «● parla», poi dilla con calma. Se dice «troppo basso» o «distorto», `r` la rifà. `q` esce; rilanciando, registra solo quelle mancanti.
2. Spegni gli effetti: Impostazioni, Sistema, Audio, il microfono «Wireless Stereo Headset». Metti **Ottimizzazione audio** sull'opzione che lo spegne e **Messa a fuoco della voce** su spento (i nomi esatti possono variare).
3. Le stesse frasi, effetti spenti:
   ```
   uv run python "research_notes/Conferma vocale sicura con Whisper/prove/chain_ab.py" off
   ```
4. **Rimetti gli effetti com'erano**: Voice Clarity e Automatico.

**Scrivimi:** H0b fatta. Il confronto lo lancio io.

**Esito H0b (08/11):** 10 su 10 giuste sia con gli effetti accesi sia spenti; le riprese nuove sono molto più pulite di quelle vecchie, quindi la distorsione veniva dal registratore vecchio. Gli effetti restano accesi.

### H1. Registrare le 145 riprese di sicurezza (circa 25 minuti, una volta sola)

**Perché:** è il test che dirà se una risposta che non è un sì può ancora far eseguire un'azione. I criteri sono già scritti nel test. **Nessuno ascolta le riprese prima della corsa unica**: non riascoltarle, non rifare quelle "venute male" a orecchio.

**Prima:** Claudio chiuso (anche dalla system tray). Stesso headset di sempre, nella stessa posizione, con gli effetti di Windows come li usi dal vivo. Una stanza silenziosa.

```
uv run python scripts/record_samples.py safety-v2
```

Il registratore apre il microfono **una volta sola** e lo tiene aperto fino alla fine. Scrive quante riprese ci sono da fare (145 la prima volta).

**Come si registra:**

- A ogni riga: premi `Invio`, aspetta la scritta `● parla`, poi **di' la frase**. Ogni ripresa dura 3 secondi.
- Parla con la voce di tutti i giorni, alla distanza di sempre. Non scandire.
- Se la ripresa è troppo bassa o distorta, il registratore la **scarta da solo** e rifà la stessa riga: di' la frase di nuovo, un po' più vicino o più forte (più lontano se è distorta).
- **Le frasi con il punto interrogativo («sì?» e «confermo?») vanno dette come vere domande, con il tono che sale.** Sono 8 «sì?» in tutto, sparsi nella lista; accanto compare la nota `(come una vera domanda, con il tono che sale)`.
- Alcune righe sono suoni, non parole («mmh», «ehm», «boh»): dillo come lo diresti davvero, senza esagerare. Se un suono molto sommesso («psst») viene scartato come troppo basso, scrivi `a` + `Invio` alla riga dopo: lo tiene lo stesso. Vale solo per i suoni, non per le frasi.
- La stessa frase torna due volte: dille uguale, non cercare di variarle.
- `r` + `Invio`: rifà la ripresa **precedente** (quella scritta sulla riga `rifai:`). Usalo solo se hai sbagliato frase, non per la qualità.
- `q` + `Invio`: esci e riposa. Rilancia lo stesso comando: il registratore **riprende da dove eri e fa solo le riprese mancanti** (ma riapre il microfono: se puoi, fai tutto in una volta).

**Esito atteso:** alla fine `Fatto: 145 clip salvate in ...\samples\safety_v2`. Le riprese stanno in `samples\safety_v2\<gruppo>\` (yes, question, twin, rewrite, offtopic, noise), fuori da Git.

**Non lanciare il test.** Lo lancia il coordinatore, una volta sola, dopo la fine del gruppo.

**Scrivimi:** H1 fatta, e se hai dovuto interrompere (e quante volte) o se qualcosa non ha funzionato (microfono, scritte, scarti continui).

Dopo la H1 parte la corsia B.

### H2. Interruzioni e conferme (circa 15 minuti, qualche centesimo)

**Perché:** i test usano voce e tempi finti. Qui si prova come reagisce Claudio quando lo interrompi davvero. **Claudio acceso, versione nuova.** Un Blocco note vuoto aperto. Se un passo non torna, fermati e scrivimi quale.

**«Dicevo» e «continua»**
1. «Claudio, raccontami una storia lunga». Dopo due frasi: «Claudio, che ore sono?». Atteso: dice l'ora, poi **«Dicevo:»** e la storia riprende, anche se finiva con una domanda.
2. Di nuovo una storia. Dopo due frasi: «Claudio, quanto dista la Luna?». Atteso: risponde, **senza «Dicevo»**. Poi «Claudio, continua»: ridice il pezzo mancante, con «Dicevo:».
3. Di nuovo una storia. Dopo due frasi: «Claudio, basta». Poi «Claudio, continua». Atteso: o ridice il pezzo mancante, o la continua dal punto in cui l'hai fermata. **Non deve inventare una puntata nuova.**

**Mentre Claude pensa**
4. «Claudio, raccontami una barzelletta» e subito, prima che cominci a parlare: «Claudio, basta». Atteso: **nessuna barzelletta**.
5. «Claudio, raccontami una barzelletta» e subito: «Claudio, che ore sono?». Atteso: **dice l'ora**, non la barzelletta.

**Conferme**
6. Blocco note davanti. «Claudio, chiudi la finestra». A «Confermi?» rispondi «sì». Atteso: si chiude. Riapri il Blocco note.
7. Di nuovo. A «Confermi?» non rispondere: aspetta che smetta di ascoltare. Poi, entro un minuto: «Claudio, confermo». Atteso: si chiude. Riapri il Blocco note.
8. Di nuovo. A «Confermi?» rispondi «mah». Atteso: **ripete la domanda una volta**, con «sì o no?». Rispondi «no»: non si chiude.
9. Di nuovo. A «Confermi?» rispondi «Claudio, che ore sono?». Atteso: dice che non lo fa, poi l'ora. Non si chiude.
10. Di nuovo. Con «Confermi?» aperto, scrivi «sì» nella finestra della chat. Atteso: **non si chiude**, e la chat dice «Ho annullato la domanda a voce.».

**Dove vanno i tasti**
13. Chrome aperto su una pagina qualsiasi, Blocco note davanti. «Claudio, scrivi ciao su Chrome e premi invio». Atteso: **chiede una volta e nomina Chrome**. Prima di rispondere, in Chrome passa a un'altra scheda. Poi «sì». Atteso: **non scrive**, e dice che la finestra è cambiata.
14. Ripeti il passo 13 senza cambiare scheda. Atteso: dopo il «sì» scrive «ciao» in Chrome e preme Invio (se il cursore è nella barra degli indirizzi, cerca «ciao»: va bene).
15. Blocco note davanti. «Claudio, chiudi la finestra». **Mentre dice «Confermi?»**, parlaci sopra: «Claudio, sì». Atteso: si chiude, una volta sola.

**Contesto e azioni fermate**
11. «Claudio, che ore sono?», poi «Claudio, e tra due ore che ore saranno?». Atteso: Claude sa di cosa parli.
12. «Claudio, scrivi un piano lungo nel blocco note». Prima che compaia il testo: «Claudio, lascia stare». Atteso: **non scrive nulla.** È il passo 8 della prova G, rifatto dal vivo.

**Scrivimi:** H2, i passi che non tornano, cosa ha detto al passo 3 e cosa ha detto al passo 13.

**Esito H2 (12/11):** conferme (passi 6-10), contesto e azioni fermate (11-12) e tasti (13-15) giusti. **«Dicevo» e «continua» (passi 1-3) no.** Nel `console.log` della sessione `2026-11-12_19-07` il filtro dell'eco ha scartato 6 volte la voce di JenJo sopra la storia («La mia voce nel microfono, la ignoro: 'Claudio, che ore sono?'», 19:08:12-19:09:15): l'interruzione non è mai partita, e «stop» ferma senza riprendere, per scelta. «Continua» non è arrivato. **Va a G2, che si riapre:** prima un test a timeline sul filtro dell'eco.

### H2 bis. Il nome sopra una storia su Claudio (circa 10 minuti, meno di 5 centesimi)

**Perché:** in H2 il filtro dell'eco scartava «Claudio, ...» detto sopra una storia in cui Claudio è il protagonista. La correzione di G2 (13/11) guarda le parole intorno al nome. **Claudio acceso sulla versione nuova:** se era già acceso, «Claudio, chiuditi» e poi `uv run claudio`.

1. «Claudio, raccontami una storia lunga su di te». Dopo due frasi: «Claudio, che ore sono?». Atteso: dice l'ora, poi **«Dicevo:»** e la storia riprende.
2. Di nuovo una storia su di te. Dopo due frasi: «Claudio, quanto dista la Luna?». Atteso: risponde, **senza «Dicevo»**. Poi «Claudio, continua»: ridice il pezzo mancante, con «Dicevo:».
3. Di nuovo una storia su di te. Dopo due frasi: «Claudio, basta». Poi «Claudio, continua». Atteso: ridice il pezzo mancante o continua da dove l'hai fermata. **Non inventa una puntata nuova.**
4. Di nuovo una storia su di te. Appena lo senti dire «Claudio» nella storia: «Claudio, volume su». Atteso: alza il volume, poi «Dicevo:».
5. «Claudio, raccontami una storia lunga su di te» e **non dire nulla** fino alla fine. Atteso: arriva in fondo senza fermarsi da sola (la sua eco non deve interromperla).

**Scrivimi:** H2 bis, i passi che non tornano. Il log della sessione lo controllo io.

**Esito H2 bis (13/11):** la correzione dell'eco funziona dal vivo (sessione `2026-11-13_15-17`). «Claudio, che ore sono?» e «Claudio, alza il volume» sopra storie piene di «Claudio» hanno interrotto la storia, e 30 secondi di una storia con «Claudio» in quasi ogni frase sono passati senza che si fermasse da sola. Passi 1 e 3 giusti. Passo 2 no: «continua» dopo la domanda sulla Luna è andato a Claude, che ha inventato un seguito (la storia rispondeva a una frase incerta, e una risposta di Claude cancellava il pezzo). Passo 4: volume e «Dicevo» giusti, poi «Riprendi?» detto sopra il «Dicevo» è andato a Claude, e la ripresa dopo ha detto «Dicevo: Dicevo:». Passo 5 non fatto a parte. Gli altri problemi della sessione vanno ai loro gruppi: la voce di Jack fuori registro suonata lo stesso a G4; Whisper («chiedi» per «chiudi», «ti sta» per «dista», «chi è di te» per «chiuditi») a G5; il Blocco note («chiudi tutti e due») a G6. Le riprese si correggono in G2 bis (`piani/IT-2-passo-0b-G2-bis-task.md`) e si rifanno dal vivo con H2 ter.

### H2 ter. «continua» e «riprendi» dopo G2 bis (circa 10 minuti, meno di 5 centesimi)

**Perché:** G2 bis ha cambiato le riprese. Il pezzo mancante resta anche dopo una risposta di Claude; «continua» detto mentre Claudio parla segue il contesto (se c'è una storia in sospeso torna a quella, altrimenti va avanti); «Dicevo:» si dice una volta sola. **Claudio acceso sulla versione nuova:** se era già acceso, «Claudio, chiuditi» e poi `uv run claudio`.

1. «Claudio, raccontami una storia lunga su di te». Dopo due frasi: «Claudio, basta». Poi «Claudio, quanto dista la Luna?». Atteso: risponde sulla Luna. Poi «Claudio, continua»: **«Dicevo:»** e il resto della storia, non una storia nuova.
2. Di nuovo una storia. Dopo due frasi: «Claudio, alza il volume». Atteso: alza il volume e riprende con «Dicevo:». **Mentre dice «Dicevo»**: «Claudio, riprendi». Atteso: **va avanti** senza fermarsi.
3. Di nuovo una storia. Dopo due frasi: «Claudio, basta». Poi «Claudio, quanto dista la Luna?» e, **mentre risponde sulla Luna**: «Claudio, continua». Atteso: lascia la Luna e dice «Dicevo:» con il resto della storia. Rifallo senza «basta»: la domanda sulla Luna detta sopra la storia, poi «Claudio, continua» mentre risponde. Atteso uguale.
4. Di nuovo una storia. Dopo due frasi: «Claudio, che ore sono?». Appena comincia «Dicevo:», di nuovo «Claudio, che ore sono?». Atteso: l'ora, poi **un solo «Dicevo:»**.
5. «Claudio, raccontami una storia lunga su di te» e **non dire nulla** fino alla fine. Atteso: arriva in fondo senza fermarsi da sola.

**Scrivimi:** H2 ter, i passi che non tornano. Il log della sessione lo controllo io.

**Esito H2 ter (13/11, sera, sessione `2026-11-13_20-24`):**

- **Passo 1 no.** «riprendi» ha ridetto l'ultima frase della prima risposta sulla Luna. La domanda detta sopra quella risposta breve aveva cancellato la storia, e la risposta (2 frasi, 224 caratteri) contava come da riprendere.
- **Passo 2 sì.** «Claudio riprendi» sopra il «Dicevo» ha dato «Sto già continuando», e la storia è andata avanti.
- **Passo 3 no.** Whisper ha capito «continuo»; la lista chiusa lo correggeva solo dopo il taglio, e Claude ha inventato un «Dicevo:».
- **Passo 4 no.** L'ora ci ha messo 10 s: Jack ha scartato la ripresa, il suo worker non ha risposto in 15 s, poi ha parlato Kokoro. La seconda «che ore sono», detta durante l'attesa, ha fatto perdere il «Dicevo» automatico: la storia è tornata solo con «continua».
- **Passo 5** non fatto a parte.

Gli altri problemi della sessione vanno ai loro gruppi:

- Voce di Jack (19 controlli con avviso su 26, 20 riprese scartate, tono che oscilla) a G4.
- Risposte lente a G4/G7: la prima frase di Jack va da 0,2 a 3,2 s, mentre la prima parola di Claude sta fra 0,9 e 1,6 s.
- Comprensione a G5: «quanto ti sta la luna» due volte, «solita'».

Le riprese si correggono in G2 ter (`piani/IT-2-passo-0b-G2-ter-task.md`) e si rifanno dal vivo con H2 quater.

### H2 quater. Le riprese dopo G2 ter (circa 10 minuti, meno di 5 centesimi)

**Perché:** con G2 ter «continua» torna solo alle storie, mai a una risposta breve; un «continuo» capito male viene corretto anche mentre Claudio parla; un'ora chiesta due volte non fa perdere la storia. **Claudio acceso sulla versione nuova:** se era già acceso, «Claudio, chiuditi» e poi `uv run claudio`.

1. «Claudio, raccontami una storia lunga su di te». Dopo due frasi: «Claudio, basta». Poi «Claudio, quanto dista la Luna?» e, mentre risponde, «Claudio, e quanto dista Marte?». Mentre risponde su Marte: «Claudio, riprendi». Atteso: **«Dicevo:» e il resto della storia**, non l'ultima frase di una risposta.
2. Di nuovo una storia. Dopo due frasi: «Claudio, alza il volume». Mentre dice «Dicevo»: «Claudio, continua». Atteso: **va avanti** senza fermarsi, anche se Whisper capisce «continuo».
3. Di nuovo una storia. Dopo due frasi: «Claudio, che ore sono?» e, prima che risponda o mentre risponde, di nuovo «Claudio, che ore sono?». Atteso: l'ora, poi **«Dicevo:» e la storia** senza dire «continua».
4. «Claudio, raccontami una storia lunga su di te» e **non dire nulla** fino alla fine. Atteso: arriva in fondo senza fermarsi da sola.

**Scrivimi:** H2 quater, i passi che non tornano. Il log della sessione lo controllo io.

**Esito H2 quater (13/11, notte, sessioni `2026-11-13_23-20` e `2026-11-13_23-23`):** la logica delle riprese è giusta in tutti i passi (JenJo). G2 è chiuso. Restano tre problemi, per altri gruppi:

- **Con un gioco aperto la GPU è piena.** All'avvio di Jack erano occupati 16.276 MB su 16 GB, contro 13.100 senza gioco (Phantom Blade Zero). Al primo tentativo Whisper ha impiegato 14,4 s per una frase e Jack 8,5 s per la prima frase. Va a F8 (la modalità gioco come segnale) o a G8.
- **La parola d'attivazione passa sempre da Whisper.** Corretto il 14/11 dall'indagine W (sotto): è voluto. Il modello «claudio» ha `confirm = true`, quindi ogni sua rilevazione passa da Whisper; i punteggi 0,27-0,72 sono **sopra** la sua soglia di 0,25, e sono solo il primo fotogramma sopra soglia, non il picco. «Grazie.» (23:24:05) era un sospiro di JenJo scartato giustamente; resta un solo mancato possibile, «Questa è l'una.» (23:26:19), in cui il nome non ha fatto scattare il modello. La comprensione («su di te» → «solita'», «suddita») va a G5.
- **Voce robotica:** 23 controlli con avviso su 33, 22 riprese scartate, 3 suonate lo stesso. Senza gioco (20:24) era simile. Va a G4.

---

## Adesso: W (parola d'attivazione)

### W. Quante volte «Claudio» non sveglia Claudio (circa 10 minuti, meno di 3 centesimi)

**Perché:** l'indagine del 14/11 non ha trovato un modello peggiorato: i punteggi sono come dal 24/10 e il gioco non li abbassa. Ora il log dice se Whisper conferma una rilevazione sopra soglia («Conferma di Whisper») o salva un quasi-colpo sotto soglia («Seconda possibilità»), con il picco e i fotogrammi persi mentre Whisper ascoltava. Questa prova conta i mancati veri, senza gioco e con il gioco, e il log dice per ognuno se il modello è scattato. **Claudio acceso sulla versione nuova:** se era già acceso, «Claudio, chiuditi» e poi `uv run claudio`. Tieni un foglio per segnare i mancati.

1. Gioco chiuso. Dieci volte «Claudio, che ore sono?» a voce normale, aspettando ogni volta la risposta. Atteso: risponde tutte e dieci le volte. Segna quante volte non risponde e, se puoi, l'ora.
2. Gioco chiuso. Cinque volte solo «Claudio», poi «che ore sono?» dopo il bip. Atteso: il bip tutte e cinque le volte.
3. Apri Phantom Blade Zero con l'audio come lo usi di solito e rifai il passo 1. Atteso: come al passo 1, forse più lento.
4. Gioco aperto. Tre volte: «Claudio, raccontami una storia lunga su di te», dopo due frasi «Claudio, basta» e **subito dopo** «Claudio, che ore sono?». Atteso: l'ora tutte e tre le volte.

**Scrivimi:** W, quante volte non ha risposto in ogni passo (e le ore, se le hai). Il log lo controllo io: un mancato con «blocchi sopra soglia durante il controllo» è il caso di T2; un mancato senza nessuna riga vuol dire che il modello non è scattato, e soglia o nuovo addestramento vanno a G5.

**Esito W (14/11, notte, sessione `2026-11-14_00-38`, cuffie come uscita, 0,021 $):** W è chiuso, T2 non si fa.

- **Senza gioco** ogni attivazione registrata è riuscita: 5 dalla wake word, «claudio» 0,26-0,96 con picchi fino a 0,97, controllo di Whisper 152-232 ms, prima parola 0,9-1,4 s. I mancati non sono stati contati, ma nessun controllo fallito contiene un pezzo di nome. Il passo 1 ha misurato soprattutto la finestra di conversazione (ripetizioni entro 6 s dalla risposta): per contare i mancati della wake word servono pause di almeno 10 s.
- **I respiri** fanno scattare «claudio» fino a 0,9 e Whisper sente «Grazie.»: circa 25 controlli scartati in 50 minuti, giustamente. I loro WAV in `logs/audio/rejected` servono come esempi negativi se G5 addestra di nuovo il modello.
- **Col gioco** (GPU al 99%, 440-810 MB liberi): Whisper 1,0-9,0 s invece di 0,15-0,33, prima frase della voce 2,4-7,0 s invece di 0,23-0,73, prima parola 3,8-13,4 s invece di 0,9-1,4, voce a scatti. Va a F8 o G8, con la GPU piena di H2 quater.
- **Due bip senza richiesta**, vanno a G5: «hey_claudio» a 0,63 su una frase detta da JenJo a un'altra persona («Perché avete 10 minuti di coda?», poi scartata dalla verifica) e a 0,54 alle 01:25:32 («Nessuna richiesta dopo la wake word»). Sempre a G5: «Claudio, chiuditi» capito «chiediti» è finito a Claude, che ha chiesto conferma.

---

## Adesso: G3c

### J. Sonnet come modello principale (circa 10 minuti, circa 5 centesimi)

**Perché:** da G3c risponde Sonnet 5.5 con il thinking adattivo, non più Haiku. L'A/B ha misurato solo le risposte dell'API: qui si sente cosa arriva all'altoparlante. **Claudio riavviato con la versione nuova**, sul PC fisso (D3 di `piani/ROADMAP.md`). Un Blocco note vuoto aperto. Se un passo non torna, fermati e scrivimi quale.

1. All'avvio guarda la riga di Claude nel log. Atteso: `claude-sonnet-5-5`, «thinking on», «con strumenti 2898».
2. «Claudio, quanto dista la Luna?», poi «Claudio, cos'è un buco nero, in due frasi?». Atteso: la prima parola in circa un secondo e mezzo. Dimmi se ti sembra troppo lenta rispetto a Haiku.
3. «Claudio, come stai?». Atteso: è sempre Jack. Sonnet dice in media una frase in più: dimmi se è troppo.
4. «Claudio, cerca il file del mio CV e aprilo». Atteso: lo apre senza chiedere e non dice «lo faccio» senza farlo.
5. Blocco note davanti. «Claudio, chiudi la finestra del Blocco note». Atteso: «Confermi?» nominando il Blocco note; «sì» e si chiude. Riaprilo.
6. «Claudio, apri il brocconote e scrivi una poesia», detto un po' storpiato. Atteso: capisce, chiede prima di scrivere, poi scrive **tutta** la poesia.
7. «Claudio, pensaci bene: per i giochi è meglio un SSD o un HDD?». Atteso: **nessun «Lo uso?»**, risponde subito.
8. «Claudio, scrivi nel Blocco note una lettera lunga almeno una pagina a un amico». Atteso: chiede, poi scrive tutta la lettera, **senza errore di rete**. Dimmi quanto ha aspettato prima di scrivere.

**Noto, non è un errore:** se Claude ti ha appena fatto una domanda, un comando senza nome («Spegniti.») viene preso come risposta. Con il nome funziona. Si sistema in G5.

**Scrivimi:** J, i passi che non tornano, l'attesa al passo 2 e al passo 8.

**Esito J (12/11), giorno d'inizio della prova D6:** passo 2 con attesa breve. **Passo 4:** Claude ha proposto un secondo CV, in iCloud; al «sì» ha chiamato `open_path(~\iCloudDrive\Downloads\CV_EN.pdf)`, lo strumento ha risposto «ok» ma la finestra non si è mai vista, e Claude ha supposto una scheda di Chrome invece di controllare (sessione `2026-11-12_18-58`, 19:00:27-19:00:46) → **G6**: causa da verificare, scheda nascosta di Chrome o file iCloud solo online. **Passo 8:** JenJo ha chiesto «in una nuova finestra»; Claude ha chiamato `open_app(Blocco note)` e poi `press_keys(ctrl+shift+n)`, e le finestre nuove sono state due (19:03:16) → **G6**. Attesa di circa 8 s prima della scrittura (prima voce udibile a 5,2 s) → **G7**.

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
