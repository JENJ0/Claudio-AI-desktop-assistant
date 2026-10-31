# IT-1 – Consegne: unione dei 7 rami nel ramo `it-1`

Ramo locale `it-1` (cartella di lavoro `C:\Users\JenJo\Desktop\PROGETTI\claudio-it1-merge`), partito
da `main` = `ba3872f` (tag `v0.4.0`, con il commit dei documenti `3d5e9db`). **Non unito a `main`**
(prima lo provi tu) e mai pushato.

## Cosa c'è dentro

| Ordine | Ramo | Commit | Conflitti risolti |
|---|---|---|---|
| 1 | `it1/memoria` | `35cbc9e` | nessuno |
| 2 | `it1/comandi` | `d98a269` | CHANGELOG |
| 3 | `it1/ascolto` | `959b056` | `app.py`, `__main__.py` (parametri nuovi di entrambi), `router/responder.py` (`understands`) |
| 4 | `it1/voce` | `d241166` | `brain/claude.py` (import), `i18n.py` |
| 5 | `it1/latenza` | `b22785d` | `sentence_splitter.py`, `speaker.py`, `jack.py`, `pc/tools.py`, `i18n.py`, `claudio.toml`, `test_speaker.py`, CHANGELOG |
| 6 | `it1/pronuncia` | `2e86568` | nessuno |
| 7 | `it1/sempreattivo` | `8f275a9` | `__main__.py`, `i18n.py`, CHANGELOG |

Poi due commit miei: `fix: what merging the IT-1 branches got wrong between them` e
`docs: one IT-1 section in the CHANGELOG, README status after the merge`.

### Comportamenti tenuti apposta (dalle consegne)

- **Ascolto**: `barge_in_min_dbfs = -42` misurato *prima* del guadagno del microfono; da spento
  `understands` vale solo per "accenditi"; la trascrizione veloce della seconda possibilità usa il
  prompt delle frasi spedite (`[stt.prompts]`).
- **Memoria + ascolto**: `understands` tiene le due regole insieme: da spento solo "accenditi"
  (niente frasi di memoria), da acceso anche le frasi di memoria.
- **Voce + latenza**: come nell'unione di prova `it1/prova-latenza-voce`: `group_stream` legge i
  pezzi di Claude da solo, quindi al segnale `CUT_SHORT` (risposta tagliata a `max_tokens`) scarta
  il frammento con `drop_unfinished()` (senza, Jack dice di nuovo "Attra"); `_next_event` usa
  `_poll()` di voce ma solleva `WorkerTimeout` di latenza (così Jack riparte quando la scheda video
  si libera); `jack_group_chars = 170` e `jack_group_pause_ms = 200` insieme.
- **Latenza + memoria**: `claudio_command` riceve la lingua della richiesta, e accanto ci sono gli
  strumenti `remember`/`forget`.
- **Sempre attivo**: `run_assistant` con l'icona della tray, il `close_when_idle` e l'istanza unica
  in `main()`, insieme a tutto quello che gli altri rami hanno aggiunto lì (memoria, scheda video
  occupata, WAV scartati, seconda opinione).

## Correzioni tra i rami (trovate unendo, ognuna con un test)

1. **Seconda opinione su una frase di memoria** (comandi × memoria): la seconda opinione guarda
   solo i comandi, quindi "dimenticalo" o "cosa ricordi di Anna?" passavano da Whisper con il prompt
   dei comandi (+0,2 s e il rischio di diventare un comando). Ora la salta quando il router capisce
   la frase (`Assistant._may_recover`). Test: `test_no_second_opinion_on_a_memory_request`, che
   falliva sul codice unito.
2. **Log del ducking** (voce × ascolto): "Abbasso la risposta: voce a X dBFS" dava il livello dopo
   il guadagno, ma da ascolto la soglia si confronta con quello prima. Ora scrive entrambi:
   "voce a -18 dBFS (-30 prima del guadagno)".
3. **Test delle take di Jack** (voce × pronuncia): la tabella dei semi usa il testo come lo legge
   Jack, che ora ha gli accenti ("un attimo" → "un àttimo"): il test usava il testo scritto.

## Verifiche

- `pytest` (tutta la suite, senza i lenti): **2452 passati**, 5 saltati (prima dell'unione:
  memoria 1962, pronuncia 1912, comandi 1843, voce 1837, sempreattivo 1833, ascolto 1790,
  latenza 1789).
- Test di integrazione con i modelli veri (`CLAUDIO_MODELS_DIR` della cartella principale): 11
  passati, 3 saltati (espeak-ng non installato, inglese non configurato), compreso quello del
  reset della wake word di ascolto.
- `ruff check` e `ruff format --check` puliti, `mypy src` pulito (86 file).
- `claudio --check` sul codice unito: tutte le righe OK, compreso "modello Whisper large-v3
  (3.1 GB)".
- Un avviso già presente prima: pywinauto "Revert to STA COM threading mode".

## Frasi fisse di Jack

Vanno rifatte dopo l'unione: la tabella dei semi usa il testo come lo legge Jack, e con la
pronuncia le frasi con una sdrucciola cambiano ("accenditi" → "accènditi"). Le 22 del 31/10
restano in `data\jack_cache\`: quelle che non combaciano più vengono solo ignorate.

JenJo (31/10, notte): "rigenerazione domani mattina appena accendo il computer". Quindi:

- la cartella principale `C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant` è sul
  ramo `it-1` (la cartella di lavoro `claudio-it1-merge` è staccata, `HEAD` sullo stesso commit);
- il task di Windows **ClaudioJackTakes** parte una volta all'accesso: aspetta 2 minuti, poi
  aspetta che la GPU abbia almeno 6000 MB liberi e sia sotto il 30 % e che Claudio sia chiuso
  (riprova ogni 5 minuti, al massimo 6 ore), lancia
  `.venv\Scripts\python.exe scripts\jack_takes.py --from-logs 3` e alla fine **si cancella da
  solo**;
- cosa è successo: `logs\jack-takes\auto-run.log`; il risultato: `logs\jack-takes\report.md`;
- lo script del task: `logs\jack-takes\run_once.ps1` (fuori da Git).

Per toglierlo prima che parta: `Unregister-ScheduledTask -TaskName ClaudioJackTakes`.

## Come provarlo (passi per JenJo)

1. Chiudi Claudio se è acceso ("Claudio, chiuditi") e i giochi.
2. Apri PowerShell (non l'app Claude) e vai nella cartella principale:
   `cd C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant`
3. Controlla di essere sul ramo unito: `git branch --show-current` deve scrivere `it-1`
   (altrimenti `git switch it-1`; i tuoi `data\`, `models\`, `logs\` e `.env` restano).
4. Installa la dipendenza nuova dell'icona: `uv sync` (aggiunge solo `pystray`).
5. `uv run claudio --check`: ogni riga OK, compresa "OK modello Whisper large-v3 (3.1 GB)".
6. Le frasi fisse di Jack: se `logs\jack-takes\auto-run.log` finisce con "exit 0" sono fatte;
   altrimenti, GPU libera, circa 10 minuti: `uv run python scripts\jack_takes.py --from-logs 3`.
   Leggi `logs\jack-takes\report.md`.
7. Avvia: `uv run claudio` e aspetta "Sono pronto". Vicino all'orologio (anche sotto la freccia ^)
   c'è l'icona: blu mentre carica, verde quando è pronto.
8. Prove, una per thread:
   - **Memoria**: "Claudio, ricordati che mia sorella si chiama Anna" (la ripete), "Claudio, cosa
     ricordi di mia sorella?" (Anna), "Claudio, dimenticalo" (la dimentica). Risposte immediate.
   - **Comandi storpiati**: "Claudio, che ore sono?", poi entro 6 s, senza nome, "...iudi la chat"
     mangiandoti l'inizio: la chiude o chiede "Hai detto: chiudi la chat?" (rispondi "sì").
   - **Ascolto**: dopo una risposta resta zitto 10 s: nessun bip da solo. Di' solo "Claudio",
     aspetta il bip, poi "apri la chat". Di' d'un fiato "Claudio spegniti": si spegne e l'icona
     diventa grigia. Da spento basta **una** volta "Claudio, accenditi": tic leggero, poi "Eccomi,
     sono acceso".
   - **Pronuncia**: dopo "Claudio, spegniti" deve dire "Per riaccèndermi di' Claudio, accènditi".
   - **Voce**: "Claudio, raccontami una storia lunga": volume uguale dall'inizio alla fine; parlaci
     sopra senza il nome: il volume scende e risale piano. Se viene tagliata finisce con "Mi fermo
     qui: se vuoi, continuo.", mai a metà parola.
   - **Latenza**: "Claudio, apri la cartella Progetti": "Ci guardo, socio." quasi subito. Rifallo e
     di' "basta" appena lo senti: si ferma subito.
   - **Sempre attivo**: in una seconda PowerShell, nella stessa cartella, `uv run claudio`: deve
     scrivere "CLAUDIO è già acceso: questa copia si chiude". Tasto destro sull'icona: "Pausa"
     (arancione), "Accendi" (verde), poi "Chiudi CLAUDIO" (l'icona sparisce).
9. Se qualcosa non torna mandami `logs\sessions\<data_ora>\console.log` della prova.
10. Va tutto bene: scrivimi **unisci** e porto `it-1` in `main`. Qualcosa non va: `git switch main`
    torna subito alla v0.4.0 (poi `uv sync`).

Il task all'avvio di Windows (avvio automatico di Claudio) non è ancora registrato: punto 4 di
`piani\IT-1-sempreattivo-consegne.md`, da fare solo dopo l'unione in `main`.

## Rischi e cose aperte

1. **Wake debole + frase di memoria senza nome** (c'era già prima dell'unione, da memoria con il
   codice di IT-0): con una wake debole "un comando conta" e la richiesta è considerata detta a
   Claudio, quindi una regola "da ora in poi..." sentita dalla TV in quel momento verrebbe salvata
   senza chiedere. Raro (serve una wake debole proprio su quella frase); da sistemare in un thread
   della memoria, non qui.
2. **2.2(c) non collegata**: la frase detta sopra la risposta non passa ancora dalla seconda
   opinione (`_recover_command` è pronta; il thread ascolto la lasciava al thread comandi).
3. **Domande ancora aperte dai thread**: sempre attivo (chi registra il task all'accesso; dopo 4
   crash in 10 minuti arrendersi o riprovare ogni 5 minuti), comandi (le sue 5 scelte motivate),
   memoria (riassunti delle conversazioni, D1, in un thread a parte).
4. **Non provato dal vivo** con microfono e GPU dopo l'unione: solo i test automatici.
5. Il CHANGELOG ha ancora molte sezioni vecchie intitolate `[0.4.0]` (erano `Unreleased` prima del
   rilascio): non toccate.

## Pulizia dopo che hai unito in `main`

Il sistema dei permessi non mi lascia cancellare rami e cartelle: i comandi sono per te, in
PowerShell, **solo dopo** che `it-1` è in `main`. `-d` (minuscolo) e `worktree remove` rifiutano da
soli se qualcosa non è unito o ha modifiche non salvate.

```powershell
cd C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant
git branch -D it1/prova-latenza-voce   # unione di prova di latenza, non serve più
foreach ($n in "memoria","comandi","ascolto","voce","latenza","pronuncia","sempreattivo") {
    git worktree remove "..\claudio-it1-$n"
    git branch -d "it1/$n"
}
git worktree remove ..\claudio-it1-merge
```

Se `worktree remove` rifiuta per dei file non tracciati (log o WAV delle prove dei thread),
guardali prima e poi ripeti con `--force`.
