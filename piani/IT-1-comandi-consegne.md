# IT-1 – Consegne: comandi storpiati (2.1), niente "Dicevo:" fuori luogo (5.4), decisioni 1-2

Branch locale `it1/comandi` (worktree `C:\Users\JenJo\Desktop\PROGETTI\claudio-it1-comandi`),
allineato a `main` v0.4.0 (rebase fatto), non pushato, da unire in `main` con gli altri thread
di IT-1. 6 commit, dal più vecchio:

| Commit | Cosa |
|---|---|
| `feat: a guessed basta stops at once, switching off stays asked` | decisioni 1 e 2 |
| `fix: no "Dicevo:" after a question, a short answer or a doubtful hearing` | 5.4 |
| `feat: Whisper's second opinion on a short phrase no command understands` | 2.1 |
| `feat(bench): stt_bench measures the second opinion and false commands` | la misura del piano |
| `fix: review findings on the second opinion and the resume rule` | le 9 correzioni della revisione |
| `docs: ...` | README (it/en), CHANGELOG, questo file |

## Fatto

| Cosa | Dove |
|---|---|
| **Decisione 1**: un "basta" dedotto da Haiku tra il 50 e l'80 % ferma subito (prima: "Hai detto: basta?"). Eccezione mia: subito dopo una domanda di Claude lo chiede ancora ("Pasta." può essere la risposta a "Che piatto vuoi?") | `src/claudio/intake.py` (`NEVER_ASKED`, `asked_or_done`), `src/claudio/router/responder.py` (`_action`) |
| **Decisione 2**: pausa, "spegniti" e "chiuditi" dedotti sono sempre chiesti, anche al 99 %; stessa regola per Haiku e per il recupero locale | `src/claudio/intake.py` (`ALWAYS_ASKED`) |
| **5.4**: una risposta interrotta riprende ("Dicevo: ...") solo se è lunga almeno 200 caratteri (una storia, una spiegazione), non finisce con una domanda e non rispondeva a una frase che Whisper aveva capito male. Una storia ripresa e interrotta di nuovo riprende ancora | `src/claudio/app.py` (`_unsaid`, `_resume_reply`, `CutReply`), `src/claudio/resume.py` (`worth_resuming`), `[conversation] resume_min_chars` |
| **2.1**: una frase breve (fino a 6 parole, anche una domanda) che nessun comando capisce riceve la "seconda opinione" di Whisper: la riascolta con i comandi come prompt e il comando proposto è giudicato sul suono da solo, senza prompt. Molto più probabile della prima lettura: lo esegue; circa uguale: "Hai detto: ...?" (si risponde in locale, zero API); meno probabile: come prima (Claude, o la stima di Haiku) | `src/claudio/stt/recovery.py` (nuovo), `src/claudio/stt/transcriber.py` (`listen_again`, `_fit`, `_forced`), `src/claudio/app.py` (`_may_recover`, `_recover_command`), `[recovery]` in `config\claudio.toml` |
| Funzione riusabile per la 2.2 (thread "Far sentire bene Claudio"): `await assistant._recover_command(audio, testo, lingua, interazione)` restituisce `Decision(ANSWER, comando)`, `Decision(ASK, "Hai detto: ...?")` o `None`. Prende da sola il lock di Whisper (non chiamarla tenendolo); se la frase vale il tentativo lo decide chi chiama, come fa `_may_recover` | `src/claudio/app.py` |
| Nelle metriche: `recovery` (testo proposto), `recovery_command`, `recovery_fit` (punteggi), `recovery_mean`, `recovery_verdict`, `recovery_ms`, `recovered_from`; nel log della sessione `recovered_from` | `logs\metrics.jsonl`, `src/claudio/session_log.py` |
| Misura del piano: `scripts\stt_bench.py --variants live live+recupero` (variante `recupero` e colonna `falsi`) | `scripts/stt_bench.py` |
| Test: 90 in più (1753 → 1843) | `tests/unit/test_recovery.py` (nuovo), `test_guess.py`, `test_conversation.py`, `test_resume.py`, `test_stt_bench.py` |

## Come funziona la seconda opinione (2.1), in breve

1. Dopo la trascrizione normale e la lista chiusa, se la frase è breve e nessun comando la capisce,
   Whisper la riascolta con il prompt "Claudio, Apri chrome. Apri la cartella download. ...
   Chiuditi." (la prima frase di ogni comando; ~200 token, sotto il limite di 223).
2. Se ne esce un comando, Whisper dà un **punteggio forzato** alle due letture, senza prompt: quanto
   è probabile che quell'audio dica proprio quelle parole e poi finisca. La fine del testo conta:
   così "Apri Chrome." non batte "Apri Chrome e cerca Dainese Moto". Una sola codifica dell'audio
   per tutto.
3. `gap` = quanto la proposta è più probabile della prima lettura; `mean` = probabilità media per
   pezzo di parola della proposta. Con `mean < -1.75` o `gap < -6`: niente; `gap >= 2`: eseguito;
   in mezzo: "Hai detto: ...?". Pausa, "spegniti", "chiuditi": sempre chiesti; "basta": mai.
4. "Svegliati" e "accenditi" non vengono mai proposti: da acceso e sveglio non cambiano nulla, e
   nella prima misura erano tutte e 4 le proposte sbagliate (Whisper le scriveva su "Grazie." ed
   "Ehem.").
5. Mai sull'eco di Claudio, su una voce lontana (la TV), su un "sì/no", sulla risposta a una
   domanda di Claude, in pausa, da spento, né quando Whisper è finito sulla CPU (lì il modello
   piccolo aggiungerebbe 1-3 s a ogni frase breve e le soglie sono di large-v3).

## Misure (GPU RTX 5070 Ti, large-v3, 30/10 sera)

- **Il banco del piano** (`scripts\stt_bench.py`, i due set, 2 microfoni, 40 frasi ciascuno):
  comandi giusti **40/48 con la sola trascrizione, 43/48 con lista chiusa e seconda opinione
  insieme** (cioè come fa Claudio adesso), **frasi non-comando scambiate per comandi: 0 e 0**.
  Cuffia set 1: 9 → 11; webcam set 2: 10 → 11; gli altri due invariati.
- **240 frasi in tutto** (i due set, 36 WAV della sessione del 30/10, i 44 pezzi detti da Jack
  come prova di eco): la seconda opinione da sola ha recuperato **3 comandi** ("Scaniti." ->
  "Hai detto: spegniti?", "Claudio Chiediti." -> "Hai detto: chiuditi?", "Chi lascia atto?" ->
  "Hai detto: chiudi la chat?"), **0 sbagliati, 0 falsi, 0 sulla voce di Jack**.
- **Latenza**: +190 ms (mediana) solo sulle frasi brevi non capite; +300 ms quando propone un
  comando; nulla su tutto il resto. Con un gioco sulla GPU tutto è circa 3 volte più lento.
- **Costo**: zero (tutto locale); le frasi recuperate non arrivano più alla stima di Haiku.
- **5.4** sulle 188 risposte vere dei log: delle 14 riprese "Dicevo:" fatte finora, le **9
  sbagliate sono tutte bloccate** (domande, battute, risposte brevi, risposte a frasi capite male).
- **Correzione dell'analisi IT-0**: i punteggi del piano ("Chi ce l'ha fatta" eseguito) venivano da
  uno script con `<|notimestamps|>` due volte. Con il punteggio giusto "chiudi la chat" è meno
  probabile della prima lettura, quindi Claudio **chiede** invece di eseguire: più sicuro, sempre
  zero API. Nessuna proposta è stata "nettamente migliore": l'esecuzione immediata non è ancora
  misurata dal vivo.

## Revisione

Revisione avversaria a 3 revisori (macchina a stati, Whisper/ctranslate2, prodotto e test) più un
verificatore: 13 segnalazioni, 9 confermate, tutte di gravità bassa, tutte corrette con un test
che prima falliva (storia ripresa due volte, "basta" di Haiku sulla risposta a una domanda, testo
da un solo pezzo come "No", punteggio mancante, nome storpiato nella domanda "Hai detto: L'audio...",
CPU senza seconda opinione, metriche, commento delle soglie). Il controllo finale a 2 agenti non è
partito per il limite d'uso: l'ho sostituito con i miei controlli (passi di ctranslate2 misurati
sulla GPU per testi da 1 a 14 pezzi, suite completa).

## Decisioni mie, motivate (dimmelo se ne vuoi un'altra)

1. **Regola dei 200 caratteri per "Dicevo:"** (oltre il piano): 19 domande su 88 dei log non hanno
   il "?" ("Dimmi cosa vuoi che faccia..."); solo la lunghezza le ferma. Tutte le storie vere dei
   log superavano i 200. Si cambia con `[conversation] resume_min_chars` (0 = come prima).
2. **La regola "frase capita male" del piano resta**, anche se nei log non ferma nessuna ripresa
   sbagliata che le altre due non fermino già, e nei log avrebbe bloccato 2 storie lunghe (richieste
   sentite con poca sicurezza): una storia non ripresa costa un "continua", una ripresa sbagliata
   disturba. Se preferisci, si toglie.
3. **"Hai detto: ...?" anche per i comandi normali** quando la seconda opinione non è netta: nessun
   comando parte senza un sì se l'audio non lo sostiene.
4. **Decisione 1 con un'eccezione**: subito dopo una domanda di Claude (45 s) un "basta" dedotto
   tra 50 e 80 % viene chiesto, non eseguito.
5. Punteggio con la fine del testo (`generate` di ctranslate2) invece di `align`: costa ~100 ms in
   più, ma solo quando c'è una proposta (3 frasi su 240).

## NON fatto, rischi, da guardare

1. **Soglie tarate su pochi casi** (3 giusti, 4 sbagliati): il margine su `min_mean` è sottile
   (-1.67 i giusti, -1.79 il peggiore sbagliato). Dopo qualche giorno d'uso si ritarano sulle righe
   con `recovery_verdict` in `logs\metrics.jsonl`.
2. **2.2 non collegata qui**: le frasi dette mentre Claudio parla passano dal recupero solo quando il
   thread "Far sentire bene Claudio" chiama `_recover_command` sulla frase ricomposta.
3. Comportamenti esistenti non toccati: la lista chiusa può ancora eseguire subito "spegniti" o
   "chiuditi" se compaiono tra le 5 trascrizioni di Whisper (nei log è successo 2 volte, entrambe
   giuste); la stima di Haiku può girare sull'eco di Claudio quando la finestra di conversazione
   la scarta come "non è una richiesta" (la seconda opinione invece la salta).
4. Il banco ha riscritto `data\stt_testset\risultati.json` e `data\stt_testset_2\risultati.json`
   con questa misura (è il suo file di uscita); le copie di prima sono in
   `C:\Users\JenJo\AppData\Local\Temp\claude\C--Users-JenJo-Desktop-PROGETTI\ea9c5433-ca60-43f6-a5f5-bdef967b7d99\scratchpad\risultati_backup`.
5. Non provato dal vivo con il microfono: la prova manuale è sotto.

## Prova manuale (dopo l'unione in main)

1. Apri PowerShell e scrivi:
   `cd C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant`
   poi `uv run claudio`. Aspetta "Sono pronto".
2. **Storia ripresa (deve ancora funzionare)**: di' "Claudio, raccontami una storia lunga". Mentre
   parla di' "Claudio, che ore sono?". Atteso: dice l'ora, poi "Dicevo: ..." e continua la storia.
3. **Domanda non ripresa (5.4)**: di' una richiesta vaga, per esempio "Claudio, cosa ne pensi?".
   Se Claude risponde con una domanda ("Di cosa?"), mentre la dice di' "Claudio, che ore sono?".
   Atteso: dice l'ora e **nessun** "Dicevo:". (Se Claude non fa una domanda, riprova con un'altra
   frase vaga.)
4. **Seconda opinione (2.1)**: di' "Claudio, che ore sono?", poi entro 6 secondi, senza "Claudio",
   di' "chiudi la chat" mangiandoti l'inizio (quasi "...iudi la chat?") oppure "...egniti". Atteso:
   o lo capisce subito come prima, o chiede "Hai detto: chiudi la chat?" / "Hai detto: spegniti?".
   Rispondi "sì": lo fa; "no": "Va bene, lascio stare.". Se ti spegne: "Claudio, accenditi".
5. **Decisione 2**: "Claudio, spegniti" detto bene si spegne subito come prima (è un comando capito,
   non dedotto); solo quando è dedotto da una frase storpiata chiede sempre. Riaccendi con
   "Claudio, accenditi".
6. Chiudi con "Claudio, chiuditi". In `C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant\logs\metrics.jsonl`
   le righe con `recovery_verdict` dicono cosa ha proposto e con che punteggi: mandamele se
   qualcosa non torna.

## Verifiche

- `pytest`: 1843 passati, 4 saltati (prima 1753); `ruff check` e `ruff format --check` puliti;
  `mypy src` pulito (sul branch dopo il rebase su v0.4.0).
- Misure: script in
  `C:\Users\JenJo\AppData\Local\Temp\claude\C--Users-JenJo-Desktop-PROGETTI\ea9c5433-ca60-43f6-a5f5-bdef967b7d99\scratchpad`
  (`recovery_bench2.py`, `recovery_report2.py`, `stt_bench_set1.txt`, `stt_bench_set2.txt`,
  `resume-rule\`).
