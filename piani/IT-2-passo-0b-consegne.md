# IT-2, Passo 0b: affidabilità (v0.4.2) e residui (v0.4.3)

> **Per chi esegue:** prima di ogni gruppo usa `superpowers:writing-plans` per i task TDD di quel gruppo, poi `superpowers:subagent-driven-development`. Vale `ponytail` (skill di progetto): cambiamento minimo completo, mai togliere validazione, gestione errori, sicurezza. I passi usano `- [ ]`.

**Obiettivo:** far sì che Claudio esegua davvero ciò che dice, non si confonda quando viene interrotto, tenga un tono stabile e capisca meglio, e che queste correzioni restino nel tempo.

**Architettura:** nessun cambio di stack. Si aggiunge uno strato di prova che oggi manca, tra i test con finti e le sessioni dal vivo: WAV reali che passano dal codice vero, con l'LLM registrato e le azioni su `Recorder().ctx()`. Le correzioni passano per tre punti unici: un corpus etichettato, una sola decisione a fine stream, un solo budget di latenza.

**Tecnologie:** Python, SDK `anthropic`, faster-whisper, Silero VAD, openWakeWord, faster-qwen3-tts, pywinauto. Graphify per orientarsi nel codice.

**Fonti (solo in locale, fuori da git):**
- `reports/Affidabilità assistente vocale Claudio.md`: la ricerca, da leggere per intero prima di G1. Distingue verificato e dedotto.
- `research_notes/Affidabilità assistente vocale Claudio/`: le note con i link.
- `piani/IT-2-passo-0-report.md`: residui e idee del Passo 0, numerati qui sotto.
- Log della sessione analizzata: `logs/sessions/2026-11-05_17-00/`. I WAV di quella sessione in `logs/audio/` stanno ruotando: **il primo commit di G0 ferma la rotazione**.

Piano approvato da JenJo il 05/11/2026 dopo grilling e deep research. I numeri di riga valgono per `main` = `df22b88`: se il codice si è spostato, fanno fede le funzioni.

## Come lavorare

1. Ramo `it2/passo-0b`, già creato. Un gruppo alla volta, nell'ordine qui sotto.
2. All'inizio di ogni gruppo: `graphify update .`, poi scrivi i task e **mostra a JenJo un piano breve del gruppo**.
3. Implementer su Sonnet, revisori indipendenti su Opus. Gli implementer **non committano**. Test prima del codice, visti fallire.
4. Fine gruppo: mostra diff e test, aspetta l'ok, poi commit piccoli in stile Conventional Commits. Un'approvazione vale solo per ciò che è stato mostrato.
5. Fine rilascio: gate di rilascio qui sotto, README e CHANGELOG, tag solo dopo l'ok. Poi riassumi a JenJo cosa è stato fatto, costi, latenza misurata, rischi, e fermati.
6. **Notifica push** (`PushNotification`) quando serve l'attenzione di JenJo o quando hai finito. Messaggio breve con ciò che deve fare. Arriva sul telefono solo con Remote Control connesso: a inizio thread JenJo digita `/remote-control`, poi tu mandi una notifica di prova. Se il tool risponde «not sent», dillo in chat: non darla per arrivata.
7. **Token:** non compattare mai. Controlla il contesto a ogni passo importante dal `usage` dell'ultima riga assistant del transcript (`input_tokens + cache_read_input_tokens + cache_creation_input_tokens`). A circa 850.000 aggiorna questo file e proponi un thread nuovo.

| Rilascio | Gruppi | API nei test (stima) | Stato |
|---|---|---|---|
| v0.4.2 | G0-G6 | ≤ 0,30 $ | G0 e G1 fatti, G2 in corso |
| v0.4.3 | G7-G10 | ≤ 0,20 $, cloud a parte | da fare |

### Stato lasciato

**Aggiornato il 07/11/2026. G1 è chiuso: commit fino a `e3b7348`, nessun push.** Prova G di JenJo fatta; manca solo la breve «G bis» in `piani/IT-2-passo-0b-prove.md`. Registro di G1: `.superpowers/sdd/IT-2-passo-0b-G1-task/progress.md` (fuori da git). Il gate di replay dà 39 superate, 3 note, 0 fallite.

**Adesso G2, corsia A (aggiornato l'08/11, 04:10). Piano approvato da JenJo il 07/11: `piani/IT-2-passo-0b-G2-task.md`.** Registro, da leggere per primo: `.superpowers/sdd/IT-2-passo-0b-G2-task/progress.md` (rulings, minori rimandati, alberi di ogni task). **Niente di G2 è committato:** tutto è nel working tree.
- Fatti e revisionati, con verifica per mutazione: task 1, 3, 5, 6, 7, 8, 9a. Il task 9 è stato diviso in 9a e 9b. Il task 3 è partito prima della prova H0: dalla catena audio dipende solo il valore della soglia, che sta in configurazione (-0.75 provvisorio).
- In corso: task 9b, secondo giro di correzioni.
- Aspettano la prova H0 di JenJo: task 2 (replay offline, può cambiare la soglia) e task 4 (registratore e 145 riprese, sblocca la corsia B).
- Poi task 10: lancio unico del test sulle 145 riprese, revisione dell'intero gruppo su Opus con mutazioni, gate di replay con cassette nuove (`probes/record_low_brake.py`, budget G2 0,05 $), prova H2 di JenJo, diff, commit dopo l'ok.
- Da dire a JenJo nel riepilogo: la soglia vale anche al livello 3; «Ok Claudio» detto per iniziare una richiesta nuova con una domanda aperta è un sì (spec); una frase lunga senza nome dopo «Confermi?» non diventa più una richiesta a Claude, riceve la seconda domanda; una risposta dopo il silenzio deve avere il nome; spegnimento e pausa annullano ogni domanda.

**Da guardare dalla prova G bis:** Haiku offre Sonnet solo a parole («dimmi solo se procedo»), senza un'offerta armata: in una scena 6 volte su 6 con la stessa cronologia, 3 su 4 con una nuova. Haiku può scrivere un ragionamento in inglese prima di un comando di livello 3, e viene detto con la domanda. Con Haiku muto Claudio dice «Spegniti. Confermi?». «Va bene. Confermi?» passa la regola senza nominare nulla. Candidati per G3.

**Detto a JenJo, sul costo del Task 10:** trattenere le prime due frasi ritarda la prima parola di **circa 250-400 ms** sulle risposte brevi (dai log: fine messaggio meno prima parola, 250 ms a p50 fino a 60 token, 391 ms fino a 120). La mia motivazione «risposte quasi invariate» era sbagliata: è confrontabile con i circa 260 ms del thinking adattivo. Il vantaggio resta altrove: doppia frase azzerata per costruzione, nessuna impostazione, nessun blocco di thinking da rimandare. Opzione per G7: trattenere solo se la richiesta contiene un verbo d'azione, così le domande non aspettano.

Budget di G1 approvato 0,10 $: spesi circa 0,09 $, più la sessione dal vivo.

### Corsie, scelta di JenJo (07/11)

| Corsia | Gruppi | Quando |
|---|---|---|
| A, in questo checkout | G2 → G3 → G6 | Subito dopo la chiusura di G1 |
| B, in un worktree | G4 → G8 | **Solo quando le 145 riprese di G2 sono fatte**, per non chiedere a JenJo due prove insieme |
| Poi, in fila | G5, infine G7 | G5 dopo G2: rifà in parte gli stessi esperimenti su Whisper. G7 per ultimo: misura il risultato di tutti |

**Preparare la corsia B** (dopo l'ok di JenJo): `git worktree add ../Claudio-voce -b it2/passo-0b-voce it2/passo-0b`. In git non ci sono `.env`, `.venv`, `models/`, `samples/`: copia `.env`, fai `uv sync`, collega `models` e `samples` con una junction. La configurazione del repo è condivisa: controlla `git config user.email` lì dentro. **Regole tra corsie:** B non cambia prompt né tool, altrimenti le cassette saltano per A; un solo Claudio acceso e una sola prova di JenJo alla volta; nessuna misura di latenza mentre l'altra corsia usa la GPU; B si fonde in `it2/passo-0b` a fine gruppo, dopo un rebase. File contesi: `i18n.py`, `config.py`.

**Decisioni di JenJo sulla prova G:** rumore fuori da log e disco (fatto; 270 clip salvate in `samples/corpus/negativi-rumore`, utili per riaddestrare il modello «claudio», debole: il rumore arriva a 0,96 e una richiesta vera parte da 0,25); per un'azione senza conferma resta solo la frase dopo; il thinking resta spento.

**Da guardare dalla prova G:** il testo si incolla al cursore, quindi può finire in mezzo a un documento vecchio (16:54, G6); Claudio ha detto «nella scheda nuova» senza che un tool l'avesse aperta (17:03, promessa inventata); 4 promesse vere senza azione nella sessione (il rilevatore le ha viste); «lascia stare» non è stato provato davvero dal vivo: il testo era già scritto 2 s prima (da rifare in G2).

**Metodo, lezione di G1:** i test verdi con i finti non bastavano. La verifica per mutazione e la revisione dell'intero gruppo hanno trovato ciò che le revisioni a lettura per singolo task non vedevano, compreso il guasto più serio. Nei gruppi con sicurezza vanno chieste entrambe. Un finto che sposta la finestra all'istante nasconde le corse tra un'azione e la successiva.

**Rimandati da G1:**

| Gruppo | Cosa |
|---|---|
| G2 | Un'offerta di Sonnet mai sentita resta armata: una frase detta sopra può chiamare Sonnet (una chiamata, nessuna azione sul PC). Serve un `played` riempito nel finto `LongSpeaker`. Dopo un sì, senti solo «Confermi?» mentre il bersaglio può essere un altro: scheda «Blocco note online» davanti, `focus_window` fallito e poi `type_text`, un lancio che batte un focus successivo. Un'azione armata tolta da `cancel_pending` perde tutto il turno dalla cronologia. «Claudio, sì» sopra un «Confermi?» che sta suonando arriva a Haiku senza contesto. «Dicevo» può ripetere la promessa di un'azione fermata. Una richiesta locale sopra una risposta ferma l'azione in silenzio. `handle_typed` può azzerare lo stop mentre uno stream vecchio gira. `BRIDGE_RUN` («Eseguo.») resta in cronologia |
| G6 | Il testo di una pagina può guidare le azioni di livello 1, e `open_website` può portare fuori quel testo in un URL senza un sì: conseguenza della scelta sul testo delle finestre, `level()` non guarda `_outside_text` per questi tool. Elenco delle finestre private: mancano nomi come `secrets.yaml`, `token.txt`. L'uscita di `run_command` e i titoli delle finestre non sono protetti dal recinto dei dati. `rich_text` non controlla i campi password |
| G7 | Le attese nuove sul percorso Claude, da misurare dal vivo: fino a 1,2 s dopo `open_app` e `open_path`, fino a 1,2 s di rilettura dopo la scrittura. Una cartella col nome italiano («Download») aspetta sempre il tetto. `GetText(-1)` non ha un tetto su un documento enorme |
| G9 | `test_pc_tools.py` è a 2600 righe: va diviso con un commit che sposta soltanto, spostando prima i finti. Rilevatore delle promesse: togliere `mi` e `vi` nudi («Devo salvarmi»). 5 `SSLError` su 240 chiamate nell'A/B, tutti a metà stream, nessuno nei log veri: ora gestiti, causa ignota, e il messaggio finirà nel log. Un retry interrotto da «basta» sottostima la spesa di 0,001 $ al massimo. `log.exception` su uno stream rotto può scrivere l'input intero del tool |

Il resto di questa sezione è del 06/11. **G0 è chiuso.** Le prove B, C e D sono fatte: la B con 105 etichette (Whisper sbaglia l'11,1% delle parole su 42 frasi per Claudio), la D con 38 clip superate e 4 guasti noti. **La C ha trovato un guasto di sicurezza al livello 2, e il suo test audio resta rosso:** la correzione è il primo lavoro di G2, vedi lì. Manca la prova F, prima di G3. I task sono in `piani/IT-2-passo-0b-G0-task.md`, le prove in `piani/IT-2-passo-0b-prove.md`. Il prossimo gruppo è G1.

`.claude/skills/graphify/` resta fuori da git per scelta di JenJo: si rigenera con `graphify install --project`. Graphify è uno strumento globale, senza versione fissata: oggi è la 0.9.103.

In `~/.claude/settings.json`: `autoCompactEnabled: false` e notifiche attive. Il grafo è in `graphify-out/`.

Budget: 2,58 $/giorno, al massimo 80 $ al mese, dai crediti Max. Le cassette del gate sono costate 0,0035 $, fuori da `data/spend.db`.

**Il gate di replay** si lancia con `uv run python scripts/replay_gate.py --report`, senza chiave e senza spesa. Oggi su 42 clip: 38 superate, 4 guasti noti, 0 fallite. Tre cose da sapere prima di usarlo:
- Quando un gruppo corregge un guasto noto, la sua clip diventa `fixed` e il gate esce con 1: togli il segno da `samples/gate/manifest.jsonl`. Le clip sono `171248` (G1, riprodotto da Claude vero), `171251` (G3), `171609` e `125104` (G2).
- Una modifica a `system`, ai tool o a `data/CLAUDIO.md` fa mancare tutte le cassette: il gate lo dice, e `--record` chiede conferma. Se Whisper scrive un testo diverso, cambia la chiave di quella clip.
- Solo quelle 4 clip sorvegliano un guasto. Le altre proteggono route, tool e sicurezza. La riaccensione non si può riprodurre: il suo WAV non contiene il nome, per scelta. Su 29 clip del 05/11 manca la nota, quindi Whisper usa il prompt di oggi e un cambio di vocabolario può cambiarne il testo e la chiave.

**Scoperte di G0 per G2**, provate da test `xfail` in `tests/unit/test_timeline.py`:
- Una frase detta mentre Claude pensa si perde, non solo «basta»: `handle_frame` non ha un ramo per `RESPONDING` (`app.py:378-387`). Con circa 3 s di attesa su Claude, questa finestra è aperta a ogni richiesta. Lo stesso vale per un «basta» durante Whisper se la richiesta va avanti. **JenJo deve decidere come trattarla.**
- La riga delle metriche può perdere il testo della risposta interrotta: `_marked` lo salva nel suo `finally`, che non scatta se `_produce` sta generando il pezzo dopo o ha la coda piena. Con Jack succede solo mentre Claude sta ancora scrivendo.
- `src/claudio/safety/confirmation.py:89`: «Sì?!», «Sì?.» e «Sì?...» risultano un sì, perché conta solo l'ultimo carattere. Da vedere con le frasi registrate di JenJo.
- Le 4 sottoclassi `LongSpeaker`, `SentenceSpeaker`, `StartSpeaker`, `CueSpeaker` restano fuori dal contratto: leggono tutto prima di `on_first_audio` e non riempiono `played`. I test di barge-in ci si appoggiano: spostali su `TimedSpeaker` in G2.
- `timeline.py` esegue Whisper inline, quindi i fotogrammi non arrivano mai durante Whisper: per provare 5.2 #5 serve dargli una durata in fotogrammi.
- Una risposta a un testo scritto non si ferma a voce: stessa radice di `handle_frame`.
- Gli helper `talk` (`test_conversation.py:460`), `feed` (`test_safety.py:359`) e `asked` (`test_own_sound.py:189`) devono aspettare `own_sound.playing` prima di un test con un conteggio esatto di fotogrammi sordi: oggi è una corsa innocua.
- `test_timeline.py:111` e altre precondizioni stanno sotto `raises=AssertionError`: una scena rotta si legge come `xfail`. Quando G2 le corregge, `strict` le fa saltare comunque. Il motivo «discovery» di quel test va rinominato quando JenJo decide.
- `build_brain` non inietta un orologio: `time.monotonic` decide `history_reset_s` (`claude.py:663`) e le scadenze da 60 s (`claude.py:591,597`). Uno stallo lungo cambia percorso e chiave. G2 rifà quelle scadenze.
- Il contratto dello speaker non copre l'errore dello stream né il produttore cancellato senza `await`; `first_audible` non ha un test per «solo la prima volta», e il punto è `_resume_reply`, che G2 rifà. `holding()` salta ogni `*.json`.

**Rimandati da G0 agli altri gruppi:**

| Gruppo | Cosa |
|---|---|
| G1 | Un `overloaded` a metà stream si legge «errore di Claude: 200» (stesso punto di 5.2 #13) |
| G3 | Manca un test sulle note dei WAV scartati; `live_row` non trova la riga delle clip in `rejected/` |
| G5 | `app.py:785` dà la lingua di un tentativo scartato (latente con una sola lingua). Clip `125043` fragile: ci arriva dalla lista chiusa, e 12 piccole variazioni dell'audio su 15 cambiano il testo. Clip `171510`, capita dal vivo e vuota nel replay: causa ignota. 3 clip con picchi 0,001-0,053 da cui il modello non scatta: causa ignota |
| G6 | `move_file` di un'intera cartella della casa finta cambia la chiave e resta per la scena dopo: nessuna clip la usa |
| G7 | In `benchmark_latency.py`, `PATHS`: `confirm_timeout` non corrisponde mai, `guess*` finiscono in «altro», la risposta a un «Confermi?» di Claude conta nel percorso Claude: **chiedere a JenJo prima di misurare.** Righe di barge-in con `m2_ms` sotto `m1_ms` e `speech_ms` negativo. Manca un test per `Udibile` |
| G9 | `label_wavs.py`, tasto `u`: con `labels.jsonl` aperto in un altro programma l'annullamento chiude lo script con un traceback, senza perdere etichette (manca un `except OSError`); nessun test copre una morte a metà annullamento, provata a mano. `config/claudio.toml:334` stima 84 KB a WAV, misurati circa 120. `test_sentence_splitter.py::test_group_stream_starts_with_what_has_arrived_and_groups_the_rest` dipende dal tempo reale: 2 fallimenti su 24 con carico estremo. `benchmark_latency.py --help` si rompe con `≈` su cp1252 se è in pipe. `error_outcome` perde il motivo di una nota illeggibile |

**Da assegnare:** la «prova LLM dal vivo con k=3» del gate di rilascio non ha ancora uno script né un gruppo. **Per JenJo, alla prova B:** 16 righe su 71 della sessione del 05/11 hanno il WAV già ruotato, quindi quelle frasi non si etichettano.

## Vincoli globali

- Priorità del progetto: affidabilità, costi, latenza, sicurezza, pulizia del codice. In conflitto vince la prima.
- Budget API 2,58 $/giorno, al massimo 80 $ al mese dai crediti Max (05/11/2026). Modelli e prezzi solo in configurazione.
- I livelli di sicurezza li decide il codice. Un retry ricostruisce gli argomenti, **non autorizza**: ripassa dalle stesse conferme.
- Repo pubblico: WAV, cassette, corpus e memoria mai in git. Da spento nessun testo né audio salvato, salvo la riaccensione.
- Commit con l'identità JENJ0 noreply: controllala prima di ogni push, mai `git push --all`.
- Ogni script di prova usa `Recorder().ctx()` e il `FakeDesktop` del gate: un `ActionContext()` di default rifiuta le azioni, ma non protegge `PcTools`, e `build_responder` costruisce comunque un `WindowsDesktop` vero. Niente API inventate: carica la skill `claude-api` prima del codice API.
- Sotto Git Bash: `uv run python -m pytest`. Usa `graphify explain` e `graphify path`, più precisi di `query`.
- `/plugin` non esiste in questo ambiente.

## Decisioni di JenJo

| Tema | Scelta |
|---|---|
| «Dicevo» | Da solo soltanto dopo comandi del router locale; dopo una richiesta a Claude mai. «Continua» o «riprendi» ridice il testo mancante esatto, in locale |
| Frasi nella finestra | Le giudica Haiku, attivo solo con ≤ 2 falsi silenzi e ≤ 6 false risposte su 40 per tipo |
| Percorsi | Cerca e apre su tutti i dischi locali, livello 1. Spostare solo in `~`. Eseguibili e rete sempre rifiutati |
| Lettura file | Se la finestra è vuota guarda lo schermo; `read_file` di livello 1 **solo in `~`**, segreti sempre rifiutati, livello 2 fuori da `~`, testo trattato come esterno |
| Conferme | Frase col nome = richiesta nuova: annulla dicendolo e risponde. Senza nome e poco chiara: chiede una volta, poi annulla. Mai azione senza sì |
| Attesa | Nessun suono nuovo: prima si tagliano i tempi. Si riapre solo se resta oltre 2 s |
| Voce | Stesso modello con registro corretto. Riaddestramento ammesso se i dati lo giustificano |
| Trascrizione | Prima i fix locali. Cloud solo dopo, ogni invio di audio approvato prima |
| Controllo WAV | Script locale prima di ogni tag, esito nelle note di rilascio. La CI pubblica non cambia |
| Etichette | Circa 2 ore di JenJo in più sedute, circa 300 frasi su 3 sessioni e 2 giorni |
| Bluey | Regole d'azione ora; volto e cursore in una fase futura |
| Contesto (06/11) | Claudio ragiona sulla richiesta attuale in relazione alle precedenti della sessione, soprattutto l'ultima, comandi locali compresi |
| Conferma vocale (06/11) | La correzione viene dopo G1, come primo lavoro di G2. Livello 2: soglia `avg_logprob` -0.75 sul sì e una sola seconda domanda, insieme. Livello 3: per ora solo voce, un'eventuale conferma non vocale si decide in G6 |
| Pubblicazione (06/11) | Push del ramo alla fine di v0.4.2, se non ci rallenta |
| Testo delle finestre (07/11) | Dopo `type_text`, `open_app`, `open_path` a Claude arrivano il verdetto del codice **e** il testo della finestra, marcato come dati. Quei tool contano come testo esterno; le finestre private non si leggono mai. Riconfermato anche per `open_app`, sapendo che i ricordi oltre 80 caratteri vengono rifiutati finché la nota resta in cronologia e che un Blocco note che riapre vecchie schede ne manda il testo |
| Attesa della finestra (07/11) | Dopo `open_app` e `open_path` fino a 1 s, solo sulle richieste passate da Claude. Da misurare dal vivo |
| «Confermi?» senza frasi (07/11) | Se in quella risposta non è stata sentita nessuna frase, il codice legge l'azione. Il «Ci guardo.» del codice non conta come frase |
| Frase mentre Claude pensa (07/11) | Come quando parla: «basta» ferma, una frase col nome prende il posto della richiesta, il resto si ignora |

## Attenzione ai casi limite

Da coprire con test nel gruppo indicato:

1. **Testo incollato nella finestra sbagliata (G1):** il fuoco cambia tra `open_app` e `type_text`. Non deve incollare altrove, deve dirlo, e gli appunti di JenJo tornano com'erano anche in caso di errore.
2. **Interruzione durante il nuovo tentativo (G1, G2):** «basta» mentre il retry è in corso. Dopo non parte nessuna azione e il risultato tardivo viene scartato. Un tool già eseguito resta in cronologia, così non si ripete.
3. **Marcatore spezzato (G3):** `[[SILENT]]` diviso tra due chunk non deve arrivare alla voce. Rifiuto, timeout ed errore API non contano mai come silenzio.
4. **`read_file` su percorsi ostili (G6):** segreti (`.env`, `.ssh`, `*.pem`, `*.key`, profili browser, archivi di credenziali), junction che escono da `~`, flussi alternativi NTFS, file enormi, binari, codifiche strane.
5. **Ricerca su dischi interi (G6):** junction che creano cicli, cartelle negate, chiavette, dischi di rete mappati, tempo e memoria dell'elenco in background. L'avvio non deve rallentare.

## Gate di rilascio

`uv run python -m pytest`, `uv run ruff check .`, `uv run mypy src` verdi; controllo WAV verde con esito nelle note; prova LLM dal vivo con k=3; sessione di 10-15 minuti di JenJo con checklist, che riprende anche i punti ancora aperti del test manuale in `IT-2-passo-0-report.md` sezione 7.

---

## v0.4.2

### G0: basi di misura

Si fa per primo perché ogni guasto perso è un caso di test che non torna.

- Commit di Graphify, ponytail ed esclusioni.
- **Conservare i WAV:** alza `keep_audio` (oggi 50, `config/claudio.toml:334`, circa 22 MB) a un tetto di circa 1 GB. Salva con ogni WAV il prompt esatto di Whisper e il guadagno applicato: `vocabulary.py` cambia il prompt nel tempo, e senza non si riproduce nulla.
- **`ActionContext` sicuro di default** (idea 2): azioni vere solo con un costruttore esplicito, `ActionContext.real()`.
- **Nuova metrica primaria:** fine del parlato → primo campione udibile, per percorso (locale, conferma, Claude). `m1_ms` parte già dalla fine del parlato ma si ferma prima dell'audio; `m2_ms` include il parlato. Restano per continuità. Aggiorna il `CLAUDE.md` del progetto, mostrando il diff.
- **Chiudi il Passo 0:** `scripts/benchmark_latency.py --log` sulla sessione del 05/11 per il punto «la latenza non peggiora», e riportalo.
- **Script di etichettatura:** riproduce un WAV, propone il testo, JenJo conferma o corregge con un tasto. Salva fuori da git, con destinatario (per Claudio o no) e sessione. Dev e test divisi per sessione.
- **Finti verificati:** una suite di contratto che gira su `FakeSpeaker` e sullo speaker vero. Oggi `speak` unisce tutto prima di `on_first_audio` e `interrupt()` è `pass` (`tests/unit/test_app.py:115-132`).
- **Frasi di sicurezza come audio vero** (idea 10), con negativi quasi uguali.
- **Controllo WAV:** `scripts/replay_gate.py`, dal callback del microfono con wake word, VAD, Whisper e router veri, LLM registrato sul client di Claudio (chiave: hash di modello, system, tool, messaggi), azioni su `Recorder().ctx()`. Il gate duro guarda l'esito (wake, intento, tool, argomenti, classe di sicurezza); le trascrizioni esatte vanno in un report di differenze. Parti da 30-60 clip tratte dai guasti veri.
- **Test a timeline:** orchestratore vero, modelli finti, orologio iniettato. Riprendi i casi di `tests/test_agent_session.py` di LiveKit come lista di controllo. Non migrare a framework.
- Limita G0 a circa una settimana: ciò che il replay non può provare (eco, barge-in reale) va in timeline o in checklist.

**Fatto quando:** lo stesso WAV ripetuto 100 volte dà esito identico, oppure è documentato cosa varia e il gate ne tiene conto; un test fallisce se il gate non usa il recorder; i WAV restano; la nuova metrica è nelle righe.

### G1: azioni vere

Causa verificata del Blocco note finto: il testo di `type_text` sfora `[pc] max_tokens = 400` (`config/claudio.toml:293`, usato a `claude.py:736`); `_Leg.read` (`claude.py:914-936`) scarta giustamente la chiamata tagliata; `_stream` (`claude.py:831-840`) dice «Mi fermo qui»; `turn.done` si riempie solo per ciò che è partito (`claude.py:540`), quindi Claude non sa nulla. 4 casi su 4 nella sessione. C'è un secondo ostacolo: `type_max_chars = 500` (`tools.py:904`) rifiuterebbe comunque un testo lungo.

- **Una sola decisione a fine stream** per: `tool_use` completo, tagliato, marcatori (G3), `refusal`, errore, frase d'azione senza chiamata.
- **Chiamata tagliata:** un solo retry con `max_tokens` alto (2048) e `tool_choice` forzato su quel tool. **Solo su Haiku:** Sonnet 5.5 e Opus 5.5 rispondono 400 su `any` e `tool`. Se è ancora tagliata: stato `not_run_cut`, a voce «non l'ho fatto». Mantieni la guardia: l'SDK analizza il parziale con `partial_mode=True`, e solo `stop_reason` prova la completezza.
- **La nota non deve ingannare:** `_read_outside` (`claude.py:631-636`) conta i tool leggendo le note, e da lì dipende un livello di sicurezza. Un tool mai partito non deve risultare eseguito: serve un formato distinguibile, con test.
- Alza `type_max_chars` in linea col tetto di token. Verifica il taglio a voce nel codice, così il limite del parlato non dipende da `max_tokens`.
- **I tool riportano ciò che osservano** (idea di Bluey): dopo `type_text`, `open_app`, `open_path`, testo letto dalla finestra con tetto di tempo e nodi. Lo screenshot resta ultima risorsa. Errori come frasi chiare per il modello.
- **Promesse senza azione** (17:04:15, 17:12:48): rilevatore su frasi italiane in prima persona, quando i tool erano offerti e non c'è `tool_use`. **All'inizio solo nei log.** Il retry automatico si attiva dopo aver visto i falsi positivi, con un'approvazione a parte. Più una riga di prompt, mai da sola.
- **A/B di `effort` e thinking:** la documentazione di Haiku 5.5 dice che il testo simile a ragionamento compare di più con thinking spento o effort `low`, cioè le impostazioni attuali (`config.py:442-443`). Misura costo e latenza prima di cambiarle.
- `_NoteFilter` con nota mai chiusa (5.2 #11); JSON o UTF-8 non validi a metà stream, `claude_error: 200` (5.2 #13); «Ci guardo» ripetuto o imitato da Haiku (5.2 #16).
- Salva e ripristina gli appunti attorno all'incolla (`paste_min_chars = 40`), se la prova regge. Altrimenti dillo a JenJo.

**Fatto quando:** con client finto, la scena del Blocco note con testo lungo scrive tutto; un taglio doppio dà «non l'ho fatto» e la nota corretta; dal vivo, «scrivi un testo lungo nel Blocco note» lo scrive davvero e Claudio verifica ciò che è scritto.

### G2: interruzioni e stato del turno

Causa verificata: `worth_resuming` (`resume.py:87`) scarta le risposte che finiscono con una domanda, e Haiku chiude le storie con «Vuoi il seguito?». Quindi «Dicevo» non parte dopo «che ore sono?» (log 17:08:15 e 17:08:35) e parte invece dopo una richiesta nuova (17:13:18 e 17:13:40). Nessun framework letto riprende da solo dopo un turno vero dell'utente.

- `_resume_reply` (`app.py:1631`) e `_unsaid` (`app.py:1608`): ripresa automatica solo se la richiesta è stata risolta dal router locale. Rivedi `worth_resuming` per le storie che finiscono con una domanda.
- «Continua» e «riprendi»: comando locale che ridice il testo mancante esatto, 0 $. Oggi Claude inventa una nuova puntata (17:08:15).
- **Cronologia:** salva ciò che è stato sentito. `_remember_cut` (`claude.py:650-659`) non riceve `played` mentre `DrainedStream` (`resume.py:23-39`) continua a leggere. Prima un test a timeline che lo dimostri: il percorso completo non è stato tracciato.
- **`turn_id` lato client:** testo vecchio scartato, risultati di tool eseguiti tenuti, tool mai partiti segnati. Consolida `CutReply`, `_Turn.done` e `_Action` prima di aggiungere codice.
- **Conferme:** `_open_question` (`responder.py:389`) e il percorso di `_confirm` in `claude.py`. Frasi vere: 17:16:09 e 17:20:17. Regola nelle decisioni qui sopra.
- Testo perso nella riga dopo «basta» (5.2 #4); richiesta persa con il controllo Whisper in corso, `_judge_stranded` (5.2 #5).
- **Prova dal vivo del 06/11, 12:50-12:52** (sessione `2026-11-06_12-48`), «chiudi la finestra» fallito 3 volte su 3. Le cause, da verificare con test a timeline:
  - **La risposta cade nella finestra sorda:** dopo «Confermi?» la voce nel tratto sordo è 64, 352 e 224 ms (`deaf_speech_ms`). Una volta si è sentita solo la coda e Whisper l'ha scritta come un rifiuto; un'altra è rimasto vuoto. `_join_deaf` esclude apposta questa finestra: sbaglia dalla parte sicura, ma la conferma è quasi inutilizzabile. Chiedi a JenJo di riascoltare `20261106-125111-875_ok.wav` per sapere cosa aveva detto.
  - **Una conferma locale non sopravvive al silenzio:** `cancel_pending(silence=True)` (`responder.py:303-313`) azzera `_pending`, mentre un'azione di Claude resta armata (`ACTION_TTL_S`). Quindi «Claudio, confermo» finisce a Claude. Allinea: un «sì» col nome, entro lo stesso tempo, risponde alla domanda aperta. Mai azione senza sì.
  - **Claude non sa dei comandi locali:** la cronologia si riempie solo in `_remember` e `_remember_cut` (`claude.py:647,659`), quindi risponde «non vedo nessuna azione in attesa». Dagli l'ultimo comando locale e la domanda aperta, con poche parole per non spendere: è la decisione «Contesto».

- **Conferma vocale, primo lavoro di G2** (prova C del 06/11). Sul percorso reale 3-5 negativi su 64 eseguono l'azione e 28 sì su 44 la eseguono. Su risposte corte e veloci Whisper toglie o sostituisce parole. **Al riascolto JenJo sente la voce distorta, e in parte quasi tagliata, in tutte le 6 riprese controllate:** il danno peggiore è in `non_va_bene__1`; in `si_fa_tardi__1` la «r» di «tardi» è chiara, e Whisper scrive comunque «Sì, va bene.». Le riprese sono grezze, quindi la distorsione nasce prima del codice di Claudio. L'attacco della voce non è tagliato, e le liste del parser non sono la causa. **Il contributo del microfono non è misurato: la diagnosi guardava solo il taglio iniziale.** Il livello 3 non ha accettato nessun falso sì, ma passa 1 ripresa su 4. Dettagli solo in locale: `reports/Conferma vocale sicura con Whisper.md` e `reports/Diagnosi prova C.md`; script e risultati in `research_notes/Conferma vocale sicura con Whisper/prove/`. In ordine:
  0. **Prima la distorsione, perché sposta la soglia.** JenJo ascolta 3 WAV dal vivo: suonano uguali alle riprese? Si spengono gli effetti audio di Windows su quel dispositivo (cuffia USB, venditore 054C, sul percorso PC non è Bluetooth), si prova la modalità raw, si confronta un altro microfono sulle stesse 10 frasi. Un indizio, non una prova: nelle 6 riprese riascoltate il 19% della voce cade in buchi oltre -30 dB, contro il 9% di tutte (sono solo 6, scelte perché fallivano). Se la catena audio cambia, soglia e prompt si riprovano da zero.
  1. **Parser, senza costi:** qualunque «?» nel testo non è un sì (`confirmation.py:89` guarda solo la fine). «No, Claudio.» deve rifiutare: oggi `only_wake_words` (`intake.py:218`) fa riascoltare e la domanda resta aperta. «Sì, Claudio.» e «Ok Claudio.» devono valere.
  2. **Replay offline, una variabile alla volta, prima di fissare la soglia:** `without_timestamps=False`, `beam_size=1` sulle sole risposte, un prompt congelato per le risposte. Si adotta solo ciò che non peggiora né i sì né i falsi sì.
  3. **Scelta di JenJo: `avg_logprob` -0.75 sul sì e una sola seconda domanda, nello stesso commit,** con la soglia in `config/claudio.toml`. Da sola la seconda domanda alza i falsi sì. Simulato sulle 108 riprese, non validato: 34-36 sì su 44 e 1-2 falsi sì su 64; circa 1 sì su 3 sente la seconda domanda. `_answer` (`responder.py:516`) richiede una volta invece di annullare, nominando azione e risposte; il floor va in `_answer` (`intake.py:268`), che oggi salta il filtro di confidenza. Lo stesso vale per il «Confermi?» di Claude (`claude.py:311,327,478,1024`), mai provato perché manca la cassetta.
  4. **Test nuovo, con le soglie scritte prima di registrare:** registratore con uno stream aperto per tutta la sessione, riprese sotto il picco rifiutate, passo naturale. 93 non-sì nuovi con al massimo 1 errore (sotto il 5%, unilaterale al 95%), 8 «sì?» a parte, 44 sì con obiettivo 33. `test_safety_audio.py` deve vedere ciò che vede il percorso reale: oggi, senza guadagno né Endpointer, esagera (24 e 6 contro 28 e 3).
  - **Da sapere:** con la soglia, `whisper_prompt` diventa configurazione di sicurezza: ogni modifica del prompt vuole il test di regressione, o il prompt congelato. «sì?» scritto `Sì.` resta irrisolto al livello 2. Altri due rischi senza una misura: una pausa oltre `end_silence_ms` chiude la risposta, quindi «sì», pausa, «anzi no» ha già eseguito; `answer_min_dbfs` legge il livello dopo il guadagno e non scarta nulla da +8 dB in su.

**Fatto quando:** le sequenze vere delle 17:08 e 17:13 sono test a timeline con esito corretto; «Dicevo» riparte dopo «che ore sono?» anche se la storia finiva con una domanda; il test nuovo delle frasi di sicurezza passa con il criterio scritto prima.

### G3: frasi per Claudio

Causa verificata: `short_request` (`intake.py:390`) vuole almeno 2 parole, quindi «Scrivi.» cade; su 32 frasi scartate nella storia molte erano per Claudio.

- **Subito:** verbo singolo accettato.
- **Poi, solo dopo le etichette di JenJo:** marcatore all'inizio della risposta normale di Haiku, `[[SILENT]]` e `[[WAIT]]` con `stop_sequences` e buffer iniziale. Niente chiamata in più: una separata aggiungerebbe circa 335 ms a p50. Bozza del prompt nel report, non testata. Prima misura offline (circa 0,10 $ per 172 frasi × 3 esecuzioni).
- Regole: nel dubbio risponde; `refusal`, timeout ed errori non sono mai silenzio; un silenzio non riapre la finestra; il nome nel testo salta il filtro (`addressed_to_claudio`); il verdetto è visibile nella chat.
- `[[WAIT]]` usa `Kind.CONTINUE`. **Prima verifica** che il microfono continui a catturare nei circa 2,4 s del verdetto: non è stato controllato. Una frase lasciata in sospeso nella sessione del 05/11 non è andata in `CONTINUE`: va capito perché.
- Smart Turn v3.2 solo se il replay offline lo giustifica: richiede `transformers` e un vetting. LiveKit escluso per licenza.

**Fatto quando:** test su ≥ 40 frasi per tipo con dev e test separati: falso silenzio ≤ 2/40, falsa risposta ≤ 6/40, meglio del filtro attuale (McNemar p < 0,05), 0 marcatori parlati, ≤ 50 ms in più al primo testo a p50. Regola di arresto: filtro disattivato se più di 5 dei successivi 50 erano per Claudio. **Non attivarlo se il test non passa:** riporta a JenJo.

### G4: voce

Cause verificate: `TakeJudge.reasons` (`voice_check.py:239-256`) non ha una regola per le take basse; dopo i tentativi suona l'ultima, non la migliore (`jack.py:464-494`, letto dalla ricerca, da riverificare); `_caps` (`spoken_text.py:110-118`) sillaba solo le sigle senza vocali. Provato: `AMD → amd`, `RAM → ram`, `NVIDIA → nvidia`. Dati: 89 pezzi suonati, da 78 a 281 Hz, 18 fuori da 100-170; 27 take scartate su 116; livello grezzo da −31,5 a −9,9 LUFS. In ordine:

1. **Sigle:** sillaba quelle impronunciabili (AMD) con un elenco per le altre. Test con `to_spoken_it`.
2. **Rimisura gli estremi** con un secondo tracciatore di pitch: 78 e 281 Hz possono essere errori d'ottava. Ricalcola `jack_usual_hz`: 132 Hz è la mediana delle take generate, Jack vero sta a 100-126 Hz. Ripartizione alte/basse delle 18 anomale. Le righe dopo dipendono da qui.
3. **Take migliore invece dell'ultima**, regola per le basse (specchio di `HIGH_ST`).
4. **Scarto di tono contro lunghezza del pezzo:** se i pezzi corti sono i peggiori, alza `first_clause_chars` e `jack_group_first_chars`. L'apertura è giudicata sui primi 0,67 s, ma la parte instabile sembra essere nei primi 1-2 s (#343 di Qwen3-TTS, un solo utente).
5. **Correzione parziale del registro**, solo sui pezzi dopo il primo: sotto 2 st nulla, tra 2 e 4 st parziale, oltre scarta. Latenza non pubblicata: misura su un take da 10 s a −4, −2, +2, +4 st, 20 ripetizioni. Candidati: `python-stretch` (MIT), `praat-parselmouth` (GPLv3+), `TimeStretch` del repo più resample. Ogni dipendenza passa da `supermatt:dependency-vetting`. Una pagina di ascolto alla cieca per JenJo, che sceglie anche sulla licenza. Il repo non ha `LICENSE` alla radice.
6. A/B di un code predictor più freddo per il «robotico». `temperature` e `top_p` arrivano solo al primo codebook: non spendere tempo su altri valori.
7. Take abbandonata che ruba gli eventi della risposta dopo (5.2 #6).

**Fatto quando:** al massimo il 5% dei pezzi suonati fuori da ±4 st dal nuovo riferimento, contro circa il 20% di oggi (soglia proposta: confermala con JenJo all'inizio del gruppo); nessuna take bassa senza segnalazione; JenJo preferisce il nuovo all'ascolto cieco; latenza riportata.

### G5: comprensione, solo locale

Nessun modello è dimostrato migliore su comandi italiani brevi. Tra cuffia e webcam, sui test di JenJo, ci sono 19,6 punti di differenza. Su registrazioni congelate, in quest'ordine:

1. **Prompt:** nessuno, solo frasi, solo nomi, frasi più 10-20 nomi. «giacca» contro «Jack» con `_fit` (`transcriber.py`), con e senza prompt. `hotwords` non è un boost: è lo stesso spazio di `initial_prompt`.
2. **Coda delle frasi:** «giacca→Jack» e «continua→continuo» colpiscono la vocale finale. Riproduci con 200-500 ms di coda in più e controlla se Silero o il gate della cuffia tagliano.
3. Beam 1 contro 5 sul primo passaggio; replay del livello a −12, −6, 0, +6 dB; spettrogramma per sapere se la banda si ferma a 4 kHz (profilo Bluetooth narrowband).
4. Tolleranza alle flessioni nel router; forme storpiate nel vocabolario (idea 7, 5.1 #5); candidati fonetici verificati con `_fit`. **Una correzione del testo non decide mai da sola** un'azione.
5. Candidati locali da provare a coppie: `large-v3-turbo`, Cohere Transcribe, Parakeet v3. Vetting e verifica Windows prima.

Niente riduzione del rumore: gli studi trovano che di solito peggiora Whisper.

**Fatto quando:** sul corpus etichettato la metrica primaria è il successo del comando, con le azioni sbagliate come limite, e il risultato è riportato a JenJo con McNemar. Le 40 frasi vecchie sono ormai un set di sviluppo: serve un test nuovo e chiuso.

### G6: PC

- **Percorsi:** `PathPolicy` (`files.py:83-151`) oggi unica. Ricerca e apertura ovunque, `files.move` (`files.py:488`, `tools.py:634`) solo in `~`: impostazione separata in `config/claudio.toml:295`. `RUNNABLE` (`files.py:31,155`) e rifiuto dei percorsi di rete restano. Misura l'elenco cartelle di background (oggi 6338 in 672 ms).
- **`read_file`:** regola nelle decisioni, conta come `OUTSIDE_TEXT` (`tools.py:71`, `claude.py:71`). `read_window` vuoto in VS Code (17:09:34): passa allo schermo.
- **Sicurezza da Bluey:** tasto di stop globale che svuota la coda e rifiuta azioni per qualche secondo; `type_text` rifiuta i campi password (UI Automation); blocco di Win+L e Ctrl+Alt+Canc in `press_keys`. Attenzione: `close_window` usa Alt+F4 di proposito.
- Haiku rinuncia a `press_keys` (17:06:55) e chiede a JenJo di farlo: prompt, verificato con una prova vera.
- **Prova dal vivo del 06/11, 12:55-12:57:** la richiesta di aprire una cartella con una sigla per nome ha aperto la cartella di un gioco, con quelle lettere dentro una parola più lunga. Tre cause: la cartella cercata esiste ma sta fuori da `~`, quindi `roots = ["~"]` non la vede; la ricerca abbina anche dentro una parola, e deve mettere prima il nome esatto e le parole intere; Whisper ha scritto male la sigla perché non è nel vocabolario (va con G5). Claude ha aperto «la prima» senza accorgersi che non c'entrava: se un nome non corrisponde, lo dice prima di aprire.
- **Router:** «apri la cartella <nome>», «ciao», «grazie» locali (idea 3, 5.1 #7). Negazioni: «non abbassare il volume» non abbassa (idea 5, 5.2 #3).

- **Livello 3, scelta di JenJo (06/11): si decide qui** se aggiungere una conferma non vocale, per esempio un pulsante in una notifica di Windows, per ciò che non si annulla. Oggi «confermo» non ha dato falsi sì ma passa 1 ripresa su 4, e un rischio sotto l'1% richiederebbe 299 prove senza errori. Il pulsante va tenuto fuori dalla portata degli strumenti di Claudio, che ha pyautogui e pywinauto. La parola parlata si sceglie sulle prove nuove di G2.

**Fatto quando:** l'apertura per nome di una cartella fuori da `~` e di quella di Steam funzionano; un test copre i livelli (dentro e fuori `~`, segreti, spostamento, eseguibile); il riassunto del file aperto funziona in VS Code; il tasto di stop è provato.

---

## v0.4.3

### G7: latenza

Dati: M1 su Claude 3086 ms di mediana (VAD 416, STT 529, Claude 984, voce 797), locale 1004. VRAM libera mediana 772 MiB nella sessione contro 11110 a Claudio chiuso, GPU al 56%: chi occupa la memoria non è stato stabilito, e `nvidia-smi` non dà la VRAM per processo. Misuralo.

- Meno passaggi Whisper su ogni frase (seconda opinione, lista chiusa, alternative): misura cosa rendono prima di togliere.
- Niente Whisper sui near miss `no_voice` mentre si gioca (idea 6, 5.1 #4).
- Voce dalla prima virgola (idea 4, senza riempitivi). TTL della cache del prompt (idea 12, 5.1 #8).
- Ogni correzione riporta il costo nella metrica di G0.

**Fatto quando:** latenza prima e dopo per percorso, su almeno 20 richieste vere. Se resta oltre 2 s, riapri con JenJo la scelta del suono.

### G8: Jack all'avvio

14 avvii a freddo tra 31/10 e 05/11: Jack fallito in 11 (`worker Jack: nessuna risposta in 90 s`). Riprova in background senza avviso a voce e caricalo prima degli altri (idea 1). Prima misura la causa (disco, antivirus, CPU, GPU): è anche nel Passo 3.

**Fatto quando:** causa scritta nella relazione; da freddo Jack arriva da solo senza riavviare Claudio.

### G9: residui

Tutti, dal report del Passo 0: idee 8 (`SAID_FIELDS` come lista di ammessi), 9 (stato del microfono nel tray), 11 (C901 ≤ 12 in ruff); 5.1 #9 (test su `sqlite3.ProgrammingError`); 5.2 #1, #2, #7, #8, #9, #10, #12, #14, #15, il resto di #16, #17; 5.3 (test con thread instabili, tempi di chiusura dello stream). Divisi in sotto-gruppi da 3-8 per approvazione. **Se uno non vale il rischio, proponi di lasciarlo con un motivo: non toglierlo in silenzio.**

### G10: solo se le soglie lo chiedono

- **Cloud:** se dopo G5 gli errori restano alti, prova Scribe v2, AssemblyAI Universal-3.6 Pro, Deepgram Nova-3 con `mip_opt_out=true`. Soglia fissata prima di vedere i risultati: almeno 5 punti in più di comandi riusciti con intervallo al 95% che esclude lo zero, azioni sbagliate non più alte, P95 di latenza nel budget, ricaduta su Whisper senza rete. Servono chiavi di JenJo e il suo ok a ogni invio di audio.
- **Riaddestramento di Jack:** se G4 non raggiunge la soglia. Prima ascolta i round 3 e 4 sullo stesso testo: un corpus troppo stretto può causare il suono robotico. `jack_finetune.py` non è sotto `scripts/`: va controllato il bug #179.

Dopo v0.4.3 si torna al Passo 1 (web) di `IT-2-consegne.md`.

## Rischi

- Il rilevatore delle promesse e `[[SILENT]]` sono progetti mai eseguiti: nessun dato su Haiku 5.5.
- Il determinismo del replay non è verificato: se non regge, il gate cambia forma.
- Un falso silenzio perde una richiesta vera. Lo limitano i test, il verdetto visibile e la regola di arresto.
- Il controllo WAV è locale: dipende dalla disciplina, non da una CI.
- Percorsi allargati: la prompt injection può far aprire più file. Gli eseguibili restano bloccati e `read_file` è ristretto.
- Whisper diverso o Jack riaddestrato possono far regredire ciò che oggi funziona: si confronta sempre a coppie.
