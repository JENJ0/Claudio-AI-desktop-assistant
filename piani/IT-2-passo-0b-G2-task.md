# G2 Interruzioni e stato del turno: task

> **Per chi esegue:** `superpowers:subagent-driven-development`, un task alla volta. Implementer su Sonnet, revisori indipendenti su Opus; verifica per mutazione e revisione dell'intero gruppo a fine G2, perché tocca la sicurezza. I passi usano `- [ ]`.

**Obiettivo:** mai un'azione senza un sì sentito davvero, e dopo un'interruzione Claudio sa cosa è stato detto, cosa è partito e cosa no.

**Architettura:** tre punti unici. La risposta a una domanda passa da una sola decisione (`RequestFilter._answer`, poi `_answer` locale o `_confirm` di Claude, con le stesse regole). Lo stato del turno sta su `_Turn`: la sua identità basta, niente id nuovo, così una richiesta nuova non può più azzerare lo stop di quella vecchia. La cronologia scrive ciò che lo speaker ha suonato, riportato dall'Assistant con `reply_cut`.

**Tecnologie:** Python 3.11+, faster-whisper, Silero VAD, SDK `anthropic`.

**Spec:** `piani/IT-2-passo-0b-consegne.md`, sezione G2, tabella «Rimandati da G1», «Scoperte di G0 per G2», decisioni «Dicevo», «Conferme», «Contesto», «Conferma vocale». Report solo in locale: `reports/Conferma vocale sicura con Whisper.md`, `reports/Diagnosi prova C.md`.

## Vincoli globali

- Prima di cercare nel codice: `graphify explain "<X>"` o `graphify path "<A>" "<B>"`, poi solo le righe indicate.
- Gli implementer **non committano**. Regole: `.superpowers/sdd/regole/implementer-rules.md`.
- Mai azione senza sì. Qualunque dubbio sul sì lascia l'azione non fatta o fa ripetere la domanda, mai il contrario.
- WAV, cassette, corpus e memoria mai in git. Da spento nessun testo né audio salvato.
- Soglie in `config/claudio.toml`, mai nel codice. Con la soglia, `whisper_prompt` diventa configurazione di sicurezza: ogni sua modifica rilancia il test del Task 4.
- Il test del Task 4 si lancia **una volta sola**, a fine gruppo, con i criteri già scritti. Se fallisce, quelle riprese diventano sviluppo e ne servono di nuove. Nessuno guarda quei WAV prima.
- Budget API di G2: 0,05 $ tra cassette e prove. Il gate di replay deve restare a 0 fallite.

## Fatti verificati (non ridiscuterli)

- Percorso reale, 108 riprese, prompt vero, da +14 a +22,4 dB: 3-5 non-sì su 64 eseguono l'azione, 28 sì su 44 la eseguono (`Diagnosi prova C`, sezione A).
- Simulato: «?» + soglia -0.75 + seconda domanda dà 34-36 sì su 44 e 1-2 falsi sì su 64; da sola la seconda domanda li alza fino a 10 su 64. Il secondo ascolto simulato è l'altra ripresa della stessa frase, e copre «non chiaro», «solo il nome» e «vuoto». Non è validato: le soglie sono state scelte su quelle riprese.
- `avg_logprob` -0.75 tiene i 4 sì veri dal vivo (-0.27...-0.46); `non_va_bene__1` era a -0.762, margine sottile.
- `deaf_speech_ms` in `logs/metrics.jsonl`: 92 righe di conversazione su 99 sono a 0, solo 5 oltre 100 ms; sulle risposte 4 su 5 a 0, una a 352. Una soglia di 100 ms non fa ripetere la domanda di continuo.
- Una sola corsa da 93 riprese con al massimo 1 errore dichiara «sotto il 5%, unilaterale al 95%». Si conta per domanda, non per frase detta.

## Attenzione (i casi che fanno più male)

1. **Testa persa nella finestra sorda:** «non va bene» con il «non» nel tratto sordo diventa «va bene». Test nel task 3.
2. **Un sì a una domanda mai sentita:** frase sopra un «Confermi?» non ancora partito. Test nel task 9.
3. **Conferma locale armata dopo il silenzio:** la TV dice «sì» con la finestra chiusa. Serve il nome, e dentro 60 s. Test nel task 9.
4. **Frase detta mentre Claude pensa:** l'eco del bip o la TV non devono fermare la risposta. Test nel task 8.
5. **«Continua» senza niente da riprendere:** deve andare a Claude, non dire un testo vecchio. Test nel task 7.

## Prova H0 di JenJo, prima del task 1

Distorsione: 3 WAV dal vivo contro le riprese, effetti audio di Windows spenti, modalità raw, un altro microfono sulle stesse 10 frasi, più il riascolto di `20261106-125111-875_ok.wav`. Passi in `piani/IT-2-passo-0b-prove.md`, sezione H. Se la catena audio cambia, servono riprese di sviluppo nuove prima del task 2.

---

## Task 1: parser

**File:** `src/claudio/safety/confirmation.py`, `src/claudio/router/responder.py`, `src/claudio/intake.py`. Test: `tests/unit/test_safety.py`, `tests/unit/test_intake.py`.

**Regole:**
1. Un «?» in qualunque punto del testo non è un sì, e il NO resta prima. Anche `_yes_then_guess`.
2. Con una domanda aperta, `decide` non fa riascoltare una frase che è già una risposta: `only_wake_words` è vero per «No, Claudio.» e fa restare aperta la domanda.

- [ ] Test che falliscono: `parse_answer("Sì?!")`, `"Sì?."`, `"Sì?..."`, `"Sì? Grazie."`, `"Va bene? Grazie."` danno `UNCLEAR`; `"No?"` dà `NO`.
- [ ] Test che falliscono: con `responder.awaiting`, `decide("No, Claudio.")`, `"Sì, Claudio."`, `"Ok Claudio."` danno `Kind.ANSWER`; `"Claudio."` da solo resta `RELISTEN`; senza domanda aperta `"No, Claudio."` non cambia.
- [ ] Toglie i tre `xfail` di G0 su questi casi, se ci sono. Suite, ruff, mypy.

## Task 2: replay offline, una variabile alla volta

**File:** solo `research_notes/Conferma vocale sicura con Whisper/prove/replay_vars.py` e il suo risultato, fuori da git. Niente `src/`. Nessuna chiamata API. Usa `build_harness`, `plan_scene`, `_replay_scene` di `scripts/replay_gate.py`, a +14 e +22,4 dB, su `samples/safety/` (le 108 di sviluppo), con il prompt vero congelato.

**Variabili:** base; `without_timestamps=False`; `beam_size=1` sulle sole risposte; prompt congelato per le risposte. Ogni variabile da sola contro la base.

- [ ] Una tabella: sì che eseguono su 44, non-sì che eseguono su 64, frasi che cambiano. Stampa solo numeri e nomi di file.
- [ ] Si adotta solo ciò che non peggiora né i sì né i falsi sì. Il coordinatore decide e scrive `Ruling:` nel registro. Se si adotta qualcosa, entra nel task 3.

## Task 3: soglia, seconda domanda, regola del nome

Nello stesso commit, per scelta di JenJo.

**File:** `src/claudio/intake.py`, `src/claudio/config.py`, `config/claudio.toml`, `src/claudio/safety/confirmation.py`, `src/claudio/router/responder.py`, `src/claudio/brain/claude.py`, `src/claudio/app.py`, `src/claudio/i18n.py`. Test: `tests/unit/test_safety.py`, `tests/unit/test_pc_tools.py`, `tests/unit/test_timeline.py`.

**Produce:**
- `NoiseSettings.answer_min_logprob: float = -0.75`, `answer_deaf_voice_ms: int = 100`, in `[noise]`.
- `Heard.deaf_voice_ms: int`: l'Assistant misura la voce nel tratto sordo prima di decidere, con il calcolo di `_measure_deaf`.
- `RequestFilter._answer`, oltre a ciò che fa: un sì (`parse_answer` a livello 2) col segmento più debole sotto `answer_min_logprob` dà `Decision(Kind.ANSWER, "", "unsure_yes")`; voce nel tratto sordo oltre `answer_deaf_voice_ms` dà `Decision(Kind.ANSWER, "", "deaf_answer")`, qualunque sia la risposta, un no compreso (ne manca la testa: «no, aspetta» sentito a metà). Un no sicuro resta un no.
- `Pending.asked_again: bool = False`, `_Action.asked_again: bool = False`.
- `PcTools.describe_calls(calls, language) -> str`: la parte di `question` che legge le azioni, usata da `question` e dalla seconda domanda. Stringa vuota se nessuna azione ha una descrizione.
- `confirmation.py`: `ASK_AGAIN = {"it": "Non ho capito. {what}: sì o no?", "en": "I didn't catch that. {what}: yes or no?"}`, `ASK_AGAIN_EXPLICIT = {"it": "Non ho capito. {what}. Per procedere dì: confermo.", "en": "I didn't catch that. {what}. To go ahead, say: I confirm."}`.

**Regole, uguali per domanda locale e di Claude:**
1. Vuoto o non chiaro, senza nome, prima volta: riarma la domanda (scadenza da capo), dice `ASK_AGAIN` con l'azione. Locale: il testo `confirm` del comando, altrimenti la frase del comando. Claude: `describe_calls`; se è vuota, solo `ASK_LEVEL_2`.
2. Vuoto o non chiaro la seconda volta: annulla con `NOT_UNDERSTOOD`.
3. Non chiaro **col nome** (`heard_wake_word`), a qualunque giro: `CANCELLED`, poi risponde alla frase come richiesta nuova, in un solo stream. Scena vera: «Claudio, che idea hai?» delle 17:20:17. La chiamata annullata resta in cronologia come non eseguita.
4. La soglia vale anche al livello 3: un «confermo» debole fa ripetere la domanda, che è più sicuro. Dillo a JenJo nel riepilogo.
5. La riga di metriche porta `asked_again`, così si conta quanti sì sentono la seconda domanda.

- [ ] Test che falliscono: soglia sul sì; un no a -1.0 resta un no; voce sorda con «va bene» non esegue; voce sorda sotto soglia esegue; regole 1, 2 e 3 su percorso locale e su percorso Claude; un secondo sì sotto soglia annulla; dopo una seconda domanda un «no» annulla; `17:16:09` «Si apre il blocco note.» chiede una volta poi annulla; `17:20:17` annulla e risponde.
- [ ] Il gate di replay: le clip con una risposta passano; aggiorna le attese con motivo scritto.

## Task 4: registratore e test nuovo

**File:** `scripts/record_samples.py`, `scripts/safety_phrases.tsv`, `tests/integration/test_safety_audio.py`. Riprese in `samples/safety_v2/`, fuori da git.

**Regole:**
1. Modalità `safety-v2`: **uno stream aperto per tutta la sessione**, che non si chiude tra una ripresa e l'altra; riprese sotto il controllo di picco rifiutate con richiesta di rifarle; passo naturale: `Invio` apre la ripresa, con la durata di oggi.
2. Frasi: le 31 non-sì di `safety_phrases.tsv` senza «sì?», due riprese ciascuna (62), più 31 trappole nuove scritte qui prima di registrare, tipo «non è ok», «non ancora», una ripresa ciascuna; 8 «sì?»; 44 sì nuovi. Totale 145. Colonna gruppo: gemello di un sì, riscrittura «si …», fuori tema, rumore.
3. Il test riusa `build_harness`, `plan_scene`, `_replay_scene`: guadagno a +14 dB e `Endpointer` veri, prompt congelato. Seconda risposta, quando la domanda si ripete: l'altra ripresa della stessa frase; per una trappola, che ha una ripresa sola, la trappola dopo nello stesso gruppo. Rifare lo stesso WAV darebbe lo stesso testo, e il conto sarebbe troppo ottimista.
4. **Criteri scritti nel test prima di registrare:** non-sì che eseguono ≤ 1 su 93; sì ≥ 33 su 44. A parte, senza criterio: gli 8 «sì?», i sì al primo colpo, il conto per frase, i risultati per gruppo.

- [ ] Test unitari del registratore, con `sounddevice` finto: una sola apertura di stream per sessione; ripresa sotto il picco rifiutata; riprende da dove si era fermata. Il test audio salta se le riprese mancano.
- [ ] **JenJo registra** (circa 25 minuti, prova H1). Da qui parte la corsia B.
- [ ] Il test **non** si lancia ora: si lancia a fine gruppo, dopo il task 9.

## Task 5: finti e tempo

Prima di toccare gli interrupt, per non scrivere test sopra finti che mentono.

**File:** `tests/unit/timeline.py`, `tests/unit/test_timeline.py`, `tests/unit/test_conversation.py`, `tests/unit/test_safety.py`, `tests/unit/test_own_sound.py`, i file dei quattro finti, `src/claudio/__main__.py` (`build_brain`).

**Regole:**
1. `LongSpeaker`, `SentenceSpeaker`, `StartSpeaker`, `CueSpeaker` passano al contratto di `TimedSpeaker`: niente letture prima di `on_first_audio`, `played` riempito.
2. `talk`, `feed`, `asked` aspettano `own_sound.playing` prima di contare i fotogrammi sordi.
3. `timeline.py`: Whisper ha una durata in fotogrammi, così arrivano fotogrammi durante la trascrizione.
4. `build_brain` riceve l'orologio, così `history_reset_s` e le scadenze da 60 s si provano senza dormire.
5. Il contratto dello speaker copre l'errore dello stream e il produttore cancellato; `first_audible` ha un test per «solo la prima volta». I motivi dei test `xfail` restano veri: `test_timeline.py:119` si rinomina dopo la scelta di JenJo.

- [ ] Nessun cambiamento di comportamento in `src/` oltre all'orologio. Suite, ruff, mypy: stessi `xfail` di prima.

## Task 6: cronologia sentita e stato del turno

**File:** `src/claudio/brain/claude.py`, `src/claudio/router/responder.py`, `src/claudio/protocols.py`, `src/claudio/app.py`. Test: `tests/unit/test_timeline.py`, `tests/unit/test_pc_tools.py`, `tests/unit/test_memory_wiring.py`.

**Produce:**
- `ClaudeBrain.reply_cut(played: Sequence[str]) -> None`, con il passante in `Responder`; l'Assistant lo chiama appena una risposta viene tagliata e di nuovo a fine `_resume_reply`, con tutto ciò che è stato suonato.
- `_Turn.stopped: bool`, `_Turn.heard: str | None`, `_Turn.written: tuple | None` (la riga di cronologia già scritta, da riscrivere). `self._turn` è il turno in corso. `stop_acting()` scrive su quel turno: `_stop_acting` sparisce.

**Regole:**
1. `reply_cut` vale per il turno già scritto (lo riscrive, con la nota di taglio) e per quello ancora in corso (`_remember` e `_remember_cut` usano `heard`). Le note di tool partiti e non partiti restano.
2. Testo tardivo di un turno tagliato: mai in cronologia. Va solo in `DrainedStream`, per «Dicevo».
3. Un'offerta di Sonnet o una domanda il cui testo non è in `played` si disarma.
4. `cancel_pending` con un'azione armata scrive il turno in cronologia, con la nota «non eseguito». Oggi lo perde.
5. `BRIDGE_RUN` fuori da `turn.parts`.
6. `_marked`: `inter.info["reply"]` si aggiorna a ogni pezzo, senza aspettare il `finally`.
7. Nuova `ask` con uno stream vecchio ancora vivo: il vecchio resta fermo. Era il guasto di `handle_typed`.

- [ ] Test che falliscono: i 2 `HISTORY_BUG` e i 4 `LOSES_THE_TEXT` di `test_timeline.py` diventano verdi, tolto `xfail`; offerta mai sentita disarmata (scena del gruppo G1: una frase sopra non chiama Sonnet); azione tolta da `cancel_pending` ancora in cronologia; `handle_typed` durante uno stream vecchio non lo riattiva.
- [ ] Il gate di replay: le cassette con Sonnet o cronologia possono mancare, chiede una registrazione.

## Task 7: «Dicevo», «continua», «riprendi»

**File:** `src/claudio/app.py`, `src/claudio/resume.py`, `src/claudio/router/responder.py`, `src/claudio/router/actions.py`, `config/commands.toml`, `src/claudio/i18n.py`. Test: `tests/unit/test_timeline.py`, `tests/unit/test_resume.py`.

**Regole:**
1. Ripresa da sola **solo** se la richiesta che ha tagliato è stata risolta dal router locale. Dopo Claude mai.
2. `worth_resuming` guarda solo la lunghezza. Una storia che finisce con «Vuoi il seguito?» si riprende; se `responder.awaiting`, no.
3. Comando locale `resume`, frasi «continua», «riprendi», «continua la storia», «riprendi la storia», «vai avanti», `go on`, `continue`: ridice il testo mancante esatto, 0 $, chi ascolta sente anche «Dicevo:». Solo se c'è un testo mancante; altrimenti la frase va a Claude come oggi. Il testo vale fino a `history_reset_s`, sparisce allo spegnimento e a una nuova risposta di Claude arrivata in fondo.
4. Dopo il testo ridetto, `reply_cut` aggiorna la cronologia, così Claude sa cosa è stato detto.
5. Una richiesta locale sopra una risposta con un'azione fermata: dice `ACTION_STOPPED = {"it": "Ho lasciato stare l'azione di prima.", "en": "I dropped the action from before."}` e non riprende, così «Dicevo» non ripete la promessa.

- [ ] Test che falliscono: `test_17_08_story_then_time_resumes` e `test_17_13_story_then_new_request_does_not_resume` verdi, tolto `xfail`; «continua» dopo «basta» ridice il testo esatto con 0 chiamate API; «continua» senza testo mancante va a Claude; scaduto `history_reset_s`, va a Claude; spento, nulla; azione fermata + «che ore sono?»: dice l'ora e la frase, niente «Dicevo».

## Task 8: ascolto mentre Claude pensa

**Scelta di JenJo (07/11):** come quando parla. «Basta» ferma, una frase col nome prende il posto della richiesta, il resto si ignora.

**File:** `src/claudio/app.py`. Test: `tests/unit/test_timeline.py`, `tests/unit/test_conversation.py`.

**Regole:**
1. In `RESPONDING`, solo dentro una richiesta a voce, `handle_frame` passa i fotogrammi allo stesso ascolto di `SPEAKING`, armato alla fine di Whisper: «basta» ferma senza parole e senza spesa ulteriore, una frase per Claudio prende il posto, il resto si ignora con lo stesso `_is_talking_to_me`. `handle_typed` e `announce` no: restano sordi.
2. Il suono di Claudio (`own`, il bip) resta escluso come in `SPEAKING`.
3. Una frase che prende il posto prima della prima parola: `stop_acting`, lo stream si chiude, niente «Dicevo», perché non c'è niente di sentito.
4. Un «basta» durante Whisper (`TRANSCRIBING`) vale allo stesso modo sulla richiesta che sta per partire.
5. `_judge_stranded`: una richiesta sentita mentre il controllo Whisper è in corso non si perde (5.2 #5).

- [ ] Test che falliscono: `test_basta_before_the_first_word_says_nothing_later` verde, rinominato il motivo; frase con il nome prima della prima parola prende il posto; TV senza nome non ferma; rumore durante il bip non ferma; `handle_typed` resta sordo; «basta» durante Whisper; il test di 5.2 #5.
- [ ] Il test a mano di «lascia stare» di G1 si rifà qui (prova H2).

## Task 9: conferme, contesto, residui di G1

**File:** `src/claudio/router/responder.py`, `src/claudio/brain/claude.py`, `src/claudio/pc/tools.py`, `src/claudio/app.py`, `src/claudio/protocols.py`. Test: `tests/unit/test_safety.py`, `tests/unit/test_pc_tools.py`, `tests/unit/test_timeline.py`, `tests/unit/test_memory_wiring.py`.

**Regole:**
1. Silenzio dopo una domanda locale: `_pending` resta armato per `ACTION_TTL_S`, come quella di Claude. Dopo il silenzio serve il nome: «Claudio, confermo» esegue; un «sì» di TV a finestra chiusa non arriva nemmeno, per la wake word. Una frase col nome che non è un sì segue la regola 3 del task 3. Scaduto: non esegue e non chiede.
2. `ClaudeBrain.note_local(question: str, answer: str) -> None`: ogni comando locale entra in cronologia come un turno normale, con la stessa potatura. Niente da spento né in pausa. Le cassette con un comando locale prima di Claude cambiano chiave.
3. `_run_calls`: dopo un `focus_window` rifiutato o fallito, un `type_text` e un `press_keys` senza finestra nello stesso giro non partono, con la frase chiara per il modello.
4. `type_text` e `press_keys` di livello 2 senza finestra: la domanda nomina sempre dove andrà il testo (titolo della finestra attiva), anche con `confirm_brief`. Copre la scheda «Blocco note online» davanti.
5. Dopo il sì, un lancio dello stesso giro che batte un focus successivo: `_typing_target` controlla la finestra subito prima di scrivere, e se non è quella detta nella domanda non scrive e lo dice.
6. Risposta sopra un «Confermi?» che sta suonando (`Claudio, sì`, `Claudio, no`): se il testo della domanda è in `played` vale come risposta, con la soglia del task 3; se non è ancora partito, Claudio ripete la domanda. Non arriva mai a Haiku senza contesto.
7. Un messaggio scritto nella chat non risponde mai a una domanda armata a voce: la annulla, lo dice nella chat e risponde come a una richiesta nuova. Trovato nella revisione del task 6, già nel codice di base: un «sì» scritto dopo la finestra chiusa confermava un'azione di livello 2 entro 60 s.

- [ ] Test che falliscono: un caso per regola, compreso «Claudio, confermo» a 59 s e a 61 s; «Che ore sono?» poi una domanda che ci si riferisce: la richiesta a Claude ha il comando locale in cronologia; da spento niente in cronologia.
- [ ] Il gate: registra le cassette che mancano, dentro il budget.

## Task 10: chiusura del gruppo

- [ ] **Lancio unico** di `tests/integration/test_safety_audio.py -m slow` sulle riprese del task 4. Esito nel registro e nel riepilogo. Se fallisce: non si tocca la soglia per farlo passare, si ferma e si chiede a JenJo.
- [ ] Revisione dell'intero gruppo su Opus, con verifica per mutazione su `_answer`, `_confirm`, `reply_cut`, l'ascolto in `RESPONDING`.
- [ ] Gate di replay: `scripts/replay_gate.py --report`, 0 fallite. Le clip `171609` e `125104` diventano `fixed`: togli il segno dal manifesto.
- [ ] Le sequenze delle 17:08 e 17:13 sono test a timeline con esito corretto. Prova H2 di JenJo in `piani/IT-2-passo-0b-prove.md`.
- [ ] Diff a JenJo, commit piccoli dopo l'ok, `docs(plan)` con le consegne aggiornate, nessun push.

## Fuori da G2, deciso qui

- Una risposta a un testo scritto non si ferma a voce: resta così, la chat ha il suo stop. Se serve lo decide G6 con il tasto di stop globale.
- `«sì?»` scritto `Sì.` resta irrisolto al livello 2: si riporta a parte, come da spec.
- «sì», pausa oltre `end_silence_ms`, «anzi no»: rischio noto, senza misura. Da riaprire se il test del task 4 o la sessione di rilascio lo mostrano.
- `answer_min_dbfs` dopo il guadagno: non scarta nulla da +8 dB in su. La soglia di confidenza è la difesa vera; se resta inutile si toglie in G9.
