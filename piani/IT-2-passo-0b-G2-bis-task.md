# G2 bis: le riprese dopo la prova H2 bis

> **Per gli agenti:** SUB-SKILL OBBLIGATORIA: superpowers:executing-plans (scelta native) o superpowers:subagent-driven-development. I passi usano le caselle (`- [ ]`).

**Obiettivo:** «continua» e «riprendi» fanno quello che la prova H2 si aspettava, e «Dicevo:» si dice una volta sola. In più una prova A/B sulla lentezza vista il 13/11.

**Cause (verificate il 13/11 su `logs/sessions/2026-11-13_15-17` e sul codice):**

1. Passo 2 di H2 bis («continua» dopo la domanda sulla Luna è andato a Claude). Le cause sono due. La storia rispondeva a una frase su cui Whisper era insicuro, e `_unsaid` (`src/claudio/app.py:1832`) ne butta il resto. Poi una risposta completa di Claude cancella `_unsaid_rest` (`app.py:790-791`, regola 3 del Task 7 di G2).
2. Passo 4 («Riprendi?» detto sopra il «Dicevo:» è andato a Claude). Il pezzo appena tagliato sta nella variabile locale `cut`, non in `_unsaid_rest`, quindi il router (`src/claudio/router/responder.py:437`) non vede niente da riprendere.
3. «Dicevo: Dicevo:». Il resto di una ripresa tagliata comincia già con «Dicevo:», e `_resume_reply` (`app.py:1861`) ne aggiunge un altro. È la scorciatoia documentata in `_heard_in_all`.

**Decisioni di JenJo (13/11):** il pezzo mancante resta fino a `rest_s` anche dopo una risposta di Claude; «skip» che fa `media_next` va bene (si corregge il criterio di C1 al passo 7); A/B sulla latenza adesso.

**Spec:** `piani/IT-2-passo-0b-prove.md`, H2 passi 1-4 e H2 bis; `piani/IT-2-passo-0b-G2-task.md`, Task 7 (la regola 3 cambia).

## Vincoli globali

- Comandi: `uv run python -m pytest -q`, `uv run python -m mypy src`, `uv run ruff check .` e `uv run ruff format --check .`.
- Mai un grep ricorsivo dalla radice: `models/` e `samples/` pesano ~135 GB. Solo `src`, `tests`, `scripts`, `config`, i README e `research_notes/<cartella>`.
- Prima di cercare nel codice: `graphify query` / `graphify explain`.
- Test prima del codice, visti fallire. Codice e commenti in inglese.

## Review Focus

1. Una storia nuova che finisce intera lascia valido il pezzo della storia vecchia: «continua» ridice la vecchia (regola scelta da JenJo, fino a `rest_s`, 300 s).
2. Leggere fino in fondo la risposta a una frase incerta costa le parole che nessuno sente (al massimo il tetto di una risposta lunga, circa 500 token di uscita, circa 0,0075 $).
3. «continua» detto mentre Claude pensa o sopra una risposta diversa (la Luna) viene ignorato: va ridetto dopo.
4. «riprendi» non deve fermare una risposta che non è una ripresa? Viene comunque ignorato: Claudio va avanti.
5. «basta» e «stop» restano più forti di tutto: `is_stop` viene controllato prima.

---

### Task 0: A/B della latenza, prompt vecchio contro nuovo (ricerca, fuori da Git)

**File:** crea `research_notes/Latenza 13-11/ab_prompt.py` e `risultati.md`. Riusa `research_notes/Sonnet e thinking/prove/ab_sonnet.py`: Claudio è costruito con `build_responder` su `FakeDesktop`, con un tracker usa e getta, e nessuna azione viene eseguita.

- [ ] Braccio «nuovo»: la richiesta come la costruisce Claudio oggi. Braccio «vecchio»: la stessa, con nel testo di sistema il paragrafo di prima di `381450a` e il tag `[it | …]` nel messaggio.
- [ ] `--dry-run` senza API: stampa le due richieste e la differenza.
- [ ] 1 chiamata di riscaldamento per braccio (scrive la cache), poi 20 chiamate alternate ABAB con la stessa frase («quanto fa dodici per sette?»). Freno: si ferma sopra 0,10 $.
- [ ] Misura apertura dello stream («primo token») e prima parola; scrive mediana, p25 e p75 per braccio in `risultati.md`. Costo atteso circa 0,06 $.

### Task 1: il pezzo resta dopo una risposta di Claude

**File:** `src/claudio/app.py:790-791`; test `tests/unit/test_timeline.py`.

- [ ] `test_a_later_claude_answer_clears_the_rest` diventa `test_a_later_claude_answer_keeps_the_rest`: dopo «Parigi.» `_unsaid_rest` non è None; «continua» va al comando locale `continue_reply`, si sente «Dicevo:», le chiamate API restano 2. Visto rosso.
- [ ] Togli la cancellazione alle righe 790-791. `test_a_newer_cut_claude_reply_clears_the_older_rest` non cambia.

### Task 2: la risposta a una frase incerta non riparte da sola, ma resta per «continua»

**File:** `app.py` (`_unsaid`, `_go_on`); test `tests/unit/test_timeline.py`.

- [ ] Test nuovo: storia chiesta con trascrizione incerta (`assistant.unsure_logprob = -0.2`, `stt_segments` come in `test_conversation.py:761-764`), tagliata da una domanda per Claude: nessun «Dicevo». Poi «continua»: comando locale, «Dicevo:» con il resto. Visto rosso. `test_the_answer_to_a_doubtful_hearing_is_not_resumed` resta verde.
- [ ] `_unsaid` tiene la risposta incerta solo se era già arrivata intera (`stream.ended`), e la restituisce con `unsure=True`; altrimenti la annulla come prima (`CUT_FINISH_S` è 5 s: aspettarla ritarderebbe la risposta nuova). `_go_on` la riprende da sola solo se non è incerta, altrimenti la mette in `_unsaid_rest`.

### Task 3: «continua» o «riprendi» sopra una risposta, secondo il contesto

**Regola di JenJo (13/11):** (A) se non c'è una risposta precedente interrotta in sospeso, Claudio va avanti con quella che sta dicendo: niente stop, niente Claude. (B) Se ce n'è una, diversa da quella che sta dicendo, si ferma e riprende quella con «Dicevo:».

**File:** `src/claudio/router/responder.py` (`is_resume`), `src/claudio/protocols.py`, `app.py` (`_judge_barge_in`, `_speak_reply` alla riga 1755, `_handle_utterance`), il finto di `tests/unit/test_app.py`; test `tests/unit/test_timeline.py`.

- [ ] Test A: storia tagliata da «che ore sono», «Dicevo:» in riproduzione, `say_over("riprendi")`: nessuna riga nuova interrotta, nessuna chiamata API in più, la ripresa arriva in fondo. Visto rosso.
- [ ] Test B: storia fermata con «basta» (pezzo in sospeso), poi una domanda a Claude; mentre risponde, `say_over("continua")`: la risposta di Claude si ferma, il comando è locale `continue_reply`, si sente «Dicevo:» con il resto della storia, nessuna chiamata API in più. Visto rosso.
- [ ] `LocalResponder.is_resume(text) -> bool`: stessa regola di `is_stop`, azione `resume`. In `_judge_barge_in`, dopo `is_stop`: se `is_resume` e `_can_resume()`, è caso B: `_take_over`, con un flag che a `_speak_reply` impedisce di cancellare il pezzo più vecchio (riga 1755); altrimenti è caso A: log «Sto già continuando» e `duck(None)`.

### Task 4: «Dicevo:» una volta sola

**File:** `app.py` (`_resume_reply`); test `tests/unit/test_timeline.py`.

- [ ] Test: storia tagliata da «che ore sono», «Dicevo:», di nuovo tagliata da «che ore sono» durante la prima frase, ripresa ancora: l'ultima ripresa contiene «Dicevo:» una volta sola. Visto rosso.
- [ ] `_resume_reply` toglie un «Dicevo:» iniziale dal resto prima di aggiungere il suo.

### Task 5: chiusura

- [ ] Suite, `ruff check`, `ruff format --check`, mypy; gate a Claudio chiuso: 38/0/3.
- [ ] Revisione finale Opus sull'intero diff, con mutazioni.
- [ ] Documenti: esito di C1 (passi 2-7 superati, criterio del passo 7 corretto, passo 8 dal 18/11) in `piani/F1-passo-C-prove.md`; esito di H2 bis e regola 3 aggiornata in `piani/IT-2-passo-0b-prove.md` e nel piano G2; prova nuova «H2 ter» (passi 2 e 4 dal vivo) e, nella stessa modifica, `piani/prove-da-fare.md`.
- [ ] Commit approvati, push di `it2/passo-0b`, CI verde.

## Fatto quando

Test nuovi prima rossi e poi verdi; suite, `ruff check`, `ruff format --check` e mypy verdi; gate 38/0/3; report dell'A/B; H2 ter scritta insieme alla sua riga in `prove-da-fare.md`; commit approvati e spinti; CI verde.
