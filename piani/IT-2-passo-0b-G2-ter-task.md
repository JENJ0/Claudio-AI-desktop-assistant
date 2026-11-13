# G2 ter: «continua» torna solo alle storie

> **Per gli agenti:** SUB-SKILL OBBLIGATORIA: superpowers:executing-plans (native, come G2 bis). I passi usano le caselle (`- [ ]`).

**Obiettivo:** dopo la prova H2 ter, «continua» e «riprendi» tornano all'ultima storia interrotta, mai a una risposta breve, e funzionano anche quando Whisper capisce «continuo».

**Cause (verificate il 13/11 su `logs/sessions/2026-11-13_20-24` e sul codice):**

1. «Dicevo» ha ridetto solo l'ultima frase (20:25:37). La storia era ferma in attesa. «Quanto dista la luna?», detto sopra la prima risposta sulla Luna, l'ha cancellata: la regola di `src/claudio/app.py:1755` vale per qualunque risposta di Claude interrotta. Poi «riprendi» ha trovato in sospeso solo la prima risposta sulla Luna (2 frasi, 224 caratteri: sopra `resume_min_chars` 200) e ne ha ridetto la coda.
2. «Claudio, continuo.» (20:27:01): il controllo sopra la risposta usa il testo grezzo. La lista chiusa che lo corregge in «continua» arriva solo dopo, in `_decide`. La regola B non è scattata, e Claude ha inventato un «Dicevo:».
3. Passo 4: un'ora detta sopra la storia, tagliata da un'altra «che ore sono» (la prima aveva impiegato 10 s: Jack fallito), non riprende la storia da sola.

**Decisioni di JenJo (13/11, sera):** «continua» torna solo alle storie: una risposta breve non prende il posto della storia e non la cancella. La lista chiusa vale anche sopra la risposta. L'ordine di F2 resta: la voce di Jack (20 riprese scartate, worker morto) va a G4.

**Spec:** `piani/IT-2-passo-0b-prove.md`, H2 ter; `piani/IT-2-passo-0b-G2-bis-task.md`.

## Vincoli globali

- Comandi: `uv run python -m pytest -q`, `uv run python -m mypy src`, `uv run ruff check .` e `uv run ruff format --check .`.
- Mai un grep ricorsivo dalla radice (`models/` e `samples/` ~135 GB). Prima `graphify query`.
- Test prima del codice, visti fallire. Codice e commenti in inglese.

## Review Focus

1. Una storia ancora in arrivo, tagliata presto, ha meno di 3 frasi: deve contare come storia (non si sa ancora quanto è lunga).
2. La lista chiusa sopra la risposta costa un passaggio di Whisper: solo frasi col nome, non comandi, che `wants_closed_list` accetta.
3. «Dicevo» automatico dopo un comando locale: resta com'è (lunghezza), cambia solo cosa resta per «continua».

---

### Task 1: solo le storie restano per «continua»

**File:** `src/claudio/resume.py` (`STORY_SENTENCES = 3`, `is_story(reply, min_chars) -> bool`: `worth_resuming` e almeno 3 frasi contate con `SentenceSplitter`), `src/claudio/app.py` (`_keep_stopped`, `_go_on`, `_pending_rest`, riga 1755); test `tests/unit/test_resume.py`, `tests/unit/test_timeline.py`.

- [ ] Test unitari di `is_story`: le due risposte sulla Luna del 13/11 no; `STORY` sì.
- [ ] Test a timeline (passo 1 di H2 ter): storia fermata con «basta»; domanda; sopra la risposta breve un'altra domanda; sopra la seconda risposta «riprendi»: «Dicevo:» con il resto della **storia**. Visto rosso.
- [ ] Test: una risposta breve fermata con «basta» non sostituisce la storia: «continua» ridice la storia.
- [ ] Codice: `_unsaid_rest` e `_pending_rest` accettano solo storie; una risposta di Claude interrotta cancella il resto più vecchio solo se è una storia, o se era ancora in arrivo (non si sa ancora). `test_a_newer_cut_claude_reply_clears_the_older_rest` resta verde.

### Task 2: la lista chiusa anche sopra la risposta

**File:** `app.py` (`_check_barge_in`, prima di `_judge_barge_in`); test `tests/unit/test_timeline.py`.

- [ ] Test (passo 2 con l'errore di Whisper): «Dicevo:» in riproduzione, `say_over("Claudio, continuo.")` con una lista chiusa finta che propone «Claudio, continua.»: va avanti (regola A), nessuna chiamata a Claude. Visto rosso.
- [ ] Codice: dopo la trascrizione, se la frase ha il nome, non è già un comando e `wants_closed_list` la accetta, chiede a Whisper le ipotesi (sotto `_stt_lock`) e usa la prima che `is_resume` riconosce. Log «Lista chiusa sopra la risposta».

### Task 3 (se JenJo lo approva): l'ora tagliata da un'altra ora riprende la storia

**File:** `app.py` (`_go_on`); test `tests/unit/test_timeline.py`.

- [ ] Test (passo 4): storia tagliata da «che ore sono»; mentre risponde, di nuovo «che ore sono»; dopo la seconda risposta: «Dicevo:» e il resto della storia, senza «continua». Visto rosso.
- [ ] Codice: la storia che una risposta locale stava per riprendere passa alla richiesta che taglia quella risposta, se anche questa è locale.

### Task 4: chiusura

- [ ] Suite, ruff ×2, mypy; gate 38/0/3; revisione Opus con mutazioni.
- [ ] Esito di H2 ter nel file delle prove; prova nuova «H2 quater» (passi 1, 2 e 4 dal vivo) e la sua riga in `prove-da-fare.md`, nella stessa modifica.
- [ ] Commit locali; push solo con l'ok di JenJo; CI verde.

## Fatto quando

Test nuovi rossi e poi verdi; suite, `ruff check`, `ruff format --check`, mypy verdi; gate 38/0/3; H2 quater scritta insieme alla sua riga in `prove-da-fare.md`; commit locali; push approvato; CI verde.
