# IT-1 – Consegne: memoria a lungo termine (sezione 6.4, fase 4)

Ramo locale `it1/memoria` (worktree `C:\Users\JenJo\Desktop\PROGETTI\claudio-it1-memoria`),
partito da `main` 62f2b79 e già in pari con `main` v0.4.0 (merge 92f1fe9). Non pushato.
Test manuale: `piani\IT-1-memoria-test.md` (da fare dopo il merge, dalla cartella principale).

## Scelte (domande A-D a JenJo, 30/10)

Risposte di JenJo (30/10 21:56): A1, B1, C1, D1.

| Domanda | Fatto così | Per cambiare |
|---|---|---|
| A. Ricordi nel prompt | **A1 ibrido**: regole + fatti più recenti in un blocco in cache (`block_chars = 2000`); gli altri solo se la richiesta ne condivide le parole | `block_chars = 0` = A2 (solo i pertinenti; le regole restano sempre) |
| B. Salvataggio da solo | **B1 acceso** (`learn = true`): Claude salva da solo le preferenze durature che gli dici, e lo dice ad alta voce | `learn = false` in `[memory]`: solo su richiesta |
| C. Regole "da ora in poi" | **C1**: nel database, `data\CLAUDIO.md` non viene mai toccato | C2 richiede codice nuovo (ricarica di CLAUDIO.md e della cache) |
| D. Riassunti delle conversazioni | **D1**: non fatti, thread a parte | vedi "Non fatto" |

## Fatto

| Cosa | Dove |
|---|---|
| Archivio SQLite `data\memory.db` (mai in Git): fatti e regole con le tue parole, chi li ha salvati (tu o Claude), data; scrittura con attesa massima 1 s, schema ricreato se il file sparisce, testo cancellato azzerato nel file (`secure_delete`) | `src/claudio/memory/store.py` |
| Rifiuto di password, PIN, codici, combinazioni, chiavi del wifi, carte, IBAN e numeri da 13+ cifre (non i telefoni con 00); i fatti normali su carte e password ("pago con la carta di credito") restano | `store.py` (`sensitive`) |
| Ricerca locale gratuita (~1 ms): per radice ("sorella"/"sorelle") e per parola intera ("Anna" non è "anni") | `src/claudio/memory/relevance.py` |
| Frasi capite in locale, prima di ogni comando: "ricordati che…", "da ora in poi…", "ricordati sempre di…", "dimentica…", "dimenticalo", "dimentica l'ultimo ricordo", "cosa ricordi di…"; con "ok", "senti", "e", "per favore" davanti; italiano e inglese | `src/claudio/memory/voice.py` |
| "Dimentica X" cancella subito solo un ricordo con tutte le parole di X dette col nome; altrimenti chiede ("Dimentico questo: …?" o "Quale dimentico?"); risposte accettate: sì/no, "la seconda", "tutti e due", parole di uno, "nessuna delle due"; mai l'eco della domanda né una richiesta più lunga ("alza il volume di due") | `voice.py` |
| Senza wake word (la TV nella finestra di conversazione) una regola o una cancellazione viene chiesta prima | `voice.py`, `intake.py` (`Hearing.addressed`), `app.py` (`_tell_hearing`) |
| Comandi fissi `memory_list` ("cosa ricordi?") e `memory_wipe` ("dimentica tutto", livello 3: "confermo") | `config/commands.toml`, `router/actions.py` |
| Claude vede regole e fatti come secondo blocco di sistema in cache, ricostruito solo quando un ricordo cambia; un'istantanea per richiesta | `src/claudio/memory/prompt.py`, `brain/claude.py` |
| I fatti fuori dal blocco vanno con la richiesta come `[memory, data: …]`, una sola volta per conversazione; dopo una cancellazione la conversazione con Claude riparte (non cita più il ricordo) | `prompt.py`, `brain/claude.py` |
| Strumenti di Claude `remember` / `forget`: un fatto è livello 1, ma livello 2 se nella stessa richiesta Claude ha letto finestre, schermo, file o un comando; una regola è livello 2; `forget` livello 2 (oltre 3 ricordi livello 3) e uno solo per risposta; la domanda dice sempre cosa si salva o si cancella | `src/claudio/memory/tools.py`, `pc/tools.py` |
| Il router "capisce" le frasi di memoria: niente seconda passata di Whisper, niente ipotesi a pagamento; "Quale dimentico?" si può rispondere anche nella chat | `router/responder.py`, `app.py`, `protocols.py` |
| Database bloccato o rotto: risposta a voce "Non riesco a usare la memoria…", CLAUDIO continua | `responder.py`, `__main__.py` (`build_memory`) |
| Configurazione `[memory]` | `config/claudio.toml`, `src/claudio/config.py` |
| README (en/it), CHANGELOG, test manuale | `README.md`, `README.it.md`, `CHANGELOG.md`, `piani/IT-1-memoria-test.md` |

## Risultati

- Test: **1927 passed**, 4 skipped (166 nuovi della memoria in `tests/unit/test_memory.py` e
  `tests/unit/test_memory_wiring.py`); ruff e mypy puliti.
- Prova vera con l'API (Haiku 5.5), due giri:
  - frasi di memoria in locale in 0-13 ms, a costo zero;
  - prima domanda a Claude: 6532 token scritti in cache, ~0,00085 $; la successiva 6532 letti e
    116 scritti, ~0,0001 $;
  - dopo una cancellazione: 6216 letti (strumenti e sistema restano in cache), 285 scritti;
  - prima parola di Claude 0,8-1,1 s (1,7 s con lo strumento `remember`);
  - totale 0,0013 $ per 9-11 richieste. Claude ha usato regole e fatti ("Tua sorella si chiama
    Anna, capo, e abita a Torino") e lo strumento `remember` per "Tienilo a mente: …".
- Revisione multi-agente (4 revisori + verifica avversaria): 23 segnalazioni, 16 confermate
  riproducendole; **tutte e 23 corrette** (commit 8055e21), ognuna con un test "review".

## Non fatto / rischi

1. **Riassunti delle conversazioni** (fase 4 di CLAUDE.md, scelta D1): da fare in un thread a
   parte (Haiku riassume a fine conversazione, ~0,0002 $, recupero per parole).
2. **Promemoria a orario** ("ricordami di chiamare la mamma alle 5"): non è memoria, arriva con
   i timer (fase 7). Oggi Claude risponde che non può.
3. **Ricerca per significato**: in locale "mamma" non trova "madre". Con A1 non serve finché i
   ricordi stanno nel blocco (Claude li vede tutti); oltre, servirebbero embedding locali
   (dipendenza nuova da valutare).
4. **Filtro dei segreti**: è euristico. Una frase strana può passare ("il numero magico della
   porta è 4512" non ha parole come codice o PIN). Non dettare segreti a CLAUDIO.
5. **Fatti detti dalla TV nella finestra di conversazione**: un fatto ("ricordati che…") si salva
   anche senza nome, ma CLAUDIO lo ripete ad alta voce; "dimenticalo" subito dopo lo toglie.
6. **Strumenti di Claude solo con `[pc] enabled`**: senza le azioni sul PC Claude vede i ricordi
   ma non può salvarli né cancellarli; le frasi locali funzionano sempre.
7. **Non provato a voce sul PC**: solo prove scritte con l'API vera. Il test manuale serve per
   vedere come Whisper scrive "ricordati che…", "dimenticalo", "la seconda".
8. File toccati che anche altri thread IT-1 toccano (possibili conflitti al merge, piccoli):
   `router/responder.py`, `brain/claude.py`, `pc/tools.py`, `app.py`, `intake.py`,
   `protocols.py`, `__main__.py`, `config.py`, `i18n.py`, `commands.toml`, `claudio.toml`,
   `README*`, `CHANGELOG.md`, `tests/unit/test_app.py` (FakeResponder.needs_voice).
