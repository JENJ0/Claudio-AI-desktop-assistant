# IT-1 – Consegne: memoria a lungo termine (sezione 6.4, fase 4)

Ramo locale `it1/memoria` (worktree `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\claudio-it1-memoria`),
partito da `main` 62f2b79 e in pari con `main` v0.4.0 (merge 92f1fe9). Non pushato.
Test manuale: `piani\IT-1-memoria-test.md` (dopo il merge, dalla cartella principale).

## Scelte di JenJo (30/10 21:56): A1, B1, C1, D1

| Domanda | Fatto così | Per cambiare |
|---|---|---|
| A. Ricordi nel prompt | **A1 ibrido**: regole + fatti più recenti in un blocco in cache (`block_chars = 2000`); gli altri solo se la richiesta ne condivide le parole | `block_chars = 0`: solo i pertinenti (le regole restano sempre) |
| B. Salvataggio da solo | **B1 acceso** (`learn = true`): Claude salva da solo le preferenze durature e lo dice | `learn = false` in `[memory]` |
| C. Regole "da ora in poi" | **C1**: nel database; `data\CLAUDIO.md` non viene mai toccato | - |
| D. Riassunti delle conversazioni | **D1**: non fatti qui, thread a parte | vedi "Non fatto" |

## Fatto

| Cosa | Dove |
|---|---|
| Archivio SQLite `data\memory.db` (mai in Git): fatti e regole con le tue parole, chi li ha salvati, data. Scrittura con attesa massima 0,25 s; file cancellato a mano: si riparte vuoti; il testo dimenticato viene azzerato nel file | `src/claudio/memory/store.py` |
| Rifiuto dei segreti: password, PIN, codici, combinazioni, chiavi del wifi e private, credenziali, documenti, carte, IBAN, frasi di recupero, "password: x". Restano i fatti normali ("pago con la carta di credito", "il codice di Claudio è su GitHub") | `store.py` (`sensitive`) |
| Ricerca locale gratuita (~1 ms) per radice ("sorella"/"sorelle") e per parola intera ("Anna" non è "anni"); il nome "Claudio" e i numeri da soli non contano | `src/claudio/memory/relevance.py` |
| Frasi capite in locale prima di ogni comando, anche con "ok", "senti", "per favore": "ricordati che…", "da ora in poi…", "ricordati sempre di…", "dimentica…", "dimenticalo", "dimentica l'ultimo ricordo", "cosa ricordi di…"; italiano e inglese | `src/claudio/memory/voice.py` |
| Dimenticare non indovina mai: subito solo se tutte le parole corrispondono e hai detto il nome, altrimenti "Dimentico questo: …?" o "Quale dimentico?". Contano solo risposte vere (sì/no, "la seconda", "tutti e due", parole di uno), mai una richiesta ("chiudi tutte le finestre", "che tempo fa a Roma?") né l'eco della domanda; il silenzio la chiude | `voice.py` |
| "Dimenticalo" = il ricordo di cui avete appena parlato (un minuto, senza altre richieste in mezzo); dopo un rifiuto "Non l'avevo salvato." | `voice.py`, `router/responder.py` (`moved_on`) |
| Senza il nome (finestra di conversazione, parole dette sopra la risposta) una regola o una cancellazione viene chiesta prima; una frase di memoria sopra la risposta serve il nome | `voice.py`, `intake.py` (`Hearing.addressed`), `app.py` |
| Comandi fissi `memory_list` ("cosa ricordi?") e `memory_wipe` ("dimentica tutto", livello 3: "confermo") | `config/commands.toml`, `router/actions.py` |
| Claude vede regole e fatti come secondo blocco di sistema in cache, un'istantanea per richiesta; i fatti fuori dal blocco vanno con la richiesta (`[memory, data: …]`) una volta per conversazione; dopo una cancellazione la conversazione con Claude riparte, e una proposta di Sonnet in sospeso cade | `src/claudio/memory/prompt.py`, `brain/claude.py` |
| Strumenti di Claude `remember`/`forget`: fatto livello 1, livello 2 se la conversazione ha letto finestre, schermo, file o comandi; regola livello 2; `forget` livello 2 (3 oltre tre ricordi, uno per risposta); ciò che chiede il sì viene letto per intero (massimo 80 caratteri) | `src/claudio/memory/tools.py`, `pc/tools.py` |
| Il router "capisce" le frasi di memoria (niente seconda passata di Whisper, niente ipotesi a pagamento); "Quale dimentico?" si risponde anche nella chat | `router/responder.py`, `app.py`, `protocols.py` |
| Database bloccato o rotto: risposta a voce "Non riesco a usare la memoria…", CLAUDIO continua | `responder.py`, `__main__.py` |
| Configurazione `[memory]`, README (en/it), CHANGELOG, test manuale | `config/claudio.toml`, `config.py`, `README*.md`, `CHANGELOG.md`, `piani/` |

## Risultati

- Test: **1962 passed**, 4 skipped (198 nuovi della memoria in `tests/unit/test_memory.py` e
  `tests/unit/test_memory_wiring.py`); ruff e mypy puliti.
- Prove vere con l'API (Haiku 5.5), quattro giri da 4-11 richieste:
  - frasi di memoria in locale in 0-13 ms, a costo zero;
  - prima domanda a Claude: ~6600 token scritti in cache (0,00085 $); le successive ~6600 letti e
    ~100 scritti (0,0001 $); dopo una cancellazione ~6300 letti e ~290 scritti;
  - prima parola di Claude 0,6-1,1 s (1,5-1,7 s quando usa `remember`);
  - circa 0,0013 $ per giro. Claude ha usato regole e fatti ("Tua sorella si chiama Anna e abita
    a Torino, capo"), ha salvato da solo "sono vegetariano" (B1) e l'ha usato per la cena.
- Revisioni multi-agente: prima 4 revisori + verifica avversaria (23 segnalazioni), poi 3
  revisori sulle correzioni (14 nuove). **Tutte e 37 corrette**, ognuna con un test.

## Non fatto / rischi

1. **Riassunti delle conversazioni** (D1): thread a parte (Haiku riassume a fine conversazione,
   ~0,0002 $, recupero per parole).
2. **Promemoria a orario** ("ricordami di chiamare la mamma alle 5"): arrivano con i timer (fase
   7). Oggi Claude risponde che non può.
3. **Ricerca per significato**: in locale "mamma" non trova "madre". Con A1 non serve finché i
   ricordi stanno nel blocco (Claude li vede tutti); oltre, servirebbero embedding locali.
4. **Filtro dei segreti euristico**: una frase strana può passare. Non dettare segreti a CLAUDIO.
5. **Haiku parla un po' troppo** dopo un salvataggio automatico, e dopo "cosa ricordi?" con più
   ricordi un "dimenticalo" va a Claude, che a volte chiede "Confermi?" da solo (il prompt glielo
   vieta). Da rivedere con il thread della latenza o del prompt.
6. **Un fatto detto dalla TV** nella finestra di conversazione si salva (la regola no), ma CLAUDIO
   lo ripete ad alta voce: "dimenticalo" subito dopo lo toglie.
7. **Strumenti di Claude solo con `[pc] enabled`**; le frasi locali funzionano sempre.
8. **Non provato a voce sul PC**: solo prove scritte con l'API vera. Il test manuale serve a vedere
   come Whisper scrive "ricordati che…", "dimenticalo", "la seconda".
9. File toccati che anche altri thread IT-1 toccano (conflitti piccoli al merge):
   `router/responder.py`, `brain/claude.py`, `pc/tools.py`, `app.py`, `intake.py`,
   `protocols.py`, `__main__.py`, `config.py`, `i18n.py`, `commands.toml`, `claudio.toml`,
   `README*`, `CHANGELOG.md` e i finti responder nei test (`understands(..., memory=True)`,
   `needs_voice`).
