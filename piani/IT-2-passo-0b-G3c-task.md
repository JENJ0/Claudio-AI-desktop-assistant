# G3c Sonnet, thinking e frasi incerte: task

> **Per chi esegue:** `superpowers:subagent-driven-development`, un task alla volta. Implementer su Sonnet, revisori indipendenti su Opus; revisione dell'intero gruppo con mutazioni a fine G3c, perché T2 e T4 toccano la sicurezza. Prima del codice API: skill `claude-api`, sezione «Migrating to Claude Sonnet 5.5». I passi usano `- [ ]`.

**Obiettivo:** Sonnet 5.5 diventa il modello principale senza rompere nulla; il thinking diventa un interruttore misurato; una frase che Whisper ha sentito male passa prima da una lettura di Haiku; il blocco di memoria sale a circa 1.800 token.

**Architettura:** nessun modulo nuovo. Il modello resta una riga di configurazione: le regole che dipendono dal modello vanno in liste di configurazione, come l'attuale `retry_models`. La pre-chiamata riusa la decisione di `guess` (`act_at`, `ask_at`, «Hai detto: ...?») e resta su Haiku.

**Tecnologie:** Python 3.11+, SDK `anthropic`, Claude Sonnet 5.5 e Haiku 5.5.

**Stato (09/11, sera):** fatto: T1-T5, revisione di gruppo e correzioni, gate 39/0/2; commit `83b9db2`..`de83088`. **Approvata da JenJo** con tre aggiunte (Attenzione 6 e il suo test in T2, prompt per Sonnet in T2, chiamata reale a 2 giri in T3). Nessun fallback lato server sui rifiuti: basta la frase fissa di oggi (scelta di JenJo: ritenta solo `cyber` e `frontier_llm`). Fatti verificati il 09/11: `_params` a riga 1119; prezzi sulla pagina ufficiale; regole di `between_tools` dalla guida di migrazione.

**Spec:** scelte di JenJo del 09/11 in chat: Sonnet subito, l'A/B serve solo a confermare; thinking come interruttore provato con un A/B; pre-chiamata Haiku sulle frasi incerte; memoria in due tempi (ora il tetto del blocco, il diario in un gruppo dopo); G3c al posto di G3a, che aspetta (il suo T3 non serve più con Sonnet principale). Fonti: `reports/Migliorare Claudio con più crediti.md`, le note in `research_notes/Migliorare Claudio con più crediti/`, prezzi verificati il 09/11 su platform.claude.com/docs/en/about-claude/pricing.

## Vincoli globali

- Prima di cercare nel codice: `graphify explain "<X>"` o `graphify path "<A>" "<B>"`, poi solo le righe indicate.
- Gli implementer **non committano**. Regole: `.superpowers/sdd/regole/implementer-rules.md`.
- Mai azione senza sì. Nessuna lettura di un LLM vale come risposta a «Confermi?»; i livelli 2-3 passano sempre dalla domanda del codice.
- Su Sonnet 5.5: `thinking: {"type": "disabled"}` e `tool_choice` `any`/`tool` danno 400; `between_tools` solo con effort `high` o meno, nessun altro campo dentro `thinking`, mai rimandato a un altro modello.
- Modelli e prezzi solo in configurazione. Script di prova con `Recorder().ctx()` e il `FakeDesktop` del gate; `data/spend.db` non toccato dalle sonde.
- Budget API del gruppo: al massimo 2 $ (tetto di JenJo 5 $). Il gate di replay resta a 0 fallite.

## Attenzione

1. **Note di avanzamento di Sonnet** (`between_tools`): arrivano come blocchi `thinking` con testo. Mai dette a voce, mai nella cronologia come testo, ripassate intatte nel giro di tool successivo. Test in T2.
2. **Offerta di Sonnet con Sonnet principale:** niente `ask_sonnet`, niente «Lo uso?», e «pensaci bene» non chiama un secondo modello uguale. Test in T2.
3. **Pre-chiamata con una domanda aperta, da spento o in pausa:** non parte. Test in T4.
4. **Lettura di Haiku che è un comando di livello 2-3:** chiede sempre, come oggi; non diventa mai un sì. Test in T4.
5. **Thinking acceso con il tetto vecchio:** risposta tagliata; il tetto del thinking si applica. Test in T2.
6. **Thinking legato alla conversazione:** dentro un giro di tool, `system`, `tools` e i messaggi precedenti restano identici byte per byte. Con `between_tools` il `drop_block` non è ammesso: un cambio dà 400 sugli account creati dal 31/08/2026. Test in T2.

---

## Task 1: prezzi

**File:** `config/pricing.toml`. Test: quelli che leggono il file, se fissano il valore.

- [ ] `[models.claude-sonnet-5-5] cache_read = 0.10` (era 0.20; pagina ufficiale, nota 2: 0.05x l'input). Commento in testa: Haiku 5.5 oltre 100.000 token costa **5 volte** (0,50 / 2,50 $), non il doppio.
- [ ] Suite, ruff, mypy.

## Task 2: Sonnet 5.5 come modello principale

**File:** `src/claudio/brain/claude.py` (`_params` riga 1119, il ramo dell'offerta e di «pensaci bene», la lettura dei blocchi nello stream), `src/claudio/config.py`, `config/claudio.toml`. Test: `tests/unit/test_claude_brain.py`, `tests/unit/test_pc_tools.py`.

**Interfaccia:** in `ClaudeSettings` due campi nuovi: `between_tools_models: list[str]` (default `["claude-sonnet-5-5"]`), `thinking_max_tokens: int = Field(1024, ge=64, le=8192)`. `[guess]` riceve `model` (default `"claude-haiku-5-5"`): guess e pre-chiamata non seguono il modello principale.

**Regole:**
1. Thinking spento: `{"type": "between_tools"}` se il modello è in `between_tools_models`, altrimenti `{"type": "disabled"}`. Acceso: nessun campo `thinking` (adattivo) e `max_tokens = thinking_max_tokens`. La configurazione rifiuta all'avvio `between_tools` con effort `xhigh` o `max`.
2. Modello principale uguale a `smart_model`: nessun `ask_sonnet` tra i tool, nessuna proposta, nessuna seconda chiamata per «pensaci bene».
3. I blocchi `thinking` con testo non vanno alla voce, al rilevatore di promesse né al testo della cronologia; nel giro di tool successivo l'intero contenuto dell'assistente ripassa intatto.
4. Il percorso del retry resta com'è: Sonnet non è in `retry_models`, quindi niente `tool_choice` forzato.
5. `effort` resta `"low"` fino all'A/B di T3.

- [ ] Test che falliscono, con il client finto: Sonnet e thinking spento → `thinking == {"type": "between_tools"}`, nessun `"disabled"`; Haiku e thinking spento → `disabled`; thinking acceso → nessuna chiave `thinking` e `max_tokens == 1024`; `model == smart_model` → nessun `ask_sonnet` nei tool e `wants_offer` falso; uno stream con un blocco `thinking` con testo, poi `tool_use` → il testo non è detto, il round dopo contiene il blocco identico; config con `between_tools` ed effort `xhigh` → errore di validazione; `guess` chiama `claude-haiku-5-5` anche con Sonnet principale; un giro a 2 round con Sonnet → `system` e `tools` identici tra i round, i messaggi del primo sono un prefisso del secondo, e con Sonnet principale nessuna chiamata con `tools=False` dentro un turno (oggi righe 417 e 697).
- [ ] Visti fallire, poi il codice. Suite, ruff, mypy.
- [ ] Prompt per Sonnet: grep nel system prompt e in `CLAUDIO.md` di istruzioni contro il pensare o contro i tool («solo se necessario», «minimizza le chiamate»): Sonnet 5.5 le segue alla lettera. Modifiche come diff a JenJo.
- [ ] Ultimo passo: `[claude] model = "claude-sonnet-5-5"` in `config/claudio.toml`.

## Task 3: A/B di conferma (a pagamento)

**File:** sonda nuova `research_notes/Sonnet e thinking/prove/ab_sonnet.py`, fuori da git; riusa `make_world`, `request_params`, `one_call`, il freno e il tracker usa e getta di `research_notes/Effort e thinking Haiku/prove/ab_effort.py`. Solo la prima chiamata; un `tool_use` si legge e non si esegue.

- Frasi: circa 50 dai log (`logs/sessions/*/conversazione.md`, solo grep): tutte quelle con promessa senza tool, passi raccontati, domanda al posto dell'azione, «Confermi?» scritto da Claude, più un campione di richieste normali e di domande. Elenco scritto prima della corsa.
- Bracci: `haiku/off` (com'era), `haiku/adaptive-medium`, `sonnet/between_tools` (low), `sonnet/adaptive-low`. 3 ripetizioni, tetto 2048 per vedere quanto pensa davvero. Il braccio Haiku con thinking dice se Haiku pensante arriva a Sonnet spendendo meno. Già misurato in G1 (20 frasi × 3, prima chiamata), non si ripete: l'effort di Haiku senza thinking non cambia nulla (`low/off` prima frase p50 559 ms, 28/30 frasi d'azione con tool; `medium/off` 563 ms, 27/30); col thinking `low` 848 ms e 29/30, `medium` 1.168 ms e 30/30, 7 tagli su 60 a 400 token.
- Misure: prima frase p50 e p90, token di output, tagli, tool chiamato o solo promesso (`promises.py`), testo prima della chiamata, frasi nelle risposte brevi, costo.
- Costo calcolato: 300 chiamate Sonnet × 0,0037 $ = 1,11 $, più thinking (ipotesi +103 token × 10 $/MTok × 150 = 0,15 $), più 150 Haiku × 0,0002 $ = 0,03 $, più 150 Haiku pensanti × 0,00026 $ = 0,04 $ → circa 1,33 $. Freno a 2 $.
- Una chiamata reale a 2 giri su `sonnet/between_tools` (un `tool_result` finto, `FakeDesktop`): prova che il blocco `thinking` ripassato è accettato sul nostro account (circa 0,01 $).
- **Allarme** (si riporta a JenJo, Sonnet resta): prima frase p50 di Sonnet oltre 1.500 ms, o più promesse senza tool di Haiku. Il valore di `thinking_max_tokens` si fissa sul p99 misurato più margine.

- [ ] `--self-test` e `--dry-run` (costo massimo, nessuna chiamata). Poi `--run --yes`, tabella per braccio, spesa vera.

## Task 4: pre-chiamata Haiku sulle frasi incerte

**File:** `src/claudio/app.py` (dove nasce l'incertezza, `_tell_hearing` riga 1186), `src/claudio/router/responder.py` (`guess` riga 245), `config/claudio.toml` (`[guess] precall = false`). Test: `tests/unit/test_guess.py`, `tests/unit/test_g2_interactions.py`.

**Regole:**
1. Parte solo se Whisper è insicuro, il router non capisce la frase, nessuna domanda è aperta, Claudio non è spento né in pausa, e `precall = true`.
2. Haiku riceve la frase, le alternative di Whisper, i nomi vicini del PC (finestre, cartelle, app) e l'ultimo comando locale; risponde con la lettura più probabile o con «non chiaro».
3. Lettura che il router capisce: la stessa decisione di `guess` (`act_at`, `ask_at`, «Hai detto: ...?»); un comando di livello 2-3 chiede sempre. Lettura che il router non capisce: va al modello principale con la frase sentita e la lettura. «Non chiaro», timeout (`timeout_s`) o errore: si va avanti come oggi.

- [ ] Test che falliscono: «Claudio, chi udite?» con l'alternativa «Claudio, chiuditi.» e la lettura di Haiku → «Hai detto: chiuditi?», nessuna chiusura senza sì; con una domanda aperta, da spento, in pausa → nessuna pre-chiamata; timeout → stesso percorso di oggi; `precall = false` → nessuna chiamata.
- [ ] Visti fallire, poi il codice. Suite, ruff, mypy.
- [ ] Replay offline delle 56 richieste incerte dei log, 3 volte (circa 0,03 $): comandi capiti, azioni sbagliate (0 ammesse ai livelli 2-3 senza sì), domande in più, latenza. Interruttore acceso solo se nessuna azione sbagliata: altrimenti si riporta a JenJo.

## Task 5: blocco di memoria più grande

**File:** `config/claudio.toml` (`[memory] block_chars`; `MemorySettings` ammette fino a 20000, nessun cambio di codice).

- [ ] `block_chars = 6000` (circa 1.800 token, in cache). Un test che fissa 2000 si aggiorna. Suite.

## Fine gruppo (coordinatore)

- [ ] Revisione dell'intero gruppo su Opus, con mutazioni su T2 e T4.
- [ ] Cassette riregistrate su Sonnet (calcolo: 0,0035 $ misurati su Haiku × 18 ≈ 0,06 $); gate a 0 fallite.
- [ ] Prova «J» in `piani/IT-2-passo-0b-prove.md`: latenza percepita, personaggio, un'azione di livello 1, «chiudi la finestra» con conferma, una frase storpiata, «pensaci bene».
- [ ] Diff e test a JenJo; commit dopo l'ok, uno per task. Diff separato per `CLAUDE.md` del progetto (oggi: «Haiku fa da smistatore... Sonnet solo se necessario»). `graphify update .`; consegne aggiornate.
