# Claudio – Piano maestro e ciclo di vita

**Stato: BOZZA del 11/11/2026. JenJo ha confermato le decisioni della sezione 3; il testo è da rivedere.** Questo file decide **ordine e perimetro** dei lavori. Il dettaglio di ogni gruppo resta nei piani già scritti (sezione 5): dove l'ordine è diverso, vale questo file.

**Obiettivo principale (JenJo, 11/11): migliorare la comprensione e le capacità di Claudio.** Quando due lavori competono, passa quello che fa capire meglio Claudio o gli fa fare una cosa nuova.

Se sei il thread che riceve la consegna, comincia dalla sezione 1 e fermati dove dice di fermarti.

## 1. Primo thread: allineamento

Un passo alla volta. Ogni modifica a un file si mostra come diff e aspetta l'ok di JenJo (regola del `CLAUDE.md`).

### Passo A — Orientarsi, senza modificare nulla

1. `git status` e `git log --oneline -3`: il punto di partenza atteso è il ramo `it2/passo-0b` a `dc808e9`, allineato a GitHub.
2. `graphify update .`
3. Controlla che esista `reports/Connettere Claudio.md` (ignorato da git). Se manca, chiedi a JenJo di copiarlo da `C:\Users\JenJo\Downloads\Connettere Claudio.md`: tutte le sigle S, M, R, I di questo piano rimandano a quel report, con file e riga.
4. Leggi `Stato lasciato` in `piani/IT-2-passo-0b-consegne.md`.

**Fatto quando:** dici a JenJo in cinque righe il commit attuale, le prove a mano ancora aperte e il primo compito del Passo B.

### In parallelo — le prove a mano di JenJo

JenJo vuole chiuderle. Non aspettano i Passi B e C: proponile appena Claudio è riavviato sul codice di G3c. I passi sono in `piani/IT-2-passo-0b-prove.md`.

| Ordine | Prova | Perché adesso |
| --- | --- | --- |
| 1 | J | è il primo uso vero di Sonnet principale: il suo giorno è l'inizio della prova D6 |
| 2 | F | dà le etichette per G3b e il corpus per G5, cioè la comprensione |
| 3 | H2 | chiude G2: conferme e interruzioni dal vivo |

**Fatto quando:** l'esito di ogni prova è scritto nel file delle prove e ciò che ha trovato è assegnato a un gruppo.

### Passo B — F0: documenti e strumenti, nessun codice di prodotto

| # | Cosa | Fatto quando |
| --- | --- | --- |
| B1 | `CLAUDE.md` del progetto: voce (Jack, poi Kokoro, Piper solo se Kokoro non si carica); `CLAUDIO.md` si copia una volta e le regole dette a voce vanno in `data/memory.db`; la chat FastAPI esiste già; solo italiano; l'elenco «Fasi» diventa un rimando a questo file | ogni punto della sezione 6 del report («Documenti e piani non più allineati») è corretto o motivato |
| B2 | `README.md` e `README.it.md`: Sonnet principale in prova (decisione D6), solo italiano, effetto voce | corrette le righe che il report elenca (6, 98, 137-144, 215, 349, 462-464) nelle due lingue |
| B3 | `piani/IT-2-consegne.md`: una nota in testa che rimanda qui per l'ordine. La riscrittura del Passo 1 aspetta il verdetto D6 | la nota c'è; nessun'altra riga toccata |
| B4 | `gh` (GitHub CLI) installato: oggi manca su questo PC e serve per le PR di rilascio | `gh auth status` risponde |
| B5 | Protezione di `main`: si entra solo con PR e CI verde (decisione D8). La imposta JenJo o la approva | una PR di prova senza CI verde non si può fondere |
| B6 | Skill: vedi sezione 7 | vedi sezione 7 |
| B7 | Prova Sonnet (D6): scrivi qui la data di inizio, cioè il giorno della prova J | data scritta nella sezione 3, riga D6 |

**Fatto quando:** B1-B7 chiusi, commit `docs:` approvati e spinti, CI verde.

### Passo C — F1: pulizia piccola, 0 $ di API

Test prima del codice per S8 e S9. Per le rimozioni vale il test della cancellazione: si toglie un test solo se provava la funzione tolta.

| # | Cosa | Nota |
| --- | --- | --- |
| C1 | S8: `purge_old_logs` copre anche `claudio.log.jsonl` e i ruotati | privacy |
| C2 | S9: all'avvio si controlla anche il prezzo di `[guess] model` | budget |
| C3 | `piper-tts` fissato a 1.8.0 in `pyproject.toml` | `kokoro.py` usa parti private di Piper |
| C4 | M8, M9: funzioni e setter usati solo dai test | |
| C5 | M7: via il codice multilingua, resta l'italiano | può cambiare il prompt di sistema: vedi sotto |
| C6 | M6: via il catalogo inglese dei log | prima `graphify explain i18n` per vedere chi lo usa |
| C7 | M4: via `tts/effects.py` e il suo testo nel README | |

**Attenzione alle cassette.** Una modifica a `system`, ai tool o a `data/CLAUDIO.md` fa mancare tutte le cassette del gate. Se C5 tocca il prompt, dillo a JenJo prima di rifare la registrazione.

**Restano fuori da F1, di proposito:** M1 (escalation Haiku→Sonnet), M2 (retry forzato) e R2 (default di `ClaudeSettings`). Servono se Haiku torna principale: si decidono al verdetto D6.

**Fatto quando:** suite, `ruff` e `mypy` verdi; gate di replay 39/0/2 sul fisso, oppure rifatto con l'ok di JenJo; righe tolte riportate; commit spinti; CI verde. Poi **fermati** e riassumi a JenJo fatto, costi, rischi.

## 2. Stato di partenza, verificato l'11/11/2026

| Cosa | Stato |
| --- | --- |
| Ramo | `it2/passo-0b` a `dc808e9`, uguale a `origin`; `main` è 47 commit indietro |
| Rilasci | ultimo tag `v0.4.1`; v0.4.2 in corso: G0, G1, G2, G3c fatti |
| Prove a mano aperte | J, H2, F (`piani/IT-2-passo-0b-prove.md`) |
| Controlli | suite 4566 passati e 4 saltati; gate 39/0/2 **solo sul fisso** (con Whisper su CPU i trascritti cambiano) |
| CI | `.github/workflows/ci.yml` a ogni push e PR: ruff, mypy, pytest, modelli, test lenti su CPU |
| Report del 10/11 | 5 affermazioni ricontrollate sul codice, tutte vere: S1, S3, S4, M1, cronologia 6 scambi e 300 s |
| Vetrina | manca `LICENSE`; i due README raccontano ancora Haiku principale e «italiano e inglese» |

## 3. Decisioni di JenJo, 10-11/11/2026

| # | Decisione |
| --- | --- |
| D1 | **Scopo:** strumento personale più una vetrina leggera: `LICENSE`, README allineati, schema dell'architettura, demo breve |
| D2 | **Linea di taglio:** si chiudono v0.4.2 e G8. G4, G7, G9, G10 passano dopo le cuciture |
| D3 | **Portatile:** solo sviluppo. Claudio e il gate di replay girano solo sul fisso |
| D4 | **Capacità nuove:** web vero, poi osservatore e sessione lunga (I3, I4), poi i momenti giusti (I5). Telefono (I8), app e agenti (I9), casa: dopo |
| D5 | **Pulizia:** via multilingua, catalogo inglese, effetti voce. Pre-call di Haiku (M3): se G5 non lo accende, si toglie |
| D6 | **Sonnet principale è in prova** per circa 10 giorni d'uso reale. Resta se la proiezione è entro **50 $ al mese**; sopra, torna Haiku principale. Fino al verdetto M1, M2 e R2 non si toccano. Inizio prova: 12/11/2026, giorno della prova J. |
| D7 | **Diario dell'osservatore:** su disco in `claudio.db`, cancellato dopo `retention_days` |
| D8 | **Git:** push a fine di ogni gruppo approvato; in `main` solo con PR a CI verde, al rilascio |
| D9 | **Skill:** vedi sezione 7 |

**Come si misura D6.** Nessun codice nuovo: una query su `data/spend.db`.

```sql
SELECT day, COUNT(*) AS calls, ROUND(SUM(cost), 4) AS usd,
       SUM(cache_read_input_tokens = 0) AS cold_calls
FROM calls
WHERE model LIKE '%sonnet%' AND day >= :start
GROUP BY day;
```

Proiezione = media di `usd` sui giorni d'uso reale × 30. Proposta da confermare con JenJo: è «d'uso reale» un giorno con almeno 10 chiamate. Riporta anche la quota di `cold_calls`: è il numero che sposta il costo (sezione 8).

## 4. Ordine dei lavori

Le sigle G sono i gruppi di `piani/IT-2-passo-0b-consegne.md`; S, M, R, I sono nel report. I nomi delle versioni dopo v0.4.3 sono proposte.

| Fase | Cosa | Dettaglio in | Fatto quando |
| --- | --- | --- | --- |
| F0 | Documenti e strumenti | sezione 1, Passo B | B1-B7 chiusi |
| F1 | Pulizia piccola | sezione 1, Passo C | suite e gate verdi, CI verde |
| F2 | **Chiudere v0.4.2**, con la comprensione per prima: G3a (verbo singolo); **G5** (comprensione, sul corpus della prova F); G3b; G6 allargato a S1-S5, S7, R6 e all'indice di Windows Search | piano 0b; report, sezione 1 | gate di rilascio del piano 0b; tag `v0.4.2`; prima PR verso `main` |
| F3 | **G8:** Jack all'avvio | piano 0b | causa scritta; da freddo Jack arriva da solo; tag `v0.4.3` |
| V | **Verdetto D6**, quando ci sono 10 giorni d'uso reale: non aspetta le altre fasi | sezione 3 | Sonnet resta: si cancellano M1 e M2, si sistema R2, si riscrive `IT-2-consegne.md` per Sonnet. Torna Haiku: modello in configurazione, cassette rifatte, M1 e M2 restano |
| F4 | **Cuciture:** registro dei tool, porta del cervello con il canale, cancello del turno con coda di annunci, oggetto `Conversation`, una sola `decide()` (S6), tipi del desktop con un solo enumeratore; i finti spostati in `tests/fakes.py` | report, «Le cuciture» | **il gate passa senza rifare le cassette**: prova che prompt e tool non sono cambiati di un byte; suite verde |
| F5 | **G7:** latenza, misurata dopo le cuciture, con la scelta del TTL della cache | piano 0b; sezione 8 | latenza prima e dopo per percorso su almeno 20 richieste vere |
| F6 | **Web vero** sopra il registro dei tool | `IT-2-consegne.md`, Passo 1 | i criteri di quel passo; costo reale per risposta nel report |
| F7 | **Osservatore e sessione lunga** (I3, I4), con il diario di D7 | report, I3 e I4 | «cosa stavo facendo?» risponde dal diario a 0 $; per il contesto, una prova che confronta finestra più larga, riassunto locale e compaction, misurando `cache_read_input_tokens`, costo e latenza: sceglie JenJo sui numeri |
| F8 | **Momenti giusti** (I5), con la modalità gioco come segnale | report, I5; `IT-2-consegne.md`, Passo 3 | tetti e canali decisi da JenJo all'inizio del gruppo |
| Vetrina | `LICENSE` dopo una scansione delle licenze delle dipendenze; schema dell'architettura; demo | D1 | al rilascio v0.4.2 i primi due, la demo dopo F6 o F7 |
| Di riserva | G4 (tono), G9 (residui); G10 solo se le soglie lo chiedono | piano 0b | come scritto lì |
| Dopo | Telefono (I8), app e agenti (I9), Passi 2, 4, 5, 6 di IT-2, casa, procedure (I6), passaggio di consegne (I7), aggiornamento automatico | report; `CLAUDE.md` per l'aggiornamento | non pianificati ora |

**Vincoli tra le fasi:**
- F7 viene dopo F2: un Claudio che osserva sempre moltiplica i buchi dello schermo S1-S4.
- F4 viene prima di F6: senza il registro, ogni connettore aggiunge rami in 6 metodi di `PcTools`.
- I refactoring R1-R10 si fanno quando un gruppo lavora già in quel file, non come gruppo a sé.
- **Proposta da confermare a fine G5:** se gli errori restano sopra la soglia già scritta in G10, il confronto con un STT cloud si fa subito e non dopo le cuciture. Il costo misurato è di centesimi al mese (sezione 9); servono le chiavi di JenJo e il suo ok a ogni invio di audio.
- Licenza: `piper-tts` 1.8.0 è `GPL-3.0-or-later` ed è importato nel processo. Va detto a JenJo prima che scelga.

## 5. Dove sta il dettaglio

| File | Leggilo quando |
| --- | --- |
| `CLAUDE.md` | sempre: metodo, priorità, sicurezza |
| `piani/IT-2-passo-0b-consegne.md` | lavori a un gruppo G; «Come lavorare» è il ciclo di ogni gruppo |
| `piani/IT-2-passo-0b-prove.md` | prepari o aggiorni una prova a mano di JenJo |
| `piani/IT-2-consegne.md` | lavori a web, modalità gioco o ai Passi 2-6. Dà ancora per scontato Haiku principale e cita righe spostate |
| `reports/Connettere Claudio.md` | questo piano nomina una sigla S, M, R o I |
| `reports/Migliorare Claudio con più crediti.md` | il report sopra lo cita |

## 6. Ciclo di vita

Il ciclo di un gruppo è già scritto in «Come lavorare» del piano 0b e nel `CLAUDE.md`. Qui ci sono solo le differenze decise ora.

- **Fine gruppo:** dopo l'ok e i commit, **push sul ramo**. La CI controlla ogni gruppo, non quattro insieme.
- **Rilascio:** gate di rilascio, README e CHANGELOG, poi **PR verso `main`** con CI verde, poi il tag.
- **Gate di replay:** resta locale e sul fisso. Usa audio privato e la GPU, e un runner in casa su un repo pubblico eseguirebbe codice di chiunque apra una PR.
- **Un thread per fase**, con questo file e la memoria come passaggio di consegne.
- **Spesa:** il credito è un tetto vero (sezione 8). Un workspace dedicato con un limite protegge anche se il contatore locale sbaglia.

## 7. Skill di sviluppo

**Assetto (D9):** superpowers è l'unico «router», cioè il pack che decide il flusso. supermatt e ponytail si usano à la carte, una skill alla volta. Superpowers va aggiornato.

**Fatti verificati l'11/11/2026:**
- I plugin qui sono «Synced from claude.ai»: superpowers 6.4.2, supermatt 0.22.0. L'ultima superpowers è la 7.0.2 del 03/11/2026, con correzioni per Windows nell'hook di avvio e il brainstorming rifatto nella 7.0.0.
- Peso fisso in ogni sessione (`claude plugin details <nome>@synced`): superpowers circa 840 token, supermatt circa 2.506. I sette plugin estranei a Claudio (brightdata-plugin, desktop-commander, data, engineering, design, productivity, cowork-plugin-management) circa 10.134.
- Il confronto di addyosmani/agent-skills è scritto dal progetto stesso e cita un solo esperimento su un compito. Consiglia un router principale e il resto à la carte: i meta-skill impilati si contendono i nomi dei comandi e il flusso.
- Prova di Vercel (01/2026): una skill lasciata all'innesco automatico non è mai stata chiamata nel 56% dei casi; con l'istruzione esplicita il risultato sale al 79%, con un indice sempre caricato al 100%.
- Studio di Snyk (02/2026): su 3.984 skill pubbliche il 36,82% ha almeno un difetto di sicurezza e 76 sono malevole confermate.

**Cosa ne segue per Claudio:**
- Una regola che deve valere sempre sta nel `CLAUDE.md` o in un hook. Le skill si chiamano per nome nei momenti della tabella sotto.
- Ogni pack nuovo e ogni aggiornamento passa da `supermatt:dependency-vetting` e dalla lettura dei `SKILL.md`, compreso `allowed-tools`.

**Compito B6:**
1. Aggiorna superpowers alla 7.0.2 da claude.ai: provato l'11/11, `claude plugin update superpowers@synced` rifiuta un plugin sincronizzato («cannot be updated here»). Controllo della 7.0.2 (11/11): nessuna skill dichiara `allowed-tools`, l'unico hook è locale, l'unica chiamata di rete è il logo del visual companion, che manda la versione e si spegne con `SUPERPOWERS_DISABLE_TELEMETRY`.
2. Spegni per questo progetto i plugin estranei con `claude plugin disable <nome> --scope local` (oppure `project`: guarda quale file cambia e mostra il diff). Chiedi a JenJo se desktop-commander gli serve qui.
3. Lancia `/doctor` e riporta cosa dice sul costo delle skill. `claude doctor` da riga di comando (11/11) controlla solo l'installazione; il costo di ogni plugin si legge con `claude plugin details <nome>@synced`.

**Fatto quando:** `claude plugin list` mostra superpowers 7.0.2; nel progetto restano attivi solo i plugin scelti da JenJo; l'esito di `/doctor` è riportato.

| Momento | Skill | Nota |
| --- | --- | --- |
| Domanda sul codice | `graphify query`, poi le righe che indica | regola del `CLAUDE.md` |
| Gruppo nuovo o requisiti poco chiari | `superpowers:brainstorming` | `supermatt:grilling` quando JenJo vuole essere messo alla prova |
| Fatti esterni: API, librerie, prezzi | `supermatt:research` | il workflow di ricerca approfondita solo per domande larghe: l'ultimo giro ha usato 5,2 milioni di token |
| Dipendenza, tool o skill nuovi | `supermatt:dependency-vetting` | |
| Dove mettere una cucitura (F4) | `supermatt:codebase-design` | test della cancellazione; due adattatori fanno una cucitura vera |
| Piano del gruppo | `superpowers:writing-plans` | |
| Implementazione | `superpowers:subagent-driven-development` con `test-driven-development`; ponytail sempre | |
| Guasto | `superpowers:systematic-debugging` | |
| Prima di dire «fatto» | `superpowers:verification-before-completion` | |
| Revisione | `superpowers:requesting-code-review` per task; `/code-review` sull'intero gruppo; `/security-review` nei gruppi di sicurezza (G6, F4, F7) | nei gruppi di sicurezza anche la verifica per mutazione |
| Pulizia (F1, verdetto D6) | ponytail e test della cancellazione | |
| Rilascio | `superpowers:finishing-a-development-branch`, `supermatt:pr` | |
| Piani, `CLAUDE.md`, documenti per agenti | `supermatt:writing-for-agents` | |

**agent-skills sì o no:** _in attesa dell'esito della ricerca dedicata (lanciata l'11/11)._

## 8. Credito API: fatti e usi possibili

**Fatti**, dalla pagina ufficiale letta l'11/11/2026 ([crediti](https://platform.claude.com/docs/en/about-claude/api-credits-for-subscribers), [prezzi](https://platform.claude.com/docs/en/about-claude/pricing)):
- Max dà 100 $ al mese. Coprono API, Agent SDK, Managed Agents e Playground. **Non** coprono Claude Code né altri fornitori, quindi nemmeno un STT o TTS cloud.
- Non si accumulano: scadono a fine ciclo. Finiti i crediti le richieste si fermano fino al mese dopo, senza addebiti, se non ci sono crediti comprati.
- Prezzi per milione di token: Sonnet 5.5 2 $ in, 10 $ out, scrittura cache 2,50 $ (5 minuti) o 4 $ (1 ora), lettura 0,10 $. Opus 5.5 4 $, 20 $, 5 $, 8 $, 0,20 $. Batch a metà prezzo. Ricerca web 10 $ ogni 1000.

**Misure** da `data/spend.db` (453 chiamate) e `logs/metrics.jsonl` (20 giorni attivi):
- Una chiamata Sonnet a cache fredda è costata 0,0169 $: 6.523 token di prefisso scritti. A cache calda il calcolo dà circa 0,0018 $.
- Il 90,1% delle chiamate arriva entro 5 minuti dalla precedente, ma sono sessioni di prova a raffiche. Chiamate al giorno: mediana 16,5, massimo 85.
- Calcolo: da 1,64 a 8,4 $ al mese con questo ritmo; 43 $ al mese con 85 chiamate al giorno tutte fredde.

**Usi possibili**, con il conto fatto sui prezzi sopra. I volumi sono parametri scelti, non previsioni.

| Uso | Conto | $ al mese |
| --- | --- | --- |
| Ricerca web (F6) | 30 ricerche al giorno × 0,01 $, più i token delle pagine | 9,00 più token |
| Momenti giusti (F8) | tabella del report | 1,48-9,35 |
| Cache a 1 ora, per l'uso sparso | una scrittura l'ora di 6.523 token × 4 $/M = 0,0261 $, per 8 ore | 6,26 |
| Opus su richiesta («pensaci bene») | 3.000 token in e 1.000 out = 0,032 $, 5 al giorno | 4,80 |
| Riassunto notturno del diario, in batch | 20.000 token in e 1.000 out su Sonnet = 0,025 $ a notte | 0,75 |
| Prove sui prompt | misurato: l'A/B di G3c è costato 0,678 $, le cassette 0,0035 $ | sotto 1 a giro |

Sommando i massimi e la voce su Sonnet si resta sotto i 40 $. Solo il lavoro di agenti via Agent SDK consuma davvero il credito: Claude Code pilotato a voce (I9) o una ricerca approfondita fatta girare sull'API invece che sull'abbonamento.

## 9. Fatti e piste dalla ricerca del 10-11/11

**Verificati** con voto 3-0 o 2-0:
- **Parakeet TDT 0.6b v3:** italiano tra 25 lingue; WER italiano 3,00% su FLEURS, 10,08% su MLS, 3,69% su CoVoST (dati del produttore); licenza CC-BY-4.0. La scheda nomina solo Linux e NeMo: su Windows va fatto il vetting del runtime. Candidato per G5.
- **Voxtral Realtime in locale:** chiede una GPU con almeno 16 GB. Escluso: la 5070 Ti ne ha 16 e li divide con Jack.
- **Deepgram Nova-3** in streaming 0,46 $ l'ora, con 200 $ di credito gratuito senza scadenza. **ElevenLabs Scribe v3 Live** 0,22 $ l'ora. Con il massimo misurato (1,33 minuti di parlato al giorno) sono 0,31 e 0,15 $ al mese: per G10 il costo non conta, contano privacy e funzionamento senza rete.
- **Smentito, da non usare:** la classifica italiana dell'Open ASR Leaderboard che metteva Whisper large-v3 davanti a Parakeet.

**Piste non verificate**, da controllare all'inizio del gruppo che le usa:
- G3b: Smart Turn v3.2 dichiara 23 lingue con l'italiano, licenza BSD a 2 clausole, modello da 8 MB per CPU.
- Eco e casse (R3): `pywebrtc-audio` 0.2.0 porta AEC3 su Windows per Python 3.12, ma è giovane e in beta.
- Voce (G4, G10): le due voci italiane di Kokoro hanno voto C dall'autore; Chatterbox si dichiara non più mantenuto; Pocket TTS 4.0.1 dichiara italiano e streaming su CPU.
- F6: tipi dei tool `web_search_20260318` e `web_fetch_20260318` senza header beta: da scegliere con la Models API, come dice già il Passo 1.
- F7: ricerca ibrida FTS5 più `sqlite-vec` in un solo file SQLite, solo se il recupero della memoria diventa un problema misurato.

## 10. Rischi e domande aperte

- **La prova Sonnet misura l'uso, non il modello:** con poche richieste sparse quasi ogni chiamata è fredda. Il verdetto va letto insieme alla quota di chiamate fredde e alla scelta del TTL.
- **Licenza** (D1): da scegliere dopo la scansione; pesa il vincolo GPL di `piper-tts`.
- **Compaction** è una beta: F7 la prova contro due alternative senza beta prima di fidarsi.
- **La coda resta lunga:** tra F2 e la prima capacità nuova (F6) ci sono G3a, G3b, G5, G6, G8, le cuciture e G7. Se una prova a mano blocca una fase, la riserva (G4, G9) riempie l'attesa.
- **Decisioni che aspettano il loro gruppo:** conferma non vocale del livello 3 (G6); frasi che attivano l'osservatore (F7); tetti e canali della proattività (F8).
