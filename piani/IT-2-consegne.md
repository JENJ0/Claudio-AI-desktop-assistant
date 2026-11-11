# IT-2 – Piano completo e consegna per i prossimi thread

> **Nota dell'11/11/2026.** Ordine e perimetro dei lavori stanno in `piani/ROADMAP.md`: dove
> questo file dice altro, vale la roadmap. Il prossimo passo lo dà la roadmap, non la tabella di
> «Come lavorare». Questo piano dà ancora per scontato Haiku come modello principale e cita righe
> di `claude.py` che si sono spostate: il Passo 1 si riscrive al verdetto D6 (roadmap, sezione 4,
> riga V).

Piano deciso con JenJo il 04/11/2026, dopo la lettura dei log e una ricerca verificata. Parte da
`main` = `110d023` (v0.4.0): i numeri di riga valgono per quel commit, le funzioni restano il
riferimento se il codice si è spostato.

## Come lavorare

Un thread fa **un solo passo**.

1. Dalla tabella qui sotto prendi il primo passo `da fare`, leggi la sua sezione e crea il ramo
   `it2/passo-N` da `main`.
2. Proponi a JenJo un piano breve del passo e aspetta la sua approvazione.
3. Implementa con test prima del codice, mostrando il diff prima di ogni modifica.
4. Chiudi il passo quando ogni punto di **Fatto quando** è dimostrato da un test o da una misura:
   `uv run pytest`, `uv run ruff check .`, `uv run mypy src` verdi, test manuale scritto passo
   passo, README e CHANGELOG aggiornati, commit e tag dopo l'approvazione.
5. Aggiorna la tabella e la sezione **Stato lasciato dal passo** qui sotto, riassumi a JenJo cosa è
   stato fatto, costi, latenza misurata e rischi, poi fermati. Il thread dopo riparte da questo file.

| Passo | Contenuto | Versione | API nei test (stima) | Stato |
|---|---|---|---|---|
| 0 | Correzioni | v0.4.1 | 0 $ | fatto |
| 0b | Affidabilità e residui: `piani/IT-2-passo-0b-consegne.md` | v0.4.2, v0.4.3 | ≤ 0,50 $ | da fare, **prima del Passo 1** |
| 1 | Web vero con Haiku | v0.5.0 | ≤ 0,15 $ | da fare |
| 2 | Risorse, cronometro, Opus, schermo | v0.5.1 | ≤ 0,10 $ | da fare |
| 3 | Modalità gioco, frasi non rivolte a Claudio, avvio a freddo | v0.6.0 | ≤ 0,05 $ | da fare |
| 4 | Buffer di 30 s | v0.6.1 | ≤ 0,02 $ | da fare |
| 5 | Clic per nome e livello 3 in chat | v0.7.0 | ≤ 0,05 $ | da fare |
| 6 | Computer use | da decidere | Opus ≤ 1 € | da fare, in fondo |

Con il budget attuale i test con chiamate vere vanno distribuiti su più giorni: le chiamate vere
sono poche, tutto il resto gira con il client finto.

### Stato lasciato dal passo

Passo 0 chiuso il 05/11/2026 (v0.4.1). Tutti i punti di **Fatto quando** hanno un test, tranne
uno, misurato a parte: **la latenza non è peggiorata** (`benchmark_latency.py --log`, 44 richieste
dal vivo: M1 su Claude da 4947 a 3062 ms, locale da 1326 a 1004 ms). Il confronto non è alla pari:
il periodo precedente aveva sessioni in gioco.

Fatto oltre il passo, su richiesta di JenJo: refactoring delle funzioni sopra C901 12 e correzione dei
bug trovati dall'audit e dalle revisioni. Dettaglio in `CHANGELOG.md`; il report completo è
`piani/IT-2-passo-0-report.md`, solo in locale (fuori da git: cita i log).

**Da fare dopo, deciso da JenJo: la voce di Jack non parte all'avvio del PC.** Su 14 avvii a freddo tra
il 31/10 e il 05/11 Jack è fallito 11 volte (`worker Jack: nessuna risposta in 90 s`), gli ultimi 6 di
fila, con Whisper a 112 s in parallelo. Non nasce da questo passo: c'era già prima del primo commit.

Per i prossimi passi: un `ActionContext()` di default rifiuta le azioni, ma non protegge `PcTools`:
gli script di prova usano `Recorder().ctx()` e il `FakeDesktop` del gate.

## Architettura e decisioni di fondo

```
voce → wake word → VAD → Whisper → router locale (0 $)
                                 → Haiku 5.5 con tool PC e web
                                 → Opus 5.5 senza tool, solo dopo il sì di JenJo
GPU sotto carico: voce Kokoro (CPU) e Whisper più leggero
```

Queste scelte sono chiuse: si riaprono solo su richiesta di JenJo o con un dato nuovo dai log.

| Decisione | Perché |
|---|---|
| Tool calling con SDK `anthropic` diretto, dopo il router locale | Nessun sovrapprezzo sui tool; 17-18 tool stanno sotto la fascia 30-50 in cui la documentazione segnala il degrado. LangChain è stata scartata: un solo provider e il suo strato può modificare la cronologia |
| MCP non per i tool interni; da riprendere per i servizi esterni (fase 7) | Un solo client nello stesso processo: aggiungerebbe un processo e un protocollo. Ragionamento, non un fatto verificato dalla ricerca |
| Terminator solo come fonte di idee | Binding Python parziali e fermi a ottobre 2025, ultimo codice di aprile 2026, server MCP con `npx @latest`, segnalazione di sicurezza #479 senza risposte |
| Nessun LLM locale, tool search o programmatic tool calling | Il primo aggiunge carico sulla GPU contesa; gli altri due rendono poco con 1-2 chiamate per richiesta |
| Budget 2,58 $/giorno, al massimo 80 $ al mese | Alzato il 05/11/2026, arrivati i crediti Max: 100 $ al mese, JenJo ne permette 80 per ora |
| I livelli di sicurezza li decide il codice | Vale per ogni tool nuovo. Una regola nel prompt è accettabile solo dove un errore non ha conseguenze di sicurezza |

## Passo 0: correzioni

**Metriche ereditate.** Le righe di false attivazioni e rumore copiano tool, modello e costo della
richiesta precedente (+20% sul costo mostrato: 8 righe, 0,0018 $ su 0,0089 $): `app.py:785-787`
fonde `responder.last_route` anche quando il router non è mai stato chiamato. La fusione deve
avvenire solo se la richiesta è arrivata al router. **Attenzione:** azzerare `last_route` a inizio
richiesta rompe `Responder.asked_question` (responder.py:194), che `RequestFilter.is_request`
(intake.py:245) legge per sapere se Claude aveva fatto una domanda al turno prima.

**Note d'azione false.** Il 01/11 alle 23:32:42 Haiku ha scritto `[Claudio ran: web_search ok]`
senza aver chiamato nessun tool, nonostante il divieto in prompts.py:91. Il codice toglie dal testo
del modello `ACTIONS_NOTE` e `CUT_NOTE` (claude.py:80, 85) prima della voce e di `_remember`
(claude.py:587). Il testo arriva a pezzi: il filtro deve reggere una nota spezzata tra due chunk.
`_read_outside` (claude.py:580) legge quelle note, quindi devono poter nascere solo dal codice.

**Stop muto e locale.** "No, Claudio, non parlare. Claudio, stop." è finito a Claude, che ha
risposto "zitto come una tomba" (01/11 17:00). Il comando `stop` (commands.toml:232) è `exact`:
aggiungi "non parlare" e fai riconoscere uno stop dentro una frase rivolta a Claudio.

**Comando ripetuto vale sì.** Il 02/11 alle 16:23 "Claudio, spegni" detto durante la conferma di
`claudio_command(off)` ha dato `UNCLEAR` e ha annullato l'azione: 4 frasi e ~25 s per spegnere. In
`_confirm` (claude.py:414) ripetere lo stesso comando di sistema in attesa vale come sì. "Spegni"
non è tra le frasi di `off`: confrontalo con le frasi del comando in attesa, non con la lista
generale.

**Niente testo da spento.** Da spento si salva il testo solo se è il comando di riaccensione. Oggi
la frase finisce in `metrics.jsonl`, `conversazione.md` e `console.log` (responder.py:372), e viene
salvato anche l'audio `*_off.wav`: toglilo insieme al testo, per lo stesso motivo.

**Meno rumore.** Le righe `near_miss` (756 su 830) restano in `metrics.jsonl` ed escono da
`conversazione.md` (`MetricsWriter.write`, metrics.py:91).

**GPU nelle metriche.** Ogni riga registra carico GPU e VRAM libera, così il Passo 3 avrà soglie
basate sui giochi veri di JenJo. `read_gpu()` (gpu.py:35) lancia un processo da ~0,1 s: leggilo in
background, fuori dal percorso della risposta.

**Budget.** `daily_budget` in `[cost]` diventa l'equivalente di 0,30 € al cambio del giorno (il
contatore lavora in USD).

**Fatto quando:**
- le frasi vere del 01/11 e 02/11 citate sopra danno in un test l'esito atteso, con zero chiamate
  API per lo stop;
- una riga `false_wake` non ha `route`, `tools` né `cost`, e `asked_question` funziona ancora;
- una nota falsa, anche spezzata in due chunk, non arriva né alla voce né alla cronologia;
- da spento nessun testo né audio resta su disco, salvo la riaccensione;
- le righe delle metriche hanno i campi GPU e la latenza non peggiora (`benchmark_latency.py --log`).

## Passo 1: web vero con Haiku

Il problema più sentito: il 01/11 JenJo ha chiesto tre volte "cercamelo tu e dimmelo" e Claudio ha
aperto Google dicendo "la lista la devi leggere tu".

- **Conflitto di nome:** il tool client `web_search` (tools.py:204, 292, 604, 744; prompts.py:79-81)
  diventa `open_search_page`, e resta per "aprimi la ricerca su Google".
- **Tool server** in `_params` (claude.py:676), con ordine fisso perché fanno parte del prefisso in
  cache. Scegli la versione con la Models API (`capabilities.server_tools`) e confronta latenza e
  token tra base e dynamic filtering su 5 domande vere. Parti con `max_uses` 2 e con
  `max_content_tokens` basso per la lettura delle pagine.
- **Risposta del modello:** `_Leg.read` (claude.py:848) tiene solo `text` e `tool_use`. Con i tool
  server arrivano `server_tool_use` e `*_tool_result`, che vanno rimandati quando nello stesso turno
  c'è un tool client. Gestisci `pause_turn`. Un errore del tool server arriva con HTTP 200, come
  oggetto al posto della lista: controllalo.
- **Contatore:** `Usage.from_api` (pricing.py:78) legge solo i token. Aggiungi le ricerche da
  `usage.server_tool_use` (10 $ ogni 1000, prezzo in `config/pricing.toml`, mai nel codice).
- **Tetto (JenJo):** al 90% del budget giornaliero Claudio risponde senza cercare e lo dice. Togliere
  i tool riscrive la cache una volta: è accettabile.
- **Testo esterno:** web search e fetch contano come `OUTSIDE_TEXT` (tools.py:71 e claude.py:71). Non
  passano da `_run_calls`, quindi vanno segnati quando si legge la risposta e nella nota in
  cronologia. Con `url_sources` lascia solo `user_input` e `server_tool_results`, così Claudio non
  apre link letti da finestre o file.
- **Controllo degli indirizzi (JenJo):** se la richiesta o la conversazione ha letto testo esterno,
  `PcTools.level()` porta `open_website` al livello 2 quando l'indirizzo ha parametri, un frammento,
  un percorso lungo o un host sospetto (etichette lunghe), e dice a voce il dominio. Lo decide il
  codice. Gli indirizzi dettati da JenJo restano di livello 1. `level()` oggi è "fisso per tool":
  aggiorna la docstring.
- **Fonti:** a voce "secondo <sito>", in chat i link. La documentazione richiede le citazioni quando
  l'output arriva all'utente.

**Fatto quando:**
- la scena del 01/11 ("dove trovo la lama spettrale scarlatta?") ottiene una risposta a voce con il
  nome del sito, in una prova vera;
- con il client finto: ricerche conteggiate, stop al 90%, errore del tool server gestito,
  `pause_turn` gestito, blocchi server rimandati in un turno misto;
- un test mostra `open_website("https://sito/?d=...")` al livello 2 dopo testo esterno e
  `open_website("youtube.com")` al livello 1;
- nel report: costo reale per risposta e latenza fino alla prima parola.

## Passo 2: risorse, cronometro, Opus, schermo

- **`system_status` (livello 1):** CPU, RAM e GPU di Claudio e del PC, con `psutil` e `read_gpu()`.
  La VRAM per processo su Windows può non essere disponibile: verificalo e, se manca, riporta il
  totale. Il 01/11 alle 23:28 questa domanda ha prodotto un PowerShell di livello 3 che misurava il
  PC e non Claudio.
- **Cronometro locale, 0 $:** "stop" è già il comando per zittire Claudio, quindi deve fermare il
  cronometro solo quando questo sta girando. Usa l'istante della wake word e di' a JenJo che
  l'errore è di circa 1 s.
- **Opus al posto di Sonnet, senza tool, dopo il sì:**
  - `smart_model = "claude-opus-5-5"`, `effort` esplicito (`low`, poi misura). `_smart_params`
    (claude.py:698) non invia `thinking`, quindi va bene. Il ragionamento conta in `max_tokens`:
    alza `smart_max_tokens` (oggi 1024) e controlla `timeout_s = 10`.
  - Fallback: `fallbacks: "default"` con la beta `server-side-fallback-2026-07-01`, che richiede
    `client.beta.messages.stream` (oggi `client.messages.stream`, claude.py:727). Il rifiuto è già
    gestito a claude.py:760.
  - Costo: ~0,03 $ a chiamata, perché la cache non si condivide tra modelli. Sonnet è stato usato
    una volta in 13 giorni.
  - Nomi: "Sonnet" compare in claude.py, prompts.py (`ASK_SONNET_TOOL`), i18n.py e nella
    configurazione, anche nelle frasi dette a voce.
  - Aggiorna il `CLAUDE.md` del progetto (Sonnet → Opus), mostrando il diff.
- **Schermo:** il 01/11 Haiku ha contato 6 armi invece di 5 e ha detto "la scritta è piccola". Alza
  `screenshot_side` (es. 1568, verifica il limite del modello). Se Haiku non riesce a leggere propone
  Opus e, dopo il sì, gli passa l'immagine: oggi `_offer` conserva solo i messaggi iniziali, senza
  l'immagine (claude.py:348). Costo ~0,04 $.

**Fatto quando:**
- "quanta memoria stai usando?" risponde senza conferma con i dati di Claudio;
- il cronometro funziona a voce e "stop" zittisce Claudio quando il cronometro è fermo;
- una domanda con Opus risponde senza errori 400 e il costo è registrato;
- una schermata di gioco vera è letta meglio di prima, con il confronto riportato a JenJo.

## Passo 3: modalità gioco, frasi non rivolte a Claudio, avvio a freddo

**Modalità gioco (JenJo: dal carico reale, nessuna lista di giochi).** Dai dati del Passo 0 fissa le
soglie di carico e VRAM, con una tolleranza nel tempo per evitare il passaggio continuo tra voci. Con
LoL, TFT e Brawlhalla Jack deve restare se la GPU è libera: verificalo su righe vere.
- Sopra soglia: `JackSynth.close()` (jack.py:267) e parla Kokoro, già sulla CPU (kokoro.py:116).
- Sotto soglia: `start_in_background` (jack.py:227); Kokoro parla finché Jack non è pronto.
- Whisper più leggero: scegli con `scripts/stt_bench.py` durante un gioco pesante vero. Il cambio
  non deve interrompere una richiesta in corso. `large-v3` era stato scelto per la precisione: riporta
  a JenJo sia i tempi sia gli errori.

**Frasi non rivolte a Claudio (JenJo: le decide Haiku).** Il 01/11 alle 16:59 una wake a 0,97 su "vieni
via dalla torre" e poi "no, ma non volevo cedere" hanno ottenuto risposte. La voce era forte (−19 e
−21 dBFS), quindi il volume non aiuta. `only_addressed` provato sulle frasi dei log blocca 27 frasi su
81, e in gioco 3 richieste vere su 4: per questo non è la prima scelta.
1. Regola nel prompt e marcatore riservato: se la frase non è rivolta a Claudio, nessuna risposta.
   Il filtro in `_stream` trattiene l'inizio del testo finché il marcatore non è escluso, non parla e
   non salva in cronologia.
2. **Prima di attivarla**, misura su ~80 frasi dei log (~0,03 $) etichettate insieme a JenJo, che
   approva anche la soglia di errore accettabile.
3. Se il test va male: con una finestra a schermo intero, `only_addressed` più la conferma del nome con
   Whisper su ogni wake (`confirm_below`, app.py:465).

**Avvio a freddo.** Prima si misura la causa, poi si corregge solo se è chiara. A freddo: 87-123 s,
whisper 86-117 s, `claude` 34-63 s, e Jack non pronto entro 90 s in 4 avvii su 6. A caldo: 15-72 s,
whisper 4-10 s. Ipotesi non verificata: antivirus o disco freddo sulla `.venv`.

**Fatto quando:**
- in un gioco pesante vero M1 è misurato prima e dopo, e il risultato è riportato (non si sa ancora
  quanto si recupera);
- con un gioco leggero Jack resta attivo; uscendo dal gioco Jack torna da solo;
- il test sulle frasi è riportato con precisione e richieste vere perse, e la modalità scelta è
  approvata da JenJo;
- la causa dell'avvio a freddo è misurata e scritta nella relazione.

## Passo 4: buffer di 30 s

JenJo vuole sapere cosa è appena successo in gioco. Claude non accetta video: una clip è una serie di
immagini.

- Cattura in background a 1 immagine al secondo, in RAM, mai su disco. Riusa `to_jpeg` (screen.py).
- Solo con Claudio acceso e con la finestra in primo piano che copre tutto il suo monitor, finestra
  senza bordi compresa: confronta il rettangolo della finestra con quello del monitor (una finestra
  ingrandita non lo copre, provato il 04/11). Cattura quel monitor e salta le finestre di
  `private_windows`.
- A "cosa è successo?" circa 10 immagini a Haiku (~0,001 $). Con la richiesta di ragionarci, Opus dopo
  il sì.
- `ImageGrab` (windows.py:245) può dare immagini nere a schermo intero esclusivo. Una libreria nuova
  per la cattura richiede `supermatt:dependency-vetting` e l'approvazione di JenJo.

**Fatto quando:** impatto su CPU e FPS misurato in un gioco vero in finestra senza bordi, con immagini
non nere; memoria del buffer limitata da un test; una domanda vera ottiene una risposta su quello che è
successo; nessuna immagine su disco.

## Passo 5: clic per nome e livello 3 in chat

**Prima una prova** su VS Code, Esplora file e Chrome: le app Chromium possono esporre i controlli solo
dopo aver rilevato una tecnologia assistiva (fonte non verificata). Verifica anche se `invoke()`
funziona senza prendere il mouse, come fa Terminator.

**Tool di clic** per ruolo e nome con pywinauto, senza immagini.
- Livello 2 di base. Lista `[pc] free_click`, come `free_typing` (config.py:536; `prepare`,
  tools.py:450), **vuota all'inizio** (JenJo).
- Pulsanti tipo "Elimina" (elimina, cancella, rimuovi, delete, remove...): livello 3 in tutte le app
  (JenJo). La lista di parole non sarà mai completa: scrivilo nei rischi.
- Dopo aver letto il web la lista libera non vale: una pagina non deve poter cliccare per JenJo.
- Finestre di `private_windows` rifiutate. Con più pulsanti dello stesso nome, Claudio chiede quale.

**Livello 3 in chat** (JenJo): riepilogo e pulsanti Conferma/Annulla, endpoint locale con il token di
`chat/server.py`, codice monouso, scadenza di 60 s. Claudio dice "Ti ho chiesto conferma nella chat" e
apre la finestra se è chiusa (`chat/launcher.py`). "Confermo" a voce o scritto non basta più: è un
cambio voluto della regola di app.py:1499. L'attesa non deve bloccare l'ascolto, e una nuova richiesta
annulla l'azione. Aggiorna il `CLAUDE.md` del progetto, mostrando il diff. Il 01/11 alle 23:28 la
conferma a voce è scaduta dopo circa 5 s mentre JenJo giocava.

**Fatto quando:**
- "clicca Salva" funziona nelle app provate e l'esito delle altre è riportato;
- un test copre i livelli: base, lista libera, "Elimina", dopo il web, finestra privata;
- la conferma rifiuta token assente, origine esterna, codice già usato e codice scaduto;
- un `run_command` vero parte solo dopo il clic in chat.

## Passo 6: computer use, in fondo

Da iniziare solo dopo gli altri passi. Prima di partire JenJo decide i livelli: la proposta è una
conferma a voce per avviare il compito, poi clic liberi con stop su terminali e `private_windows`, al
massimo 25 passi, "basta" ferma tutto.

Ordine delle prove: risposte registrate e riprodotte, conteggio token, Haiku 5.5 su compiti fissi,
infine Opus su 3-5 compiti fissi con **stop automatico nel codice a 1 €** (JenJo può cambiarlo con i
crediti Max). Regole dell'API qui sotto, in **Fatti API**.

## Fatti API

Letti nella documentazione il 04/11/2026. Prima di scrivere codice API carica la skill `claude-api`,
che ha i modelli correnti e gli esempi. Qui ci sono solo i fatti che hanno guidato il piano.

- **Prezzi** per milione di token (input/output): Haiku 5.5 0,10/0,50 $ con lettura cache 0,01 $;
  Sonnet 5.5 2/10 $; Opus 5.5 4/20 $ con lettura cache 0,20 $. Scrittura cache: 1,25x a 5 minuti. La
  cache non si condivide tra modelli. Ricerca web 10 $ ogni 1000, lettura pagine solo token.
- **Web su Haiku 5.5:** ricerca e lettura sono disponibili su tutti i modelli; il dynamic filtering è
  indicato per Claude 4.6 e successivi senza nominare Haiku, quindi va verificato. Parametri:
  `max_uses`, `allowed_domains` o `blocked_domains`; per la lettura anche `max_content_tokens` e
  `url_sources`. Il fetch non legge i siti resi con JavaScript e non apre URL presenti solo nell'output
  del modello.
- **Opus 5.5:** `thinking: disabled` dà errore 400 (`claude.py:685-686` va usato solo per Haiku); si
  regola con `effort`, default `medium`. `tool_choice` `any` o `tool` danno 400, `auto` e `none` vanno
  bene. Nei giri dei tool i blocchi `thinking` vanno rimandati identici: `_Leg.read` oggi li perde. La
  cronologia già inviata va solo allungata, altrimenti i blocchi non valgono più.
- **Computer use:** su Opus 5.5 solo `{"type": "computer_toolset_20260801"}`, senza `name` né
  dimensioni. Disponibile anche su Haiku 5.5. Aggiunge ~4.500 token a richiesta (~410 in meno senza
  `zoom`), screenshot esclusi. L'azione è il `name` del blocco `tool_use`; ogni risultato deve ripetere
  `"toolset_name": "computer"`. Le coordinate sono nei pixel dello screenshot. Lo screenshot deve già
  rispettare i limiti dell'immagine. La documentazione raccomanda una VM isolata, e qui gira sul PC vero.
- **Crediti Max:** 100 $ al mese per API e Agent SDK, non per Claude Code. JenJo deve collegare
  un'organizzazione della Console (dopo 7 giorni per i nuovi abbonati, non modificabile da solo): deve
  essere quella della chiave in `.env`. Scadono a fine ciclo. Un workspace dedicato con un tetto di
  spesa protegge anche se il contatore locale sbaglia.

Stime, non misure: Opus su 10 passi di computer use 0,15-0,30 $, Haiku ~0,005 $ con affidabilità non
misurata; una giornata tipo ~0,15 $ (100 richieste Haiku, 5 risposte con ricerca, 1 Opus).

Fonti: [prezzi](https://platform.claude.com/docs/en/about-claude/pricing),
[ricerca web](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool),
[lettura web](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool),
[computer use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool),
[crediti](https://platform.claude.com/docs/en/about-claude/api-credits-for-subscribers).
La ricerca non ha verificato MCP, sicurezza né UI Automation contro screenshot: su questi punti ci sono
solo ragionamenti e fonti non votate.

## Dati di partenza dai log

Ultime 10 sessioni (31/10 23:17 → 02/11 17:13), con le righe ereditate escluse dalle mediane.

| Misura | Valore |
|---|---|
| M1 richieste a Claude | mediana 5,0 s, p90 9,6 s (obiettivo ~1 s) |
| Claude, prima parola | mediana 0,6 s |
| Trascrizione senza gioco / in gioco | 0,3 s / mediana 2,2 s, picco 11 s |
| Voce, prima parola su richieste a Claude | mediana 1,7 s, p90 5,1 s |
| GPU in gioco | 98-99% di carico, VRAM fino a 16,3 GB su 16,3 |
| VRAM di Claudio, senza gioco | ~4,6 GB (stima: 7177 MB con Jack pronto, ~2500 MB con Claudio chiuso; il README dice ~1,2 GB per Jack: da riallineare) |
| Spesa dal 20/10 al 02/11 | 234 chiamate, 0,065 $ (Sonnet: 1 chiamata) |
| Esiti delle attivazioni | 10 wake senza richiesta, 7 false attivazioni scartate |

## Rischi

- Il controllo sugli indirizzi è parziale: un percorso breve porta fuori pochi dati.
- La prompt injection non è risolta: i livelli nel codice la limitano. Sul confronto tra le difese la
  ricerca non ha dati verificati.
- Whisper più leggero farà più errori in gioco.
- Una frase vera può essere taciuta se Haiku sbaglia. Una frase al gioco può ancora ottenere risposta.
- Le soglie GPU e l'esito delle prove in gioco dipendono da sessioni vere di JenJo.

## Fuori da questo piano

- **Presenza (JenJo, 05/11; riferimento: bluey-by-riley):** volto animato, occhi e cursore che indica, didascalie e stati nel tray. Fase futura, dopo gli altri passi. Le regole d'azione di Bluey entrano invece nel Passo 0b.
- **Spostati nel Passo 0b:** dal Passo 3 il giudizio di Haiku sulle frasi non rivolte a Claudio e la misura dell'avvio a freddo.
- **Procedure ricordate** (idea di Terminator: compito riuscito ripetuto a 0 $, con il modello solo se un
  passo fallisce): da riprendere quando i log mostrano compiti ripetuti.
- **Commento continuo in gioco** (~0,05-0,10 $ l'ora, stima): dopo i crediti Max.
- **Take basse di Jack** (JenJo, 31/10; `piani/IT-1-merge-consegne.md`): lavoro a parte di IT-2, non legato
  a questi passi.
- Altri aperti in quel file: wake debole con frase di memoria, seconda opinione sulla frase detta sopra la
  risposta.
