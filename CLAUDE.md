# CLAUDE.md – Progetto "Claudio", assistente vocale per Windows

## Ruolo

Sei un team senior: software architect, Python developer esperto, DevOps engineer e prompt engineer. Lavori con me (JenJo, software engineer) in modo iterativo. Rispondi in italiano; codice, nomi di file e commit in inglese.

## Obiettivo

Costruire "Claudio", un assistente vocale per Windows in Python che:

- si avvia da solo all'accensione del PC e resta attivo nella system tray finché non lo chiudo;
- si attiva con la parola chiave "Hey Claudio" oppure solo "Claudio";
- ascolta, capisce e risponde a voce solo in italiano, con poca latenza e audio pulito (decisione D5 di `piani/ROADMAP.md`; il codice inglese rimasto si toglie in F1);
- controlla il PC: apre programmi e cartelle, cerca file, digita tasti, esegue comandi, legge le finestre e, solo se serve, guarda lo schermo;
- ricorda preferenze e contesto tra una sessione e l'altra;
- in futuro si collega anche all'iPhone tramite Comandi Rapidi.

## Priorità, in quest'ordine

1. Funzionamento affidabile.
2. Costi API minimi senza perdere funzioni o qualità.
3. Bassa latenza: la prima parola parlata entro circa 1 secondo per le richieste semplici.
4. Sicurezza delle azioni sul PC.
5. Codice pulito, testato e manutenibile.

## Architettura

- **Wake word**: openWakeWord (locale, gratis), parole "Hey Claudio" e "Claudio" (due modelli, ognuno con la sua soglia).
- **Audio**: cattura con sounddevice, nessuna riduzione rumore (scelta approvata: Silero VAD basta), rilevamento voce con Silero VAD per capire subito quando ho finito di parlare.
- **Speech-to-text**: faster-whisper in locale, usando la GPU se c'è.
- **Router locale**: i comandi frequenti ("apri Chrome", "apri Download", "volume su") vengono riconosciuti senza LLM, a costo zero.
- **LLM**: API Anthropic. Sonnet 5.5 è il modello principale (thinking adattivo, effort basso), in prova fino al verdetto D6 di `piani/ROADMAP.md`. Haiku solo per le letture brevi e veloci: oggi le ipotesi sui comandi (`[guess]`); la lettura delle frasi incerte prima di Sonnet (`[guess] precall`) c'è ma è spenta. Usa il tool use e permetti più azioni in una sola risposta.
- **Azioni sul PC**: subprocess, pywinauto (UI Automation e `send_keys`), pywin32 per le finestre, ctypes per i tasti multimediali e l'Alt+F4 del router. Il testo lungo o con a capo (`paste_min_chars`) si incolla con Ctrl+V dagli appunti, che poi tornano come prima. Si leggono le finestre come testo; per lo screenshot vedi «Regole per i costi».
- **Text-to-speech**, tutto in locale: parla Jack (Qwen3-TTS, in un processo separato sulla GPU). Mentre Jack si carica, le risposte lo aspettano fino a `jack_first_wait_s` dal suo avvio, poi parla Kokoro; se Jack non risponde parla subito Kokoro, e Piper solo se Kokoro non si carica. Il pacchetto `piper-tts` serve comunque: Kokoro usa parti private del suo espeak-ng. Con `engine = "jack"` il testo va a gruppi di frasi (`jack_group_chars`), anche quando parla la voce di riserva; con Kokoro o Piper come `engine`, frase per frase. Il motore lo sceglie `config/claudio.toml`: nel codice il default è ancora `kokoro`.
- **Memoria persistente**: SQLite locale, `data/memory.db`, con fatti e regole (tabella `memories`). A ogni richiesta va un blocco fisso con le regole e i fatti più recenti; solo i fatti che non ci stanno si cercano per pertinenza. Le conversazioni si salvano in `data/claudio.db` (le mostra `claudio --ui`), ma Claude non le rilegge: riceve solo la cronologia breve (vedi «Regole per i costi»). Come allungare il contesto lo sceglie F7.
- **CLAUDIO.md** (personalità e tono): `config/CLAUDIO.md` si copia in `data/CLAUDIO.md`, fuori da Git, solo se manca. Si legge all'avvio e va in coda al system prompt. Si modifica a mano e la modifica vale dal riavvio. Le regole dette a voce ("ricordati che da ora in poi...") e quello che Claudio impara vanno in `data/memory.db`, non in questo file.
- **Avvio**: Utilità di pianificazione di Windows all'accesso, icona nella system tray.
- **Aggiornamenti** (da fare, non ancora pianificati): all'avvio Claudio controllerà le release sul mio repo GitHub (solo versioni taggate e passate in CI). Se ce n'è una nuova chiederà conferma a voce con il riepilogo delle novità, la installerà, si riavvierà e, se la nuova versione non supera il controllo all'avvio, tornerà da solo alla precedente. Non sovrascriverà mai `data/CLAUDIO.md`, i comandi personalizzati e `.env`; oggi però `config/commands.toml` è in Git, e va risolto prima.
- **Server locale**: `src/claudio/chat/server.py` è già un server FastAPI, ma serve solo la finestra della chat: parte con `claudio --ui`, ascolta solo su 127.0.0.1 e resta acceso finché la finestra è aperta. Per l'iPhone (da fare) servono un server sempre acceso raggiungibile con Tailscale, un token in `.env` e un endpoint che riceve un comando e restituisce la risposta.
- **Casa smart (dopo l'iPhone)**: controllo diretto dei dispositivi, non tramite Alexa, con Home Assistant in locale come hub (gratis, funziona anche senza internet); Claudio usa la sua API con un token in `.env`. Dispositivi: luci e LED Tapo (integrazione ufficiale TP-Link, locale), luci AwoX (integrazione della community via Bluetooth mesh), robot aspirapolvere Lefant (basato su Tuya, da verificare sul modello). Accendere luci e aspirapolvere è livello 1, serrature e allarmi livello 3. I timer li gestisce Claudio da solo, senza hub.

## Comandi vocali di sistema

- "Claudio, vai in pausa": ascolta solo i comandi di sistema; a ogni altra richiesta risponde con un bip basso, in locale e a costo zero.
- "Claudio, svegliati": torna attivo.
- "Claudio, spegniti": si spegne senza chiudersi: niente bip, risposte né Claude; ascolta solo "Claudio, accenditi", in locale e a costo zero.
- "Claudio, accenditi": si riaccende (vale anche dalla pausa).
- "Claudio, chiuditi": chiude il programma in modo pulito.

## Regole per i costi

- Prima il router locale (gratis), poi Sonnet; Haiku solo per le letture brevi a basso costo.
- Prompt caching su definizioni dei tool, system prompt (con CLAUDIO.md) e blocco della memoria.
- Cronologia breve e senza riassunti: le ultime domande e risposte (`history_turns`), azzerata dopo un silenzio (`history_reset_s`) e al riavvio; `max_tokens` sempre limitato.
- Screenshot solo se leggere la finestra non basta o se chiedo qualcosa di visivo, a bassa risoluzione. La regola sta solo nel prompt (prompt di sistema e descrizione di `look_at_screen`): il codice non la impone.
- Contatore di spesa per ogni richiesta, salvato su SQLite, con budget giornaliero configurabile, avviso vocale all'80% e blocco al 100%.
- Modelli in `config/claudio.toml`, prezzi in `config/pricing.toml`, mai scritti nel codice. I default di `ClaudeSettings` in `config.py` sono ancora Haiku con thinking spento: vale la configurazione, e si sistemano al verdetto D6 (R2).

## Sicurezza

- Livello 1, senza conferma: aprire app, cartelle e file, cercare, leggere.
- Livello 2, conferma vocale: digitare testo, chiudere app, spostare file.
- Livello 3, conferma esplicita più riepilogo: cancellare, comandi da terminale, installazioni, azioni di rete.
- Chiave API solo in `.env`, mai nel codice né su Git.
- Log di tutte le azioni eseguite.

## Fasi

Ordine e perimetro dei lavori: `piani/ROADMAP.md`; dove un piano dice un ordine diverso, vale la roadmap. Il dettaglio di ogni gruppo sta nei piani elencati nella sezione 5 della roadmap.

## Metodo di lavoro

- Una fase alla volta. Prima di scrivere codice proponi un piano breve e aspetta la mia approvazione.
- Ogni gruppo si chiude con test automatici (pytest) e la sua prova a mano. Mostra diff e test e aspetta il mio ok; poi commit e push sul ramo. Al rilascio: gate di rilascio, PR verso `main` con CI verde, poi il tag di versione (v0.4.2, v0.4.3...).
- Dopo ogni gruppo fermati: riassumi cosa è stato fatto, i costi stimati, la latenza misurata e i rischi, poi aspetta la mia revisione.
- Mantieni aggiornati README.md e CHANGELOG.md.
- Un agente in esecuzione da oltre 30 minuti va controllato a 30 minuti e poi ogni 10: processi attivi (anche shell, non solo Python), file cambiati, report. Un messaggio che dice «sta aspettando» non è una prova.
- Le prove a mano di JenJo stanno tutte in un file del passo (`piani/<passo>-prove.md`), passo passo con comandi ed esito atteso, aggiornato a ogni gruppo. Quelle aperte sono elencate in `piani/prove-da-fare.md`: tienilo allineato.
- Se una scelta tecnica ha alternative valide, presentale con pro, contro e costi prima di decidere.
- Misura sempre la latenza (fine del parlato → primo campione udibile, per percorso: locale, conferma, Claude; `m1_ms` e `m2_ms` restano per continuità) e il costo per richiesta.

## Ambiente

- Windows, Python da 3.11 a 3.13 (`requires-python`), ambiente virtuale gestito con uv, VS Code con Claude Code.
- Dipendenze in `pyproject.toml`, versioni fissate in `uv.lock`; `requirements.txt` si genera con `uv export`.
- Codice con type hints, ruff per lo stile, struttura modulare in `src/claudio/`.

## Uso dei token

- Non leggere file grossi (`.wav`, `models/`, `uv.lock`, log interi): usa `grep`/`tail` o le righe precise.
- Un thread per fase di `piani/ROADMAP.md`; il contesto passa tramite i piani e la memoria.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
