# IT-1 – Test manuale della memoria (fase 4)

Durata: circa 10 minuti. Costo: meno di 0,01 $ (solo le domande che vanno a Claude).
Il database della memoria è `data\memory.db`: per ripartire da zero basta cancellarlo a
CLAUDIO chiuso (o dire "dimentica tutto" e poi "confermo").

## 0. Avvio (dopo il merge del ramo `it1/memoria` in `main`)

Il worktree `C:\Users\JenJo\Desktop\PROGETTI\claudio-it1-memoria` non ha `models\`, `.env` né il
tuo `data\`: prova dalla cartella principale dopo il merge. Nessuna dipendenza nuova.

1. Apri **PowerShell** (Start, scrivi `PowerShell`, Invio). Non quello dentro Claude.
2. `cd C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant`
3. Test automatici: `uv run pytest -q tests\unit` → deve finire con `passed` e nessun `failed`.
4. Avvia: `uv run claudio`. Nel log all'avvio deve comparire
   `Memoria: 0 ricordi in C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant\data\memory.db`.

## 1. Salvare (in locale, gratis)

| Dici | Risposta attesa |
|---|---|
| "Claudio, ricordati che mia sorella si chiama Anna" | "Me lo ricordo: mia sorella si chiama Anna." |
| "Claudio, ricordati che mia sorella si chiama Anna" (di nuovo) | "Lo sapevo già." |
| "Claudio, da ora in poi chiamami capo" | "D'accordo, da ora in poi: chiamami capo." |
| "Claudio, ricordati che la password del wifi è gatto" | "Questo non lo salvo: ..." |

Nel log: `route=memory`, nessuna riga `Claude haiku...` per queste frasi.

## 2. Claude li usa

| Dici | Atteso |
|---|---|
| "Claudio, come si chiama mia sorella?" | dice Anna e ti chiama capo |
| "Claudio, che regalo le faccio?" | risposta su tua sorella; nel log `cache ... letti` alto |
| "Claudio, segnatelo: il mio colore preferito è il verde" | Claude usa lo strumento `remember` e lo dice |

## 3. Ricordare e dimenticare

| Dici | Atteso |
|---|---|
| "Claudio, cosa ricordi?" | "Ricordo 3 cose. Le ultime: ..." (le regole lette con "da ora in poi") |
| "Claudio, cosa ricordi di mia sorella?" | "Mi hai detto: mia sorella si chiama Anna." |
| "Claudio, ricordati che mia sorella abita a Torino", poi "Claudio, dimentica mia sorella" | "Ne ricordo 2: ... Quale dimentico?" e resta in ascolto senza wake word |
| "nessuna delle due" (entro 45 s) | "Va bene, non dimentico niente." (nessun ricordo perso) |
| di nuovo "Claudio, dimentica mia sorella", poi "la prima" | "Dimenticato: ..." quella letta per prima |
| "Claudio, ricordati che il treno parte alle 8", poi subito "Claudio, dimenticalo" | "Dimenticato: il treno parte alle 8." ("dimenticalo" vale solo per il ricordo di cui avete appena parlato, entro un minuto) |
| "Claudio, dimentica l'ultimo ricordo" | dimentica l'ultimo salvato in questa sessione, anche dopo molto |
| "Claudio, dimentica Anna" con salvato solo "mio padre ha 60 anni" | "Dimentico questo: mio padre ha 60 anni?" (chiede, perché "Anna" e "anni" non sono la stessa parola); rispondi "no" |
| "Claudio, dimentica tutto" | "Cancello tutti i ricordi, per sempre. Per procedere dì: confermo." |
| "sì" | "Non ho capito: non lo faccio." (serve proprio "confermo") |
| "Claudio, dimentica tutto", poi "confermo" | "Fatto: ho dimenticato tutto." |

## 3b. Senza wake word (la TV nella finestra di conversazione)

Subito dopo una risposta di Claudio, entro 6 secondi e **senza** dire "Claudio", di':
"da ora in poi parla in rima". Atteso: "Da ora in poi: parla in rima. La tengo come regola?".
Rispondi "no": "Va bene, non la salvo." Una regola detta col nome si salva subito.

## 4. Dopo un riavvio

Chiudi ("Claudio, chiuditi"), riavvia come al passo 0, poi "Claudio, cosa ricordi?": i ricordi
salvati prima del riavvio ci sono ancora.

## Cosa annotare

- frasi di memoria capite male da Whisper (cosa hai detto e cosa ha scritto, dal log);
- se un comando normale è stato preso per un ricordo, o il contrario;
- la riga `Claude haiku-5-5: ... cache N letti / M scritti` delle domande al passo 2.
