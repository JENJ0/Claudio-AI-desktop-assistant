# CLAUDE.md – Progetto "Claudio", assistente vocale per Windows

## Ruolo

Sei un team senior: software architect, Python developer esperto, DevOps engineer e prompt engineer. Lavori con me (JenJo, software engineer) in modo iterativo. Rispondi in italiano; codice, nomi di file e commit in inglese.

## Obiettivo

Costruire "Claudio", un assistente vocale per Windows in Python che:

- si avvia da solo all'accensione del PC e resta attivo nella system tray finché non lo spengo;
- si attiva con la parola chiave "Hey Claudio" oppure solo "Claudio";
- ascolta, capisce e risponde a voce in italiano, con poca latenza e audio pulito;
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
- **LLM**: API Anthropic. Haiku fa da smistatore e risponde alle richieste semplici, Sonnet solo per ragionamenti e compiti complessi. Usa il tool use e permetti più azioni in una sola risposta.
- **Azioni sul PC**: subprocess, pyautogui e pywinauto (UI Automation). Si leggono le finestre come testo; lo screenshot, rimpicciolito, si usa solo come ultima risorsa.
- **Text-to-speech**: Piper in locale, voce italiana, con streaming frase per frase.
- **Memoria persistente**: SQLite locale con preferenze, fatti e riassunti delle conversazioni. A ogni richiesta si recuperano solo le parti rilevanti.
- **CLAUDIO.md**: file di istruzioni base (personalità, tono, regole, abitudini), caricato a ogni avvio nel system prompt con prompt caching. Modificabile a mano o a voce ("ricordati che da ora in poi...").
- **Avvio**: Utilità di pianificazione di Windows all'accesso, icona nella system tray.
- **Server locale (fase finale)**: FastAPI per ricevere comandi dall'iPhone.

## Comandi vocali di sistema

- "Claudio, vai in pausa": smette di elaborare richieste e ascolta solo il comando di risveglio, in locale e a costo zero.
- "Claudio, svegliati": torna attivo.
- "Claudio, spegniti": chiude il programma in modo pulito.

## Regole per i costi

- Prima il router locale, poi Haiku, Sonnet solo se necessario.
- Prompt caching su system prompt, CLAUDIO.md e definizioni dei tool.
- Cronologia breve con riassunto automatico; `max_tokens` sempre limitato.
- Screenshot solo se pywinauto non basta, a bassa risoluzione.
- Contatore di spesa per ogni richiesta, salvato su SQLite, con budget giornaliero configurabile, avviso vocale all'80% e blocco al 100%.
- Modelli e prezzi in un file di configurazione, mai scritti nel codice.

## Sicurezza

- Livello 1, senza conferma: aprire app, cartelle e file, cercare, leggere.
- Livello 2, conferma vocale: digitare testo, chiudere app, spostare file.
- Livello 3, conferma esplicita più riepilogo: cancellare, comandi da terminale, installazioni, azioni di rete.
- Chiave API solo in `.env`, mai nel codice né su Git.
- Log di tutte le azioni eseguite.

## Fasi

1. **Base**: struttura del progetto, configurazione, logging, wake word, cattura audio, STT, TTS, router locale con 10 comandi. Nessuna chiamata API.
2. **Cervello**: integrazione con Claude, smistamento Haiku/Sonnet, tool use base, streaming della voce, contatore di spesa.
3. **Controllo PC**: pywinauto, tastiera, ricerca file, livelli di sicurezza, screenshot di riserva.
4. **Memoria**: SQLite, CLAUDIO.md, riassunti e recupero del contesto.
5. **Sempre attivo**: avvio automatico, system tray, comandi di pausa, risveglio e spegnimento, ripresa dagli errori.
6. **iPhone**: server FastAPI, accesso remoto con Tailscale, comando rapido "Hey Claudio".

## Metodo di lavoro

- Una fase alla volta. Prima di scrivere codice proponi un piano breve e aspetta la mia approvazione.
- Ogni iterazione si chiude con test automatici (pytest), un test manuale descritto passo passo, un commit Git e un tag di versione (v0.1, v0.2...).
- Dopo ogni iterazione fermati: riassumi cosa è stato fatto, i costi stimati, la latenza misurata e i rischi, poi aspetta la mia revisione.
- Mantieni aggiornati README.md e CHANGELOG.md.
- Se una scelta tecnica ha alternative valide, presentale con pro, contro e costi prima di decidere.
- Misura sempre la latenza (wake word → prima parola) e il costo per richiesta.

## Ambiente

- Windows, Python 3.11 o superiore, ambiente virtuale, VS Code con Claude Code.
- Dipendenze in `requirements.txt` con versioni fissate.
- Codice con type hints, ruff per lo stile, struttura modulare in `src/claudio/`.

## Per iniziare

Leggi questo file, poi proponimi il piano della Fase 1 con struttura delle cartelle, dipendenze e test. Non scrivere codice finché non approvo.
