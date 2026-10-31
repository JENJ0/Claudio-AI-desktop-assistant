# IT-1 – Consegne: sempre attivo (istanza unica, riavvio dopo un crash, system tray, check)

Ramo locale `it1/sempreattivo` (worktree `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\claudio-it1-sempreattivo`),
partito da `main` = `ba3872f` (tag `v0.4.0`), non pushato. Sezioni 6.1, 6.2, 6.3, 6.6 e 6.7 di
`piani/IT-1-consegne.md`, più la decisione 10.

## Fatto

| Cosa | Dove |
|---|---|
| **Decisione 10**: versione 0.4.0 (pyproject, `__init__`, uv.lock, CHANGELOG) e tag annotato `v0.4.0` su `main` locale | `main`: `ba3872f` |
| **6.7** README (riga di stato, en/it) e CLAUDE.md: spegniti = spento, accenditi = riacceso, chiuditi = chiude | `main`: `3d5e9db` |
| **6.1** Istanza unica: mutex di Windows `Local\CLAUDIO-voice-assistant`; la seconda copia scrive "CLAUDIO è già acceso" e esce con codice 3 prima di caricare i modelli e senza cartella di sessione | `src/claudio/single_instance.py`, `__main__.py` |
| **6.2** Riavvio dopo un crash: `python -m claudio.supervisor` riavvia CLAUDIO dopo 5, 30, 60 s; si arrende al 4° crash in 10 minuti; si ferma con 0 (chiuso apposta), 2 (configurazione) e 3 (già acceso). Il task all'accesso ora esegue il supervisore | `src/claudio/supervisor.py`, `autostart.py`, `scripts/install_autostart.ps1` |
| **6.3** Icona nella system tray: blu in avvio, verde acceso, arancione in pausa, grigio spento (segue anche la voce); clic sinistro = chat; menu: Spegni/Accendi, Pausa, Apri la chat, Chiudi CLAUDIO. `pystray==0.19.5` fissato (ultima versione, nessun avviso OSV), `[startup] tray = true` | `src/claudio/tray.py`, `system.py` (`subscribe`), `config.py`, `config/claudio.toml`, `pyproject.toml`, `uv.lock` |
| **6.6** `claudio --check` controlla il modello Whisper che verrà caricato (large-v3 con la GPU, small senza o con `device = "cpu"`), offline, file per file; l'altro modello è solo un WARN | `src/claudio/checks.py` |
| `requirements.txt` rigenerato: mancavano pillow, pywin32, pywinauto, six | `requirements.txt` |
| Test: 66 nuovi (mutex anche tra processi e dopo un crash, supervisore con processi veri, menu e stati della tray, icona vera di pystray, modello Whisper) | `tests/unit/test_single_instance.py`, `test_supervisor.py`, `test_tray.py`, `test_checks.py`, `test_autostart.py` |

Risultati: `pytest` 1833 passed, 4 skipped (prima 1753), anche dopo le correzioni della
revisione; test lenti nuovi passati (icona vera); ruff e mypy puliti. Anche i commit intermedi
del ramo passano ruff e i loro test.

## Revisione multi-agente (31/10)

3 revisori indipendenti (fili e processi, check e integrazione, test e documenti), ognuno con
esperimenti propri: 11 segnalazioni, 8 distinte, tutte verificate da me e corrette nel commit
`fix: review findings` (lo scettico finale è stato fermato per chiudere in tempo).

| Problema | Correzione |
|---|---|
| Un errore imprevisto sotto pythonw si perdeva (nessuna console): il supervisore diceva "il motivo è nel log" ma non c'era | `main()` lo scrive nel log con il traceback, codice 1 |
| "Chiudi CLAUDIO" e "chiuditi" non chiudevano se il microfono non manda audio (staccato) | `close_when_idle`: 1 s dopo che CLAUDIO è fermo, la corsa viene annullata |
| Con un CLAUDIO avviato come amministratore, la seconda copia andava in crash (codice 1) e veniva riavviata | `ERROR_ACCESS_DENIED` = già acceso, codice 3 |
| Ctrl+C non interrompeva l'attesa del supervisore su Windows; un Ctrl+C all'avvio faceva ripartire CLAUDIO | attesa a fette di 0,5 s; dopo Ctrl+C il supervisore si ferma sempre |
| `--check` con `device = "cuda"` senza GPU diceva OK (Whisper non partirebbe) | ora è ERR |
| `--check` si interrompeva con un traceback se faster_whisper non si importa | la riga è ERR e il resto del rapporto continua |
| Il test dell'icona vera non poteva fallire (controllava una variabile azzerata sempre) | controlla che il thread di pystray sia davvero finito; avviso nel log se non si ferma |

## Verificato sul PC (30/10)

1. **Il "riavvia in caso di errore" di Windows non funziona per i crash**: task di prova che esce
   con codice 1, 3 riavvii a 1 minuto: in 3 minuti non è mai ripartito. Da qui il supervisore.
2. **Il `.venv` funziona fuori dalla app Claude**: lo stesso task (fuori dal contenitore MSIX) ha
   trovato il Python di base (`base_exists=True`).
3. **Prova completa dal task** (Utilità di pianificazione, impostazioni solo CPU, senza Claude):
   modelli caricati in 3,8 s, icona creata; una seconda copia lanciata dalla app Claude esce con
   codice 3 in 0,4 s (il mutex vale anche tra dentro e fuori il contenitore); CLAUDIO ucciso a
   forza → "lo riavvio tra 5 s" → di nuovo pronto; "chiuditi" scritto nella chat → codice 0 → il
   supervisore si ferma, nessun processo rimasto.
4. `claudio --check` sui modelli veri: "OK modello Whisper large-v3 (3.1 GB)"; con
   `CLAUDIO_STT__DEVICE=cpu` "small (0.5 GB)"; con `medium` (non scaricato) ERR con il comando.

I task di prova (`ClaudioProbeRestart`, `ClaudioProbeSupervisor`) sono stati rimossi: oggi non
c'è nessun task "Claudio".

## Test manuale per JenJo (dopo il merge in main)

1. Apri PowerShell (non la app Claude) e vai nel progetto:
   `cd C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant`
2. Installa la dipendenza nuova: `uv sync` (aggiunge solo `pystray`).
3. `uv run claudio --check`: deve esserci la riga "OK modello Whisper large-v3 (3.1 GB)".
4. Registra l'avvio all'accesso:
   `powershell -ExecutionPolicy Bypass -File C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant\scripts\install_autostart.ps1`
5. Esci da Windows e rientra (o riavvia il PC). Dopo circa 30 s, vicino all'orologio (anche
   sotto la freccia ^ delle icone nascoste) compare l'icona: blu, poi verde quando è pronto.
6. Tasto destro sull'icona: prova "Spegni" (grigia), "Accendi" (verde), "Pausa" (arancione).
   Di' "Claudio, spegniti": l'icona diventa grigia anche così. Clic sinistro: si apre la chat.
7. In PowerShell `uv run claudio`: deve scrivere "CLAUDIO è già acceso: questa copia si chiude".
8. (facoltativo) Gestione attività → Dettagli → termina il `pythonw.exe` che usa più memoria (è
   CLAUDIO; gli altri tre sono il supervisore e i due lanciatori del `.venv`): dopo circa 30 s
   CLAUDIO è di nuovo pronto (in `logs\supervisor.log.jsonl` c'è "lo riavvio tra 5 s").
9. Tasto destro → "Chiudi CLAUDIO": l'icona sparisce e non riparte.
10. Per togliere l'avvio automatico: stesso comando del punto 4 con `-Uninstall` alla fine.

## Note e rischi

- **Icone fantasma**: se CLAUDIO viene ucciso a forza (Gestione attività, crash nel codice C)
  la sua icona resta finché non ci passi sopra il mouse. È un limite di Windows; con "chiuditi",
  il menu o un errore Python l'icona sparisce (hook `atexit`). Le prove di oggi possono averne
  lasciate due: spariscono passandoci sopra.
- "Termina" sul task nell'Utilità di pianificazione chiude solo il supervisore: CLAUDIO resta
  acceso (il mutex impedisce comunque un doppione). Per chiuderlo: "chiuditi" o il menu.
- Il supervisore si arrende dopo 4 crash in 10 minuti (per non ricaricare i modelli all'infinito);
  il motivo è in `logs\claudio.log.jsonl`.
- Bug piccolo trovato, non corretto (fuori ambito): `metrics.py` `_rotate` scrive un avviso con
  traceback "Rotazione di metrics.jsonl non riuscita" quando `logs\metrics.jsonl` non esiste
  ancora (solo alla primissima richiesta su una cartella di log vuota). Correzione di una riga:
  uscire se il file non c'è.
- `uv.lock` passa alla revisione 5 (uv 0.13.1); se un altro thread IT-1 aggiunge dipendenze ci
  sarà un conflitto nel lock: dopo il merge basta `uv lock` e `uv export` (comando in testa a
  `requirements.txt`).
- `main` ha due commit nuovi (`3d5e9db` docs, `ba3872f` versione 0.4.0 + tag): gli altri thread
  IT-1 partono da `62f2b79`, il merge li unisce senza problemi (conflitti possibili solo in testa
  al CHANGELOG e nella riga di stato del README).

## Decisioni per JenJo

1. Registrare il task all'accesso: lo fai tu (punto 4 del test, consigliato: così è certo che
   parte dal tuo terminale) o lo registro io appena il ramo è unito in `main`?
2. Dopo 4 crash in 10 minuti il supervisore si arrende: va bene, o preferisci che riprovi ogni
   5 minuti per sempre?
