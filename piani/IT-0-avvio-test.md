# IT-0 – Test manuale: "spegniti / accenditi" e avvio all'accesso

Cartella del progetto (dopo il merge del branch `it0/avvio` in `main`):
`C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant`

Prima di provare senza merge, il codice è nel worktree
`C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\claudio-it0-avvio` (lì però mancano `.venv` e `models\`:
prova dalla cartella principale dopo il merge).

## 0. Preparazione (una volta)

1. Apri **PowerShell** (Start → scrivi `PowerShell` → Invio). Non quello dentro Claude.
2. Vai nella cartella:
   `cd C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant`
3. Test automatici: `uv run pytest -q tests\unit` → deve finire con `passed` e nessun `failed`.
4. Controllo: `uv run claudio --check` → tutte le righe `OK` (le `WARN` di Jack vanno bene).

## 1. Spegniti / accenditi (voce)

1. Avvia: `uv run claudio` e aspetta "Di' "Hey Claudio" per iniziare".
2. Di' **"Claudio, che ore sono?"** → beep e risposta (CLAUDIO è acceso).
3. Di' **"Claudio, spegniti."** → risponde "Mi spengo. Per riaccendermi di' Claudio, accenditi."
4. Di' **"Claudio, apri Chrome."** → **nessun beep, nessuna risposta, Chrome non si apre.**
5. Di' **"Hey Claudio, che ore sono?"** → ancora silenzio.
6. Di' **"Claudio, svegliati."** → silenzio (da spento vale solo "accenditi").
7. Di' **"Claudio, accenditi."** (chiaro, vicino al microfono) → "Eccomi, sono acceso."
8. Di' **"Claudio, apri Chrome."** → Chrome si apre: tutto torna normale.
9. Nel log `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant\logs\claudio.log.jsonl`
   cerca le righe `Spento, ignoro` (passi 4-6): ci devono essere.
10. Se "accenditi" non viene sentito: ripeti più vicino. Se capita spesso, abbassa in
    `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant\config\claudio.toml`
    la riga `off_threshold_boost = 0.1` (per esempio `0.05`) e riavvia.

## 2. Chiuditi (invariato)

1. Con CLAUDIO acceso di' **"Claudio, chiuditi."** → "A presto." e il programma si chiude.
2. Funziona anche da spento? **No, per scelta**: da spento si sente solo "accenditi".
   Per chiuderlo da spento: Ctrl+C nella finestra di PowerShell.

## 3. Avvio all'accesso (solo dopo il tuo OK sulla card)

1. Prova a vuoto (non cambia niente):
   `powershell -ExecutionPolicy Bypass -File C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant\scripts\install_autostart.ps1 -DryRun`
   → mostra utente, `...\.venv\Scripts\pythonw.exe -m claudio`, cartella, "after 30 s".
2. Installa:
   `powershell -ExecutionPolicy Bypass -File C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant\scripts\install_autostart.ps1`
   → "Task 'Claudio' registered".
3. Controlla: Start → **Utilità di pianificazione** → Libreria → deve esserci **Claudio**.
4. Chiudi eventuali CLAUDIO aperti, poi **esci da Windows e rientra** (o riavvia).
5. Dopo ~30 s di' **"Claudio, che ore sono?"** → risponde. Nessuna finestra nera: è normale.
6. Se non risponde: apri l'ultimo file in
   `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant\logs\` e cerca `ERROR`;
   in Utilità di pianificazione guarda "Risultato ultima esecuzione" di Claudio.
7. Per fermarlo senza disinstallare: "Claudio, chiuditi." (oppure Gestione attività →
   `pythonw.exe` → Termina attività).
8. Disinstalla:
   `powershell -ExecutionPolicy Bypass -File C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant\scripts\install_autostart.ps1 -Uninstall`
   → "removed". Lanciarlo una seconda volta dice "nothing to remove" (va bene).

**Attenzione**: non avviare `uv run claudio` a mano mentre gira quello dell'avvio automatico:
due CLAUDIO sullo stesso microfono rispondono entrambi.
