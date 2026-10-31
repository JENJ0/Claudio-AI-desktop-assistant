# IT-0 – Consegne: "spegniti / accenditi" e avvio all'accesso

Branch locale `it0/avvio` (worktree `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\claudio-it0-avvio`),
non pushato. Test manuale: `piani\IT-0-avvio-test.md`.

## Fatto

| Cosa | Dove |
|---|---|
| "Claudio, spegniti" = spento: niente beep, risposte, Claude, second chance, retry/closed list di Whisper; resta solo la wake word | `src/claudio/system.py`, `src/claudio/app.py`, `src/claudio/router/responder.py` |
| Da spento la wake word è più severa: ogni soglia + `off_threshold_boost` (0.1, max 0.99) | `config/claudio.toml`, `src/claudio/config.py`, `src/claudio/wakeword/detector.py` (`strict`), `src/claudio/protocols.py` (`StrictWake`) |
| "Claudio, accenditi" (anche "riaccenditi", "torna ad ascoltare"...) riaccende ed esce anche dalla pausa | `config/commands.toml` (comandi `off`, `on`) |
| "Chiuditi" chiude come prima (+ "ora chiuditi", "puoi chiuderti", "esci dal programma") | `config/commands.toml` |
| Varianti di Whisper per suono, non per lista: il confronto fonetico ora ignora anche gli spazi ("spegni ti", "accendi ti"), oltre a doppie, y/i, ecc. già gestite | `src/claudio/router/matcher.py` |
| Avvio all'accesso **preparato, non registrato**: PowerShell + Python, idempotenti, `-DryRun`/`--dry-run` | `scripts/install_autostart.ps1`, `src/claudio/autostart.py` |
| Con `pythonw` (nessuna console) il log va solo su file | `src/claudio/logging_setup.py` |
| Test: 58 nuovi (spento/acceso, varianti, assistente completo silenzioso, soglie, task XML, dry run, log senza console) | `tests/unit/test_off_mode.py`, `tests/unit/test_autostart.py`, aggiornati `test_matcher.py`, `test_actions.py` |
| README (en/it) e CHANGELOG | `README.md`, `README.it.md`, `CHANGELOG.md` |

Risultati: `pytest tests` 1592 passed, 4 skipped (prima, solo unit: 1534 passed). ruff e mypy puliti.
Dry run di entrambi gli installer verificato su una cartella valida; nessun task registrato.

## NON fatto, in ordine di importanza

1. **Registrare il task** sul PC: serve il tuo OK (card nel thread), poi passo 3 del test.
2. **Icona nella system tray** (CLAUDE.md fase 5): non fatta. Serve una dipendenza nuova
   (`pystray` + Pillow) da verificare e fissare; oggi da avvio automatico non c'è nulla di
   visibile, si chiude con "chiuditi" o da Gestione attività.
3. **Istanza singola**: se avvii `uv run claudio` mentre gira quello automatico, due CLAUDIO
   usano lo stesso microfono. Serve un lock (mutex di Windows o file).
4. **CLAUDE.md non aggiornato**: dice ancora "spegniti: chiude il programma". Va cambiato in
   "spegniti: spento, accenditi: riacceso, chiuditi: chiude" – lo lascio a te perché è il tuo
   documento di specifiche.
5. **Non verificati sul PC reale** (nessun microfono né registrazione in questa sessione):
   - che `schtasks` accetti l'XML (il test controlla solo struttura e contenuto);
   - che "Riavvia in caso di errore" scatti anche quando CLAUDIO esce con codice ≠ 0
     (la documentazione Microsoft non è chiara su questo);
   - che il `.venv` funzioni fuori dalla app Claude (punta a un Python uv in `AppData`:
     se lo script dice "not found" o il task parte e muore, esegui `uv sync` nel tuo terminale);
   - la soglia 0.1 in più: da tarare con le tue prove ("accenditi" sentito sì/no).
6. **Da spento "chiuditi" non funziona** (per scelta: solo "accenditi"). Se lo vuoi, è una riga.
7. **Inglese cambiato**: "turn yourself off" / "switch yourself off" ora spengono invece di
   chiudere; "shut down" e "quit" chiudono.
8. Nessun tag di versione creato (va fatto dopo la tua revisione e il merge).
