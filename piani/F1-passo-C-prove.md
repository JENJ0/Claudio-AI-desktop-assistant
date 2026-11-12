# F1, Passo C: prove a mano

Le prove di JenJo per il Passo C (pulizia piccola). Ordine e stato stanno in
[prove-da-fare.md](prove-da-fare.md); il piano dei task in [F1-passo-C-task.md](F1-passo-C-task.md).

## Prima di ogni prova

Sul PC fisso, con il codice nuovo: se Claudio è acceso chiudilo («Claudio, chiuditi») e riavvialo.

```powershell
cd C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant
uv run claudio
```

Facoltativo: nel tuo `.env` la riga `CLAUDIO_TTS__EFFECT=none` non serve più. Claudio la ignora
senza errori, puoi toglierla.

## C1. Claudio dopo la pulizia (circa 5 minuti, meno di 1 centesimo)

**Perché:** il Passo C ha tolto il codice multilingua, gli effetti voce e il catalogo inglese dei
log, e ha cambiato il prompt di sistema. I test e il gate usano finti: qui si prova che il Claudio
vero parte e risponde in italiano. Se un passo non torna, fermati e scrivimi quale.

1. Con Claudio chiuso:

   ```powershell
   uv run claudio --check
   ```

   Atteso: nessun «Errore di configurazione», finisce senza errori. Non controlla più le voci per
   lingua (tolte): una voce Kokoro sbagliata in `config/claudio.toml` ora si vede solo all'avvio.
2. Avvia Claudio. Atteso: dice «Sono pronto.» e nel log di avvio non c'è nessun avviso su lingue,
   voci o effetti.
3. «Claudio, che ore sono?» Atteso: risposta locale, in italiano.
4. «Claudio, apri il blocco note.» Atteso: si apre.
5. «Claudio, quanto fa dodici per sette?» Atteso: risponde Claude, in italiano.
6. «Claudio, raccontami una storia lunga.» Mentre parla: «Claudio, stop.» Atteso: si ferma.
7. «Claudio, skip.» Atteso: non è più un comando (prima saltava la canzone). Va bene se risponde
   Claude, se dice che non lo sa fare o se ti chiede se intendevi «canzone successiva»; non deve
   saltare la canzone da solo.
8. Privacy dei log. Dal 18/11/2026 in poi (30 giorni dopo la prima riga, del 18/10), all'avvio
   Claudio toglie dai log le righe più vecchie di `retention_days`. Dopo un avvio:

   ```powershell
   Get-Content logs\claudio.log.jsonl -TotalCount 1
   Get-Content logs\metrics.jsonl -TotalCount 1
   ```

   Atteso: la data della prima riga di ciascun file non è più vecchia di 30 giorni; nel log di
   avvio ci sono «Log: N righe più vecchie di 30 giorni tolte da claudio.log.jsonl» e
   «Metriche: N righe vecchie tolte da metrics.jsonl». Prima del 18/11 non c'è niente da
   togliere: salta questo passo.

**Scrivimi:** C1, i passi che non tornano e, se l'hai fatto, cosa mostra il passo 8.
