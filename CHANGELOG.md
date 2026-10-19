# Changelog

## [Unreleased]

### Aggiunto
- Voce maschile Nicola (Kokoro-82M, ONNX, in locale) come voce predefinita; Piper resta come
  ripiego automatico se Kokoro non parte. I fonemi arrivano dall'espeak-ng già incluso in Piper,
  quindi niente da installare in più. `tts.length_scale` regola la velocità di entrambe le voci
  (predefinita 0.77, cioè +30%).
- Tono robotico (`tts.effect`): `ultron_estremo` predefinito (copia più grave, distorsione,
  modulazione ad anello ed eco metallica), più `estremo`, `ultron`, `jarvis` e `none`; circa 5 ms
  a frase.

- Livello del microfono automatico (`audio.auto_gain`, attivo): dopo ogni richiesta Claudio
  corregge il guadagno digitale verso picchi di -20 dBFS (+3 dB alla volta, -6 dB se satura),
  tra 0 e +30 dB, e lo salva per ogni microfono in `data/audio_level.json`. Il livello misura
  solo la voce, e ogni richiesta registra anche saturazione (`clipped`) e guadagno (`gain_db`).
- Senza `audio.output_device` Claudio parla dall'uscita predefinita di Windows del momento
  (letta con Core Audio a ogni frase): cambiarla con Claudio acceso non richiede più un riavvio.
- `benchmark_latency.py --replay` fa ripassare le richieste salvate in `logs/audio` per la
  pipeline vera (senza eseguire azioni), per confrontare sulla propria voce quanto audio prima
  della wake word dare a Whisper (`--lead-in`) e il beam (`--beam`). Sulle prime 22 registrazioni
  di JenJo togliere la wake word dall'audio peggiorava (10 comandi capiti invece di 13), perché
  il comando le è attaccato: resta quindi tutto, e il prompt di Whisper scrive "Hey Jarvis".

### Cambiato
- Microfono e uscita predefiniti passano da MME a WASAPI, che ha meno latenza.
- Whisper con beam 5: sulle 33 registrazioni di JenJo capisce 25 comandi invece di 19, con
  appena 2 ms in più sulla GPU (circa 106 ms).
- Wake word più sensibile: soglia di `hey_jarvis` a 0,22. Nell'ultima prova di JenJo i tentativi
  non riconosciuti arrivavano a 0,23-0,31 e lui li aveva dovuti ripetere.
- Il prompt di Whisper non contiene più "Claudio" finché la wake word è "hey jarvis": Whisper lo
  ricopiava nel testo ("E il Jarvis, Claudio, sono").

### Corretto
- La wake word detta in fretta e storpiata da Whisper all'inizio della frase ("E il jarviso che
  ore sono", "Ejci Arvis, …") ora viene tolta prima di cercare il comando.
- Whisper poteva entrare in un loop ("Ciao, ciao, ciao…" per 100 parole) sull'audio confuso: ora
  si ferma a 80 token e un testo così ripetitivo viene scartato.
- Kokoro e Piper nello stesso processo (test, `benchmark_tts.py`, Piper di ripiego) inizializzavano
  espeak-ng due volte: su Windows la seconda volta ne rovinava le tabelle dei fonemi, ed entrambe
  le voci dicevano parole senza senso (o il programma si chiudeva). Ora Kokoro usa lo stesso
  espeak-ng di Piper.
- Nicola strascicava le sillabe con "c" e "g" dolci ("calcolatrice", "dieci"): ora le affricate
  arrivano a Kokoro come un unico suono, come nel suo addestramento.

## [0.1.0] - 2026-10-18

Prima iterazione della Fase 1: scheletro funzionante, tutto locale, nessuna chiamata API.

### Aggiunto
- Pipeline completa: microfono (sounddevice, 16 kHz, blocchi da 80 ms) → openWakeWord (ONNX)
  → bip → Silero VAD v6 via onnxruntime (senza PyTorch) con pre-roll di 800 ms → faster-whisper
  → router locale → Piper con streaming frase per frase.
- Macchina a stati `IDLE → LISTENING → TRANSCRIBING → RESPONDING → SPEAKING`, timeout di 4 s
  senza voce e 8 s di frase massima; la wake word è ignorata mentre Claudio lavora o parla.
- Più wake word insieme, ognuna con la sua soglia (pronto per "Hey Claudio" + "Claudio");
  segnaposto `hey_jarvis` finché i modelli custom non sono addestrati.
- STT: `large-v3-turbo` float16 su CUDA, ripiego automatico su `small` int8 CPU; filtro delle
  allucinazioni note di Whisper ("Sottotitoli creati dalla comunità Amara.org", eco del prompt).
- Router con 10 comandi in `config/commands.toml` (ora e data separati), normalizzazione del testo
  e confronto esatto + fuzzy (rapidfuzz); numeri per il volume ("alza il volume di 10").
- Azioni di livello 1: Chrome, Download, Documenti (cartelle reali anche se spostate su OneDrive),
  Esplora file, Blocco note, Calcolatrice, tasti volume/muto via API Windows, ora e data a voce.
- Dispositivi audio per nome: a parità di nome vince WASAPI (latenza più bassa, con conversione
  di frequenza di Windows); il nome viene risolto di nuovo a ogni riapertura del microfono.
- Audio quasi muto non arriva a Whisper (evita "Buon appetito!" e simili sul silenzio).
- Configurazione TOML + variabili d'ambiente con messaggi d'errore chiari (pydantic-settings).
- Log su console e file JSON a rotazione; metriche M1/M2 per ogni richiesta in `logs/metrics.jsonl`.
- Diagnostica dell'ascolto: per ogni richiesta volume del microfono (`input_dbfs`), sicurezza
  massima del VAD (`vad_max`) e WAV di quello che Claudio ha sentito in `logs/audio/` (ultimi 50).
- Ascolto dopo il bip più tollerante (dal primo test dal vivo): soglia VAD 0,4, 64 ms di voce per
  iniziare, 800 ms di audio tenuto prima dell'inizio, 4 s per cominciare a parlare.
- Whisper parte senza contattare huggingface.co quando il modello è già scaricato.
- Wake word più sensibile: soglia di `hey_jarvis` a 0,35. Le attivazioni mancate (picco sopra
  0,15 senza superare la soglia) finiscono in `metrics.jsonl` come `near_miss`, con audio.
- Guadagno digitale del microfono (`audio.input_gain_db`) e avviso nel log quando la voce arriva
  troppo bassa; ogni richiesta registra anche il picco della voce (`speech_dbfs`).
- Comandi detti tutto d'un fiato ("hey jarvis che ore sono"): se la wake word scatta in
  ritardo, le parole dette subito prima vengono aggiunte alla frase e la wake word viene tolta
  dal testo trascritto. Se la frase contiene solo la wake word, Claudio ascolta di nuovo la
  richiesta senza perdere quello che hai detto nel frattempo.
- Script: `download_models.py`, `check_gpu.py`, `list_audio_devices.py`, `record_samples.py`,
  `benchmark_latency.py`, `benchmark_tts.py`; istruzioni per addestrare le wake word.
- `uv` con `uv.lock` (risolto per Windows) ed export `requirements.txt`.
- 208 test veloci + test sui modelli reali; CI su `windows-latest`.

### Note
- Da verificare sul PC di JenJo: CTranslate2 4.8.3 sulla RTX 5070 Ti (Blackwell) con
  `scripts/check_gpu.py`, latenza reale M1/M2.
