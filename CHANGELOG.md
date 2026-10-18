# Changelog

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
- Script: `download_models.py`, `check_gpu.py`, `list_audio_devices.py`, `record_samples.py`,
  `benchmark_latency.py`, `benchmark_tts.py`; istruzioni per addestrare le wake word.
- `uv` con `uv.lock` (risolto per Windows) ed export `requirements.txt`.
- 170 test veloci + test sui modelli reali; CI su `windows-latest`.

### Note
- Da verificare sul PC di JenJo: CTranslate2 4.8.3 sulla RTX 5070 Ti (Blackwell) con
  `scripts/check_gpu.py`, latenza reale M1/M2.
