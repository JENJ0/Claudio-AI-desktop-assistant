# Addestrare le wake word "Hey Claudio" e "Claudio" (v0.1.1)

In v0.1.0 CLAUDIO usa il modello pre-addestrato `hey_jarvis` come segnaposto.
I due modelli si addestrano gratis su Google Colab con il notebook ufficiale di openWakeWord,
**uno per frase**: così "Claudio" (una sola parola, più esposta a falsi positivi) ha una soglia
sua, più severa, senza peggiorare "Hey Claudio". CLAUDIO li carica insieme e condividono
l'elaborazione audio, quindi il secondo modello costa pochissimo.

1. **Le tue registrazioni** (≈5 minuti per frase, col microfono che userai davvero):
   `uv run python scripts/record_samples.py wakeword --phrase "Hey Claudio" --count 50`
   `uv run python scripts/record_samples.py wakeword --phrase "Claudio" --count 50`
   → `samples/hey_claudio/*.wav`, `samples/claudio/*.wav` (metà in silenzio, metà con un po' di rumore).
2. **Notebook**: apri `notebooks/automatic_model_training.ipynb` dal repository
   [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord) su Colab (runtime GPU).
   Parametri da usare (due esecuzioni):
   - modello `hey_claudio`: `target_phrase = ["hey claudio", "ehi claudio"]`
   - modello `claudio`: `target_phrase = ["claudio"]`
   - per entrambi `custom_negative_phrases`: `["claudia", "audio", "chiudi", "ehi claudia", "hey cloud", "eh caldo", "claudine", "laudi"]`
   - `n_samples`: 20000, `steps`: 30000 (default del notebook se l'addestramento è lento)
   - voci sintetiche: aggiungi le voci italiane di Piper (`it_IT-paola-medium`, `it_IT-riccardo-x_low`)
     per la pronuncia "Clàudio" all'italiana.
3. **Consegna**: scarica `hey_claudio.onnx` e `claudio.onnx` in `models/wakeword/` e imposta in
   `config/claudio.toml` (al posto di `hey_jarvis`):
   ```toml
   [[wakeword.models]]
   name = "hey_claudio"
   threshold = 0.5

   [[wakeword.models]]
   name = "claudio"
   threshold = 0.6
   ```
4. **Verifier** (v0.1.1): un piccolo modello sulla tua voce, addestrato con le registrazioni del punto 1,
   per tagliare i falsi positivi.

Obiettivi, per ciascuna frase: < 1 falso positivo ogni 2 ore di audio di sottofondo,
≥ 9 attivazioni su 10 a 1-2 m. Con "Claudio" da solo puoi anche dire il comando di fila
("Claudio apri Chrome"): il router ignora la parola "Claudio" nel testo trascritto.
I parametri effettivamente usati vanno annotati qui per poter riaddestrare.
