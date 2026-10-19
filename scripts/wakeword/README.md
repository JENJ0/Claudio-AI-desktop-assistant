# Wake word training

Trains CLAUDIO's two wake word models, `hey_claudio.onnx` ("Hey Claudio") and `claudio.onnx`
("Claudio"), for [openWakeWord](https://github.com/dscripka/openWakeWord). It runs for free on
Google Colab (2-3 hours, CPU runtime) or on any Linux PC.

| File | What it does |
|---|---|
| `phrases.py` | What the models learn: wake word spellings, similar words to ignore, ordinary sentences, the recording styles and prompts |
| `augment.py` | Speed, level, room echo, microphone colour and background noise (numpy and scipy only) |
| `generate_clips.py` | Synthetic voices: Piper (Italian voices and 904 English speakers reading Italian) and Kokoro (54 voices, blended in pairs) |
| `train_wakeword.py` | The whole run: downloads, synthesis, features, training, ONNX export and the threshold report |
| `train_claudio.ipynb` | The Colab notebook that runs `train_wakeword.py` for both models |

## 1. Record (on the PC, about 20 minutes)

From the CLAUDIO folder, with the usual microphone:

```powershell
uv run python scripts/record_samples.py wakeword --phrase "Hey Claudio"
uv run python scripts/record_samples.py wakeword --phrase "Claudio"
uv run python scripts/record_samples.py negatives
uv run python scripts/record_samples.py noise
uv run python scripts/record_samples.py pack
```

- `wakeword`: 50 takes per wake word. Every five takes the script asks for a different way of
  saying it: normal, fast, slow, quiet, loud, from 2 m, "Ehi" instead of "Hey", with a missing
  letter, absent-minded, followed by a command. Takes followed by a command are only used to
  measure the model, never to train it.
- `negatives`: 40 ordinary sentences without the wake word (about 3 minutes).
- `noise`: one minute of the quiet room and one with the TV or music on, nobody speaking.
- `pack`: writes `samples/claudio_training.zip` with the recordings and these scripts.

Each mode adds new takes after the existing ones, so running it again never overwrites. Type
`r` + Enter to redo the previous take (after the last one the script asks once more), `q` +
Enter to stop. The "● parla" cue appears only once the microphone is recording, and the
microphone is the one CLAUDIO uses (`audio.input_device`).

## 2. Train on Colab

1. Open <https://colab.research.google.com>, then **File → Upload notebook** and pick
   `scripts/wakeword/train_claudio.ipynb`.
2. **Runtime → Run all**. When the upload button appears, choose `claudio_training.zip`.
3. Keep the tab open. At the end the browser downloads `claudio_wakeword.zip`.

If a step fails, Run all stops there with the reason in red. After a disconnect, Run all again
picks up where it stopped, finished models included. To train on new recordings in the same
session, tick `new_recordings` in the upload cell before Run all.

## 3. Install the models

Copy `hey_claudio.onnx` and `claudio.onnx` from the zip into `models/wakeword/`. The two
`*_report.json` files (also printed as tables at the end of each training) give, for every
threshold, how many of your held-out takes wake CLAUDIO and how many false wakes per hour the
validation audio causes, plus the recommended threshold. Put that threshold in
`config/claudio.toml`, in place of the `hey_jarvis` placeholder:

```toml
[[wakeword.models]]
name = "hey_claudio"
threshold = 0.5    # recommended_threshold from hey_claudio_report.json

[[wakeword.models]]
name = "claudio"
threshold = 0.6    # a single word: usually stricter
```

With "Claudio" alone you can say the command straight after it ("Claudio apri Chrome"): the
router ignores the word "Claudio" in the transcript.

## On a Linux PC instead of Colab

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch scipy soundfile scikit-learn tqdm requests \
    piper-tts==1.8.0 onnxruntime==1.31.0 onnx==1.23.2
pip install --no-deps openwakeword==0.6.0
unzip claudio_training.zip -d claudio && cd claudio
python wakeword/train_wakeword.py download --data data
python wakeword/train_wakeword.py train --model hey_claudio --workers 8
python wakeword/train_wakeword.py train --model claudio --workers 8
```

Every stage keeps its output in `work/` (the models in `output/`) and is skipped when it is
already there and was made from the same recordings: the same command resumes after an
interruption, and new or redone takes are picked up. `train` checks that `download` got
everything before it starts; `--force` trains a finished model again. `--help` lists the
options (number of clips, steps, target false wakes per hour).

## How it works

- **Positives**: 6000 synthetic clips per model, from very slow to very fast speech, with light
  mispronunciations ("hey cladio", "clodio"), plus your takes repeated with different noise.
- **Negatives**: similar words ("Claudia", "audio", "chiudi", "hey Claude"), ordinary Italian
  and English sentences, your own sentences and room noise, and about 2000 hours of varied
  audio (openWakeWord's precomputed ACAV100M features).
- **Augmentation**: silence and room noise trimmed around the voice, speed 0.85-1.25x,
  synthetic room echo from desk-dry to across the room, microphone filters, AudioSet and
  ESC-50 background sounds at 0-20 dB SNR; the wake word ends near the end of each 2 s window,
  sometimes followed by the start of a command, as the detector hears it while streaming.
- **Training**: openWakeWord's small network (16 frames x 96 features), two rounds with
  negatives weighing more and more; the checkpoint kept is the one with the best recall at no
  more than 0.5 false wakes per hour (`--target-fph`).
- **Report**: your held-out takes go through the real streaming detector, as in CLAUDIO.

The training data has its own licences (openWakeWord features CC BY-NC-SA 4.0, ESC-50
CC BY-NC 3.0, AudioSet CC BY 4.0): the resulting models are for personal, non-commercial use.
