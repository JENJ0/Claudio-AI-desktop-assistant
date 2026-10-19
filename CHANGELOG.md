# Changelog

## [Unreleased] - 0.1.1

### Added
- Wake word training kit in `scripts/wakeword/` for "Hey Claudio" and "Claudio": synthetic
  voices (Piper Italian voices, 904 English speakers reading Italian, Kokoro voices blended in
  pairs) from slow to very fast speech and with light mispronunciations, similar words as
  negatives ("Claudia", "audio", "chiudi", "hey Claude"), augmentation with numpy and scipy only,
  openWakeWord's network trained in two rounds, checkpoint chosen for the best recall at no more
  than 0.5 false wakes per hour, and a per-threshold report on JenJo's held-out takes through the
  real streaming detector. A Colab notebook (`train_claudio.ipynb`, CPU runtime) runs it all.
- `record_samples.py` guides the recordings: a different way of speaking every five takes
  (fast, quiet, from 2 m, "Ehi", a missing letter, followed by a command), ordinary sentences
  without the wake word, room noise with and without the TV, and `pack` for the Colab zip. New
  takes never overwrite old ones; `r` redoes the previous take.
- System voice commands, local and free: "vai in pausa" or "vai a dormire" (CLAUDIO then
  answers only "svegliati" and "spegniti", and every other request gets a low beep), "svegliati",
  and "spegniti", which closes CLAUDIO cleanly after saying "A presto.". Shutting down needs one
  of its exact phrases (`exact = true` in `commands.toml`), so "spegni" or "spegni la luce" never
  close it.

### Changed
- README in English, with the Italian version in `README.it.md`; CHANGELOG in English.

## [0.1.0] - 2026-10-19

First iteration of Phase 1: a working skeleton, all local, no API calls, tuned with live tests
on JenJo's PC.

### Added
- Full pipeline: microphone (sounddevice, 16 kHz, 80 ms blocks) → openWakeWord (ONNX) → beep →
  Silero VAD v6 through onnxruntime (no PyTorch) with 800 ms of pre-roll → faster-whisper →
  local router → Piper, streamed sentence by sentence.
- State machine `IDLE → LISTENING → TRANSCRIBING → RESPONDING → SPEAKING`, 4 s timeout without
  speech and 8 s maximum per request; the wake word is ignored while CLAUDIO works or speaks.
- Several wake words at once, each with its own threshold (ready for "Hey Claudio" + "Claudio");
  `hey_jarvis` as a placeholder until the custom models are trained.
- STT: `large-v3-turbo` float16 on CUDA, automatic fallback to `small` int8 on the CPU; filter for
  Whisper's known hallucinations ("Sottotitoli creati dalla comunità Amara.org", prompt echo).
- Router with 10 commands in `config/commands.toml` (time and date separate), text normalisation
  and exact + fuzzy matching (rapidfuzz); numbers for the volume ("alza il volume di 10").
- Level 1 actions: Chrome, Downloads, Documents (the real folders, even when moved to OneDrive),
  File Explorer, Notepad, Calculator, volume/mute keys through the Windows API, time and date
  aloud.
- Audio devices by name: when names tie WASAPI wins (lower latency, with Windows' sample rate
  conversion); the name is resolved again every time the microphone is reopened.
- Near-silent audio never reaches Whisper (avoids "Buon appetito!" and the like on silence).
- TOML configuration + environment variables with clear error messages (pydantic-settings).
- Console logs and rotating JSON files; M1/M2 metrics for every request in `logs/metrics.jsonl`.
- Listening diagnostics: for every request the microphone level (`input_dbfs`), the VAD's highest
  confidence (`vad_max`) and a WAV of what CLAUDIO heard in `logs/audio/` (last 50).
- More tolerant listening after the beep (from the first live test): VAD threshold 0.4, 64 ms of
  speech to start, 800 ms of audio kept before the start, 4 s to begin speaking.
- Whisper starts without contacting huggingface.co when the model is already downloaded.
- Missed wake words (a peak above 0.15 that did not reach the threshold) are logged in
  `metrics.jsonl` as `near_miss`, with audio.
- Digital microphone gain (`audio.input_gain_db`) and a log warning when the voice arrives too
  quiet; every request also logs the voice's peak (`speech_dbfs`).
- Commands said in one breath ("hey jarvis che ore sono"): if the wake word fires late, the words
  said just before it are added to the request and the wake word is removed from the
  transcript. If the request holds only the wake word, CLAUDIO listens again without losing what
  was said in the meantime.
- Scripts: `download_models.py`, `check_gpu.py`, `list_audio_devices.py`, `record_samples.py`,
  `benchmark_latency.py`, `benchmark_tts.py`; instructions to train the wake words.
- `uv` with `uv.lock` (resolved for Windows) and a `requirements.txt` export.
- 252 fast tests + 15 on the real models; CI on `windows-latest`.
- Male voice Nicola (Kokoro-82M, ONNX, local) as the default voice; Piper stays as the automatic
  fallback if Kokoro does not start. The phonemes come from the espeak-ng already bundled with
  Piper, so nothing else to install. `tts.length_scale` sets the speed of both voices (default
  0.77, i.e. +30%).
- Robotic tone (`tts.effect`): `ultron_estremo` by default (a lower copy, distortion, ring
  modulation and metallic echo), plus `estremo`, `ultron`, `jarvis` and `none`; about 5 ms per
  sentence.
- Automatic microphone level (`audio.auto_gain`, on): after every request CLAUDIO corrects the
  digital gain towards -20 dBFS peaks (+3 dB at a time, -6 dB when clipping), between 0 and
  +30 dB, and saves it per microphone in `data/audio_level.json`. The level measures only the
  voice, and every request also logs clipping (`clipped`) and gain (`gain_db`).
- Without `audio.output_device` CLAUDIO speaks through Windows' default output of the moment
  (read with Core Audio for every sentence): switching it while CLAUDIO runs no longer needs a
  restart.
- `benchmark_latency.py --replay` runs the requests saved in `logs/audio` through the real
  pipeline (without running actions), to compare on one's own voice how much audio before the
  wake word to give Whisper (`--lead-in`) and the beam (`--beam`). On JenJo's first 22
  recordings, cutting the wake word out of the audio made things worse (10 commands understood
  instead of 13), because the command is glued to it: so everything stays, and Whisper's prompt
  spells "Hey Jarvis".

### Changed
- The default microphone and output move from MME to WASAPI, which has less latency.
- Whisper with beam 5: on JenJo's 33 recordings it understands 25 commands instead of 19, with
  only 2 ms more on the GPU (about 106 ms).
- A more sensitive wake word: `hey_jarvis` threshold at 0.22. In JenJo's last test the missed
  attempts reached 0.23-0.31 and had to be repeated.
- Whisper's prompt no longer contains "Claudio" while the wake word is "hey jarvis": Whisper
  copied it into the text ("E il Jarvis, Claudio, sono").

### Fixed
- The wake word said fast and mangled by Whisper at the start of the request ("E il jarviso che
  ore sono", "Ejci Arvis, …") is now removed before looking for the command.
- Whisper could loop ("Ciao, ciao, ciao…" for 100 words) on confusing audio: it now stops at 80
  tokens and such repetitive text is discarded.
- Kokoro and Piper in the same process (tests, `benchmark_tts.py`, the Piper fallback)
  initialised espeak-ng twice: on Windows the second time corrupted its phoneme tables, and both
  voices spoke nonsense (or the program closed). Kokoro now uses Piper's espeak-ng.
- Nicola dragged the syllables with soft "c" and "g" ("calcolatrice", "dieci"): the affricates
  now reach Kokoro as one sound, as in its training.

### Notes
- Checked on JenJo's PC: CTranslate2 4.8.3 runs on the RTX 5070 Ti; median M1 564 ms (p90
  764 ms) over 44 requests, with Whisper at about 128 ms.
- Still to try: 30 minutes of TV or music to count false wakes at the 0.22 threshold, and
  unplugging and replugging the microphone while CLAUDIO runs.
