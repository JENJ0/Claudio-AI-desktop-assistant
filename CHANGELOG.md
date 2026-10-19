# Changelog

## [Unreleased] - 0.1.1

### Added
- Wake word training kit in `scripts/wakeword/` for "Hey Claudio" and "Claudio": synthetic
  voices (Piper Italian voices, 904 English speakers reading Italian, Kokoro voices blended in
  pairs) from slow to very fast speech and with light mispronunciations, similar words as
  negatives ("audio", "chiudi", "hey Claude"), augmentation with numpy and scipy only,
  openWakeWord's network trained in two rounds in three variants, and a per-threshold report on
  JenJo's held-out takes through the real streaming detector. A Colab notebook
  (`train_claudio.ipynb`, CPU runtime) runs it all, stops at the first failed step, and after a disconnect resumes where it stopped; new or redone
  takes are picked up without starting again from scratch.
- `record_samples.py` guides the recordings: a different way of speaking every five takes
  (fast, quiet, from 2 m, "Ehi", a missing letter, followed by a command), ordinary sentences
  without the wake word, room noise with and without the TV, and `pack` for the Colab zip. New
  takes never overwrite old ones; `r` redoes the previous take, the last one included.
- Second chance for the wake word: when the model's score peaks between `near_miss_floor`
  (0.15) and the threshold, Whisper listens to the last 1.5 s (no prompt, greedy, a few tokens;
  nothing kept for the language fallback) and, if it hears the name, CLAUDIO wakes as if the
  model had fired. At most one check runs at a time, never over a request already under way, and
  Whisper is used by one thing at a time. `wakeword.second_chance = false` turns it off. The
  `metrics.jsonl` row keeps what Whisper heard (`second_chance`, `second_chance_ms`).
- System voice commands, local and free: "vai in pausa" or "vai a dormire" (CLAUDIO then
  answers only "svegliati" and "spegniti", and every other request gets a low beep), "svegliati",
  and "spegniti", which closes CLAUDIO cleanly after saying "A presto.". Shutting down needs one
  of its exact phrases (`exact = true` in `commands.toml`), so "spegni" or "spegni la luce" never
  close it.

- English next to Italian. Whisper picks the language of every request among
  `stt.languages` (`["it", "en"]`), reusing the encoder pass of the transcription, so it costs
  almost nothing; below `stt.language_confidence` (0.6) the previous request's language stays.
  Every command has English phrases and replies (`phrases_en`, `reply_en` in `commands.toml`),
  time and date are said in English too, and the reply follows the language of the phrase
  recognised ("open chrome" is answered in English even if Whisper guessed Italian). English
  replies have their own voice with the same effect: Kokoro `am_michael` (one model in memory for
  both languages) or Piper `en_US-ryan-low`, downloaded by `download_models.py`. Whisper's known
  English hallucinations ("Thanks for watching!") are dropped like the Italian ones.
- CLAUDIO's own messages (console, log files, `--check`, errors) in Italian or English with the
  `language` setting; a test checks that every message has its English version.

### Changed
- The wake word is now our own "Hey Claudio" (`hey_claudio`, threshold 0.30), trained on JenJo's
  voice: on his 50 takes it hears 44 (the first training heard 31), 13 of 14 that it never
  learned from, and wakes 0 times in the room noise; the weakest ways of speaking (fast,
  distracted, "Ehi", a missing letter, quiet) are heard 4 times in 5. "Claudio" alone is trained
  too but hears 9 takes in 50 and wakes 60 times an hour in the noise, so it is left out of the
  config. Known: "Hey, ciao, come stai?" wakes it (0.98), it starts like the wake word. The
  training ran without AudioSet's sounds (its download fails) and with ESC-50 and JenJo's noise.
  Whisper's prompt spells "Hey Claudio" again.
- Wake word training, second round, after the first models heard JenJo's own voice badly
  ("Hey Claudio" 31 takes in 50, "Claudio" alone none): two fifths of every batch of positives
  now come from his takes (each repeated 60 times with different noise) instead of being lost
  among thousands of synthetic windows; each checkpoint is judged at its best threshold within
  the false wake limit, on his held-out takes as well as the synthetic ones (before, at a fixed
  0.5, where a model that was quiet and deaf could win); each model tries three settings and
  keeps the best; "Claudia" is no longer a negative (it differs from "Claudio" by one vowel and
  refusing it taught the single-word model to refuse the real word); the false wake limit is
  one per validation hour instead of one every two. The report also says how many of the takes
  it learned from it hears, which ordinary sentences wake it (with the sentence said), whether
  the model works at all (`healthy`, `problems`) and which variant and settings won.
- The training keeps what it made and says what it was made from: every variant trained is
  saved as soon as it ends (a disconnect costs only the one in progress, and a variant that
  fails does not stop the others), and clips, features and models are made again by themselves
  when the recordings, the noise recordings, the options (`--steps`, `--real-copies`,
  `--target-fph`...) or the scripts change, not only when someone remembers to bump a number.
  Options that cannot work are refused before the hours of synthesis. The notebook checks the
  zip before touching the session, prints the version of the scripts, and has a `force` box.
- The ordinary sentences to record are 38: the two with "Claudia" are out. A take of a sentence
  with "Claudia" is left out of the zip and of the benchmark, however it was recorded: the
  script now writes what each take says (`prompts.json`; the first version's takes are
  recognised by their number). `record_samples.py wakeword --styles veloce,distratto` records
  only some ways of speaking.
- "Claudia" is no longer refused by the router: after a wake it is dropped like any other
  spelling of the name ("Claudia, apri Chrome" opens Chrome).
- The tests ignore the `.env` of the PC they run on.
- README in English, with the Italian version in `README.it.md`; CHANGELOG in English.
- `stt.language` is now `stt.languages` (`language` still fixes a single one), and Whisper's
  prompt is per language in `[stt.prompts]`. `router.unknown_reply` and the new
  `router.error_reply` take one text per language (a plain string still sets the Italian one).
- A command said before the name is understood: "Svegliati, Claudio", "Apri Chrome, Claudio",
  "Wake up, Claudio" (before, the name was taken as the start of the request and the command
  was lost).
- `keywords_xx` in `commands.toml`: words a fuzzy match must contain, so "What's the weather
  today?" is no longer answered with the date.

### Fixed
- An unsure language detection falls back to the language of the last real request, never to
  that of the start-up warm-up, a lone wake word or a dropped hallucination.
- "Shut down the..." (a pause mid-sentence) and "shut down by ten" no longer close CLAUDIO:
  a command with `exact = true` must be all that was said.
- "Turn the volume on" turned the volume down; it now turns it up.
- A reply without its English text, or a language without its own voice, is said in Italian
  with the Italian voice instead of mixing them. `--text` starts from the first of
  `stt.languages`, like the live assistant.
- Settings in `.env` from v0.1.0 keep working: a plain-text `CLAUDIO_ROUTER__UNKNOWN_REPLY` no
  longer stops CLAUDIO from starting, and `CLAUDIO_STT__INITIAL_PROMPT` again replaces the
  first language's prompt. `CLAUDIO_STT__LANGUAGES` takes `it,en` as well as JSON, and a value
  that cannot be read gives a configuration error instead of a traceback.
- Configuration errors are in English too with `language = "en"`, and `--check` no longer warns
  about a missing Piper English voice when Kokoro speaks English.

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
