# CLAUDIO

**English** | [Italiano](README.it.md)

A voice assistant for Windows that runs **entirely on the PC**: wake word, transcription,
commands and voice. Phase 1 makes no API calls, so it costs nothing to run. It understands
Italian and English and answers in the language you spoke.

```
Microphone ─► openWakeWord ("Hey Claudio") ─► beep
           └─► Silero VAD (end of speech) ─► faster-whisper (GPU) ─► local router ─► Kokoro, Nicola's voice (sentence by sentence)
```

Status: **v0.1.1**, Phase 1, second iteration. The wake word is our own "Hey Claudio", trained
on the owner's voice (see [scripts/wakeword/README.md](scripts/wakeword/README.md); the model
file is not in Git). "Claudio" alone is trained too but not reliable enough yet, so it is off.
Full specification (in Italian): [CLAUDE.md](CLAUDE.md).

## Requirements

- Windows 10/11, Python 3.12 (3.11-3.13 work), [uv](https://docs.astral.sh/uv/)
- For the GPU (recommended): a recent NVIDIA driver (570 series or later for the RTX 50xx).
  The CUDA 12.8+ and cuDNN 9 libraries come from pip; the CUDA Toolkit is not needed.
- About 3 GB of disk for the models.

## Installation

```powershell
# once: install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/JENJ0/Claudio-AI-desktop-assistant.git
cd Claudio-AI-desktop-assistant
uv sync --extra gpu                       # without a GPU: uv sync
uv run python scripts/download_models.py  # Silero, openWakeWord, Kokoro, Piper, Whisper turbo + small
uv run claudio --check                    # every line must say OK
uv run python scripts/check_gpu.py        # checks that Whisper really runs on the RTX
```

If PowerShell says `uv` is not recognised, close and reopen the terminal (PATH only updates in
new windows). If that is not enough, add uv's folder to PATH once:

```powershell
$p=[Environment]::GetEnvironmentVariable('Path','User'); if ($p -notlike "*\.local\bin*") { [Environment]::SetEnvironmentVariable('Path', "$p;$env:USERPROFILE\.local\bin", 'User') }
```

With pip instead: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
and `pip install -e . --no-deps`.

## Usage

```powershell
uv run claudio                         # the full assistant (Ctrl+C to quit)
uv run claudio --text "che ore sono"   # try the router and the voice without a microphone
uv run python scripts/list_audio_devices.py
```

Say the wake word, wait for the beep, then the command. If you say nothing within 4 s a low
beep plays and CLAUDIO goes back to waiting. You can also say it all in one breath ("hey claudio
che ore sono"): the words said before the beep are recovered. If the model just misses, Whisper
gets a second chance to hear the name (see `wakeword.second_chance`).

| Command (examples) | What it does |
|---|---|
| "apri Chrome", "apri il browser" | opens Chrome |
| "apri la cartella Download" | File Explorer on Downloads |
| "apri Documenti" | File Explorer on Documents |
| "apri Esplora file" | File Explorer |
| "apri il Blocco note" | Notepad |
| "apri la calcolatrice" | Calculator |
| "alza il volume" / "alza il volume di 10" | volume up (5 steps, or the number said) |
| "abbassa il volume" | volume down |
| "muto", "togli l'audio" | mute / unmute |
| "che ore sono", "che giorno è oggi" | answers aloud |
| "vai in pausa", "vai a dormire" | pause: until "svegliati" CLAUDIO answers nothing else |
| "svegliati" | back from the pause |
| "spegniti" | closes CLAUDIO |
| "quanto ho speso oggi" | today's Claude spending and the share of the daily budget |
| "chiudi la finestra" | closes the window in front, after "Confermi?" and a "sì" |
| "metti in pausa la musica", "canzone successiva", "canzone precedente" | media keys: play/pause, next, previous (Spotify, YouTube...) |
| "ripeti", "non ho capito" | says the last reply again |
| "basta", "stop" | ends the conversation; said while CLAUDIO speaks, cuts it short |

Every command works in English too: "open Chrome", "open my downloads", "turn up the volume by
10", "mute", "what time is it", "what day is it", "go to sleep", "wake up", "shut down". Whisper
decides for each request whether it is Italian or English, and CLAUDIO answers in that language
with an English voice.

The phrases live in [config/commands.toml](config/commands.toml): adding more needs no code
(`phrases_en` and `reply_en` are the English ones). Every other request gets "Questo non lo so
ancora fare" or "I can't do that yet"; in Phase 2 they will go to Claude Haiku.

Actions have a safety level (`level` in `commands.toml`). Level 1 runs at once. Level 2 asks
"Confermi?" and CLAUDIO listens for the answer right away, without the wake word: "sì", "ok",
"va bene" (or "yes") run it, anything else (a "no", silence, a word it does not get) cancels.
Level 3 reads a summary of what will happen and only "confermo" / "I confirm" runs it. Every
question and answer goes to the log.

After a reply CLAUDIO keeps listening for 6 seconds without the wake word (a beep says so):
another command needs no "Hey Claudio". The window closes on silence, on "basta", or on
anything that is not a command (the TV, someone else talking), with a low beep and no reply.
While it speaks, CLAUDIO also listens for "basta" / "stop" and stops at once: with speakers its
own voice reaches the microphone too, so headphones work best. Both are in `[conversation]`
(`window_s = 0` turns the window off, `barge_in = false` the listening while speaking).

## Configuration

Everything is in [config/claudio.toml](config/claudio.toml): microphone and output, wake word
thresholds, VAD, STT model, voice. Any value can be overridden with an environment variable or
in `.env`, for example `CLAUDIO_STT__DEVICE=cpu`.

The microphone is chosen by name (part of it is enough; `scripts/list_audio_devices.py` lists
them); when names tie, CLAUDIO uses WASAPI:

```
CLAUDIO_AUDIO__INPUT_DEVICE=C270
```

Without `OUTPUT_DEVICE` CLAUDIO speaks through whatever output Windows is using at that moment:
if you switch it while CLAUDIO runs (headphones, monitor, speakers), the next sentence already
plays there. A device plugged in after start-up needs a restart (CLAUDIO says so in the log).
With `CLAUDIO_AUDIO__OUTPUT_DEVICE=...` it stays on that device instead.

CLAUDIO sets the microphone level itself: after every request it checks how loud the voice
arrived and corrects the digital gain (up 3 dB at a time, down 6 if the voice clips), aiming at
peaks around -20 dBFS. The value is saved per microphone in `data/audio_level.json` and reused
at the next start; `INPUT_GAIN_DB` is only the starting point. Windows' own microphone level is
not touched. To set it by hand: `CLAUDIO_AUDIO__AUTO_GAIN=false` and
`CLAUDIO_AUDIO__INPUT_GAIN_DB=12`. The `voce` and `guad` columns of `benchmark_latency.py --log`
show the level and gain of every request.

Spending (ready for Phase 2, when Claude answers): `[cost]` sets the daily budget (`daily_budget`,
in dollars, 1.0 by default). Every Claude call will be written to `data/spend.db` (SQLite) with
its model, input and output tokens, cache writes and reads (`cache_creation_input_tokens`,
`cache_read_input_tokens`) and TTL, and its cost from [config/pricing.toml](config/pricing.toml),
where the prices and cache multipliers live (never in the code). At 80 % CLAUDIO says so once a
day; at 100 % it stops calling Claude until the next day, while local commands keep working.

If the GPU cannot be used, CLAUDIO switches to Whisper `small` on the CPU by itself and logs it.

Languages: `[stt] languages = ["it", "en"]` are the ones Whisper chooses from. Detection reuses
the encoder pass of the transcription, so it costs almost nothing; when Whisper is unsure (below
`language_confidence`, 0.6; for English `other_confidence`, 0.8, because fast Italian often sounds
English to Whisper) CLAUDIO keeps the language of the previous request. A request heard as English
that no command understands is transcribed again in Italian (about 120 ms, only then): "cue the
calculator" becomes "apri la calcolatrice". `retry_in_first_language = false` turns it off. With one language
(`CLAUDIO_STT__LANGUAGES=it` in `.env`) there is no detection at all. Whisper's prompt is per language,
in `[stt.prompts]`. CLAUDIO's own messages (console, log, `--check`) are Italian by default;
`language = "en"` (or `CLAUDIO_LANGUAGE=en`) switches them to English.

The voice is Nicola (Kokoro, male); `CLAUDIO_TTS__ENGINE=piper` switches to Paola (Piper),
which is also the automatic fallback (`CLAUDIO_TTS__VOICE=it_IT-riccardo-x_low` for Riccardo,
male and the fastest). English replies use Michael with Kokoro and Ryan (`en_US-ryan-low`) with
Piper, set in `[tts.voices.en]`, with the same effect. `CLAUDIO_TTS__LENGTH_SCALE` sets the speed of
both (lower = faster; 0.77 = +30%, the default). The robotic tone is `CLAUDIO_TTS__EFFECT`:
`ultron_estremo` (default), `estremo`, `ultron`, `jarvis`, or `none` for the clean voice. The
effect costs about 5 ms per sentence.

## Latency

Every request ends up in `logs/metrics.jsonl` with:

- **M1** = end of speech → first sound of the reply (target ≤ 1 s);
- **M2** = wake word → first sound.

To tune listening, every line also has `input_dbfs` (average microphone level after the beep)
and `vad_max` (how sure Silero was of hearing a voice, 0-1). `logs/audio/` keeps the last 50
WAVs of what CLAUDIO heard (1.5 s before the wake word, then everything else), named by outcome
(`ok`, `timeout`, `empty`, `error`). They stay on the PC; `[logging] save_audio = false` turns
them off.

Attempts where the wake word did not fire are logged too, as `near_miss` with the highest score
reached: `benchmark_latency.py --log` lists them, to see how far to lower the threshold.

A near miss also gets a second chance (`wakeword.second_chance`): Whisper listens to the last
1.5 s, about 130 ms on the GPU and only at those moments, and if it hears the name CLAUDIO wakes
anyway. The row says what it heard (`second_chance`, `second_chance_ms`), so the near misses
that were not the wake word are visible too.

```powershell
uv run python scripts/benchmark_latency.py --log                      # live
uv run python scripts/record_samples.py commands --repeat 2           # record the commands
uv run python scripts/benchmark_latency.py --wav-dir samples/commands  # turbo vs small vs base
uv run python scripts/benchmark_tts.py                                 # Kokoro and Piper voices
uv run python scripts/benchmark_latency.py --replay                    # replay the saved requests
```

`--replay` runs the WAVs in `logs/audio/` through the same live pipeline (without running any
action) and shows, for each one, the text then, the text now and the command recognised, then
Whisper's median time. `--lead-in 0.4` (seconds) or `--lead-in none` changes how much of the
audio said before the beep reaches Whisper (by default all of it, wake word included); with
`--beam 1` Whisper tries one hypothesis instead of 5, and `--model large-v3` uses another model
(download it first with `download_models.py --whisper large-v3`).

`compare_stt.py` compares speech-to-text engines on the same saved requests and counts how many
each one turns into a known command, with its median and p90 time:

```powershell
uv run python scripts/compare_stt.py                                   # turbo vs closed list
uv run python scripts/compare_stt.py --engines turbo large-v3 lista parakeet --details
```

`lista` is the closed list CLAUDIO uses live (`stt.closed_list`): when no command understands
the best transcription, Whisper proposes 5 and the first known command wins. `parakeet` needs
`uv pip install onnx-asr` (NVIDIA Parakeet TDT 0.6B v3, downloaded on first use). The last line
says how many requests the wake word verifier (`wakeword.verify`) would drop: a wake is kept
only when Whisper heard the name at the start or the request is a local command, so "Hey, ciao,
come stai?" ends in silence and, from v0.2.0, never costs a Claude call.

## Development

```powershell
uv run pytest                 # fast tests (no microphone or models)
uv run pytest -m slow -s      # tests on the real models (after download_models.py)
uv run ruff check . ; uv run ruff format . ; uv run mypy src
```

CI (GitHub Actions, Windows) runs lint, types, the fast tests and a full round trip Kokoro and
Piper → Silero → Whisper `small` (the CPU fallback) → router on the real models.

Layout: `src/claudio/` (audio, wakeword, stt, tts, router, `app.py` with the state machine),
`config/`, `scripts/`, `tests/`. Models in `models/` and logs in `logs/`, both outside Git.
