# CLAUDIO

**English** | [Italiano](README.it.md)

A voice assistant for Windows: wake word, transcription, commands and voice run **on the PC**
for free. What no local command understands is answered by Claude (Anthropic API: Sonnet 5.5 as
the main model, on trial; Haiku 5.5 to guess misheard commands), with a daily budget. It
understands and speaks Italian.

```
Microphone ─► openWakeWord ("Hey Claudio", "Claudio") ─► beep
           └─► Silero VAD (end of speech) ─► faster-whisper (GPU) ─► local router or Claude ─► Jack's voice (Qwen3-TTS, sentences in groups; Kokoro if it fails)
```

Status: **v0.4.1** (05/11/2026, IT-2 step 0: fixes, see the changelog). v0.4.0 (30/10/2026): Phases 1-3 (local commands, Claude, PC control) and the voice
commands to pause, turn off ("spegniti"), turn on ("accenditi") and close ("chiuditi"); the
reliability iteration IT-0 is closed. Since then (IT-1, branch `it-1`, being tested): long-term
memory (Phase 4), one CLAUDIO at a time, restarted after a crash, an icon in the system tray,
misheard commands heard again, steadier listening, an even voice for Jack. Not there yet:
conversation summaries (Phase 4) and automatic updates (Phase 5). The wake words are our own, trained on the owner's voice (see
[scripts/wakeword/README.md](scripts/wakeword/README.md); the model files are not in Git):
"Hey Claudio" and "Claudio" alone, which is weaker, so Whisper confirms each of its detections.
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
uv sync                                   # without a GPU: uv sync --no-group gpu
uv run python scripts/download_models.py  # Silero, openWakeWord, Kokoro, Piper, Whisper large-v3 + small (~3 GB)
uv run claudio --check                    # every line must say OK (the Whisper model it will load too)
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
| "spegniti" | off: no beeps, no replies, nothing goes to Whisper twice or to Claude; only the wake word listens (`[wakeword] off_threshold_boost` stricter) for "Claudio, accenditi"; what is said while off stays off the disk, except the turn-on |
| "accenditi" | back from off (or from the pause) |
| "chiuditi" | closes CLAUDIO |
| "quanto ho speso oggi" | today's Claude spending and the share of the daily budget |
| "chiudi la finestra" | closes the window in front, after "Confermi?" and a "sì" |
| "metti in pausa la musica", "canzone successiva", "canzone precedente" | media keys: play/pause, next, previous (Spotify, YouTube...) |
| "ripeti", "non ho capito" | says the last reply again |
| "basta", "stop", "non parlare" | ends the conversation; said while CLAUDIO speaks, cuts it short; also at the end of a phrase that names Claudio ("No, Claudio, non parlare. Claudio, stop."), with no API call |

The phrases live in [config/commands.toml](config/commands.toml): adding more needs no code.
Every other request goes to Claude; with `[claude] enabled = false` it gets "Questo non lo so
ancora fare".

Actions have a safety level (`level` in `commands.toml`). Level 1 runs at once. Level 2 asks
"Confermi?" and CLAUDIO listens for the answer right away, without the wake word: "sì", "ok",
"va bene" (or "yes") run it, anything else (a "no", silence, a word it does not get) cancels. A yes holds nothing else: "sì, grazie mille" counts, "si chiama Anna", "sì, ma dopo" and "Sì?" do not. At a "Confermi?" asked by Claude, repeating Claudio's own command ("Claudio, spegni") also counts.
Level 3 reads a summary of what will happen and only "confermo" / "I confirm" runs it. Every
question and answer goes to the log.

After a reply CLAUDIO keeps listening for 6 seconds without the wake word (a beep says so):
another command needs no "Hey Claudio". The window closes on silence, on "basta", or on
anything that is not a command (the TV, someone else talking), with a low beep and no reply.
While it speaks, CLAUDIO keeps listening. A voice close to the microphone (`barge_in_min_dbfs`,
measured before the microphone's gain, at least 0.3 s) lowers the reply at once; when Whisper hears words the reply stops and what you
said becomes the next request if it starts with the name ("Claudio, che ore sono") or is a
command ("basta" / "stop" only stops; `barge_in_needs_name = false` takes any phrase); once it
is answered, the cut reply goes on ("Dicevo: ...", `resume_s`) if it was a long one (a story,
`resume_min_chars`), not a question nor the answer to a phrase Whisper doubted. Noise brings
the volume back. Its own voice coming back through speakers is ignored, also when Whisper
mishears it ("Sono le 19 e 42" -> "Se non le diciamo"): the phrase is compared letter by letter
with what was just said (`echo_similarity`). After every sound of its own (a reply, a beep,
"Sono pronto") nothing listens until the speakers have played it out (CLAUDIO measures their
delay) plus `echo_tail_ms` for the room: not the wake word, not the conversation window, not the
answer to "Confermi?" (it never confirms its own question). Headphones still work best. All of it is in `[conversation]` (`window_s = 0` turns the window
off, `barge_in = false` the listening while speaking, `barge_in_mode` how it reacts).

## Claude (questions)

1. Create an API key at <https://console.anthropic.com> and add some prepaid credit (the
   Claude Pro subscription does not include the API).
2. Copy `.env.example` to `.env` in the project folder (next to `pyproject.toml`) and write the
   key after `ANTHROPIC_API_KEY=`. `.env` is in `.gitignore`: it never reaches GitHub.
3. Try it without the microphone: `uv run claudio --text "chi era Leonardo da Vinci"`.

Local commands always come first and cost nothing; Claude answers the rest, streamed
(Jack speaks it in groups of sentences). Each call is logged with its tokens (cache included) and cost in `data/spend.db`;
`[cost] daily_budget` (2.58 USD) is announced at 80 % and blocks Claude at 100 %. In the
conversation window (no wake word) a phrase goes to Claude only when it sounds addressed to it
(a question, "dimmi...", "spiegami...", or the name), so the TV costs nothing. "basta" stops a
reply and closes the API stream; "ripeti" repeats it. Settings: `[claude]` in
`config/claudio.toml` (`enabled = false` for local only).

The main model is **Sonnet 5.5** (adaptive thinking, low effort), on trial: if real use costs
too much, Haiku 5.5 goes back to being the main model, as in every release so far. A Sonnet
reply with a cold cache cost 0.017 USD in the one call measured so far, almost all of it for
writing the ~6,500-token prompt to the cache (2.50 USD per million tokens). Within 5 minutes of
the previous call the prompt is read from the cache instead, at 0.10 USD per million (5 % of the
input price): about 0.002 USD a reply, by calculation. "pensaci bene" / "think hard" need no
second model: Sonnet reads them as part of the request. Claude also knows the current date and
time.

With Haiku as the main model (`model = "claude-haiku-5-5"`, `thinking = false`), Sonnet is a
second model, never used without your yes: Haiku proposes it and Claudio asks "Lo uso?" ("sì" / "no"), or start the
request with "pensaci bene". `smart_model = ""` keeps Haiku only.

**Misheard commands.** A short phrase no command understands (up to 6 words, a question too:
"Scaniti.", "Chi ce l'ha fatta?" said for "chiudi la chat") first gets Whisper's second opinion,
local and free: Whisper hears it again with the commands as its prompt, and the command it
proposes is judged on the sound alone. Much likelier than the first hearing: done; about as
likely: "Hai detto: ...?" (answered here, no API); otherwise the phrase goes on as before
(about +0.2 s, only on those phrases; `[recovery]`, every attempt in `logs/metrics.jsonl`).
Then a phrase Claudio was about to ignore (a false wake, a phrase in the
conversation window that did not look addressed to it, noise) that sounds like a command
("Spenditi.", "Audio Bust.") goes to Haiku, which says which command you most likely asked and
how sure it is: from 80 % Claudio does it, from 50 % it asks "Hai detto: ...?" first ("sì" does
it, "no" drops it, "no, alza il volume" does that instead), below it ignores the phrase as
before. Pausing, "spegniti" and "chiuditi" are always asked, "basta" never (it stops at once).
Only phrases close to a command cost a call (about 0.0001 USD); settings in `[guess]`. Claude can also run Claudio's own commands (volume,
music keys, chat window, spending, pause, "spegniti") when a request reaches it garbled
("Claudio, ok, la chata"). `uv run python scripts/replay_guess.py --dry` shows how close the
phrases dropped in `logs/metrics.jsonl` were to a command; without `--dry` it asks Haiku too.

## Personality (CLAUDIO.md)

Claudio's character is in `data/CLAUDIO.md`, made at the first start from `config/CLAUDIO.md`
(Handsome Jack from Borderlands: cocky, ironic, helpful). Edit it by hand and restart; updates
never touch it, and `<!-- notes -->` are not sent. It is read once and goes after the system
prompt, so it is cached: keep it short (`personality_max_chars = 3000`). Claudio answers in
Italian unless the request is plainly English or you ask for English.

## Memory

Claudio remembers what you tell it between sessions, in `data/memory.db` (SQLite, local).
These phrases are understood locally, for free, also offline or with the budget spent:

| You say | Claudio |
|---|---|
| "Claudio, ricordati che mia sorella si chiama Anna" | keeps the fact and reads it back |
| "Da ora in poi chiamami capo" (or "ricordati sempre di...") | keeps a standing rule |
| "Cosa ricordi di mia sorella?" | reads what matches (Claude answers when nothing does) |
| "Cosa ricordi?" | how many memories, and the latest three |
| "Dimentica che mia sorella abita a Torino" | forgets it when every word matches; otherwise asks first ("Dimentico questo: ...?", "Quale dimentico?") |
| "Dimenticalo" | forgets the memory just talked about (within a minute) |
| "Dimentica tutto" | asks for "confermo", then forgets everything |

Claude sees every rule and the newest facts after the system prompt (cached: about 0.0002 $ a
request on Sonnet); the facts that do not fit in `[memory] block_chars` go with a request only when they
share its words. Claude can also save ("segnatelo", a misheard "ricordati") and forget with its
tools: a rule or a forget it proposes needs your yes. It also keeps, by itself, a lasting
preference you mention, and says so ("Me lo segno"): `[memory] learn = false` turns that off. Passwords, PINs, cards and IBANs are never
kept. "Ricordami di... alle 5" is a reminder, not a memory: not available yet.

## Acting on the PC

With Claude on, Claudio also acts on Windows (`[pc]` in `config/claudio.toml`): it opens apps
from the Start menu and files or folders, searches files with the Windows Search index, lists
and reads windows (UI Automation, text only, ~1500 characters), brings a window to the front,
presses keys, types text, closes windows and moves files. Several actions can come in one
reply, up to 4 round trips per request.

Safety levels are fixed in the code (`src/claudio/pc/tools.py`), never chosen by Claude:
opening, searching, reading and harmless keys (arrows, Page Down, Ctrl+C, Ctrl+T...) run at once;
typing, closing a window, moving a file and every other key make Claudio ask just "Confermi?"
(`confirm_brief = false` reads the action out first). Only a short spoken yes runs it (never from the chat window).
Files: only inside your user folder (`roots = ["~"]`), never overwritten; programs and scripts
are never opened as files. Password managers are never read. Every action is logged
(`metrics.jsonl` field `tools`, "Azioni" in `conversazione.md`). An action with one tool cost
about 0.0003-0.0007 USD with Haiku as the main model; on Sonnet it is not measured yet.

Since v0.3.2: typing in Notepad needs no yes (`free_typing`, by the window's process, not its title), long text is pasted (one Ctrl+V,
your clipboard is put back), and a folder said by name ("apri la cartella Progetti") is found on
the Desktop, in Documents or Downloads. Before a slow action Claudio says "Ci guardo".

**Screen (last resort):** when a window cannot be read as text, Claude may look at a small
screenshot (long side 1024 px, JPEG, ~0.002 USD on Sonnet for a 16:9 screen; never saved). Password managers are
never photographed.

**Terminal:** Claude may run one PowerShell command per reply, always at level 3: Claudio says
the purpose, shows the exact command in the chat window (opened if needed) and runs it only after
you say "confermo" aloud (never from the chat, never with a plain "sì"). PowerShell without
profile, never as administrator (if Claudio itself runs elevated, commands are off), stopped
after 30 s with its whole process tree, output cut at 2000 characters. A deny list in
`src/claudio/pc/terminal.py` refuses, before asking, formatting, recursive deletes of drives
and system folders, machine registry writes, Defender/firewall changes, download-and-run,
encoded commands and elevation. It is a safety net, not a sandbox: read the command before
saying "confermo". `commands = false` in `[pc]` removes the tool.

## Jack's voice (default engine, personal use; model not included)

The shipped config has `engine = "jack"` in `[tts]`: a Qwen3-TTS model fine-tuned on Handsome
Jack's Italian lines (RVC is loaded but not applied, `jack_rvc = "none"`: the worker still needs Applio and
the RVC model from step 1). It needs an NVIDIA
GPU (~1.2 GB VRAM) and runs in a **separate worker process** with its own venv, so Claudio itself
does not depend on torch. Without the trained voice Claudio says "La voce di Jack non è partita"
at start and Kokoro speaks; `engine = "kokoro"` skips the attempt.

1. The trained voice is not in the repo. Put it under `models/voices/jack/`: the Qwen3
   checkpoint (`ft/1.7B-calmo/checkpoint-epoch-7`, `jack_finetune.py` round 4: calm lines only, joined up to 12 s), Applio 3.7.0 with the RVC model
   (`rvc/Applio/logs/jack/jack_100e_2800s.pth`); other paths: `jack_*` keys in `[tts]`.
2. Build the worker venv once: `powershell -File scripts\setup_jack_worker.ps1`.
3. Try it alone: `models\voices\jack\worker\env\Scripts\python.exe scripts\try_jack.py --play`.
4. Nothing to set: the shipped `config/claudio.toml` already has `engine = "jack"`.
5. Pick the pronunciation settings by ear: `uv run python scripts/jack_variants.py` (close
   Claudio first) makes every variant speak the same replies and writes the WAVs plus a
   report (Whisper's word error rate, first audio, longest pause) to `logs/jack-variants/`.
   The keys: `jack_temperature` (lower = steadier), `jack_rvc` (`"none"` in the shipped config; or
   `"blocks"`, `"sentence"`, `"auto"`). Qwen3 is always told "Italian" (with "Auto" it guessed the language per sentence).

Before Jack speaks, symbols, acronyms, Markdown and emoji become Italian words ("50%" ->
"cinquanta per cento") and the words in `[tts.jack_words]` are respelled ("file" -> "fàiil"):
`src/claudio/tts/spoken_text.py`. Add a word there when Jack says it wrong. Jack also writes
the stress on sdrucciole ("accènditi", "dìmmelo"; lexicon `data/stress_it.tsv` from Wiktionary,
CC BY-SA 4.0), never on the other words, and an English word not in the table is logged once
("Jack: parola sconosciuta"); `scripts/jack_variants.py --words WORD...` lets you hear the
options.

Every take is brought to `jack_lufs` (-24 LUFS, peaks limited) and checked before it plays:
one too high, shouted or babbling is generated again with the next seed (`jack_take_check`),
and the seed of a good take is kept for that text (`data/jack_takes.json`). The fixed replies
(commands, confirmations, "Sono pronto.") can be made in advance, with the GPU free and
Claudio closed: `uv run python scripts/jack_takes.py --from-logs 3` (about 10 minutes, report
in `logs/jack-takes/report.md`); they then play from `data/jack_cache/` with no TTS. Run it
again after changing `[tts.jack_words]` or Jack's checkpoint.

The worker loads in 10-30 s, in the background: Claudio listens as soon as Whisper and the
wake word are loaded, and a reply asked meanwhile waits for Jack until `jack_first_wait_s`
after the worker started (30 s, Claudio's own loading included; 0 = the fallback speaks at
once). With less than `jack_min_free_vram_mb` (6000 MB) free on the graphics card at start,
a game for instance, replies do not wait at all: the fallback speaks until Jack is ready, and
a worker that timed out is tried again once the card has room. When Jack is ready Claudio
says "Sono pronto" (`[startup] announce_ready`). Kokoro (Piper if Kokoro is missing) is the fallback for any sentence
the worker cannot speak. If the worker does not start, Claudio says so
out loud ("La voce di Jack non è partita") and the reason is in
`console.log` (`Jack non disponibile (...)`). The same happens when Whisper cannot use the GPU.

## Chat window

With Claudio running (`uv run claudio`), say **"apri la chat"**, "chat" or "fammi scrivere"
("open the chat"): the window opens, Claudio keeps listening. "chiudi la chat" closes it, and it
closes with Claudio. The phrases are in `config/commands.toml` (`open_chat`, `close_chat`), matched
locally at no cost. `uv run claudio --ui` opens the same window alone (read-only, no voice).

The window shows the conversations: pick a session on the left, search
all of them at the top, see what each answer cost. It reads `data/claudio.db`, so it works with
Claudio running (it updates by itself) or not. Messages older than `logging.retention_days`
(30) are deleted when Claudio starts, and so are the older session logs (`logs/sessions/`), the
recordings in `logs/audio/` with their notes (missed wake words too) and the rotated `logs/metrics.jsonl.*`.

You can also **write to Claudio** in the box at the bottom (Enter sends): with Claudio running,
it answers in the chat, in silence (`[chat] speak_typed = true` to hear it too). Commands work
as well ("apri chrome"); actions that need a confirmation must be asked by voice.

## Session logs

Every run writes `logs/sessions/<date>_<time>/console.log` and `conversazione.md` (requests,
replies, latency, cost): send that folder, or let Claude read it, instead of copying the
console. The newest 30 sessions are kept.

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

CLAUDIO sets the microphone level itself: after every request in which Whisper heard words it
checks how loud the voice arrived and corrects the digital gain, aiming at peaks around
-20 dBFS. A microphone's first request sets it at once; after that it moves gradually (at most
+3 dB or -6 dB a request, a louder voice counting more than a quieter one), and -6 dB at once
if the voice clips. Each microphone really in use keeps its own value in
`data/audio_level.json` (the headset does not start at the webcam's gain), reused at the next
start; `INPUT_GAIN_DB` is only the starting point. A peak louder than the rest is bent softly
under full scale instead of clipping. Windows' own microphone level is not touched. To set it
by hand: `CLAUDIO_AUDIO__AUTO_GAIN=false` and `CLAUDIO_AUDIO__INPUT_GAIN_DB=12`. The `voce` and
`guad` columns of `benchmark_latency.py --log` show the level and gain of every request.

Spending (ready for Phase 2, when Claude answers): `[cost]` sets the daily budget (`daily_budget`,
in dollars: 2.58 in the shipped config, 1.0 if the line is missing). Every Claude call will be written to `data/spend.db` (SQLite) with
its model, input and output tokens, cache writes and reads (`cache_creation_input_tokens`,
`cache_read_input_tokens`) and TTL, and its cost from [config/pricing.toml](config/pricing.toml),
where the prices and cache multipliers live (never in the code). At 80 % CLAUDIO says so once a
day; at 100 % it stops calling Claude until the next day, while local commands keep working.

If the GPU cannot be used, CLAUDIO switches to Whisper `small` on the CPU by itself and logs it.

Language: Whisper always transcribes in Italian: there is no language detection and no second
transcription in another language. Its prompt, the sentences it reads as the words said just
before, is `[stt] initial_prompt`.

The voice is Jack (see "Jack's voice"); the fallback is Nicola (Kokoro, male), and Paola (Piper)
when Kokoro does not load. `CLAUDIO_TTS__ENGINE=kokoro` or `piper` makes one of them the main
voice (`CLAUDIO_TTS__VOICE=it_IT-riccardo-x_low` for Riccardo with Piper, male and the fastest).
`CLAUDIO_TTS__LENGTH_SCALE` sets the speed of
Kokoro and Piper (lower = faster; 0.77 = +30%, the shipped value; Jack's is `jack_speed`).

## Start at login

CLAUDIO can start by itself 30 s after you log in to Windows (Task Scheduler, your user only,
no admin rights, no console window: `pythonw.exe`, logs in `logs/`). Nothing is registered
until you run one of these from the project folder (`install` and `uninstall` are safe to run
twice):

```powershell
uv run python -m claudio.autostart install --dry-run   # show only
uv run python -m claudio.autostart install
uv run python -m claudio.autostart status
uv run python -m claudio.autostart uninstall
```

The task runs `python -m claudio.supervisor`, which starts CLAUDIO and starts it again after a
crash: 5 s, 30 s and 60 s later, then it gives up if CLAUDIO crashed 4 times in 10 minutes (the
reason is in `logs/claudio.log.jsonl`, the supervisor's own lines in
`logs/supervisor.log.jsonl`). "Chiuditi", the tray menu and a configuration error end it too.
Task Scheduler's own "restart on failure" is not enough: it only retries a program that cannot
start, never one that ends with an error (tried on the PC, 30/10/2026).

Only one CLAUDIO runs at a time (a named Windows mutex): a second `uv run claudio`, typed while
the one started at login is running, writes "CLAUDIO è già acceso" in the log and closes with
exit code 3 before loading anything, so two of them never share the microphone.

## System tray

While CLAUDIO runs, its icon sits next to the clock (`[startup] tray`, on by default; it needs
`pystray`, installed by `uv sync`). The color is the state: blue while the models load, green
when on, amber when paused, gray when off ("spegniti"), and it follows the voice commands too.
A left click opens the chat window; the right-click menu turns CLAUDIO off or on, pauses it,
opens the chat and closes CLAUDIO. Without `pystray` CLAUDIO runs as before, with a warning in
the log.

## Startup time

Whisper, the wake word, the voice and Claude load at the same time (`[startup] parallel`;
`false` = one after the other, if a model misbehaves when loaded so). `console.log` shows what
each one took and when everything, Jack included, was ready:

```
Avvio: whisper 3.1 s, wake word 0.4 s, voce 1.2 s, claude 0.6 s (in parallelo: sì)
Modelli caricati in 3.4 s. Di' "Hey Claudio" per iniziare (Ctrl+C per uscire).
Worker Jack pronto in 14.2 s (VRAM GPU 1200 MB)
Avvio completo in 15.0 s
```

## Latency

Every request ends up in `logs/metrics.jsonl` with:

- **M1** = end of speech → first sound of the reply (target ≤ 1 s);
- **M2** = wake word → first sound.

Every line also has `gpu_free_mb` and `gpu_busy_pct`, read in the background at most every 5 s
(None on the first line, or with no NVIDIA card).

To tune listening, every line also has `input_dbfs` (average microphone level after the beep)
and `vad_max` (how sure Silero was of hearing a voice, 0-1). `logs/audio/` keeps the last 8000
WAVs (`[logging] keep_audio`, about 940 MB) of what CLAUDIO heard (1.5 s before the wake word, then
everything else), named by outcome (`ok`, `timeout`, `empty`, `error`, `interrupted` and others), and
up to 4000 more in `logs/audio/rejected/` (`keep_rejected`, about 300 MB). Beside each request's WAV
there is a small note (`.json`) with the prompt given to Whisper, the language and the input gain;
that prompt lists the names of your folders, programs and open windows. They stay on the PC and are
never committed; `[logging] save_audio = false` turns them off.

Attempts where the wake word did not fire are logged too (here, not in `conversazione.md`), as `near_miss` with the highest score
reached: `benchmark_latency.py --log` lists them, to see how far to lower the threshold.

A near miss also gets a second chance (`wakeword.second_chance`): Whisper listens to the last
1.5 s, about 130 ms on the GPU with turbo (more with large-v3) and only at those moments, and if
it hears the name CLAUDIO wakes anyway. The row says what it heard (`second_chance`,
`second_chance_ms`), so the near misses that were not the wake word are visible too.

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
`--beam 1` Whisper tries one hypothesis instead of 5, and `--model large-v3-turbo` uses the
faster model CLAUDIO had until 29/10 (`download_models.py --whisper large-v3-turbo` if missing).

`compare_stt.py` compares speech-to-text engines on the same saved requests and counts how many
each one turns into a known command, with its median and p90 time:

```powershell
uv run python scripts/compare_stt.py                                   # turbo vs closed list
uv run python scripts/compare_stt.py --engines turbo large-v3 lista parakeet --details
```

`nomi` is turbo with the names on your PC first in its prompt, as `[stt] vocabulary` does live
(the most recent folders of Desktop and Documents, two levels deep, and the shortcuts of the
Desktop and the Start menu: "Cartelle: Tesi, PSS. Programmi: Rocket League."), so short names
are written right; `--engines frase nomi --details` shows what the names change. Your own words
go in `[stt] vocabulary_words`.

`lista` is the closed list CLAUDIO uses live (`stt.closed_list`): when no command understands
the best transcription, Whisper proposes 5 and the first known command wins. `parakeet` needs
`uv pip install onnx-asr` (NVIDIA Parakeet TDT 0.6B v3, downloaded on first use). The last line
says how many requests the wake word verifier (`wakeword.verify`) would drop: a wake is kept
only when Whisper heard the name at the start or the request is a local command, so "Hey, ciao,
come stai?" ends in silence and, from v0.2.0, never costs a Claude call.

When Whisper is unsure of a request that goes to Claude (a segment's `avg_logprob` below
`stt.unsure_logprob`, -0.2), Claude is told so in the request's tag, with up to
`stt.alternatives` (2) other transcriptions from Whisper's n-best: `[it | unsure: "Apri la
cartella Tesi."] Apri la cartella desi.` The system prompt asks Claude to work out what was
most likely said. The n-best of the closed list is reused; otherwise one more Whisper pass
(~0.2 s) runs, only on these requests. `alternatives = 0` keeps only the note.

To measure how well CLAUDIO understands *you*, record a test set once and score it:
`stt_testset.py` shows the 40 sentences of `scripts/stt_frasi_it.txt` (what you really say to
Claudio) one at a time and records each from every microphone at once, without gain, into
`data/stt_testset/`; `stt_bench.py` then gives, for every microphone, the word error rate (WER),
the commands understood and the time per sentence of what CLAUDIO does today (`live`: its gain,
its prompt, Silero's cut) and of its variants (`grezzo` no gain, `frasi`/`vuoto` other prompts,
`beam1`, `large`/`turbo` other models, `novad`, `parakeet`); with `--details` every wrong
sentence, and for an empty one why (Silero heard no voice, Whisper took it for silence or a
known hallucination). Close CLAUDIO first.

```powershell
uv run python scripts/stt_testset.py          # Enter, read the sentence, Enter (r = again, q = stop)
uv run python scripts/stt_testset.py --controlla    # recordings mute or cut at the end
uv run python scripts/stt_testset.py --rifai 10 25  # record those sentences again
uv run python scripts/stt_bench.py --details
uv run python scripts/stt_bench.py --variants live live+beam1 parakeet
uv run python scripts/stt_bench.py --set data/stt_testset_2   # another set
```

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
