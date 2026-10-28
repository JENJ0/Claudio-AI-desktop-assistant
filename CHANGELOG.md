# Changelog

## [Unreleased] - Jack at his pace without RVC

### Added
- `jack_speed` works with `jack_rvc = "none"` too: a streaming WSOLA time stretch in the
  worker (`TimeStretch`) slows Qwen3's audio (0.9 = 10% slower) and keeps its pitch, chunk by
  chunk, with no extra wait for the first audio.

## [Unreleased] - Jack speaks with Qwen3 alone

### Changed
- `[tts] jack_rvc` default `"none"` (new value): Qwen3 alone, no RVC. JenJo's A/B (28/10,
  `jack_variants.py` C/F/T/Q): only Q had the right stress and melody on every sentence; RVC
  (even with `jack_pitch_fix`) brought the robotic tone and the wrong stresses. The timbre is
  less Jack's and `jack_speed` does not apply (RVC did the slowing) until Qwen3 is retrained
  on more of Jack's lines.

## [Unreleased] - no needless confirmations

### Added
- `web_search(query, browser?)` (level 1): opens the browser on the search results page
  (`[pc] web_search_url`, Google by default, https only). The query only goes into the URL,
  encoded; `browser` (Chrome, Edge, Firefox, Brave, Opera) is found in the Windows "App Paths",
  otherwise the default browser opens. "Apri Chrome e cerca X" no longer types the search
  (`type_text`, level 2, "Confermi?") and no longer loses the first word while Chrome starts.
- `open_website(url, browser?)` (level 1, https only; "youtube.com" and http:// become
  https://, `file:`/`ms-settings:` and words are refused). The prompt now prefers a tool that
  does the whole job to driving a window with typing and keys.

### Fixed (the general cause, not only Chrome)
- Typing right after an app was started went to whatever window was in front, before the new
  window existed (only "Moto" was typed): `type_text` now waits up to 3 s for the window of an
  app started in the last 6 s (or for the window it names), focuses it and lets it settle.

### Fixed (JenJo's log, 26/10 00:25)
- Haiku asked "Confermi?" in words without calling the tool: the prompt now says to call the
  tool in the same reply and never ask for a yes itself.
- "Confermo." said after the conversation window closed woke the detector and was dropped as a
  false wake: a phrase of at most 4 words within 45 s of a reply that asked something is its
  answer (`responder.answer_expected`).
- An empty phrase in the conversation window (noise Whisper drops, like "Grazie a tutti")
  closed it while a question waited: the window now stays open once more.

## [Unreleased] - Whisper hears names and whole phrases

### Added
- `[stt] vocabulary`: the names on the PC go first in Whisper's prompt (`stt/vocabulary.py`):
  the 25 most recent folders of Desktop and Documents (two levels, no generic or long names,
  "TESI" written "Tesi") and 12 shortcuts of the Desktop and Start menu, read in a thread every
  10 minutes, plus `vocabulary_words`. Never in the wake word's quick look.
  `compare_stt.py --engines nomi` measures it on the saved requests.
- A phrase left halfway (Whisper's "...", or a last word such as "e", "la", "cerca": "Apri
  Chrome e cerca..." on 26/10) waits up to `[vad] continue_s` (2 s) for the rest; both parts are
  transcribed together. Nothing more said: the first text is answered, without a second pass.
  `continued_from` in `metrics.jsonl`.
- The open windows' names go in Whisper's prompt too ("Finestre aperte: Blocco note."), read
  every 3 s (`[stt] vocabulary_windows`, 8): Store apps such as Blocco note have no shortcut and
  came out as "brocconote", "bacanoate" (26/10).
- Local commands also match by sound (`router/phonetic.py`): Whisper's capital letters are read
  as Italian letter names ("Claudio QDT" = cu-di-ti = "chiuditi", "QD" = "chiudi") and spellings
  that sound alike meet ("kiudi", "chiudi"). On JenJo's 294 saved requests it changes only the
  two "QDT"s.

### Fixed (logs of 26/10 evening)
- CLAUDIO's own voice coming back into the microphone stopped the reply when Whisper misheard
  it ("ciao cappiore" for "Ciao, campione", "dell'università e lì" for "dell'universo è in
  linea", "Ah, che grande vista" for "Ah, il grande mistero"). The echo check now compares the
  sounds of the words (`intake.echo_of`), not only identical words; a phrase with the name is
  never echo unless the reply says the name.
- A short phrase with the name in the conversation window ("Claudio QDT.") was dropped as the
  TV: the name makes it a request.
- Haiku's system prompt: capital letters are spelled words, nonsense words are names of open
  windows, programs or folders that sound alike.
- Prompt caching: an explicit marker on the system prompt besides automatic caching. With 6
  turns of history, every request from the 7th dropped the oldest turn, changed the prefix and
  wrote ~5,000 tokens again (0.0007 $ instead of ~0.0002 $ with Haiku).

## [Unreleased] - file and folder search

### Fixed (JenJo's logs, 25-26/10)
- "Università" did not match the folder `UNIVERSITA'` (accent, apostrophe): the walk ran its
  3 s to the end. Names and searches are now compared folded (no case, accents, apostrophes,
  `_ - .` separators) and without spaces ("BB Port Windows", "BBPortWindows" -> `bbport-windows`).
- Names Whisper spells the English way are matched by sound ("Projecti" -> `PROGETTI`,
  "roket" -> `Rocket League`); a name holding some of the words is offered as similar
  ("Project Work" -> `PROGETTO IUM`).
- Filler words left in a search ("cartella progetti") are dropped.

### Added
- `[pc] folders_refresh_s` (900): every folder name under the roots is listed in the background
  at start and again every 15 minutes. A folder is found in milliseconds wherever it is, and a
  misheard name is compared with all of them. The Windows index and the walk still run when
  the list has nothing (a folder made after the last read, files). `open_path` with a wrong
  folder name uses the same list.

## [Unreleased] - Sonnet only when needed, "Claudio" wake word, Jack less robotic

### Changed
- `ask_sonnet` tool text: Sonnet is a rare exception (hard maths or logic, code, long
  analysis or plans); Haiku answers stories, jokes, chat and explanations itself.
- New `[claude] no_offer_phrases` (storia, barzelletta, poesia, raccontami...): if Haiku still
  asks for Sonnet on such a request, or a detail phrase matches, Claudio does not ask "Lo uso?"
  and Haiku answers (log "Sonnet chiesto per una storia", metric `escalated = skipped`).

### Added
- `[tts] jack_pitch_fix` (off until JenJo's A/B): RVC pitch cleanup against Jack's
  occasional robotic tone. Streamed first sentence: Applio 3.7.0's realtime pipeline bucketed
  the pitch with Hz limits on mel values (~6 buckets, ~2 semitones above training), now the
  training scale. Whole sentences: octave errors folded back, 1-2 frame unvoiced holes in a
  vowel filled, median-of-3 jitter filter (`smooth_f0` in `tts/jack_worker.py`).
- Worker `rvc: "none"` (Qwen3 alone) for diagnosis.
- `scripts/jack_variants.py`: variants F (pitch fix), T (F + temperature 0.8), Q (no RVC);
  `--from-log` now compares C/F/T/Q; the report measures every reply's pitch (octave jumps,
  flicker, melody spread) and lists the suspicious replies to listen to first.

### Fixed (JenJo's test, 25/10: "Claudio" alone fires rarely)
- The "claudio" model fires late, on the word after the name when it is said in one breath:
  Whisper's check got only the last 1.5 s, the "C" was cut off and it heard "l'audio" (8 of the
  rejected checks in the logs), so JenJo had to repeat. The check now hears the last
  `WAKE_CHECK_S` 2.5 s; the request keeps its 1.5 s lead-in.
- After the "claudio" model, a transcript starting with a word at most 2 edits from "claudio"
  ("l'audio", "audio", "fraudio") wakes too, but the request is verified (name or command). Not after "hey claudio", not in the middle
  ("alza l'audio"), not "radio" or "studio".
- A strong "claudio" (`[wakeword] unclear_min` 0.6) that Whisper hears as nothing or "Grazie."
  (its guess on a lone word: half of JenJo's "Claudio" on 26/10, at 0.86-0.95) now beeps and
  listens; the request is verified (name or local command), else dropped in silence (JenJo
  chose this over retraining).

## [0.3.2] - screen and terminal

### Added
- `look_at_screen` (level 1, last resort): a screenshot of the active window or of the screen,
  shrunk to `[pc] screenshot_side` (1024 px) JPEG, sent to Haiku as an image in the tool result.
  Never of a password manager, never saved.
- `run_command` (level 3, always): one PowerShell command per reply (`-NoProfile
  -NonInteractive`, encoded, never elevated, `command_timeout_s` 30 with the process tree
  killed, output cut at `command_output_chars` 2000). Deny list in `pc/terminal.py`, checked
  before asking and again before running. The purpose is spoken, the exact command is shown in
  the chat window (opened if needed) and in `conversazione.md`; only a spoken "confermo" runs it.
  `[pc] commands = false` removes the tool; Claudio running as administrator switches it off.
- "Ci guardo" before a slow action (search, read, screenshot) when Haiku said nothing yet,
  "Eseguo" after "confermo"; `tool_ms` in `metrics.jsonl`.

### Fixed (JenJo's v0.3.1 test, 24/10)
- File search read the Windows index display path, which on an Italian Windows is
  `C:\Utenti\...`: every result looked outside the home folder and was dropped, and the folder
  walk never ran. It now reads `System.ItemUrl` and walks whenever nothing allowed is left (25/10).
- A misheard name ("desi", "attesi" for "Tesi") finds nothing exact: the walk now offers
  similar folder and file names, and Haiku opens the only one or asks which (25/10).
- Replies with tools were long: Haiku narrated each step and read paths aloud. It now stays
  silent before a tool, says a few words after it and names folders without their path (25/10).
- Games from Steam or Epic ("Apri Rocket League") open: `.lnk`/`.url` shortcuts in the Start
  menus and on the desktop count as apps. Sentences of two tool steps keep a space between them,
  so the voice pauses ("La apro. Aperta.") (25/10).
- "Apri la cartella Progetti": a name Claude puts in the wrong folder is looked for in the
  user folder, Desktop, Documents, Downloads (also under OneDrive).
- Typing in Notepad needs no yes (`[pc] free_typing`); text of 40+ characters or with new
  lines is pasted (Ctrl+V, the clipboard is put back) instead of typed key by key.
- Haiku writes at once when asked to write something (no "che tema vuoi?") and says what it is
  about to do before "Confermi?".
- Short follow-ups in the conversation window ("Allora scrivi.") are answered, and a reply with
  a question near its end ("Che tema vuoi? Dimmelo e la scrivo.") waits for the answer.
- Level 3 question shorter: "Per procedere dì: confermo."
- (2nd test, 24/10 03:09) "Hey Claudio, confermo" said after the follow-up window closed now
  answers the pending action (it stays armed for 60 s after silence); a longer new request
  instead drops the action and is answered.
- Haiku no longer "admits" that a command it ran never ran: the history keeps a short note of
  the actions that really ran ("[Claudio ran: run_command ok]").
- Any folder or file is found by name (24/10: "Tesi" in Desktop\UNIVERSITA'): when the Windows
  index finds nothing the folders are walked breadth first, Desktop/Documents/Downloads first,
  skipping big trees (miniconda3, npm-cache...); `open_path` with a wrong folder looks the
  name up the same way.

## [Unreleased] - faster startup

### Changed
- Claudio listens much sooner (JenJo: "ci mette troppo tempo ad avviarsi"). Jack's worker
  (8-30 s) starts first and loads in the background instead of blocking everything; Whisper,
  wake word, voice and Claude load at the same time (`[startup] parallel`). A reply asked
  while Jack is still loading waits for him up to `[tts] jack_first_wait_s` (30 s), then the
  fallback voice speaks it; "basta" stops the wait.
- Once everything is loaded Claudio says "Sono pronto" (`[startup] announce_ready`), with the
  notices of a degraded start before it, between two requests, never over one.
- `console.log`: `Avvio: whisper 3.1 s, ...` (each loader) and `Avvio completo in N s`.

## [Unreleased] - 0.3.2 (understanding and interrupting)

### Added
- Misheard requests are understood (JenJo: "deve capire quello che ho detto veramente anche se
  Whisper capta male"). When Whisper doubted a request for Claude (`[stt] unsure_logprob`
  -0.2: on 25/10 the misheard ones scored -0.25..-1.1, the right ones -0.01..-0.17), Haiku
  gets `unsure` and up to `[stt] alternatives` (2) other transcriptions in the request's tag
  (~10-25 tokens, ~0.000003 $). The closed list's n-best is reused; otherwise one more Whisper
  pass (~120 ms), only on these requests. `stt_unsure`, `stt_alternatives`, `alternatives_ms`
  in metrics.jsonl.
- System prompt: real examples of Whisper's mistakes ("desi"/"attesi" for "Tesi", "pure sono"
  for "che ore sono", the name mangled at the start, noise as short English); Haiku acts on the
  likely meaning and asks only when the readings lead to different actions (~200 cached tokens).
- `search_files`: when the Windows index finds nothing (on 25/10 it missed Desktop\PROGETTI
  and Desktop\UNIVERSITA'\TESI, then found them later) the folders are walked, breadth first,
  and with no match the 5 names that sound most alike are listed ("desi" -> TESI).

### Fixed
- A reply cut short ("basta", or a new request said over it) stays in the conversation with a
  `[cut off]` note: "continua la storia" and "ripeti la storia" used to get "non ho nessuna
  storia" because only finished replies were kept.

### Changed
- Talking over a reply interrupts it (JenJo: "mentre sta rispondendo se gli parlo mi deve
  sentire"): a loud, close voice (`barge_in_min_dbfs` -30, stricter VAD 0.6, at least 300 ms)
  lowers the reply to `duck_volume` at once; when Whisper hears words (two or more, the name or
  a command) the reply stops and the phrase, heard to its end, is the next request
  (`wake_word = "barge_in"` in metrics). "basta" still only stops. Noise restores the volume;
  words mostly from the reply itself are its echo and are ignored. `barge_in_mode` = duck |
  stop | confirm. Slices checked every 1.2 s instead of 2 s.
- Long phrases are no longer cut at a short pause: after 1.5 s of speech the phrase needs
  800 ms of silence to end (`long_end_silence_ms`, `long_after_s`); commands still end after
  400 ms. `max_utterance_s` 8 -> 15 s.

### Added
- A reply cut by a new request goes on after it is answered (JenJo, 25/10): "raccontami una
  storia" ... "che ore sono?" ... "Sono le 19 e 44. Dicevo: ..." from the sentence that was cut.
  The cut reply is still read to its end from Claude (at most `max_tokens`, no extra call); not
  resumed after "basta", after a closer ("ok, va bene, grazie"), after a question it was asking
  ("Lo uso?"), or after `[conversation] resume_s` (60 s; 0 = never).

### Fixed (JenJo's test, 25/10)
- A noise Whisper writes as an interjection ("Mm-hmm.", "Ehm, mh.") no longer stops the reply:
  interjections do not count among the two words needed to take over.
- The level of the voice heard over the reply is logged (`barge_dbfs` in metrics.jsonl, and the
  dBFS of each phrase ignored as too quiet) to tune `barge_in_min_dbfs` on real data; what is
  heard but not taken as a request is logged at INFO.
- `scripts/compare_stt.py --engines turbo frase large-v3 --details`: the `frase` engine is turbo
  with a natural-sentence prompt (`--prompt` to try another).
- Whisper's prompt is now a few natural sentences instead of a list of commands: on JenJo's 23
  requests it fixed "pure sono" / "Chiaro son" -> "che ore sono", "Il punto è" -> "raccontami",
  "Projecti" -> "Progetti", at the same speed (108 ms). large-v3 understood about as much at
  twice the time (232 ms): turbo stays.

## [Unreleased] - personality

### Added
- CLAUDIO.md personality (`config/CLAUDIO.md`, copied to `data/CLAUDIO.md` at first start,
  never overwritten): Handsome Jack, "full but helpful", free swearing (JenJo's choices). Read
  once and sent after the system prompt (frozen, cached); HTML comments are stripped, the text
  is cut at `[claude] personality_max_chars`.
- `[tts] jack_language` ("Auto" or "Italian") for Qwen3: try "Italian" if accent or pace drift.

### Changed
- Replies: one short sentence, two at most; questions to JenJo in a few words.
- Language: Italian unless the request is plainly English or JenJo asks for English; never two
  languages in one reply. `[stt] other_confidence` 0.8 -> 0.9.
- Level 2 confirmation is just "Confermi?" (`[pc] confirm_brief`); level 3 still says what runs.

## [Unreleased] - 0.3.1

### Jack's pronunciation (after JenJo's test, 23/10)
- Italian only for now (JenJo, 24/10): `[stt] languages = ["it"]` (no language detection, no
  English replies; the English voices are commented out), and Qwen3 is always told "Italian"
  (with "Auto" it guessed the language per sentence: Spanish/English stresses).
- `[tts] jack_temperature` (Qwen3 sampling) and `jack_rvc`:
  "blocks" (as before), "sentence" (Applio's offline pipeline on the whole sentence, rmvpe:
  fewer mangled words) or "auto" (blocks for a reply's first sentence, whole for the rest).
- Text for Jack goes through `tts/spoken_text.py`: symbols (%, €, °, &, +), acronyms (PC, USB,
  PDF), Markdown, links and emoji become Italian words before the numbers.
- The speaker renders up to 16 pieces ahead of playback (was 2), so whole-sentence RVC is
  ready before the sentence is due.
- `scripts/jack_variants.py`: the same replies in 5 settings, with WER, first audio and pauses;
  `--words` speaks candidate spellings of hard words.
- JenJo's pick by ear (24/10, variant C): `jack_temperature = 0.7`, `jack_rvc = "auto"`.
- `[tts.jack_words]`: respellings for words Jack said wrong ("PC" heard as "pace", English
  words like "file" read in English): `PC = "pì cì"`, `file = "fàiil"`, `download = "dàunlòud"` (JenJo's picks by ear), `Claude = "clòd"`, `desktop = "dèsk-top"` (25/10, from JenJo's real replies replayed with `--from-log`), ...
- Fixed: the whole-sentence RVC warm-up on silence gave a NaN pitch and a CUDA assert at worker
  startup; the pitch is now clamped (NaN = unvoiced).

### Added
- Claude acts on the PC (`src/claudio/pc/`, `[pc]` in `config/claudio.toml`): Haiku's tools
  `open_app` (Start menu apps, classic and Store), `open_path`, `search_files` (Windows Search
  index, folder walk as fallback), `list_windows`, `read_window` (UI Automation text, cut at
  `read_chars`, marked as data), `focus_window`, `press_keys`, `type_text`, `close_window`,
  `move_file`. Several tools per reply, up to `max_rounds` (4) round trips; the last round
  answers in words only. Tool definitions are part of the cached prefix.
- Safety levels fixed in code: level 1 runs at once; level 2 (type, close, move, any key not in
  `safe_keys`) makes Claudio say exactly what will run and ask "Confermi?"; the answer goes
  through the same checks as a local command's (short yes, voice near the microphone, voice
  only). JenJo's choice "Equilibrata": harmless keys without asking, apps from the Start menu,
  files only inside the user folder (`roots`), never overwritten, no programs opened as files,
  password managers never read.
- Every action is logged with level, outcome and `tool_ms`; `metrics.jsonl` gets `tools`,
  `conversazione.md` an "Azioni" line.
- Dependencies (Windows only): `pywinauto`, `psutil`.

### Fixed after JenJo's test of v0.3.0 (23/10)
- Self-activation (a 0.34 detection right after a reply): a wake below `[wakeword]
  confirm_below` (0.45) wakes only if Whisper hears the name or a local command.
- "Spiegami in dettaglio..." was answered by Haiku: `[claude] offer_phrases` propose Sonnet
  at once ("Lo uso?"), without a Haiku call; the tool description mentions detail too.
- Jack is the default voice (`engine = "jack"`, `effect = "none"`), 10% slower
  (`[tts] jack_speed = 0.9`: stretched before RVC, RVC raises the pitch back).
- Microphone: `input_device = "auto"` listens to every microphone for 0.6 s and uses the one
  that hears the most (Windows' own unless another is 10 dB louder or it is silent). After
  60 s below -50 dBFS it listens to the others with the stream still open and moves only to
  a louder one, without re-scanning devices (second test: the headset sends digital silence
  between phrases, and the re-scan broke the speaker's stream, PaErrorCode -9988).
  `input_device = ""` follows Windows' default microphone while running.
- "Claudio" alone wakes again: the `claudio` model is back (`threshold = 0.25`, `confirm =
  true`): Whisper checks every detection, so its noise costs a Whisper pass, not a wake.
  One "Claudio" fires the model on many frames: while Whisper checks, the rest of the burst
  is dropped (no extra checks, near miss rows or WAVs).

## [Unreleased] - 0.3.0

### Changed
- The rules that decide whether a phrase is a request (false wake, conversation window, retry,
  closed list) moved from `app.py` to `src/claudio/intake.py` (`RequestFilter.decide()` returns
  answer, relisten or drop and why). `Assistant` is the audio state machine only.
- Phrases ending with "?" skip the closed list (~100 ms): a question is for Claude.
- A level 2 "yes" is at most 3 words ("sì, va bene"): "sì, ma poi è andato via" is the TV.
  An answer to "Confermi?" quieter than `[noise] answer_min_dbfs` is not counted.

### Added
- Noise filter (`[noise]` in `config/claudio.toml`): a phrase no command understands does not
  reach Claude when it is short and Whisper was unsure of all of it, or when its speech is
  quieter than `min_speech_dbfs`. Drops are logged as `outcome = "noise"`, `drop_reason =
  "noise"`; every request now logs `stt_no_speech` and `stt_logprob` to tune the thresholds.
- `logging.retention_days` also deletes old `logs/sessions/`, `logs/audio/` WAVs and rotated
  `logs/metrics.jsonl.*` at startup.

## 0.2.2

### Added
- Sonnet when a request needs reasoning: Haiku hands it over with the `ask_sonnet` tool (after a
  short "Ci penso un attimo."), or JenJo asks directly ("pensaci bene", "think hard").
  `[claude] smart_model`, `smart_effort`, `smart_max_tokens`, `escalate`, `smart_phrases`; both
  calls are counted by the spending counter. Sonnet without prices in `pricing.toml` stays off.
- Prompt caching (automatic `cache_control`): from the second turn of a conversation the history
  is read from the cache. `[claude] cache`.
- Claude knows the local date and time (in the user message, after the cached prefix).
- Writing to Claudio from the chat window: a text box under the chat. The window puts the message
  in the `inbox` table of `data/claudio.db`, the assistant answers it between spoken requests
  (written reply only, `[chat] speak_typed`). `POST /api/send` needs JSON, no foreign Origin and
  the random token of the page. Level 2-3 actions written there need a spoken confirmation.

### Fixed (JenJo's test, 23/10)
- After Haiku's "Ci penso un attimo." the voice waited for Sonnet (no space between the two
  replies kept the first sentence open).
- Sonnet never runs without a yes (JenJo, 23/10): Haiku proposes it with `ask_sonnet`, Claudio
  asks "Lo uso?"; "sì" -> Sonnet answers the original question, "no" -> Haiku does. "pensaci
  bene" still asks Sonnet directly (that is the request). Haiku keeps short or unclear requests.
- In the conversation window every phrase of 3+ words is answered until "basta" or silence
  (`[conversation] only_addressed = false`; true brings back the TV filter).
- The chat window says why a message could not be sent; the server logs refused messages.

### Changed (latency)
- The HTTPS connection to the API opens at the wake word (free models lookup) and stays open 60 s
  (`[claude] keepalive_s`; httpx closed it after 5 s): no TCP/TLS handshake on the request.
- The first piece of a reply may end at a comma once it is 25 characters long
  (`[tts] first_clause_chars`), and the prompt asks for a short first sentence: the voice starts
  before the whole first sentence has arrived.

## 0.2.1

### Added (chat window)
- `claudio --ui` opens the chat in its own window (pywebview/WebView2 over a local FastAPI
  server on 127.0.0.1, read-only, Host header checked): every conversation, search across all of
  them, Claudio's answers with route/model/cost/latency, today's spending. Updates live while
  Claudio runs. The window and the assistant are separate processes sharing `data/claudio.db`.
- `src/claudio/chat/`: `ChatStore` (SQLite, WAL) fed by `MetricsWriter`; also the base for the
  Phase 4 memory.
- `logging.retention_days` (30): the chat database forgets older messages at startup (privacy).
- The running assistant opens the chat window by voice: "apri la chat", "chat", "fammi
  scrivere" (and "chiudi la chat"), local commands at no cost (`action = "chat"`,
  `src/claudio/chat/launcher.py`). The window is `claudio --ui` in a child process and closes
  with Claudio; `--ui` alone still works, read-only. It logs to `logs/chat-window.log.jsonl`.
- Spoken notice at startup when Claudio runs degraded: Jack's worker did not start, or Whisper
  fell back to the CPU.

### Fixed
- Jack's worker died at startup (`ImportError: cannot import name 'Number' from 'numbers'`):
  `tts/numbers.py` shadowed the standard library for the worker script, so Claudio spoke with
  Kokoro. Renamed to `tts/number_words.py`; a test keeps stdlib names out of `tts/`.
- The CUDA libraries are a default dependency group (`gpu`): a plain `uv sync` removed the old
  `--extra gpu` and Whisper ran on the CPU (STT 1.5-2.9 s instead of 0.1-0.3 s). CI skips them.
- Jack read some numbers in Spanish ("22", "51" with language Auto): digits are now spelled out
  in Italian before the worker (`tts/number_words.py`).

## 0.2.0

### Fixed (code review)
- `Assistant.history` keeps the last 200 interactions only (was unbounded).
- `console.log` (5 MB x 3 copies) and `logs/metrics.jsonl` (5 MB, 3 copies) rotate.
- CI: `permissions: contents: read`; actions pinned to a commit SHA.
- `Responder` and `Speaker` protocols declare everything `Assistant` uses: no more `getattr`.
- "go" alone no longer counts as a yes.

### Added
- Jack's voice (`engine = "jack"` in `[tts]`): Qwen3-TTS 0.6B fine-tuned on Handsome Jack's
  Italian lines + RVC, in a separate worker process (`src/claudio/tts/jack_worker.py`, own venv
  built by `scripts/setup_jack_worker.ps1`: torch stays out of Claudio's venv). The audio is
  played as it is generated (first chunk ~0.3 s). If the worker cannot start, dies or does not
  answer in time, Kokoro/Piper speaks and the worker restarts in the background. "basta" stops
  the generation too. English replies keep Kokoro/Piper. `scripts/try_jack.py` is the manual test.
- Claude answers what no local command understands (`src/claudio/brain/`): Haiku 5.5 by
  default, streamed sentence by sentence to the voice, in the language of the request (or
  Italian when that language has no voice). Short spoken style, no Markdown. `[claude]` in
  `config/claudio.toml` (model, `max_tokens` 300, effort low, thinking off, timeout 10 s,
  history of 6 turns reset after 5 min of silence); `enabled = false` keeps CLAUDIO local-only.
- The API key comes only from `.env` (`ANTHROPIC_API_KEY`, see `.env.example`), never logged.
- Every call goes through the spending counter: blocked at 100 % of the daily budget (a spoken
  sentence, no request sent), the 80 % warning said before the reply once a day, tokens
  (cache read/write included), model and TTL logged and stored in `data/spend.db`.
- "basta" while Claude speaks closes the API stream; the tokens already generated are still
  counted (output estimated from the text). "ripeti" repeats Claude's reply too.
- Missing or invalid key, network errors, API errors and refusals become one short spoken
  sentence: Claudio keeps working. A model missing from `pricing.toml` switches Claude off.

### Changed
- Conversation window: a phrase that no command understands goes to Claude only when it sounds
  addressed to it (a question, a request verb such as "dimmi" / "spiegami", or the name) and has
  at least `[claude] conversation_min_words` words; anything else still ends the window in
  silence, so the TV costs nothing.
- Closed list only for requests of at most `[stt] closed_list_max_words` (5) words: longer ones
  go straight to Claude, ~150 ms sooner and without being turned into a similar command.
- The wake word verifier still runs before Claude: a false wake never reaches the API.

- Every run gets a folder `logs/sessions/<date>_<time>/` (e.g. `2026-10-20_16-32`) with
  `console.log` (the console, line by line) and `conversazione.md` (each request: time, wake
  word, what was heard, Claudio's reply, latency, Claude's tokens and cost). The newest 30 are
  kept (`[logging] keep_sessions`). The reply text is also in `metrics.jsonl` (`reply`).
- After a Claude reply that ends with a question ("Da dove parti?"), the answer in the
  conversation window goes to Claude even when it is one word ("Milano.").

### Fixed
- A question said after a pause ("Hey Claudio" ... beep ... "come ti chiami?") was dropped as
  a false wake: its transcript has no name. Now, when the request is not a command, a quick
  Whisper pass on the wake word audio (~100 ms) keeps the wake if the name is there
  (`wake_check`, `wake_check_ms` in `metrics.jsonl`). "Hey, ciao, come stai?" is still dropped.

## 0.1.2

### Added
- Spending counter, ready for Phase 2: prices and cache multipliers in `config/pricing.toml`
  (Haiku 5.5, Sonnet 5.5, Opus 5.5; cache writes 1.25x at 5 min and 2x at 1 h, reads 0.1x or the
  model's own price), a SQLite ledger in `data/spend.db` with one row per call (model, TTL,
  input/output tokens, `cache_creation_input_tokens`, `cache_read_input_tokens` and their 5 min /
  1 h split, cost), a daily budget (`[cost] daily_budget`, 1 USD) announced once a day at 80 %
  and blocking Claude at 100 %. Local commands never count and keep working. Voice command
  "quanto ho speso oggi" / "how much did I spend today". Tested with fake `usage` objects only.
- Safety levels with spoken confirmation: `level = 2` in `commands.toml` asks the command's
  `confirm` question and listens for the answer without the wake word; only a clear yes runs
  the action, while a no, silence or anything unclear cancels it. `level = 3` accepts only an
  explicit "confermo" / "I confirm". Every question and answer is logged. "chiudi la finestra"
  (Alt+F4 on the window in front) is the level-2 test command.
- Conversation window: after a reply CLAUDIO listens for 6 s more without the wake word
  (`[conversation] window_s`, 0 = off). It closes on silence, on "basta", or on anything no
  command understands (no "Questo non lo so fare" for the TV).
- Barge-in: while speaking, a second Silero VAD and a quick Whisper pass listen for "basta" /
  "stop" / "zitto" and cut the reply short (the audio is queued in 0.1 s slices). An
  interrupted confirmation question cancels its action. `[conversation] barge_in`.
- "ripeti" / "non ho capito" says the last reply again, in its language and voice.
- Media keys: play/pause, next and previous track ("metti in pausa la musica", "canzone
  successiva", "salta 3 canzoni", "canzone precedente").
- Wake word verifier (`wakeword.verify`, on): a wake is kept only when Whisper heard the name
  among the first words or the request is a local command; otherwise it ends in silence
  (`outcome = false_wake`). Fixes "Hey, ciao, come stai?" (0.98) answering "non ho capito".
- Closed list (`stt.closed_list`, on): a request no command understands gets one more Whisper
  pass with 5 hypotheses, and the first that is a known command wins (`closed_list_from`,
  `closed_list_ms` in `metrics.jsonl`).
- `scripts/compare_stt.py`: the saved requests through Whisper turbo, large-v3, the closed list
  and Parakeet TDT 0.6B v3 (onnx-asr, optional), with commands understood, median and p90 time,
  and how many the wake verifier would drop.

### Fixed (robustness)
- An unplugged microphone logged a warning every second: now one warning, one every ~30 s while
  it stays missing, and a line when audio flows again.

### Fixed
- A reply stopped before its end (an error in the audio output) could hang the speaker forever
  while the next sentences were queued.

### Changed
- Fast Italian heard as English (live test of 19/10: "cue the calculator", "come with me"):
  English must now be 80 % sure (`stt.other_confidence`), and a request heard as English that no
  command understands is transcribed again in Italian (`stt.retry_in_first_language`, only then,
  about 120 ms). The `metrics.jsonl` row keeps the first text (`retried_from`, `retry_ms`).
- Version 0.1.2 (`claudio --check` printed 0.1.0 during v0.1.1).

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
- The message for a missing wake word model now says to copy a model you trained into
  `models/wakeword/` (it used to point only at `download_models.py`, which cannot fetch it).
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
