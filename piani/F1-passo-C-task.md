# F1, Passo C: pulizia piccola, task

> **Per chi esegue:** `superpowers:subagent-driven-development`, un task alla volta, con `superpowers:test-driven-development` (T1 e T2: test visti fallire prima del codice) e `ponytail`. Implementer su Sonnet, revisori indipendenti su Opus (`superpowers:requesting-code-review` dopo ogni task); a fine passo `/code-review` sull'intero diff e `superpowers:verification-before-completion`. I passi usano `- [ ]`.

**Obiettivo:** il Passo C della roadmap (C1-C7): privacy dei log, prezzo del modello `[guess]`, dipendenze dichiarate, codice morto via, Claudio solo italiano senza effetti voce né catalogo inglese.

**Architettura:** nessun modulo nuovo, una funzione nuova (`drop_old_lines`). Il resto è cancellazione: ogni task toglie insieme il codice, i suoi test (solo quelli che provavano la funzione tolta), le chiavi di configurazione e il suo testo nei due README.

**Tecnologie:** Python 3.11-3.13, pydantic-settings, faster-whisper, pytest.

**Spec:** `piani/ROADMAP.md`, sezione 1, Passo C; report locale `reports/Connettere Claudio.md` (S8, S9, M4, M6, M7, M8, M9); scelte di JenJo del 12/11 in chat (sotto).

**Scelte di JenJo (12/11):** C5 rimozione **totale** (anche il parametro `language`); prompt di sistema solo italiano, con cassette da rifare dopo il suo ok sul tetto di spesa; C1 copre anche i file di log vivi; dell'avvio automatico resta il Python (`claudio.autostart`), via il `.ps1`; `pywin32` dichiarato in `pyproject.toml`; codice e testo dei README si tolgono nello stesso task.

## Vincoli globali

- Prima di cercare nel codice: `graphify explain "<X>"` o `graphify query`, poi solo le righe indicate. Regole: `.superpowers/sdd/regole/implementer-rules.md` e `reviewer-rules.md`. Gli implementer **non committano**.
- Comandi: `uv run python -m pytest`, `uv run python -m mypy src`, `uv run ruff check .` (il lanciatore `pytest.exe` di `.venv` è rotto: «uv trampoline failed»).
- **0 $ di API fino a T12.** Nessuna chiamata vera nei test; sonde e test con `Recorder().ctx()` (una sonda può premere tasti veri).
- **Test della cancellazione:** si toglie un test solo se provava la funzione tolta; un test che la usava solo come appoggio si riscrive.
- `Settings` ha `extra="forbid"`, i modelli annidati ignorano le chiavi ignote: togliere un campo in cima (es. `language`) richiede di toglierlo da `config/claudio.toml` nello stesso task.
- Messaggi di log in italiano. Base: suite 4566 passati, 4 saltati; gate 39/0/2 (12/11).

## Da guardare (nessun test dei task lo copre da solo)

1. **Il `.env` di JenJo ha `CLAUDIO_TTS__EFFECT=none`:** tolto `effect`, Claudio deve partire lo stesso. Test in T5.
2. **Claudio già acceso e una seconda copia che parte:** il taglio di `claudio.log.jsonl` non deve rompere né svuotare il log della copia accesa. Test in T1.
3. **Una riga rotta da un crash in mezzo alle vecchie:** si taglia lo stesso; un file di sole righe illeggibili resta intatto. Test in T1.
4. **Parole inglesi che JenJo dice in italiano:** «stop», «play», «chat» restano comandi; «ok» e «confermo» restano sì, «stop» e «lascia stare» restano no. Test in T8 e T9.
5. **Il prompt di Whisper:** spostato da `[stt.prompts] it` a `initial_prompt`, non deve sparire in silenzio (una chiave annidata ignota non dà errore). Test in T7.

---

## Task 1 (C1, S8): i log dimenticano anche i file vivi

**File:** `src/claudio/session_log.py` (`purge_old_logs`, nuova `drop_old_lines`), `src/claudio/__main__.py` (`main`). Test: `tests/unit/test_session_log.py`.

**Produce:** `drop_old_lines(path: Path, cutoff: float) -> int`: righe tolte; 0 se il file manca, non ha righe vecchie o non si può sostituire.

- [ ] Test, visti fallire: `test_drop_old_lines_cuts_the_old_head` (righe come `claudio.log.jsonl`, `"ts"` ISO con `+00:00`: 2 vecchie e 1 recente → 2, resta solo la recente, byte identici); `test_drop_old_lines_reads_metrics_rows` (`"ts"` float); `test_drop_old_lines_drops_a_broken_line_among_old_ones` (vecchia, `garbage{`, vecchia, recente → 3); `test_drop_old_lines_keeps_unreadable_files` (solo righe non JSON → 0, file intatto); `test_drop_old_lines_leaves_recent_and_missing_files_alone` (→ 0, `st_mtime` invariato); `test_drop_old_lines_survives_a_file_held_open` (`os.replace` che solleva `PermissionError` → 0, file intatto, nessun `.tmp` rimasto); `test_purge_old_logs_removes_rotated_logs_and_trims_metrics` (`claudio.log.jsonl.1` vecchio cancellato, `.2` recente resta, testa vecchia di `metrics.jsonl` tolta); `test_main_trims_the_log_before_logging_opens_it` (`CLAUDIO_LOGS_DIR` = `tmp_path`, `setup_logging` sostituito da uno stub che fotografa il file e solleva; nella foto la riga vecchia non c'è).
- [ ] `drop_old_lines`: le righe sono in ordine di tempo, quindi si taglia fino all'**ultima** riga leggibile più vecchia di `cutoff`; le illeggibili prima di lei sono vecchie anche loro, quelle dopo restano. Lettura e scrittura in byte (`read_bytes().splitlines(keepends=True)`); scrittura su `<nome>.tmp` e `os.replace`; su `OSError` avviso, `.tmp` cancellato, 0.

  ```python
  cut = 0
  for i, line in enumerate(lines):
      ts = _line_time(line)  # float, or None when unreadable
      if ts is None:
          continue
      if ts >= cutoff:
          break
      cut = i + 1
  ```

- [ ] `purge_old_logs`: `logs_dir.glob("*.jsonl.*")` al posto di `metrics.jsonl.*` (ruotati di log, metriche, chat e supervisore) e `drop_old_lines` su `metrics.jsonl`, con una riga di log se ha tolto qualcosa. Docstring aggiornata.
- [ ] `main()`: il taglio di `claudio.log.jsonl` (o `chat-window.log.jsonl` con `--ui`) avviene **prima** di `setup_logging`, perché Windows non sostituisce un file che il nostro handler tiene aperto; il conteggio si scrive nel log subito dopo `setup_logging`.
- [ ] Suite, ruff, mypy.

## Task 2 (C2, S9): il prezzo del modello `[guess]` si controlla all'avvio

**File:** `src/claudio/__main__.py` (`build_responder`, il blocco `settings.guess.enabled`). Test: `tests/unit/test_command_tool.py`, accanto ai test di `build_responder`.

- [ ] Test, visti fallire: `test_build_responder_drops_the_guess_without_its_price` (`guess.enabled = True`, `guess.model = "claude-nope"` → `responder.fallback_guess is None` e un `ERROR` nel log con `claude-nope`); controllo positivo con il modello di `config/claudio.toml` → `fallback_guess is not None`. Stesso schema dei test vicini: `spend.db` in `tmp_path`, memoria spenta, `brain.tools.close()` nel `finally`.
- [ ] Codice come il controllo di `smart_model`: `tracker.pricing.price(settings.guess.model)` prima di creare `Guesser`; su `PricingError` → `log.error("Ipotesi di Claude spente: %s", exc)` e nessun `Guesser`.
- [ ] Suite, ruff, mypy.

## Task 3 (C3): dipendenze fissate e dichiarate (la fa il coordinatore)

**File:** `pyproject.toml`, `uv.lock` (solo tramite `uv lock`).

- [ ] `supermatt:dependency-vetting` su `piper-tts` 1.8.0 e `pywin32` 312: pagina PyPI, progetto, avvisi OSV.
- [ ] `"piper-tts==1.8.0",  # kokoro.py uses piper's private espeak-ng phonemizer (piper.voice._ESPEAK_PHONEMIZER)`.
- [ ] `"pywin32>=312; sys_platform == 'win32'",  # pc/windows.py imports win32gui, win32com, win32clipboard: until now only pywinauto's dependency`.
- [ ] `uv lock`, poi `uv lock --check` e `uv sync --locked`; `git diff --stat uv.lock` (il file non si legge). Suite.

## Task 4 (C4, M8, M9): codice usato solo dai test

**File:** `src/claudio/stt/transcriber.py`, `scripts/compare_stt.py`, `src/claudio/pc/tools.py`, `src/claudio/pc/files.py`, `src/claudio/app.py`, `src/claudio/autostart.py`, `scripts/install_autostart.ps1`, `README.md`, `README.it.md`; test indicati sotto.

- [ ] M8: `first_known` si sposta in `scripts/compare_stt.py`, il suo unico utente; via da `transcriber.py` e via `test_first_known` (`tests/unit/test_closed_list.py`).
- [ ] M9: via le chiavi `"type_text"` e `"press_keys"` di `_SAY` (mai lette). Via `files.walk_search`: il chiamante (`pc/tools.py`, ricerca della cartella) usa `files.walk(...).found`; `test_walk_search` si riscrive su `walk(...).found` con le stesse asserzioni.
- [ ] M9: via le proprietà `verify_wake`, `closed_list_max_words`, `claude_min_words`, `only_addressed`, `echo_tail_ms` di `Assistant` (lettura e setter). `app.py` legge `self.intake.verify_wake`; i test (anche `tests/unit/timeline.py`) scrivono `assistant.intake.<nome>` e `assistant.own_sound.tail_ms`. Via `test_states_are_strings`.
- [ ] Autostart: via `scripts/install_autostart.ps1` e la sua menzione in `autostart.py`; nei README restano solo `uv run python -m claudio.autostart install|uninstall|status [--dry-run]`.
- [ ] `grep` dei nomi tolti in `src`, `tests`, `scripts`, README: vuoto. Suite, ruff, mypy.

## Task 5 (C7, M4): via gli effetti voce

**File:** `src/claudio/tts/effects.py` (via), `src/claudio/tts/speaker.py`, `src/claudio/__main__.py` (`build_speaker`), `src/claudio/config.py` (`TtsSettings.effect`), `src/claudio/i18n.py` (le due voci dell'effetto), `config/claudio.toml` (righe 112-116), `pyproject.toml` (commento di `scipy`), `README.md`, `README.it.md`. Test: `tests/unit/test_effects.py` (via), `tests/unit/test_speaker.py`, `tests/unit/test_config.py`.

- [ ] `StreamingSpeaker` perde il parametro `effect` e i suoi rami: Jack va sempre in streaming quando il synth ha `stream()`. Via i test che provano l'effetto; un test che passava un effetto identità solo come appoggio si riscrive.
- [ ] Commento di `scipy`: `# Jack's loudness and filters (already required by openwakeword)`.
- [ ] Test nuovo `test_an_old_effect_line_in_dotenv_is_ignored`: un `.env` con `CLAUDIO_TTS__EFFECT=none` carica senza errori.
- [ ] README: via le frasi sul tono robotico e su `CLAUDIO_TTS__EFFECT`. `grep -rn "effect" src config README*` mostra solo usi estranei (es. `side_effect`). Suite, ruff, mypy.

## Task 6 (C6, M6): via il catalogo inglese dei log

**File:** `src/claudio/i18n.py` (via), i 19 file che importano `tr`, `src/claudio/logging_setup.py`, `src/claudio/session_log.py`, `src/claudio/supervisor.py`, `src/claudio/__main__.py`, `src/claudio/config.py` (`Settings.language`, `configured_language`), `config/claudio.toml` (riga 5), README. Test: `tests/unit/test_i18n.py` (via), `tests/unit/test_config.py`.

- [ ] Ogni `tr("…")` diventa la stringa; ogni `tr("… {x}", x=v)` una f-string. Via `TranslateFilter`, `set_language`, `configured_language`, `Settings.language`.
- [ ] Nessun byte verso l'API cambia: in italiano `tr` era l'identità.
- [ ] `grep -rn "i18n\|\btr(\|set_language\|configured_language" src tests scripts`: vuoto. Suite, ruff, mypy.

## Task 7 (C5a, M7): Whisper solo in italiano

**File:** `src/claudio/stt/transcriber.py`, `src/claudio/intake.py`, `src/claudio/app.py`, `src/claudio/__main__.py`, `src/claudio/config.py` (`SttSettings`), `config/claudio.toml`, `scripts/replay_gate.py`, `scripts/download_models.py`, `scripts/compare_stt.py`, `scripts/benchmark_latency.py`, README. Test: `tests/unit/test_retry_language.py` (via), `tests/unit/test_responder_metrics_stt.py`, `tests/unit/test_config.py`, `tests/unit/test_intake.py`, `tests/integration/test_tts_stt_roundtrip.py`.

- [ ] Whisper riceve sempre `language="it"` (costante del modulo). Via rilevamento della lingua, stato della lingua, ritentativo (`wants_retry`, `_retry`, la proprietà `retry_language`) e il parametro `language=` di `transcribe`. L'attributo `language` del trascrittore resta fisso a `"it"` fino a T11.
- [ ] `SttSettings`: via `languages`, `language`, `language_confidence`, `other_confidence`, `retry_in_first_language`, `prompts`, `prompt()`. Il prompt italiano di `[stt.prompts]` passa in `initial_prompt`.
- [ ] Test nuovo `test_shipped_config_keeps_the_whisper_prompt`: `load_settings().stt.initial_prompt` è il testo che era in `[stt.prompts] it`.
- [ ] Via i test multilingua elencati dall'agente (`FakeMultilingualWhisper` e i suoi, `test_roundtrip_english`, ecc.). Suite, ruff, mypy.

## Task 8 (C5b, M7): testi, voci e comandi solo italiani

**File:** i dizionari `{"it": …, "en": …}` di `brain/prompts.py`, `router/responder.py`, `safety/confirmation.py`, `resume.py`, `app.py`, `memory/voice.py`, `memory/tools.py`, `pc/tools.py`, `pc/keys.py`, `router/actions.py`, `stt/vocabulary.py`; le voci di `config.py`, `tts/kokoro.py`, `tts/speaker.py`, `__main__.py` (`build_voices`), `checks.py` (`check_voices`); `config/commands.toml`, `config/claudio.toml`; README.

- [ ] Ogni dizionario per lingua diventa il suo valore italiano; via i rami `if language == "en"` e la grammatica inglese della memoria. Via `in_language`; `unknown_reply`, `error_reply` diventano `str` e le frasi di `smart_phrases`, `offer_phrases`, `no_offer_phrases` liste (le vecchie forme a dizionario non si accettano più: via i loro test).
- [ ] Voci: via `voices`, `VoiceSettings`, `kokoro_voice_for`, `piper_voice_for`, `for_language`, il parametro `voices` di `StreamingSpeaker`, `build_voices`, `check_voices`.
- [ ] `config/commands.toml`: via ogni chiave `*_en` e l'intestazione sull'inglese; il caricatore non le legge più.
- [ ] I parametri `language` restano (inutilizzati) fino a T10-T11.
- [ ] `test_jenjo_answers_stay_after_english_goes` (in `tests/unit/test_safety.py`, se nessun test esistente lo fissa già): «ok» e «confermo» sono sì, «stop» e «lascia stare» sono no. Via i test dei testi e delle voci inglesi. Suite, ruff, mypy.

## Task 9 (C5c, M7): router solo italiano, poi il gate

**File:** `src/claudio/router/normalizer.py`, `router/matcher.py`, `stt/recovery.py`, `router/command_tool.py`, `brain/claude.py` (`_says`). Test: `test_matcher.py`, `test_normalizer.py`, `test_command_tool.py`, `test_recovery.py`.

- [ ] Una sola tabella di regole italiane: via `EN_NUMBERS`, la `Language` inglese, l'unione dei riempitivi, le tabelle e i giri tra lingue del matcher.
- [ ] Test nuovo `test_italian_loanwords_still_match`: «stop», «play», «chat» trovano i loro comandi; «skip» e «quit» no.
- [ ] Suite, ruff, mypy.
- [ ] **Gate (coordinatore, sul fisso, Claudio chiuso):** `uv run python scripts/replay_gate.py --report` → 39/0/2 e **0 cassette mancanti**: prova che T1-T9 non hanno cambiato un byte verso l'API.

## Task 10 (C5d): `language` esce dal cervello

**File:** `src/claudio/brain/claude.py`, `brain/guess.py`, `pc/tools.py`, `memory/tools.py`, `router/responder.py` (smette di passarlo). Test: `test_claude_brain.py`, `test_pc_tools.py`, `test_guess.py` e gli altri che chiamano questi metodi.

- [ ] I metodi pubblici del cervello e dei tool perdono il parametro `language`.
- [ ] Il tag del messaggio perde la lingua: `[Thu 2026-10-23 17:15]`, `[Thu … | unsure: …]` o `[unsure: …]`; senza campi niente tag e il messaggio è solo il testo. Le asserzioni su `[it` nei test seguono.
- [ ] Da qui ogni cassetta manca: è atteso, si rifanno in T12. Suite, ruff, mypy.

## Task 11 (C5e): `language` esce da tutto il resto

**File:** `src/claudio/protocols.py`, `intake.py` (`Heard`), `router/matcher.py` (`Match`), `router/responder.py`, `app.py`, `tts/speaker.py`, `stt/transcriber.py`, `stt/recovery.py`, `router/command_tool.py`, `metrics.py`, `scripts/replay_gate.py` (`Clip.language`), `config.py` (`DEFAULT_LANGUAGE`). Test: i fake e i circa 250 punti che passano `"it"`; via `test_replay_gate.py` 193-196.

- [ ] Via il campo `language` da protocolli, `Heard`, `Match`, risposte, domande in sospeso, `last_route`, righe delle metriche e note dei WAV.
- [ ] Fatto quando `grep -rn "language" src/claudio` mostra solo il `language="it"` di Whisper e le impostazioni delle voci italiane. Suite, ruff, mypy.

## Task 12 (C5f): prompt solo italiano e cassette

**File:** `src/claudio/brain/prompts.py` (righe 14-18 e 253), `src/claudio/pc/tools.py` (descrizione di `purpose`, riga 300), i test che fissano quel testo, README, `CLAUDE.md` del progetto.

- [ ] Righe 14-18 diventano (testo da approvare con il piano):

  ```text
  - Every user message may start with a tag like [Fri 2026-10-23 17:15]: the current local \
  date and time; use it when asked. Always answer in Italian, even when a request sounds \
  English: speech recognition often turns Italian words into English-looking ones.
  ```

  Riga 253: «in his language» → «in Italian». Riga 300: «in a few spoken words in JenJo's language» → «in a few spoken Italian words». Restano le righe 30-31, 214 e 248 (non parlano di risposte in inglese).
- [ ] **Stop:** `replay_gate.py --report` senza `--record` elenca le cassette mancanti; card a JenJo con quante sono e il tetto di spesa. Solo dopo il suo sì: `--record --report`, poi gate 39/0/2 o diagnosi dei falliti; il manifesto si cambia solo con il suo ok. Spesa riportata.
- [ ] `CLAUDE.md` del progetto: via «; il codice inglese rimasto si toglie in F1», con un diff a parte.

## Chiusura

- [ ] `piani/F1-passo-C-prove.md`: prova breve sul fisso (avvio senza errori di configurazione, riga del taglio dei log, «Claudio, che ore sono?», un comando locale, «Claudio, stop»), più la riga in `piani/prove-da-fare.md`.
- [ ] `graphify update .`; `/code-review` sull'intero diff; `superpowers:verification-before-completion` (suite, ruff, mypy, gate).
- [ ] Diff e test a JenJo con card; dopo il sì un commit per task (Conventional Commits). Identità noreply controllata, push solo di `it2/passo-0b`, `gh run watch` fino alla CI verde.
- [ ] Memoria `it2-step-0b-in-progress` aggiornata. Riassunto a JenJo: fatto, righe tolte, costi, rischi. **Poi fermati:** dopo viene la correzione di G2.
