# G0 Basi di misura: task

> **Per chi esegue:** `superpowers:subagent-driven-development`, un task alla volta. Implementer su Sonnet, revisori indipendenti su Opus. I passi usano `- [ ]`.

**Obiettivo:** ogni guasto dal vivo diventa un caso di prova che resta: WAV conservati e riproducibili, finti verificati, azioni vere impossibili per sbaglio, gate di replay e test a timeline.

**Architettura:** nessun cambio di stack. Il gate riusa il cablaggio di produzione (`build_assistant`, `build_responder`) e sostituisce solo i bordi: client Anthropic, desktop, contesto delle azioni, cartelle dei dati.

**Tecnologie:** Python 3.11+, pytest, numpy, soundfile, sounddevice, SDK `anthropic`, faster-whisper, openWakeWord, Silero VAD.

**Spec:** `piani/IT-2-passo-0b-consegne.md`, sezione G0. Scelte di JenJo del 05/11/2026: retention resta a 30 giorni; `audible_ms` si aggiunge e M1 resta; `.claude/skills/graphify/` fuori da git.

## Vincoli globali

- Prima di cercare nel codice: `graphify explain "<X>"` o `graphify path "<A>" "<B>"`, poi leggi solo le righe indicate.
- Gli implementer **non committano**. Commit solo dal coordinatore, a fine gruppo, dopo l'ok di JenJo.
- Non avviare Claudio. Nessuna chiamata API vera nei test. Nessuna azione vera: mai `ActionContext.real()` fuori da `run_assistant` e `run_text`.
- Repo pubblico: WAV, cassette, corpus e memoria mai in git. Tutto in `samples/`, già esclusa.
- Da spento nessun testo né audio salvato, salvo la riaccensione.
- Non leggere `.wav`, `models/`, `uv.lock`, log interi.
- Comandi: `uv run python -m pytest <file> -q`, `uv run ruff check .`, `uv run python -m mypy src` (`uv run mypy` non parte: errore del launcher di uv).
- Stile: type hints, funzioni piccole, commenti solo sul perché, messaggi di log in italiano come il codice attorno, codice in inglese.

## Attenzione (casi che nessun test copre da solo)

1. **Disco pieno o file bloccato mentre si salva la nota del WAV:** la risposta prosegue, l'errore va nel log. Test nel task 3.
2. **Etichettatura interrotta** (Ctrl+C, WAV illeggibile, nessuna uscita audio): nessuna etichetta persa, si riprende da dove eri. Test nel task 7.
3. **WAV senza nota** (le sessioni di oggi) **e clip sparita:** il gate usa il prompt attuale e lo segnala; una clip mancante fallisce col suo nome. Test nel task 10.
4. **Whisper su CPU invece che GPU:** il testo può cambiare senza che sia una regressione. Il report dice sempre dispositivo e modello. Test nel task 9.
5. **Cassetta mancante:** mai una chiamata API di nascosto. Fallisce e dice di rilanciare con `--record`. Test nel task 10.

---

## Task 1: ferma la rotazione

Fatto: commit `5633022`; sessione copiata in `samples/corpus/2026-11-05_17-00/` (50 richieste, 56 scartate, 90 pezzi di Jack, 71 righe di metriche).

## Task 2: strumenti ed esclusioni

**File:** `.gitignore` (già modificato: aggiungi `.claude/skills/graphify/`).

- [ ] Aggiungi l'esclusione e verifica: `git check-ignore .claude/skills/graphify/SKILL.md` risponde col percorso; `git check-ignore .claude/skills/ponytail/SKILL.md` non risponde.
- [ ] A fine gruppo, commit separati: `chore(tooling): add Graphify rules, hooks and ponytail`, `docs(plan): IT-2 step 0b and G0 tasks`.

## Task 3: nota accanto a ogni WAV

**File:** `src/claudio/audio/recorder.py`, `src/claudio/session_log.py`, `src/claudio/stt/transcriber.py`, `src/claudio/app.py`. Test: `tests/unit/test_recorder.py`, `tests/unit/test_off_mode.py`, più uno per `purge_old_logs` dove stanno i suoi.

**Produce:**
- `UtteranceRecorder.note(name: str, meta: dict[str, Any]) -> None`: scrive `<stem>.json` in UTF-8 accanto al WAV.
- `WhisperTranscriber.last_prompt: str`: il prompt del passaggio vero (mai quello di `quick`); `""` se l'audio era silenzio.
- `Assistant._note_wav(sink, name: str | None, **meta) -> None`: chiama `getattr(sink, "note", None)` se c'è e se `name` non è None. Non cambia la firma di `audio_sink`: i `lambda` nei test restano validi.

**Regole:**
- Richieste (`_finish`): `{"whisper_prompt", "language", "gain_db"}`. `gain_db` da `inter.info` (letto prima di `observe`), altrimenti `self.input_gain_db`. `language` da `inter.info.get("stt_language")`.
- Scartate e near miss (`_save_rejected`, `sink(heard, kind)`): solo `{"gain_db": self._gain_db()}`, dentro i `try` già presenti.
- La rotazione del recorder cancella anche `<stem>.json`. `purge_old_logs` cancella anche i `*.json` in `logs/audio/`: i prompt hanno i nomi delle cartelle di JenJo.

- [ ] **Test che falliscono:**
  - `test_note_writes_the_context_beside_the_wav`: `json.loads` di `<stem>.json` dà il dizionario passato.
  - `test_rotation_removes_the_note_with_its_wav`: con `keep=2` e 3 WAV annotati restano 2 WAV e 2 note.
  - `test_purge_old_logs_removes_old_notes`: una nota vecchia sparisce, una recente resta.
  - `test_finish_saves_the_gain_before_the_level_moves`: un `level` finto che alza `gain_db` in `observe`: nella nota c'è il valore di prima.
  - `test_last_prompt_is_the_request_prompt_not_the_quick_one`.
  - `test_a_note_that_cannot_be_written_does_not_stop_the_reply`: `note` che lancia `OSError`: `outcome == "ok"`, `audio_file` presente, errore nel log.
  - In `test_off_mode.py`, scenario `on_disk`: da spento nessun `*.json` su disco.
- [ ] Eseguili: FAIL. Implementa. Eseguili: PASS. Ruff e mypy.
- [ ] **Misura** il costo di `sorted(glob)` con 8000 file vuoti in una cartella temporanea e riportalo. Oltre 50 ms: pota solo ogni 50 salvataggi, con test.

## Task 4: `ActionContext` sicuro di default

**File:** `src/claudio/router/actions.py`, `src/claudio/__main__.py`, `scripts/benchmark_latency.py`, `scripts/replay_guess.py`, test che chiamano `build_responder`. Test: `tests/unit/test_actions.py`.

**Produce:**
- `ActionContext()`: `run`, `open_path`, `press_key`, `press_combo` valgono `_refuse`, che lancia `RuntimeError("a real action needs ActionContext.real()")`. `now`, `folder_lookup`, `home` restano veri: leggono e basta.
- `ActionContext.real() -> ActionContext`, classmethod con le quattro funzioni vere, lette dal modulo al momento della chiamata.
- `build_responder(settings, ctx: ActionContext, system=None, chat=None, tracker=None)`: `ctx` **obbligatorio**, così mypy trova chi lo dimentica.

**Chiamanti:** `run_assistant` e `run_text` passano `ActionContext.real()`. `benchmark_latency.py` (`offline`, `replay`) e `replay_guess.py`: `ActionContext()`, perché usano solo `matcher` e `guess`, senza eseguire comandi. Test: `Recorder().ctx()`.

`tests/conftest.py::no_real_key_presses` resta: è una seconda difesa.

- [ ] **Test che falliscono:**
  - `test_default_context_refuses_every_real_action`, parametrizzato sui 4 campi: `pytest.raises(RuntimeError)`.
  - `test_default_context_still_reads_clock_and_folders`.
  - `test_real_context_uses_the_real_functions`: `monkeypatch` su `actions._run`, poi `ActionContext.real().run` lo chiama.
  - `test_runner_without_context_refuses_alt_f4`: `ActionRunner().execute(<chiudi la finestra>)` lancia e `keybd_event` non è chiamato.
  - `test_run_text_acts_for_real`: `monkeypatch` su `actions._run`, `run_text(settings, "apri il blocco note")` lo chiama, con speaker finto.
- [ ] Eseguili: FAIL. Implementa. Eseguili con tutta la suite: PASS. Ruff e mypy.

## Task 5: `audible_ms` e latenza per percorso

**Dati:** M1 va già da `speech_end` a `first_audio`; è M2 a includere il parlato. Silenzio iniziale dei pezzi di Jack: mediana 49 ms, p90 100, massimo 483. Uscita: 25 ms nominali.

**File:** `src/claudio/audio/player.py`, `src/claudio/tts/speaker.py`, `src/claudio/protocols.py`, `src/claudio/app.py`, `src/claudio/metrics.py`, `src/claudio/session_log.py`, `scripts/benchmark_latency.py`, `CLAUDE.md`, `piani/IT-2-passo-0b-consegne.md`. Test: `tests/unit/test_speaker.py`, `tests/unit/test_responder_metrics_stt.py`, `tests/unit/test_app.py` (`FakeSpeaker`), player di prova in `test_speaker.py` e `test_conversation.py`.

**Produce:**
- `leading_silence_s(audio: np.ndarray, rate: int) -> float` in `speaker.py`: primo campione con `abs(x) > 328` (-40 dBFS su int16); tutto silenzio: durata del pezzo. Commento: `shortcut: only the first chunk is scanned, upgrade if audible_ms outliers need it`.
- `AudioPlayer.starts_in_s() -> float`: `max(self._ends_at - self.clock(), 0.0) + self._latency_s`.
- `StreamingSpeaker.first_sound_delay_s: float`, azzerato in `_reset_for_reply`, impostato in `_play` subito prima di `on_first_audio()`. Attributo anche nel protocollo `Speaker` e in `FakeSpeaker` (`0.0`).
- `Assistant._mark_first_audio(inter)`: `inter.mark("first_audio")` e, solo la prima volta, `inter.marks["first_audible"] = inter.clock() + self.speaker.first_sound_delay_s`. Usato in `_on_first_audio` e in `handle_typed`.
- `Interaction.summary()["audible_ms"] = self.ms("speech_end", "first_audible")`. In `summarize` (chiavi predefinite), nel log «Latenza», in `session_log.py:102`.
- `benchmark_latency.from_log`: dopo il riepilogo, `audible_ms` e `m1_ms` per percorso: `PATHS = {"locale": {"local", "memory", "unknown"}, "conferma": {"confirm", "confirmed", "refused", "confirm_timeout"}, "Claude": {"claude"}}`, il resto in «altro». Le righe vecchie non hanno `audible_ms`: `summarize` le salta.

**Documenti (diff a JenJo a fine gruppo):**
- `CLAUDE.md`: «Misura sempre la latenza (fine del parlato → primo campione udibile, per percorso: locale, conferma, Claude; `m1_ms` e `m2_ms` restano per continuità) e il costo per richiesta.»
- Consegne, riga 97: sostituisci «`m1_ms` include il tempo in cui JenJo parla: resta per continuità» con «`m1_ms` parte già dalla fine del parlato ma si ferma prima dell'audio; `m2_ms` include il parlato. Restano per continuità».

- [ ] **Test che falliscono:**
  - `test_leading_silence_finds_the_first_audible_sample`: 800 campioni a zero poi uno a 1000, a 16000 Hz: `0.05`. Tutto zero: durata intera. Primo campione udibile: `0.0`.
  - `test_starts_in_counts_what_is_queued_and_the_device`.
  - `test_first_sound_delay_is_ready_when_first_audio_fires`.
  - `test_audible_ms_adds_the_delay_to_m1`: ritardo 0,125 s: `audible_ms == m1_ms + 125`. Senza risposta: `None`.
  - `test_from_log_groups_latency_by_path`: `capsys`, righe locale, conferma, Claude, più una vecchia senza `audible_ms`.
- [ ] Eseguili: FAIL. Implementa. Eseguili: PASS. Ruff e mypy.

## Task 6: finti verificati

**File:** nuovo `tests/unit/test_speaker_contract.py`; `tests/unit/test_app.py` (`FakeSpeaker`).

**Contratto:** fixture parametrizzata `speaker`: `FakeSpeaker()` e `StreamingSpeaker(<synth finto>, <FakePlayer da test_speaker.py>)`. Ogni attesa dentro `asyncio.wait_for(..., 1.0)`.

- `test_first_audio_does_not_wait_for_the_rest_of_the_text`: il flusso dà «Uno. » e poi aspetta un `asyncio.Event` che imposta `on_first_audio`. Se lo speaker vuole prima tutto il testo, scatta il timeout.
- `test_first_audio_fires_once`.
- `test_no_text_no_first_audio`.
- `test_played_lists_the_sentences_that_started`.
- `test_interrupt_stops_the_reply`: `interrupt()` dentro `on_first_audio`: in `played` non c'è «Due.».
- `test_interrupt_while_waiting_for_text_returns_at_once`: flusso bloccato dopo la prima frase.
- `test_a_reply_after_an_interrupt_starts_clean`.
- `test_language_callable_is_read_once_the_text_has_started`.
- `test_first_sound_delay_is_set_when_first_audio_fires`: float ≥ 0.

**`FakeSpeaker`:** taglia il flusso con `split_stream` (`src/claudio/tts/sentence_splitter.py:164`), come il vero; `on_first_audio` alla prima frase; `played` con le frasi iniziate; `interrupt()` vero. `spoken` resta il testo intero che ha letto, con gli stessi spazi: centinaia di test lo confrontano. Le sottoclassi (`LongSpeaker`, `SentenceSpeaker`, `StartSpeaker`, `CueSpeaker`) sovrascrivono `speak`: restano come sono.

- [ ] Scrivi il contratto. Eseguilo: `FakeSpeaker` FAIL, `StreamingSpeaker` PASS. Se fallisce sullo speaker vero, **fermati e riporta**: è un bug di produzione.
- [ ] Riscrivi `FakeSpeaker`. Contratto PASS su entrambi.
- [ ] Tutta la suite. **Non forzare i test che si rompono:** per ognuno riporta se l'assunzione era falsa o se ha trovato un bug in `src/`. I bug in `src/` non si correggono qui: restano `xfail(strict=True)` col gruppo del piano.

## Task 7: etichettatura

**File:** nuovo `scripts/label_wavs.py`. Test: nuovo `tests/unit/test_label_wavs.py`, importato con `sys.path.insert(0, str(PROJECT_ROOT / "scripts"))` come `test_compare_stt.py`.

**Dati:** `samples/corpus/<sessione>/` come quella già copiata; `samples/corpus/labels.jsonl`; `samples/corpus/splits.json`.

**Produce (funzioni pure, testate):**
- `import_session(logs: Path, corpus: Path, name: str) -> int`: copia dalla sessione `name` fino a quella dopo console, conversazione, Jack, WAV e note (per timestamp nel nome), righe di metriche. Non sovrascrive.
- `load_rows(session_dir) -> dict[str, dict]`: `audio_file` → riga.
- `propose(row: dict | None) -> str`: `text`, altrimenti `second_chance`, altrimenti `""`.
- `pending(session_dir, labels_path) -> list[Path]`: WAV ordinati di `audio/` e `audio/rejected/` senza etichetta.
- `assign_split(splits_path, session, wanted: str | None) -> str`: la prima volta serve `--split dev|test`; poi rifiuta un valore diverso e valori sconosciuti con `SystemExit`.
- `append_label(labels_path, record) -> None`: una riga JSON, subito su disco. Campi: `wav`, `session`, `split`, `proposed`, `text`, `for_claudio`, `labeled_at`.

**Interfaccia:** `uv run python scripts/label_wavs.py --session NOME [--split dev|test] [--import]`. Riproduce il WAV con `sounddevice`, mostra testo e riga, legge un tasto con `msvcrt.getwch`: `Invio` conferma per Claudio, `n` conferma ma non per Claudio, `c` corregge il testo, `r` riascolta, `s` salta, `q` esce. Mostra quanti ne restano. Senza uscita audio lo dice e prosegue.

- [ ] **Test che falliscono:**
  - `test_import_session_takes_only_its_own_wavs_and_never_overwrites`.
  - `test_propose_prefers_the_text`.
  - `test_pending_skips_what_is_labelled`.
  - `test_a_session_cannot_change_split`.
  - `test_a_new_session_needs_a_split`.
  - `test_labels_survive_an_interrupted_run`: due etichette, `KeyboardInterrupt` alla terza: il file ha 2 righe valide e `pending` riparte dalla terza.
  - `test_an_unreadable_wav_is_skipped_and_reported`.
  - `test_the_corpus_is_ignored_by_git`: `git check-ignore samples/corpus/labels.jsonl` risponde; saltato se manca git.
- [ ] Eseguili: FAIL. Implementa. Eseguili: PASS. Ruff.
- [ ] **Fermati per JenJo:** prima seduta sulla sessione del 05/11, `--split dev`. Spiega i tasti in chat e manda la notifica.

## Task 8: frasi di sicurezza come audio

**File:** nuovo `scripts/safety_phrases.tsv` (`atteso<TAB>frase`, atteso = `yes`, `no`, `unclear`, come `Answer`); `scripts/record_samples.py` (modo `safety`, sul modello di `commands`); nuovo `tests/integration/test_safety_audio.py` (`pytestmark = pytest.mark.slow`, come `test_real_models.py`).

**Frasi:** circa 20 sì e 20 negativi quasi uguali, partendo da `NOT_A_YES` e dai parametri di `tests/unit/test_safety.py`, più «sì ma dopo», «sei sicuro», «sì?», «no aspetta». `--repeat 2`, in `samples/safety/<atteso>/`. Circa 10 minuti di JenJo (stima).

- [ ] **Test che falliscono:**
  - In `tests/unit/test_record_samples.py` (o dove stanno i suoi): `test_safety_prompts_read_the_phrases_file`.
  - `test_every_safety_phrase_parses_as_expected`: **in `tests/unit/`**, sul testo: `parse_answer(frase)` dà l'atteso. Trova subito errori nella lista.
- [ ] Eseguili: FAIL. Implementa. Eseguili: PASS. Ruff.
- [ ] **Fermati per JenJo:** registrazione. Manda la notifica con il comando.
- [ ] `test_safety_audio.py`: Whisper vero, una riga per WAV. **Duro:** nessun negativo diventa `Answer.YES`. **Solo riportato:** quanti sì vengono capiti. Saltato se `samples/safety/` è vuota. Eseguilo con `-m slow` e riporta l'esito a JenJo. Se un negativo diventa un sì è un guasto di sicurezza: fermati e riporta.

**Esito (06/11/2026):** registrazione di JenJo fatta, e il test fallisce: 6 negativi su 64 diventano un sì, 24 sì su 44 capiti. Sul percorso reale sono 3-5 e 28. Il task si chiude con il guasto scritto in G2, delle consegne: il test resta rosso fino ad allora.

## Task 9: prova di determinismo

**Rischio maggiore del gruppo:** non è verificato che la wake word vera scatti sui WAV salvati, né che lo stesso WAV dia lo stesso esito. Il piano prevede che se non regge il gate cambia forma.

**File:** `src/claudio/__main__.py` (estrai `build_assistant`); nuovo `scripts/replay_gate.py`; nuovo `tests/unit/test_replay_gate.py`.

**Produce:**
- In `__main__.py`, `build_assistant(settings, *, wake, vad, wake_vad, transcriber, responder, speaker, system, level=None, metrics=None, audio_sink=None, rejected_sink=None, clock=None) -> Assistant`: il corpo di `Assistant(...)` di `run_assistant`, che ora la chiama. **Il gate non ha un cablaggio suo.** Come argomenti riceve solo ciò che il gate deve sostituire; `clock=None` non passa nulla ad `Assistant`.
- In `replay_gate.py`:
  - `Clip(wav: Path, prompt: str | None, language: str | None, gain_db: float | None, session: str)`, letto dalla nota se c'è.
  - `Outcome(woke: bool, wake_word: str | None, text: str, route: str | None, command: str | None, tools: tuple[str, ...], actions: tuple[str, ...], safety: int | None, outcome: str | None)`, con `gate_key()` che esclude `text`.
  - `install_tripwire() -> None`: `keybd_event`, `pywinauto.keyboard.send_keys`, `os.startfile`, `subprocess.Popen` lanciano. Solo nel processo del gate.
  - `Harness`, `build_harness(settings, work_dir, wake=None, stt=None, vad=None)`: `settings` con `data_dir` e `logs_dir` in `work_dir`; `ActionRunner` su `Recorder().ctx()` (`from tests.unit.test_actions import Recorder`, `sys.path` sulla radice).
  - `replay_clip(harness, clip) -> Outcome`: 3 s di zeri, WAV, 3 s di zeri, a blocchi di `settings.audio.block_size`, passando da `AudioCapture(...)._frame_callback` con guadagno 1 (il WAV è già amplificato). **Tempo logico:** orologio che avanza coi frame; tra un frame e l'altro aspetta che `assistant.state` non sia `TRANSCRIBING`, `RESPONDING`, `SPEAKING`. Limite da scrivere in cima: barge-in ed eco non si provano qui.
  - `determinism_report(outcomes) -> dict[str, int]`: campo → valori distinti.
  - Argomenti: `--determinism N --clips <wav...>`. Stampa dispositivo e modello di Whisper, wake word scattata o no per clip, campi che variano.
- Speaker muto nel gate: legge il testo per frasi e chiama `on_first_audio`; riusa il `FakeSpeaker` verificato nel task 6.

- [ ] **Test che falliscono**, con finti, senza modelli:
  - `test_harness_refuses_real_actions`: `harness.responder.actions.ctx` è del `Recorder`; `ActionContext.real()` non compare.
  - `test_harness_uses_production_wiring`: `monkeypatch` di `claudio.__main__.build_assistant`, il gate lo chiama.
  - `test_tripwire_blocks_a_real_key_press`.
  - `test_harness_keeps_its_data_in_the_work_dir`: `settings.data_dir` e `logs_dir` sotto `work_dir`.
  - `test_replay_feeds_the_whole_wav_in_blocks`: wake finto, conta i campioni.
  - `test_report_names_the_device`.
  - `test_build_assistant_passes_the_settings`: un paio di campi (`conversation_s`, `continue_s`) arrivano all'`Assistant`.
- [ ] Eseguili: FAIL. Implementa. Eseguili con tutta la suite: PASS. Ruff e mypy.
- [ ] **Prova vera, nessuna API:** 5 clip da `samples/corpus/2026-11-05_17-00/audio/`: 3 comandi locali `_ok`, 1 Claude (solo fino al router), 1 `timeout`. `--determinism 100`. Se servono più di 15 minuti, scendi a 30 ripetizioni e dillo.
- [ ] **Riporta e fermati:** quante wake word scattano, quali campi variano, quanto tempo. **Regola:** se la wake word vera scatta su meno di 4 clip su 5, il gate usa `ReplayWake` col `prewake_ms` della riga (già in `benchmark_latency.py`) e una prova separata per la wake word; lo decide il coordinatore con JenJo.

## Task 10: gate di replay

**Scelta di JenJo (06/11/2026), dopo la prova del task 9: gate ibrido.** Ogni clip fa scattare la wake word nel punto registrato (`ReplayWake`, col `prewake_ms` della riga), così VAD, Whisper, router e tool sono provati su tutte; `ReplayWake` non passa dalla seconda possibilità. A parte, le clip che scattano davvero devono continuare a scattare: se una smette, il gate fallisce. Quasi due terzi delle righe (47 su 71) non hanno una wake word da far scattare: `conversation`, `follow_up`, `barge_in`. Le prime due entrano come **scene**: più clip consecutive sullo stesso `Assistant`, così la finestra si apre davvero dopo la risposta. Il barge-in resta al task 11.

**Cambiato durante il task (06/11/2026):**
- Niente `start: "off"`: una riaccensione non si può riprodurre, perché il suo WAV non contiene il nome, per scelta (`c4038cb`). Il comportamento è coperto da `tests/unit/test_off_mode.py`.
- Campi in più nel manifest: `scene`; `real_wake` a tre valori (assente, deve svegliarsi, non deve); `no_wake`; `known_diff`, le chiavi che un guasto noto sbaglia oggi. In `expect`: `actions` e `tool_errors`. Una chiave sconosciuta è un errore.
- `--record` si ferma al primo errore API e quando il costo registrato raggiunge `[cost] daily_budget`.
- Due passate: la prima con `ReplayWake`, la seconda con il modello vero per le clip segnate.

**Prima di iniziare:** skill `claude-api`. Interfacce vere del client: `client.messages.stream(**params)` come context manager con `.text_stream`, `.current_message_snapshot`, `get_final_message()`; `client.messages.create(**params)` in `guess.py`; `client.models.retrieve(model)` nel warm-up.

**File:** `src/claudio/__main__.py`, `scripts/replay_gate.py`. Test: `tests/unit/test_replay_gate.py`.

**Produce:**
- Bordi nel cablaggio: `build_responder(..., client=None, desktop=None)`, `build_brain(..., client=None, desktop=None)`, `build_pc_tools(..., desktop=None)`. `None`: come oggi.
- `CassetteClient(folder: Path, real: Any | None = None)`, stessa forma del client. Chiave: SHA-256 del JSON ordinato di `model`, `system`, `tools`, `messages`. Con `real`: registra `chunks`, `stop_reason`, `content`, `usage` in `<chiave>.json`; solo le risposte riuscite. Senza `real`: una chiave mancante si aggiunge a `client.misses` e lancia `CassetteMiss`. Nel `content` niente oggetti: JSON ricostruito in `SimpleNamespace`, come `ScriptedStream`.
- Desktop del gate: `FakeDesktop` di `tests/unit/test_pc_tools.py`. `home` in `work_dir`.
- `settings.claude.tell_time = False` nel gate: l'ora nella richiesta cambierebbe la chiave a ogni esecuzione (`claude.py:694`). Budget alto: il blocco giornaliero non è ciò che si prova.
- **Manifest** `samples/gate/manifest.jsonl`: `{"clip", "expect": {"woke", "route", "command", "tools", "args", "safety"}, "known_bug": "G1" | null, "why"}`.
- `run_gate(manifest, harness) -> GateResult`: `passed`, `failed`, `known`, `fixed` (un `known_bug` che ora passa: va tolto il segno), `misses`, `no_note`, `missing`. Uscita 1 se `failed`, `fixed`, `misses` o `missing` non sono vuoti.
- Report `samples/gate/report.md`: tabella esiti, differenze di testo rispetto a `text` della riga dal vivo (non bloccanti), dispositivo e modello, clip senza nota. Una riga adatta alle note di rilascio.
- Argomenti: `--record` (usa la chiave, stampa prima quante chiamate e la stima di costo, chiede conferma con `input()`), `--report`.

- [ ] **Test che falliscono**, con finti:
  - `test_cassette_replays_what_was_recorded`: `ScriptedClient` come `real`, poi senza.
  - `test_a_missing_cassette_never_calls_the_api`: `real=None`, chiave ignota: `misses` ha la chiave, nessuna rete; il gate esce con 1.
  - `test_cassette_key_changes_with_the_system_prompt_and_not_with_dict_order`.
  - `test_gate_compares_outcome_not_exact_text`: `text` diverso, stesso `gate_key`: passa e finisce tra le differenze.
  - `test_gate_fails_on_a_wrong_tool_argument`.
  - `test_known_bug_does_not_fail_and_a_fixed_one_does`.
  - `test_clip_without_note_is_flagged`.
  - `test_missing_clip_fails_with_its_name`.
  - `test_gate_fails_if_it_is_not_on_the_recorder`: `Harness` con un contesto che non è del `Recorder`: `run_gate` rifiuta prima di partire.
  - `test_real_tools_are_never_built_in_the_gate`: il desktop è il finto; `WindowsDesktop` non viene istanziato.
- [ ] Eseguili: FAIL. Implementa. Eseguili con tutta la suite: PASS. Ruff e mypy.
- [ ] **Scegli 30-60 clip** dai guasti: 17:08:15, 17:08:35, 17:09:34, 17:12:48, 17:13:18, 17:13:40, 17:16:09, 17:20:17, i casi del Blocco note; più comandi locali che oggi funzionano e frasi scartate. Il primo WAV è delle 17:05:16: quello delle 17:04:15 è perso, dillo. Le clip senza wake word (`conversation`, `follow_up`) entrano solo se il gate le apre allo stesso stato: altrimenti vanno al task 11.
- [ ] **Fermati per JenJo:** tabella clip → esito atteso → `known_bug`, per approvazione. Costo di `--record` prima di spendere; budget 2,58 $/giorno, al massimo 80 $ al mese.
- [ ] `--record`, poi due esecuzioni senza chiave: stesso esito. Riporta spesa vera.

## Task 11: test a timeline

**File:** nuovo `tests/unit/timeline.py` (non raccolto da pytest); nuovo `tests/unit/test_timeline.py`.

**Produce:** `Scene`, che usa `Assistant`, `LocalResponder`, `ClaudeBrain` veri con `ScriptedClient`, `PcTools` su `FakeDesktop`, `ActionRunner` su `Recorder().ctx()`, `FakeStt`, `FakeWake`, `FakeVad`, `AudioClock`, e un `TimedSpeaker(FakeSpeaker)` che parla per un tempo in frame e rispetta il contratto del task 6. Riusa `build`, `talk`, `speak_over`, `barge` di `test_conversation.py` invece di riscriverli.

```python
scene = Scene(replies=[...], heard=[...])
await scene.say("che ore sono")
await scene.until_speaking()
await scene.say_over("basta")
await scene.advance(2.0)
assert scene.row(0)["outcome"] == "interrupted"
```

**Casi** (`LiveKit tests/test_agent_session.py` è solo la lista di controllo: nessun codice copiato):

| Test | Da LiveKit | Cosa dimostra |
|---|---|---|
| `test_basta_while_speaking_keeps_what_was_heard` | `test_interruption` | in cronologia c'è ciò che è stato sentito (`_remember_cut`, G2) |
| `test_basta_before_the_first_word_says_nothing_later` | `test_interruption_before_speaking` | niente voce dopo lo stop |
| `test_noise_over_the_reply_lets_it_finish` | `test_false_interruption_before_speaking_resumes` | abbassa e torna a volume pieno |
| `test_tool_runs_once_and_claude_gets_its_result` | `test_tool_call` | una sola esecuzione |
| `test_slow_tool_keeps_the_first_sentence_said` | `test_slow_tool_keeps_agent_thinking_after_filler` | la voce non resta muta |
| `test_tool_without_a_reply_goes_back_to_idle` | `test_tool_without_a_reply_returns_the_agent_to_listening` | |
| `test_interrupted_row_keeps_its_first_audio` | `test_interrupted_say_uses_speech_start_timestamp` | metriche giuste |
| `test_typed_message_waits_for_the_spoken_reply` | `test_interruption_by_text_input` | Claudio aspetta, non taglia |
| `test_phrase_left_halfway_is_joined` | `test_stt_resume_*` | due parti, una frase |
| `test_stt_error_says_so_and_goes_back` | `test_stt_errors_use_unrecoverable_error_tolerance` | si riprende |
| `test_voice_latency_excludes_claude_wait` | `test_tts_node_ttfb_excludes_upstream_latency` | `tts_first_ms` |
| `test_typed_reply_cut_short_keeps_what_was_said` | scoperto nella revisione del task 6 | `_marked` salva il testo nel suo `finally`, ma dopo un'interruzione il flusso può non essere ancora chiuso quando si scrive la riga (`app.py:1688`, `:1575`): vale anche per la voce? |
| `test_17_08_story_then_time_resumes` | sequenza vera | «Dicevo» dopo «che ore sono?» |
| `test_17_13_story_then_new_request_does_not_resume` | sequenza vera | niente «Dicevo» dopo una richiesta nuova |

Le sequenze vere: `samples/corpus/2026-11-05_17-00/conversazione.md`, solo le righe di quelle ore.

- [ ] Scrivi `Scene` e i casi. Eseguili. **I casi rossi sono risultati, non errori:** quelli che dimostrano un guasto noto restano `xfail(strict=True, reason="G2: ...")`, così G2 li vede passare. Non correggere `src/` qui.
- [ ] Riporta: quanti verdi, quanti `xfail`, e per ognuno il gruppo. Se un caso rosso non corrisponde a nulla nel piano è una scoperta: riportala, non toglierla.

## Fine del gruppo

- [ ] Suite, ruff, mypy, gate. `graphify update .`.
- [ ] Revisione di tutto il ramo su Opus.
- [ ] Aggiorna `Stato lasciato` nelle consegne e la riga di `IT-2-consegne.md` sul Passo 0: la latenza è misurata (M1 Claude 4947 → 3062 ms, locale 1326 → 1004 ms, 44 richieste; confronto non alla pari, il periodo precedente aveva sessioni in gioco).
- [ ] Mostra a JenJo diff e test, aspetta l'ok, poi commit piccoli. Notifica.
