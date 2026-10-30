# IT-1 – Consegne (problemi aperti dopo IT-0, con causa e soluzione)

Stato al 30/10: IT-0 unito in `main` locale (`f40bcc7`, nessun push), worktree e rami IT-0
rimossi. 1753 test passano (+8 lenti), ruff e mypy puliti.

Fonti: la prova dal vivo di JenJo (30/10 14:00-14:08, `logs\sessions\2026-10-30_14-00\`: console,
44 pezzi di Jack in `jack\`, 29 WAV di ascolto in `logs\audio\`, righe di `logs\metrics.jsonl`),
le cinque consegne dei thread IT-0 e il codice di `main`. Ogni causa indicata come **verificata**
è stata riprodotta su quei file o letta nel codice; le altre sono marcate come ipotesi.
Gli script di analisi sono nello scratchpad della sessione (non nel repo).

Regola di lavoro (JenJo 30/10): ogni iterazione ha uno o più thread che **portano avanti i
problemi dell'iterazione precedente** e **aggiungono funzioni** verso la versione stabile finale.
Le sezioni 1-5 sono i problemi, la 6 le funzioni ancora mancanti rispetto a CLAUDE.md.

Legenda sforzo: piccolo (ore), medio (circa un giorno), grande (più giorni). Costo API: zero dove
non indicato.

---

## 1. Ascolto: "non sente nulla" e "si attiva da solo"

### 1.1 Dopo ogni bip o risposta la wake word scatta da sola circa 0,5 s dopo
- **Prova**: 4 wake con `prewake_ms = 480` (14:02:25, 14:04:13, 14:04:16, 14:04:33): scattano
  esattamente al sesto frame da 80 ms dopo un reset, su silenzio o rumore. Da spento sono la causa
  principale del minuto perso a dire "accenditi": ogni falsa wake apre 4 s di ascolto alla cieca
  (circa 18 s sordi su 57 tra 14:04:13 e 14:04:58). Riprodotto su 104 spezzoni tranquilli della
  sessione: il sesto frame dopo un reset fa scattare una wake nel 47 % dei casi.
- **Causa (verificata nel codice)**: `wakeword/detector.py:94-99` passa 25 frame di silenzio al
  modello e **poi** chiama `self._model.reset()`. In openWakeWord 0.6.0 `AudioFeatures.reset()`
  riempie il buffer con gli embedding di 4 s di **rumore casuale** (`np.random.randint(-1000, 1000)`,
  `openwakeword/utils.py:178`) e cancella il silenzio appena passato. `reset()` viene chiamato dopo
  ogni bip e ogni risposta (`app.py` `_finish`, `_hears_itself`).
- **Soluzione**: invertire l'ordine (prima `_model.reset()`, poi i frame di silenzio, senza leggerne
  i punteggi) oppure ignorare le detection finché la finestra del modello non contiene solo audio
  vero. Test di integrazione con i modelli veri: dopo `reset()`, 480 ms di audio tranquillo devono
  restare sotto soglia.
- **Misura**: tasso di scatto dopo reset 47 % → 0 %; nei log nessuna wake con `prewake_ms == 480`.
- **Sforzo**: piccolo (3-4 righe + test). **Priorità massima**: è un bug, non una taratura.

### 1.2 "Claudio" detto da solo: la seconda possibilità sente "Grazie." e lo perde
- **Prova**: 22 seconde possibilità nel log; il nome sentito solo 4 volte (sempre con la parola
  dopo), "Grazie" 12 volte, "" 4 volte. Sulle 11 wake "claudio" riuscite, il prewake senza prompt
  dà il nome 6/11; **con il prompt delle richieste 11/11**, e 0 nomi falsi su 90 finestre negative
  (silenzio, frasi senza nome, eco di Jack). 14:00:43: wake "unclear" con bip su audio sotto
  `MIN_RMS` (Whisper non ha neanche girato) → "Nessuna richiesta".
- **Causa (verificata)**: la seconda possibilità usa `quick=True`, cioè **nessun prompt**
  (`stt/transcriber.py:264`): una parola isolata con la "C" tagliata dal gate della cuffia diventa
  l'allucinazione tipica "Grazie.". La regola `unclear_min` (`app.py:390-398`,
  `wakeword/verifier.py:32-37`) tratta allo stesso modo "" da silenzio e "Grazie." da voce.
- **Soluzione**: (1) prompt delle richieste anche nella seconda possibilità, con il controllo
  `is_prompt_echo` già esistente; (2) prima di Whisper, controllo Silero sui 2,5 s: senza almeno
  64-128 ms di voce si rifiuta subito (risparmia GPU, mai wake su silenzio o clic); (3) poi ridurre o
  togliere `unclear_min`, e comunque mai una wake se l'audio era sotto `MIN_RMS`; (4) salvare i WAV
  delle seconde possibilità rifiutate in una cartella a parte, con un limite.
- **Misura**: nome sentito / "Wake word debole" da 4/22 verso la quasi totalità; "Nessuna richiesta"
  dopo una wake "unclear" = 0. Confidenza alta sul meccanismo, da confermare su più sessioni.
- **Sforzo**: piccolo (2-3 h). Latenza invariata (~325 ms con o senza prompt).

### 1.3 Nome e richiesta detti d'un fiato: si sveglia quando hai già finito
- **Prova**: 14:06:40 la seconda possibilità sente "Claudio Spegni", attiva, poi 4 s di silenzio e
  timeout; il WAV intero contiene "Claudio, spegniti.". Replay con l'endpointer del progetto su
  tutte le 21 wake: riproduce gli esiti reali; con la correzione sotto, 14:06:40 diventa
  "Claudio, spegniti." e nessun'altra wake cambia.
- **Causa (verificata)**: `audio/vad.py:141-142` mette il lead-in davanti alla frase solo se la voce
  riparte entro `ONE_BREATH_S = 0.3` s dal reset. La wake "claudio" scatta tardi e la seconda
  possibilità aggiunge 120-350 ms: se hai già finito, l'endpointer non vede mai partire la voce.
- **Soluzione**: in `Endpointer.reset(lead_in)` far passare gli ultimi 1,5 s per il VAD senza
  contarli nel timeout: la frase finita durante la wake si chiude normalmente e va a Whisper con il
  nome. Il caso "nome, pausa, richiesta" resta gestito dal RELISTEN esistente.
- **Sforzo**: piccolo-medio (mezza giornata con i test).

### 1.4 Da spento "Claudio, accenditi" va ripetuto molte volte
- **Prova**: circa 1 minuto (14:04:13-14:05:10) e 14:02:24 ("Grazie a te." scartato; la seconda
  trascrizione era "Dai, accenditi.", che il router capisce). Da spento non c'è nessun bip.
- **Causa**: oltre a 1.1, il percorso da spento è volutamente minimo (`app.py:343-348`, `524-526`,
  `708`): soglie +0,1, niente seconda possibilità, una sola passata Whisper, niente lista chiusa.
  Il log scrive la soglia normale invece di quella alta (`detector.py:85-92`).
- **Soluzione** (dopo 1.1): da spento stessa seconda possibilità di 1.2 ma accettando solo nome +
  "accenditi"; lista chiusa limitata al comando `on` (~150 ms); lead-in di 1.3; WAV dei near miss
  da spento; log della soglia attiva. Un segnale sonoro da spento solo dopo, e solo quando Whisper
  ha sentito il nome (oggi suonerebbe ~6 volte al minuto per i falsi).
- **Misura**: prova da spento con 10 "Claudio, accenditi": tentativi necessari (oggi fino a ~5) e
  zero accensioni false in 10 minuti di rumore.
- **Sforzo**: piccolo (lista chiusa, log, WAV) + medio (seconda possibilità da spento).

### 1.5 Il modello "claudio" scatta su rumori che non sono voce
- **Prova**: le wake da spento delle 14:04:32/37/58 hanno 85-96 % dell'energia sopra 1 kHz
  (la tua voce in questa cuffia: ≤ 0,7 %), Silero max 0,06. Cosa fossero (tasti, cuffia toccata,
  apertura del gate) non è verificato.
- **Causa**: il modello "claudio" è debole per costruzione (config: "~60 false wake all'ora");
  da acceso lo filtra solo Whisper, da spento nessuno.
- **Soluzione**: (1) controllo Silero sulle detection di "claudio", acceso e spento (vedi 1.2);
  (2) riaddestrare "claudio" con le tue prese vere dai log come positivi e i WAV rifiutati (clic,
  gate, voce di Jack) come negativi; la pipeline è già in `scripts/wakeword`.
- **Sforzo**: piccolo (1) / grande (2, qualche giorno di raccolta + training).

> Nota: il VAD e l'endpointer **non** sono la causa dei "Nessuna richiesta": in tutte le 21 wake,
> quando parlavi dopo il bip, Silero ti ha presa (0,55-1,0). Il bip arriva al microfono 230-370 ms
> dopo la wake (latenza HDMI): utile per tarare la coda dell'eco (2.4).

---

## 2. Ascolto: "capisce male"

### 2.1 Comandi brevi storpiati e nessun recupero acustico
- **Prova**: 14:08:13 "Chi ce l'ha fatta?" (era "chiudi la chat"), 14:01:59 "Scaniti." (spegniti,
  la stima di Haiku al 35 % l'ha scartata), 14:01:18 "Chiudila chat." (passato solo per il fuzzy).
  La frase è arrivata **intera** a Whisper (voce 1,4-5,5 s dopo l'inizio del WAV), senza clipping;
  ±6 dB di guadagno o un prompt diverso non cambiano il testo. La voce in cuffia ha il 97-99,5 %
  dell'energia sotto 1 kHz.
- **Causa**: il canale della cuffia (banda stretta + gate che apre di colpo) toglie le consonanti
  iniziali; poi il recupero lavora solo sul testo: la lista chiusa prende l'n-best con lo stesso
  prompt (se le parole giuste non sono nel beam il comando non arriva), il fuzzy esclude i comandi
  `exact` (`router/matcher.py:214-221`), e `wants_closed_list` salta le frasi con "?"
  (`intake.py:184-194`).
- **Soluzione (locale, gratis)**: per le frasi brevi non capite (≤ 5-6 parole, anche con "?"):
  (1) seconda decodifica con prompt = "Claudio, " + la prima frase di ogni comando; usata solo come
  proposta. (2) Verifica acustica del comando proposto: punteggio forzato senza prompt
  (`ctranslate2` `Whisper.align`, ~13 ms a candidato) contro la prima trascrizione: esegue se
  nettamente migliore, chiede "Hai detto: ...?" in locale (zero API) se incerto; per `off` e
  `shutdown` sempre la domanda. (3) In metrics il testo proposto e i punteggi, per tarare.
  Sui set di prova (158 registrazioni): +4 comandi recuperati, 0 falsi, 0 sbagliati; in sessione
  "Scaniti" → "Spegniti.", "Chi ce l'ha fatta" → "Chiudi la chat.". Meno chiamate a Haiku.
- **Misura**: `scripts/stt_bench.py` sui due set (comandi giusti, falsi = 0); dal vivo meno stime di
  Haiku e meno comandi da ripetere.
- **Sforzo**: medio (~1 giorno). +~200 ms solo sulle frasi brevi non capite.

### 2.2 Mentre Claudio parla, quello che dici diventa frammenti
- **Prova**: "Grazie a te" ×2, "alla tesi", "A te Giotto", "A presto" (14:07:10-14:08:05). Replay di
  6 frasi vere della sessione attraverso il listener del barge-in: solo 1 su 6 resta intera
  ("Apri la chat." → "apri la ciotola", "Chiudila chat." → "Chi di lasciate?").
- **Causa (verificata)**: il listener taglia a `barge_in_max_s = 1.2` s **contando anche gli 800 ms di
  pre-roll** (`audio/vad.py:143` + `:158`; `__main__.py:383-412`), poi trascrive ogni pezzo da solo,
  senza prompt e con 24 token: Whisper su < 1 s di voce restituisce i suoi congedi tipici.
- **Soluzione**: (a) contare `barge_in_max_s` solo dalla voce e pre-roll del barge-in a ~250 ms;
  (b) le fette corte solo per "stop/basta"; finché la voce continua accumulare e ritrascrivere la
  frase intera con il prompt normale; (c) passarla dal recupero di 2.1; (d) salvare le frasi
  sentite mentre parla (`barge_ignored`) con testo e dBFS.
- **Misura**: replay dei set di prova in modalità barge-in (oggi 1/6 comandi interi).
- **Sforzo**: piccolo (a, d) + medio (b).

### 2.3 La soglia del barge-in si misura dopo il guadagno
- **Prova/causa**: quei frammenti per livello (-19/-23 dBFS) non sono eco (l'eco vera sta 20-45 dB
  sotto la voce); ma `barge_in_min_dbfs = -30` si misura **dopo** AutoGain: a +12 dB l'eco
  massima stimata arriva a ~-35 dBFS, solo 5 dB di margine.
- **Soluzione**: confrontare la soglia con il livello prima del guadagno (dBFS meno `gain_db`).
  La cancellazione d'eco (AEC) **non** è giustificata da questi dati: resta un'idea per le casse.
- **Sforzo**: piccolo.

### 2.4 Rischio: la sordità dopo il bip o la risposta può tagliare l'inizio
- **Prova**: oggi non confermato (le frasi analizzate erano intere), ma la coda è calcolata con la
  latenza dichiarata + `echo_tail_ms` 400, mentre il ritorno reale HDMI + cuffia è 0,17-0,41 s:
  quasi tutta la coda se ne va lì e ciò che si dice in quel mezzo secondo si perde senza log.
- **Soluzione**: (1) tenere i frame sordi in un buffer (≤ 1 s) e anteporli se la voce parte entro
  ~300 ms dalla fine della sordità o se il VAD vi trova voce sopra il livello d'eco; (2) misurare
  all'avvio il ritorno reale ("Sono pronto" ritrovato nel microfono) e usare ritorno + 150 ms;
  (3) `deaf_speech_ms` in metrics.
- **Sforzo**: piccolo-medio.

---

## 3. Voce di Jack

### 3.1 Il volume non è normalizzato: salti fino a +15,7 dB e clipping
- **Prova**: 35 take uniche tra -28,1 e -9,5 LUFS (mediana -24,2, dev. std 3,8 dB). Salti tra pezzi
  della stessa risposta: 027→028 +15,7 dB, 038→039 +12,8 dB (subito prima del tuo 040), 017→018
  +8,1 dB; 028 ha 658 campioni clippati. Il 040 "La chat è già aperta." è +7,4 dB sul 038.
- **Causa (verificata nel codice)**: nessuna normalizzazione dal modello alle casse; il worker fa
  solo `np.clip(-1, 1)` (`tts/jack_worker.py:246`, taglio netto); l'unica normalizzazione
  (`effects.py`) vale solo con `effect != "none"`, e Jack usa "none".
- **Soluzione**: "level guard" nel worker su ogni chunk (~0,67 s), prima della conversione a int16:
  livello attivo verso ~-24 dB (la mediana di oggi, così il volume di Windows resta giusto),
  guadagno -15..+6 dB, attacco veloce, rilascio lento con rampa, limiter morbido a -1 dBFS al posto
  del taglio. Latenza invariata, va bene in streaming.
- **Misura**: dev. std LUFS per pezzo ≤ 1,5 dB, nessun salto > 3 dB tra pezzi, 0 campioni clippati;
  test con toni sintetici.
- **Sforzo**: piccolo (~60 righe + test).

### 3.2 Il seme fisso ripete per sempre una take cattiva
- **Prova (verificata)**: stesso testo = stesso audio bit per bit: 003=005 "Apro la chat.",
  004=043 "Chiudo la chat." (acuta, +11 st), 008=019=024=029 "Mi spengo... accenditi." (Whisper
  sente 4 volte "Accendete"), 009=020=025=030. Il seme **non** lega il timbro tra testi diversi
  (registro da 81 a 229 Hz con lo stesso seme 7).
- **Causa**: `torch.manual_seed(7)` prima di ogni pezzo (`jack_worker.py:468-469`): il commento in
  config "keeps one timbre between replies" vale solo per testi identici.
- **Soluzione**: (a) tabella persistente `data/jack_takes.json` (testo + checkpoint + parametri →
  seme che ha superato il controllo di 3.3; se fallisce si prova seme+1, +2...); (b) per le risposte
  fisse del router uno script offline che genera N semi per frase, li valuta (registro, livello,
  durata, Whisper) e mette in cache il WAV migliore: a runtime 0 ms di TTS e take sempre buona.
- **Misura**: frasi fisse entro ±3 st dalla mediana e WER 0 al ritorno con Whisper; poi il tuo
  ascolto.
- **Sforzo**: medio.

### 3.3 Take acute, tese o urlate suonano senza controllo
- **Prova**: 5 take uniche su 35 oltre +5 st (002, 004/043, 028 urlato, 038 incomprensibile, 040);
  più frequenti nelle frasi corte (3/8 contro 2/27, campione piccolo). Acuto e forte vanno insieme
  (correlazione 0,48). "Robotica" non è misurata direttamente.
- **Causa**: ogni pezzo è un'estrazione di Qwen3; il monitor valuta la take solo **dopo** averla
  suonata (`tts/jack.py:334-335`).
- **Soluzione**: controllo della take in pochi ms (registro > +5 st, livello > +6 dB sull'obiettivo,
  clipping, caratteri/s fuori 12-24): per i pezzi successivi al primo si controlla l'intero pezzo
  mentre il precedente suona e si rigenera con seme+1; per il primo si guardano i primi 1-2 chunk e
  si rigenera solo se cattivo (+0,3-0,5 s, circa 1 take su 7).
- **Sforzo**: medio.

### 3.4 Il monitor della voce dà avvisi falsi e non misura il volume
- **Prova**: "f0 0 Hz" su 003/005, che si sentono bene (YIN trova il 3 % di frame sonori, meno di 5);
  circa metà dei "registro alto" non confermati; le frasi corte risultano "piatte" per costruzione;
  nessuna misura di livello.
- **Causa**: `tts/voice_check.py:24` (YIN 0,15, troppo severo), `:67-68` (meno di 5 frame → zeri),
  soglia di piattezza fissa (`:161-177`).
- **Soluzione**: voicing più largo, "n/d" sotto ~20 frame, registro sulla mediana più il massimo su
  finestre di 1-2 s, piattezza solo sopra ~8 sillabe, LUFS/picco/clipping nel log con avviso
  "volume". Le 44 WAV di oggi come fixture di regressione. È la base di 3.3.
- **Sforzo**: piccolo.

### 3.5 Risposta troncata: Jack dice "Attra"
- **Prova**: 016.wav, 0,56 s, "Attra": la storia delle 14:03:37 si è fermata a 400 token ("4 in,
  400 out") a metà di "Attraverso".
- **Causa (verificata)**: `brain/claude.py` gestisce solo `stop_reason` "refusal" (`:637`) e
  "tool_use" (`:721`); "max_tokens" no, e il resto senza punteggiatura finisce in voce dal flush
  dello splitter.
- **Soluzione**: con `max_tokens` scartare il frammento finale senza punteggiatura (flag dal brain
  allo splitter) o chiudere con una frase breve ("Il resto alla prossima."). Test sullo splitter.
- **Sforzo**: piccolo.

### 3.6 I pezzi lunghi (12-13,5 s) cambiano volume e registro a metà
- **Prova/ipotesi**: 018 (13,5 s) ha un urlo a 6-8 s; 023 e 015 calano di 7-8 dB verso la fine.
  Con `jack_group_chars = 220` un pezzo pieno supera i 12 s delle clip di training. Non dimostrato.
- **Soluzione**: A/B `jack_group_chars` 170 contro 220 con `jack_variants.py` (~10 min di GPU), poi
  il tuo ascolto; il level guard di 3.1 contiene comunque gli sbalzi.
- **Sforzo**: piccolo.

### 3.7 Volume a scatti quando parli sopra la risposta (ducking)
- **Causa**: un inizio di voce sopra -30 dBFS abbassa a 0,3 (-10,5 dB a gradino, `app.py:842-847`)
  e "non è per me" riporta subito a 1,0 (`:937-938`); il player cambia guadagno a fette da 100 ms
  senza rampa. Non verificato che sia scattato oggi (nessun log del ducking).
- **Soluzione**: log "duck on/off" con il livello, rampa di 50-100 ms; con 3.1 l'eco diventa più
  debole e uniforme e il ducking scatta meno.
- **Sforzo**: piccolo.

---

## 4. Pronuncia di Jack

### 4.1 Nessun accento sulle parole italiane: "accenditi" diventa "accendete"
- **Prova**: 008/019/024/029 → Whisper "Accendete" 4/4 (è una sola generazione ripetuta).
  **A/B fatto oggi** (seme 7 e 42): "accenditi" scritto normale → "accendete" 2/2; scritto
  "accènditi" o "accénditi" → giusto **6/6**. Ma l'accento su una parola piana peggiora:
  "sòcio" → "soccio" 4/4.
- **Causa (verificata)**: Qwen3-TTS accetta solo testo (niente fonemi/IPA/SSML: input `text`,
  `language`, `speaker`, `instruct`); il testo passa da `tts/spoken_text.py` `to_spoken_it`, che
  conosce solo la tabella `[tts.jack_words]` di 14 parole quasi tutte inglesi. Davanti a una forma
  rara il modello mette l'accento sulla penultima. Il problema riguarda le **sdrucciole**, soprattutto
  verbo + pronome attaccato (accènditi, riaccèndermi, accòmodati).
- **Soluzione**: passo `mark_stress` alla fine di `to_spoken_it` che scrive l'accento **solo** sulle
  parole con l'accento prima della penultima sillaba; non tocca le parole già accentate o nella tua
  tabella (che resta l'autorità), né gli omografi (àncora/ancóra). Per le forme con pronomi: staccare
  il pronome, cercare la base accentata, riattaccare. Lessico costruito una volta sola da Wiktionary
  (vedi decisione D.5) in `data/stress_it.tsv`; lookup < 1 ms a pezzo.
- **Misura**: test pytest senza GPU (`to_spoken_it("... accenditi.")` contiene "accènditi", "socio"
  invariato); lista fissa di ~60 parole difficili + gruppo di controllo di piane, semi 7 e 42,
  normali contro accentate, con Whisper; poi il tuo orecchio (Whisper non sente gli accenti che
  lasciano uguali le vocali).
- **Sforzo**: medio (~1 giorno).

### 4.2 Parole inglesi fuori tabella storpiate
- **Prova**: 021 "subprime è" → "supprime" (la "è" sparisce); A/B "sàbpràim" → giusto 2/2.
- **Causa**: `respell()` usa una lista chiusa di 14 parole; nessun log delle parole sconosciute.
- **Soluzione**: (1) la tua tabella resta l'autorità; (2) log una tantum delle parole non italiane e
  non in tabella ("Jack: parola sconosciuta 'subprime'") come lista di candidati; (3) proposte di
  riscrittura generate dall'IPA italiana delle ~1.088 parole prese dall'inglese su Wiktionary, da
  confermare con `jack_variants.py --words`.
- **Sforzo**: piccolo (log + prime voci) / medio (proposte automatiche).

> Nota di misura: alcuni errori sentiti da Whisper ("socio, che" → "so ciò che", "banditi" →
> "bandietti") sono difetti della singola estrazione, non del lessico. Una parola conta come errore
> di lessico solo se sbaglia con due semi e migliora con l'accento scritto.

---

## 5. Latenza

### 5.1 Con gli strumenti la prima parola aspetta tutte le chiamate (fino a 3,6 s)
- **Prova**: 14:07:15 M1 3586 ms, "TTS prima frase" 2104 ms: "Ci guardo, socio." (17 caratteri)
  è uscita solo dopo 3 chiamate a Claude (search_files, open_path, risposta finale), insieme a
  "Trovata sul desktop, la apro.". Stesso schema alle 14:07:48 e 14:08:05 (1312-1365 ms).
- **Causa (verificata nel codice)**: `tts/sentence_splitter.py` `group_stream` aspetta
  (`await ready.get()`, senza limite di tempo) che il primo pezzo arrivi a
  `jack_group_first_chars = 40` caratteri, anche mentre Claude sta eseguendo uno strumento.
- **Soluzione**: far partire il primo pezzo anche quando il flusso si ferma (nessun testo nuovo per
  ~150-200 ms, o un segnale "tool in corso" dal brain).
- **Misura**: "TTS prima frase" sulle richieste con strumenti da 1,3-2,1 s a ~0,3 s.
- **Sforzo**: piccolo.

### 5.2 Claude alla prima parola (~1,2-1,4 s), il resto di M1
- Oggi M1 locale ~0,8 s (ok), risposte di Claude 1,6-2,3 s. Manca la misura del TTFT di Haiku nel
  log per separare rete e modello; idee non provate: interiezione locale ("Mh.") mentre Claude
  pensa, richiesta partita sulla trascrizione parziale. (da IT-0 fluidità)
- **Sforzo**: piccolo (misura) / medio (idee).

### 5.3 Altre voci di latenza ereditate da IT-0
- large-v3: oggi STT 144-460 ms; seconda possibilità ~325 ms. Se pesa: turbo solo per i controlli
  veloci (+1,6 GB VRAM). (microfono)
- Frasi recuperate da Haiku +1-1,5 s; dopo `claudio_command` una seconda chiamata solo per dire
  "Fatto" (+0,6-1 s): dire subito la risposta del comando. (incerte) Con 2.1 le stime diminuiscono.
- Worker di Jack con la GPU occupata (un gioco): oggi pronto in 21 s, il 30/10 mattina nessuna
  risposta in 90 s e ripiego su Kokoro: avviso o attesa più corta se la GPU è piena.

### 5.4 Ripresa "Dicevo:" fuori luogo
- **Prova**: 14:08:19, dopo "chiudi la chat", Claudio riprende "Dicevo: Non è chiaro cosa intendi,
  socio: chi ha fatto cosa?", cioè la risposta a una frase capita male ("Chi ce l'ha fatta?").
- **Causa (ipotesi)**: la ripresa vale per ogni risposta interrotta entro `resume_s`.
- **Soluzione**: non riprendere una risposta che chiedeva un chiarimento o che rispondeva a una
  trascrizione con "Whisper insicuro". **Sforzo**: piccolo.

---

## 6. Funzioni mancanti verso la versione stabile (CLAUDE.md)

1. **Istanza singola** (nessun lock nel codice): con l'avvio automatico due CLAUDIO userebbero lo
   stesso microfono. Mutex di Windows con nome. Piccolo, prima di registrare il task.
2. **Registrare l'avvio all'accesso** (`scripts\install_autostart.ps1`, oggi nessun task "Claudio"),
   poi verificare il riavvio in caso di errore e il `.venv` fuori dalla app Claude.
3. **Icona nella system tray** (fase 5): serve `pystray`, dipendenza nuova da verificare e fissare.
4. **Memoria persistente** (fase 4): oggi c'è solo la cronologia breve della conversazione; mancano
   fatti e preferenze su SQLite e "ricordati che..." (verificato con grep, nessun modulo).
5. **Aggiornamento automatico con ritorno alla versione precedente** (fase 5): non presente.
6. `claudio --check` non controlla che il modello Whisper scelto (large-v3, ~3 GB) sia scaricato.
7. README: la riga di stato dice ancora "v0.2.2, Phase 2"; CLAUDE.md dice ancora "spegniti chiude il
   programma" (ora spegniti = spento, chiuditi = chiude).
8. Microfono migliore: il limite di 2.1 è l'hardware (cuffia a banda stretta con gate). Consiglio:
   microfono USB a filo vicino alla bocca. Banco STT: Parakeet, rumore vero dopo la frase, lista
   chiusa simulata (bassa priorità).

---

## 7. Decisioni per JenJo (rispondi con il numero e la scelta)

1. "basta" dedotto tra 50 e 80 %: chiedere "Hai detto: basta?" (oggi) o fermarsi subito?
2. "spegniti" dedotto sopra il 90 %: chiedere sempre (oggi) o spegnersi subito?
3. Ascolto di Jack R contro K (`logs\jack-variants\it0-fluidita\R` e `\K`): confermi K (seme 7 +
   top_p 0,9)? Con 3.2 il seme diventa per frase, non più globale.
4. Livello della voce (3.1): tenere ~-24 LUFS, cioè il volume medio di oggi?
5. Lessico degli accenti (4.1): kaikki.org (Wiktionary inglese, JSONL da 743 MB scaricato una volta,
   CC BY-SA 4.0, copre le forme flesse) oppure WikiPron (pochi MB, 89.608 voci, copertura dei
   verbi con pronomi non verificata)?
6. "desktop": "dèsk-top" (tua scelta di oggi) o "desk-tòp" (Wiktionary)?
7. Prima take cattiva (3.3): accetti +0,3-0,5 s solo quando viene rigenerata (circa 1 su 7)?
8. Segnale da spento (1.4): un tic leggero solo quando Whisper ha sentito il nome, dopo la
   correzione 1.1?
9. Salvare con un limite i WAV delle seconde possibilità rifiutate e di quello che senti mentre
   Claudio parla (1.2, 2.2)? Restano solo sul PC, come gli altri.
10. Tag di versione per la chiusura di IT-0 (es. `v0.4.0`)?

## 8. Da chiarire alla prossima prova

- 14:07:38 "apri la cartella di oggi": hai detto "tesi"? Per Whisper l'audio somiglia più a
  "di oggi".
- 14:07:58-14:08:05: cosa hai detto mentre Jack parlava? (audio non salvato, vedi decisione 9)
- La cuffia Sony permette di togliere il gate o allargare la banda (firmware, modalità)?
- Da spento hai detto "Hey Claudio, accenditi" o "Claudio, accenditi"?

## Ordine consigliato (proposta, la classifica finale la fa il coordinatore)

1. Bug e correzioni piccole ad alto effetto: 1.1, 1.2, 1.3, 3.1, 3.5, 5.1, 2.2a, 2.3, 3.4.
2. Ascolto robusto: 2.1, 2.2b-d, 1.4, 2.4.
3. Voce: 3.2, 3.3, 4.1, 4.2, 3.6, 3.7.
4. Sempre attivo: 6.1, 6.2, 6.3, poi 6.4 memoria e 6.5 aggiornamenti.
5. Riaddestramento wake word (1.5), latenza di Claude (5.2), resto di 5.3.
