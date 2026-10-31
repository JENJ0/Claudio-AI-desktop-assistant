# IT-1 – Voce di Jack (sezioni 3.1-3.7): consegna

Ramo `it1/voce` (worktree `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\claudio-it1-voce`), solo locale, non
unito: base `62f2b79`, 4 commit.

| Commit | Cosa |
|---|---|
| `667bd75` | 3.5: mai più il frammento di una risposta tagliata da max_tokens ("Attra") |
| `de6b7da` | 3.7: ducking con dissolvenza e log |
| `e6324e2` | 3.1 + 3.2 + 3.3 + 3.4: level guard, controllo della take, seme per testo, frasi fisse |
| `2252619` | latenza della prima frase (misurata sulla GPU) e semi cattivi ricordati |
| (dopo) | 3.6: `jack_group_chars` 220 → 170 (A/B, scelta di JenJo) e questa consegna |

Test: 1837 unit test passano (+84 rispetto a IT-0), ruff e mypy puliti. Ogni commit passa i suoi test
anche da solo. Fonti delle misure: le 44 WAV di Jack della prova del 30/10
(`logs\sessions\2026-10-30_14-00\jack\`) e due prove sulla GPU con il worker vero sulle stesse
risposte (gli script sono nello scratchpad della sessione, non nel repo).

## Cosa cambia per JenJo

- Il volume di Jack è uguale per tutte le frasi (circa -24 LUFS, la media di oggi): niente più
  salti di volume né distorsione sui picchi.
- Le take acute, urlate o che "si perdono" vengono rigenerate prima che tu le senta. La prima
  frase di una risposta viene giudicata sul suo primo pezzo: se è buona parte subito, se è cattiva
  costa 0,5-0,8 s in più (scelta 1a di JenJo, 30/10).
- Un testo venuto bene una volta viene ridetto sempre uguale (stesso seme); uno venuto male non
  riparte più dallo stesso seme.
- Le risposte fisse (comandi, "Sono pronto.", conferme) si possono preparare in anticipo con
  `scripts\jack_takes.py`: Jack le dice senza TTS (0 ms) e sempre con la take migliore.
- Una risposta di Claude tagliata dal limite di token non finisce più con un pezzo di parola: Jack
  chiude con "Mi fermo qui: se vuoi, continuo.". Vale anche per "ripeti" e per "Dicevo: ...".
- Quando parli sopra la risposta, il volume scende e risale con una dissolvenza di 80 ms invece che
  a scatti; nel log c'è la riga con il livello della tua voce.

## 3.1 Volume (level guard) – fatto

- **Come**: `src/claudio/tts/loudness.py` (solo numpy e scipy). Contiene il loudness BS.1770: il
  K-weighting per ogni frequenza di campionamento, verificato sulla tabella dello standard a 48 kHz.
  `LevelGuard` gira nel worker su ogni pezzo audio prima della conversione in int16. Porta il volume
  a -24 LUFS misurando gli ultimi 2 s di voce (le pause non contano), con guadagno tra -15 e +6 dB.
  Il guadagno scende in 50 ms e risale al massimo di 4 dB/s, sempre con una rampa. Un blocco più
  forte di 8 dB sull'obiettivo viene abbassato subito. Un limiter con look-ahead tiene i picchi
  sotto -1 dBFS invece di tagliarli. I convertitori del worker ora restituiscono audio float, e il
  livello grezzo della take (`raw_lufs`, `raw_clipped`) viaggia negli eventi `audio` e `done`.
- **Misura (44 WAV del 30/10)**: differenza tra i pezzi da 4,1 a 0,9 dB, salto massimo da 15,7 a
  2,6 dB, campioni clippati da 204 a 0.
- **Misura (worker vero, 27 pezzi)**: differenza 0,90 dB, salto massimo 2,9 dB (nessuno sopra
  3 dB), 0 clippati, mediana -24,4 LUFS.
- **Config**: `jack_lufs = -24` (0 = spento).

## 3.2 Seme per testo e frasi fisse – fatto (frasi fisse da generare sulla GPU)

- `src/claudio/tts/takes.py`, file `data/jack_takes.json`. Per ogni testo (il testo come lo legge
  Jack, cioè dopo `to_spoken_it`) salva il seme di una take giudicata buona e i semi giudicati
  cattivi. Il testo successivo uguale parte dal seme buono e salta quelli cattivi. Ogni chiave
  include un profilo della voce: checkpoint, campionamento, velocità, `jack_lufs` e `TAKES_VERSION`.
  Così un altro checkpoint non riusa take vecchie. I percorsi sono relativi a `models/`, quindi la
  tabella vale per qualsiasi copia del repo. Tiene 2000 testi al massimo (via i meno usati) e non
  cancella mai le WAV salvate.
- `scripts/jack_takes.py` prepara le risposte fisse. Prende i `reply`/`confirm` di
  `commands.toml`, le frasi di conferma e "Sono pronto."; con `--from-logs N` aggiunge le risposte
  dette almeno N volte nei log. Genera N semi per frase (8 di default), li giudica con registro,
  livello, velocità e WER di Whisper, e salva la migliore in `data/jack_cache/`. Report e tutte le
  take finiscono in `logs\jack-takes\`. A runtime una frase salvata si suona dalla WAV, anche con il
  worker spento o ancora in caricamento.
- **Lanciato il 31/10** (`--from-logs 3`): 22 frasi fisse × 8 semi, salvate in
  `C:\Users\JenJo\Desktop\PROGETTI\CLAUDIO\Claudio-AI-desktop-assistant\data\jack_cache\` più
  `data\jack_takes.json` del checkout principale: il codice attuale di main le ignora, dopo l'unione
  si usano. Le take scartate vanno da 0 a 5 su 8 per frase. Il registro mediano delle take buone è
  129 Hz, coerente con `jack_usual_hz = 132`. Whisper sente giuste 18 frasi su 22; "Apro Chrome."
  resta "apro croma" anche nella migliore delle 8 (lavoro per il thread pronuncia). Le altre tre
  differenze sono di punteggiatura o accento ("dì", "di'"). "accenditi" ora viene sentito giusto
  (il 30/10 era "Accendete" quattro volte su quattro). Report: `logs\jack-takes\report.md` del
  worktree.
- **Va rilanciato dopo aver unito `it1/pronuncia`**, perché la chiave usa il testo pronunciato: le
  frasi con parole cambiate (per esempio "accènditi") non combacerebbero più e verrebbero generate
  dal vivo.

## 3.3 Controllo della take prima di suonarla – fatto

- `TakeJudge` (`voice_check.py`) scarta una take quando:
  - è acuta: mediana più di 5 semitoni sopra `jack_usual_hz` = 132 Hz, oppure 1,5 s qualsiasi più
    di 9 semitoni sopra (l'urlo dentro il pezzo 018);
  - è uscita dal modello più di 6 dB sopra l'obiettivo (urlata: il level guard pareggia il volume,
    non l'urlo) o oltre il fondo scala;
  - va a meno di 12 caratteri/s (con 0,8 s di margine per le pause), che si vede anche prima che
    sia finita, oppure, quando è intera, a più di 24 caratteri/s.
- Il riferimento è fisso: la mediana appresa dalle take suonate scendeva a 115 Hz e scartava metà
  delle take. Sulle 36 take uniche del 30/10 ne scarta 8: le cinque acute che avevi sentito (002,
  004, 028, 038, 040), più 012, 018 e 039.
- In `jack.py`:
  - Un pezzo che apre una risposta aspetta solo il suo primo chunk (~0,67 s di audio). Viene giudicato
    lì e rigenerato con il seme dopo al massimo `jack_first_retries` = 1 volta.
  - I pezzi successivi hanno un primo controllo a 0,6 s. Poi vengono trattenuti finché sono interi
    o finché a ciò che suona prima restano 0,4 s, e rigenerati (`jack_take_retries` = 2) solo se
    resta più di 1 s da suonare: mai un buco.
  - Ogni take giudicata aggiorna la tabella del 3.2.
- **Prima prova GPU (sbagliata, corretta)**: il pezzo di apertura aspettava di poter misurare il
  registro, e la prima parola arrivava a 0,8-1,7 s. Ora (misurato con League of Legends aperto
  sulla GPU, quindi tutto un po' più lento del normale):
  - senza controllo la prima parola arriva in 375 ms (mediana);
  - con il controllo e la take buona, +30 ms;
  - quando la take viene rigenerata, +0,5-0,8 s. È successo in 2 risposte su 16 delle risposte del
    30/10 e in fino a 1 su 3 con frasi molto corte. La tabella dei semi lo evita per i testi
    ripetuti, la cache delle frasi fisse per i comandi.
- Il worker genera a circa 0,55 volte il tempo reale (mediana; massimo 0,77): i pezzi dopo il
  primo vengono giudicati interi senza pause.

## 3.4 Monitor della voce – fatto

- YIN calcolato con la FFT su tutti i frame insieme. Su 17.636 frame dà gli stessi valori del YIN
  di IT-0, frame per frame (c'è un test), ed è 11 volte più veloce.
- Soglia di voicing 0,3 invece di 0,15: i pezzi misurabili passano da 26 a 34 su 36. Sotto 20
  frame sonori il log scrive "f0 n/d" invece di "0 Hz".
- Il registro si misura sulla mediana e sul massimo delle finestre di 1,5 s. La "melodia piatta"
  conta solo sopra 8 sillabe.
- Il log riporta LUFS e picco dopo il level guard e il livello grezzo della take, e avvisa
  "volume" se un pezzo resta oltre ±3 dB dall'obiettivo.
- Il monitor usa lo stesso `TakeJudge` del controllo, quindi un avviso nel log significa che quella
  take sarebbe stata scartata (o è stata suonata per mancanza di tempo).

## 3.5 Risposta tagliata ("Attra") – fatto

- `CUT_SHORT` (`protocols.py`) è un marcatore che vale "" e si riconosce per identità. Il cervello
  lo emette con `stop_reason == "max_tokens"`, seguito da "Mi fermo qui: se vuoi, continuo." (in
  inglese per le risposte inglesi), e scrive `cut: max_tokens` nelle metriche.
- Lo splitter, al marcatore, dice le frasi completate e scarta il frammento.
- `joined_reply()` fa lo stesso per "ripeti" e per la ripresa "Dicevo: ...".
- Causa a monte: il prompt di sistema ora dice che una risposta lunga sta sotto le 150 parole circa
  e propone di continuare.
- **Attenzione all'unione con `it1/latenza`**: lì `group_stream` legge i chunk grezzi, quindi deve
  gestire anche lui `chunk is CUT_SHORT` con `drop_unfinished()`, altrimenti "Attra" torna.
  Unione di riferimento: il ramo `it1/prova-latenza-voce` del thread latenza.

## 3.6 Pezzi lunghi: A/B `jack_group_chars` 170 contro 220 – fatto, scelto 170

Le 8 risposte lunghe più recenti dei log, divise come fa lo speaker, con il level guard acceso e il
controllo delle take spento (worker vero, 31/10):

| Lunghezza | Pezzi | Durata media | Oltre 12 s | Sbalzo di volume nel pezzo | Sbalzo di registro | WER |
|---|---|---|---|---|---|---|
| 170 | 19 | 6,4 s | 1 | 1,8 dB | 3,6 st | 1,4 % |
| 220 | 14 | 8,9 s | 3 | 2,5 dB | 3,8 st | 1,3 % |

Un pezzo pieno a 220 caratteri dura 13-13,5 s, oltre i 12 s delle clip di addestramento; a 170
resta sotto, tranne una frase sola più lunga che non si può dividere. In cambio i pezzi sono un
terzo in più. JenJo ha scelto 170 (31/10): `jack_group_chars = 170`. Le WAV dell'A/B sono in
`logs\jack-ab-group\` del worktree (`g170_*` contro `g220_*`).

## 3.7 Ducking – fatto

- `AudioPlayer`: il guadagno della risposta sfuma campione per campione (80 ms per un salto intero,
  56 ms per scendere a 0,3) e prosegue tra una fetta e l'altra. A volume pieno l'audio resta
  identico al bit, i bip non vengono mai toccati, e dopo un silenzio il volume chiesto vale subito.
- Log: "Abbasso la risposta: voce a X dBFS" (in `app.py`) e "Risposta di nuovo a volume pieno"
  (solo quando era abbassata).

## Configurazione nuova (`config/claudio.toml`, sezione `[tts]`)

```toml
jack_lufs = -24            # level guard (0 = spento)
jack_take_check = true     # controllo della take prima di suonarla
jack_take_retries = 2      # take in più per i pezzi dopo il primo
jack_first_retries = 1     # take in più per il primo pezzo (0 = mai trattenuto)
jack_usual_hz = 132        # registro abituale di Jack (riferimento di "troppo acuta")
jack_group_chars = 170     # cambiato: era 220 (3.6)
```

## Prova in locale (dopo l'unione, GPU libera, nessun gioco aperto)

1. Le frasi fisse sono già pronte nel `data\` del checkout principale. Dopo l'unione di
   `it1/pronuncia` rifalle una volta: `uv run python scripts/jack_takes.py --from-logs 3`
   (circa 10 minuti, Claudio chiuso, nessun gioco aperto), poi leggi `logs\jack-takes\report.md`.
2. Avvia Claudio come sempre.
3. Prova:
   - "Claudio, apri la chat": con le frasi fisse pronte, la risposta parte subito.
   - "Claudio, raccontami una storia lunga": volume costante dall'inizio alla fine; se viene tagliata
     finisce con "Mi fermo qui: se vuoi, continuo.", mai con mezza parola. Poi "ripeti".
   - Parla sopra una risposta lunga senza dire il nome: il volume scende e risale senza scatti.
4. Nel log della sessione (`logs\sessions\<data>\console.log`) cerca:
   - "Voce Jack NNN.wav: f0 ..., -24.x LUFS ..." (volume a posto, "n/d" invece di 0 Hz);
   - "Take di Jack scartata (seme N): ..." (quante take sono state rigenerate e perché);
   - "Abbasso la risposta: voce a X dBFS" (quando è scattato il ducking).

## Rischi e cose aperte

- Prima frase rigenerata: +0,5-0,8 s quando capita (scelta 1a). Con `jack_first_retries = 0` sparisce
  l'attesa, e il seme cattivo viene comunque ricordato per la volta dopo.
- Con la GPU occupata da un gioco tutto il TTS rallenta (prima parola da ~220 a ~380 ms, misurato).
- La chiave della tabella usa il testo pronunciato: dopo l'unione di `it1/pronuncia` le take salvate
  di prima non combaciano più (nessun danno, si rigenerano), quindi `jack_takes.py` va rilanciato.
- Se il worker cambia il modo di produrre l'audio (level guard, convertitori), alza `TAKES_VERSION` in
  `takes.py`.
