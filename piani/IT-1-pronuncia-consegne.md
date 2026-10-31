# IT-1 – Consegne: pronuncia di Jack (4.1 accenti, 4.2 parole inglesi)

Branch locale `it1/pronuncia` (worktree `C:\Users\JenJo\Desktop\PROGETTI\claudio-it1-pronuncia`),
non pushato, aggiornato con `main` (v0.4.0). Decisioni usate: D.5 kaikki.org, D.6 "dèsk-top"
(era già nella tua tabella).

## In breve

- Jack ora scrive l'accento **solo sulle sdrucciole** (accento prima della penultima sillaba):
  "accenditi" → "accènditi", "riaccendermi" → "riaccèndermi", "dimmelo" → "dìmmelo".
  Le parole piane restano come sono, perché l'accento scritto le peggiora ("sòcio" → "soccio").
- Lessico da Wiktionary (kaikki.org), costruito una volta sola in `data\stress_it.tsv` (4,7 MB,
  versionato, licenza CC BY-SA 4.0 nell'intestazione): 110.341 sdrucciole e 61.515 forme verbali
  che prendono i pronomi. Costo: 0,08 ms a frase; 0,1 s di caricamento all'avvio di Jack.
- La tua tabella `[tts.jack_words]` resta l'autorità: le sue parole e le sue riscritture non
  vengono mai toccate. Per togliere un accento che non ti piace basta una riga in tabella
  (es. `numero = "numero"`).
- Parole inglesi (4.2): quelle che sembrano straniere e non sono in tabella finiscono nel log
  **una volta** ("Jack: parola sconosciuta 'subprime'"), con una proposta da Wiktionary quando
  c'è (556 parole in `data\respell_en.tsv`). Le proposte **non** vengono applicate: le senti con
  `jack_variants.py --words` e scegli tu.

## Misure (A/B con Jack e Whisper, 30-31/10)

Whisper scrive cosa sente al posto di ogni parola; "giusta" = ha scritto la parola giusta.
Non sente gli accenti che lasciano uguali le vocali, quindi conta il tuo orecchio (WAV sotto).

| Prova | Senza accento | Con accento |
|---|---|---|
| **La frase del 30/10**: "Per riaccendermi di' Claudio, accenditi." → "accenditi" | 1/3 (2 volte "accendete") | **3/3** |
| 26 frasi come quelle di Claudio, 40 parole, semi 7, 42, 99 | 123/129 | 123/129 |
| di cui verbo + pronomi (accomodati, sbrigati, siediti...) | 5 errori | 3 errori (2 su "càlmati") |
| 90 parole da sole in "Socio, ho detto: ...", semi 7 e 42 | 142/152 | 139/152 |

- L'errore vero nasce **dentro la frase**: la stessa parola da sola Jack la dice giusta anche senza
  accento. L'accento sistema proprio il caso del 30/10 e non peggiora le parole comuni.
- Eccezioni misurate, lasciate senza accento: le parole che iniziano con "ù" ("ùtile" → "etile"
  3 volte su 5, "ùltimo" → "giltimo") e "càlmati" ("chiamati"/"cialmati" 3 volte su 5).
- Nessun segno dell'effetto "soccio" sulle sdrucciole (nùmero, mùsica, vìsita, pàgina giuste).
- **"Apro Chrome" → "apro croma"** (segnalato dal thread voce): non riprodotto in 8 prove. La causa
  comune è la consonante finale **a fine frase**: "Apro clòd." diventa "clado"/"proclode" 2 volte
  su 5, "clòdd" mai; a metà frase "cròm" e "clòd" sono giusti 10 su 10 (decisione 3a, applicata).

WAV da ascoltare, in `C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant\logs\pronuncia-it1\`:
- `frasi-vere\7\` (e `\42\`, `\99\`): `frasi_01_riaccendermi_scritta.wav` contro
  `frasi_01_riaccendermi_accentata.wav`, e così per le 26 frasi; report in `frasi-vere\report.md`.
- `parole-isolate\`: le 90 parole da sole, con `report.md`.
- `parole-inglesi\`: per ogni parola `_1` = come scritta, `_2` = la proposta (elenco in `elenco.md`).
- `consonante-finale\1\` (e `\7\`, `\42\`, `\99\`, `\123\`): `apro_claude_clòd.wav` contro
  `apro_claude_clòdd.wav`, e lo stesso per Chrome.

## Fatto

| Cosa | Dove |
|---|---|
| `mark_stress`: accento sulle sdrucciole; verbo + pronomi spezzato al volo (imperativo, infinito senza la e finale, gerundio); omografi lasciati (àncora/ancóra, sùbito/subìto, càpiti/capìti); "Fermati!", "Ricordati di..." accentati solo a inizio frase e solo per i 18 imperativi che Jack dice; mai su una "ù" iniziale né su "calmati" | `src\claudio\tts\stress_it.py`, `scripts\build_lexicon_it.py` |
| Gli accenti entrano in `to_spoken_it` dopo la tabella e **prima** dei numeri; i numeri in lettere ("tredici") restano come sono | `src\claudio\tts\spoken_text.py` |
| Log delle parole che sembrano straniere fuori tabella, una volta ciascuna (massimo 300 a sessione); niente "ed", "oh", "opzioni", "anch'" | `src\claudio\tts\loanwords.py` |
| Costruzione del lessico e delle proposte da kaikki.org | `scripts\build_lexicon_it.py` |
| Lessico e proposte (versionati con un'eccezione in `.gitignore`) | `data\stress_it.tsv`, `data\respell_en.tsv` |
| A/B con Jack e Whisper: parole isolate o 26 frasi vere (`--frasi`) | `scripts\stress_ab.py` |
| `jack_variants.py --words PAROLA...`: ogni parola come scritta (davvero, senza accenti), come in tabella e come proposta; voce come in Claudio; il WER non conta gli accenti scritti | `scripts\jack_variants.py` |
| Traduzioni inglesi dei messaggi di log nuovi | `src\claudio\i18n.py` |
| Test: 159 nuovi | `tests\unit\test_stress_it.py`, `test_loanwords.py`, `test_build_lexicon_it.py`, `test_stress_ab.py`, `test_jack_variants_words.py`, `test_spoken_text.py` |

Revisione indipendente (3 revisori + 1 verificatore): 20 difetti trovati e riprodotti, tutti
corretti, tra cui "famósissimo", "bènone", "Ci siamo càpiti", "sono ora collégati",
"salvati e órdinati", "1.250 risùltati", proposte come "agngnòstico".
Risultati finali: `pytest` 1912 passed, 4 skipped; ruff e mypy puliti.

## Prova in locale (dopo il merge in `main`)

1. Apri PowerShell nella cartella del progetto:
   `cd C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant`
2. Prova veloce senza GPU: `uv run python -c "from claudio.tts.spoken_text import to_spoken_it; print(to_spoken_it('Socio, accenditi e dimmelo subito.'))"`
   Deve stampare: `Socio, accènditi e dìmmelo subito.`
3. Avvia Claudio come sempre, di' "Claudio, spegniti" e ascolta la risposta
   "Per riaccendermi di' Claudio, accenditi.": deve dire "riaccèndermi" e "accènditi".
4. Parole inglesi nuove: alla fine della sessione cerca nel log
   `Select-String -Path logs\sessions\*\*.log -Pattern "parola sconosciuta"`.
5. Per ascoltare le alternative di una parola (chiudi prima Claudio, serve la GPU):
   `uv run python scripts\jack_variants.py --words subprime chat link`
   I WAV sono in `logs\jack-variants\parole\` con l'elenco in `elenco.md`.

## Decisioni di JenJo (31/10, applicate)

1. **Tutte** le proposte inglesi in tabella: on/òn, off/òf, offline/oflàin, link/lìnk, app/èpp,
   chat/ciàt, software/sòftuer, weekend/uichènd, meeting/mìting, team/tìm, backup/becàp,
   server/sèrver, update/apdèit, cloud/clàud (`config\claudio.toml`, `[tts.jack_words]`).
   "subprime" non ha proposta: resta nel log delle parole sconosciute.
2. **2a**: accento su tutte le sdrucciole (eccezioni misurate escluse), nessuna modifica.
3. **3a**: a fine frase una riscrittura della tabella che finisce con vocale + d, t, p, k, b, g
   raddoppia la consonante: "Apro Claude." → "Apro clòdd.", "Sul desktop" → "Sul dèsk-topp",
   "Apri la chat." → "ciàtt". A metà frase resta com'è ("Chiedo a clòd, socio."). Dopo un'altra
   consonante non la raddoppio ("pàssuord" era già giusta 3 su 3 e l'italiano non scrive "rdd");
   "cròm" finisce con m, che nell'A/B non cambiava nulla (`closed()` in `spoken_text.py`).

## Per chi fa il merge

- Il ramo `it1/voce` salva le take fisse in `data\jack_takes.json` indicizzate dal testo di
  `to_spoken_it`: dopo il merge di questo ramo le frasi con una sdrucciola cambiano testo, quindi
  vanno rigenerate (`scripts\jack_takes.py`). La firma di `to_spoken_it(text, words)` non cambia
  (c'è solo `stress=` opzionale).
- Righe pronte per il CHANGELOG:

```
### Added
- Jack writes the stress on sdrucciole (accènditi, riaccèndermi, dìmmelo) from a lexicon built
  once from Wiktionary (data/stress_it.tsv, CC BY-SA 4.0); regular words stay as written.
- English words outside [tts.jack_words] are logged once as candidates, with a proposal from
  Wiktionary's Italian IPA (data/respell_en.tsv, never applied).
- scripts/build_lexicon_it.py, scripts/stress_ab.py, jack_variants.py --words WORD...
```

## Dati e licenza

- Sorgente: `https://kaikki.org/dictionary/Italian/kaikki.org-dictionary-Italian.jsonl.gz`
  (77 MB compressi, stessi dati del file da 743 MB), scaricato una volta in
  `C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant\models\kaikki\` (fuori da git).
- Per ricostruire il lessico (~20 s): `uv run python scripts\build_lexicon_it.py models\kaikki\kaikki.org-dictionary-Italian.jsonl.gz`
- Se il file scaricato non ti serve più, cancellalo tu:
  `Remove-Item C:\Users\JenJo\Desktop\PROGETTI\Claudio-AI-desktop-assistant\models\kaikki -Recurse`
- `data\stress_it.tsv` e `data\respell_en.tsv` sono adattamenti di Wiktionary: restano
  CC BY-SA 4.0 (intestazione nel file), separati dal codice di CLAUDIO.
