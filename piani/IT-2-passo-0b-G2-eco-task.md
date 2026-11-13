# G2, correzione H2: il nome sopra una risposta che lo dice

> **Per gli agenti:** SUB-SKILL OBBLIGATORIA: superpowers:subagent-driven-development (o superpowers:executing-plans). I passi usano le caselle (`- [ ]`).

**Obiettivo:** «Claudio, che ore sono?» detto sopra una risposta che nomina Claudio la interrompe, invece di finire scartato come eco di Claudio stesso.

**Causa (verificata il 13/11 sul testo vero della sessione `2026-11-12_19-07`):** `Assistant._is_echo` (`src/claudio/app.py:1573-1582`) protegge la frase con il nome solo se la risposta non dice «Claudio». Le due storie di H2 avevano Claudio come protagonista, quindi la frase passava ai controlli sulle parole, che contano «claudio» come parola della risposta, più parole brevi presenti in ogni storia lunga:

- storia 1: `word_share` 0,75 (claudio, che, ore da «tre ore», non ancora detto) ≥ 0,6;
- storia 2: `echo_of` vero (claudio, «sono» simile a «solo»).

Il controllo sulle lettere, invece, separa bene: le frasi di JenJo stanno a 47-59 contro ciò che suonava, l'eco vera delle frasi col nome a 97-100 (soglia `echo_similarity` 70).

**Correzione (D, scelta di JenJo il 13/11 dopo le due revisioni):** quando la frase sentita e la risposta dicono tutte e due il nome, decide `echo_around_name(heard, said, now)` in `src/claudio/echo.py`. Confronta le parole intorno al nome nella frase sentita con le parole negli stessi posti intorno a ogni «Claudio» già detto (`said`, tutto ciò che è stato suonato); è eco se una coppia arriva a `NAME_ECHO` 70 (rapidfuzz `ratio`). Il nome da solo è eco se ciò che suona ora (`now`) lo dice. `_is_echo` in `app.py` la chiama con `" ".join(speaker.played) or recent` e `recent`; il resto non cambia. Vale anche per la finestra dopo la risposta (`_window_echo`, `app.py:959`), che usa la stessa funzione.

**Perché non la prima correzione (A, solo le lettere di ciò che suona):** le 7 lettere di «claudio» pesano su una richiesta corta, e le revisioni Opus e `/code-review` hanno trovato comandi persi sopra una frase col nome («Claudio, pausa» 75, «Claudio, no» 89). Banco del 13/11 (4 storie, 2 vere di H2, 34 richieste su ogni frase, 48 eco col nome): A perdeva 53 richieste su 460 e 19 eco in ritardo; le lettere senza il nome a soglia 80 perdevano 78 richieste su 782 (le corte: «no», «sì», «ancora»); D ne perde 0 su 782 e prende 47 eco su 48. Punteggi di D: richieste al massimo 62, eco almeno 92; l'eco persa è una storpiatura estrema («Claudio, sole e fine») che nessuna variante prende.

**Seconda revisione Opus (13/11), due affinamenti di D:** la finestra intorno al nome è elastica (da n-1 a n+2 parole per lato), perché Whisper perde le parole corte («Claudio salvò» per «Claudio lo salvò»); con una sola parola intorno al nome («Claudio, sì») il confronto è solo con ciò che suona ora, perché in una storia lunga «Claudio si alzò» si trova sempre. Banco allargato (una storia in più piena di «Claudio si / non / continuò», le eco del revisore): 21 richieste perse su 1092, tutte di una parola o con le parole stesse della storia intorno al suo nome («Claudio, era bello» sopra «Claudio. Era così bello»), e 1 eco persa su 74.

**Spec:** `piani/IT-2-passo-0b-prove.md`, sezione «H2», riga «Esito H2 (12/11)»; `piani/IT-2-passo-0b-G2-task.md`, Task 7.

## Vincoli globali

- Comandi: `uv run python -m pytest -q`, `uv run python -m mypy src`, `uv run ruff check .` e `uv run ruff format --check .` (la CI li esegue tutti).
- Mai un grep ricorsivo dalla radice del repo: `models/` e `samples/` pesano ~135 GB. Solo `src`, `tests`, `scripts`, `config` e i README.
- Prima di cercare nel codice: `graphify query` / `graphify explain`.
- Gli implementer non committano. Codice e commenti in inglese.
- Nessuna chiamata API: i test usano finti; il gate usa le cassette.

## Review Focus

1. Eco molto storpiata da Whisper di una frase col nome (parole intorno al nome sotto 70): Claudio si interromperebbe da solo e manderebbe la sua eco a Claude. Guardie: test delle eco esatta, storpiata e in ritardo; gate di replay invariato a 38/0/3; passo «storia senza parlare» nella prova H2 bis.
2. «Claudio?» o «Ehi Claudio» da soli mentre la frase che suona dice il nome: restano eco, ignorati. Limite accettato, fissato da un test.
3. Finestra dopo la risposta (`_window_echo`): stessa funzione, stessa regola; la coprono i test unitari della funzione.
4. `echo_similarity = 0`: spegne ancora il controllo sulle lettere delle frasi senza nome; la regola del nome non dipende più da quel valore.
5. Nome solo nella parte di risposta non ancora detta: non c'è un «Claudio» suonato a cui ancorarsi, quindi JenJo è sentito (test unitario).

---

### Task 1: il nome di JenJo sopra una risposta che lo dice

**File:**

- Modifica: `src/claudio/echo.py` (`NAME_ECHO`, `echo_around_name`), `src/claudio/app.py:22` (import) e `src/claudio/app.py:1573-1586` (`_is_echo`)
- Test: `tests/unit/test_timeline.py`, dopo `test_17_13_story_then_new_request_does_not_resume`; `tests/unit/test_echo.py` in fondo. Con D si aggiungono: l'eco in ritardo (terzo caso della guardia), `test_a_short_command_over_a_sentence_with_the_name_is_jenjo` e i test unitari di `echo_around_name`.

**Interfacce:** nuova `claudio.echo.echo_around_name(heard: str, said: str, now: str) -> bool` (D). La prima versione (A) usava `claudio.echo.similarity(heard: str, reply: str) -> float`.

- [ ] **Passo 1: scrivi i test che falliscono** (frasi prese dalle due storie vere di H2)

```python
HERO = [  # H2 12/11, first story: CLAUDIO is the hero (word_share: claudio, che, ore)
    "C'era una volta un eroe bellissimo di nome Claudio. ",
    "Era così bello che gli specchi chiedevano un autografo. ",
    "Trovò un socio che cercava da tre ore la porta che aveva davanti. ",
    "Claudio lo salvò, naturalmente. ",
    "Il robot esplose di ammirazione, letteralmente.",
]
MOON = [  # H2 12/11, second story (echo_of: claudio, "sono" like "solo")
    "Un'altra? Insaziabile, socio, ma il genio non si nega. ",
    "C'era una volta una luna abbandonata, dove un solo ristorante serviva zuppa di roccia. ",
    "Ci arrivò Claudio, ovviamente il cliente più bello della storia. ",
    "Il cuoco, un robot arrugginito, gli disse che nessuno finiva mai il piatto. ",
    "Claudio sorrise e lo finì in tre cucchiaiate.",
]


@pytest.mark.parametrize("story", [HERO, MOON], ids=["word-share", "echo-of"])
async def test_the_name_over_a_story_about_claudio_is_jenjo(make_scene, story) -> None:
    scene = make_scene(text_reply(*story), sentence_frames=PLAY, window_s=4.0)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.advance(1.5)
    await scene.say_over("Claudio, che ore sono?")
    await idle(scene)
    assert [r["outcome"] for r in scene.rows[:2]] == ["interrupted", "ok"]
    assert "Sono le 14 e 5." in scene.speaker.spoken
    assert resumed(scene)


@pytest.mark.parametrize("echo", ["Claudio lo salvò naturalmente", "Claudio?"])
async def test_the_echo_of_a_sentence_with_the_name_is_still_echo(make_scene, echo) -> None:
    scene = make_scene(text_reply(*HERO), sentence_frames=PLAY, window_s=4.0)
    await scene.say("raccontami una storia")
    await scene.until_speaking()
    await scene.until(lambda: scene.speaker.played[-1:] == [HERO[3].strip()])
    await scene.say_over(echo)
    await idle(scene)
    assert scene.row(0)["outcome"] == "ok"
```

- [ ] **Passo 2: lancia e guarda il rosso**

`uv run python -m pytest -q tests/unit/test_timeline.py -k "story_about_claudio or still_echo"`
Atteso: 2 falliti (`word-share`, `echo-of`, `['ok', 'timeout'] != ['interrupted', 'ok']`, nel log «La mia voce nel microfono, la ignoro: 'Claudio, che ore sono?'»), 2 passati (le guardie).

- [ ] **Passo 3: correggi `_is_echo`**

Dopo la riga `if heard_wake_word(text) and not heard_wake_word(reply): return False`, e dopo aver letto `min_similarity`: se `heard_wake_word(text)` e `min_similarity > 0`, restituisci `similarity(text, recent) >= min_similarity`. Commento di una riga sul perché (H2 12/11: una storia su Claudio dice il nome e le parole brevi di ogni richiesta).

- [ ] **Passo 4: lancia e guarda il verde**

Stesso comando. Atteso: 4 passati.

- [ ] **Passo 5: controlli completi**

`uv run python -m pytest -q`, `uv run python -m mypy src`, `uv run ruff check .`, `uv run ruff format --check .`. Atteso: tutto verde, conto dei test = prima + 4.

### Task 2: chiusura (coordinatore)

- [ ] Revisione del task (Opus), poi `/code-review` sull'intero diff e `superpowers:verification-before-completion`.
- [ ] Gate a Claudio chiuso: `uv run python scripts/replay_gate.py --report`. Atteso: 38 superate, 0 fallite, 3 note.
- [ ] Prova a mano «H2 bis» (i passi 1-3 di H2 dal vivo) in `piani/IT-2-passo-0b-prove.md` e, nella stessa modifica, la sua riga in `piani/prove-da-fare.md`.
- [ ] `graphify update .`
- [ ] Commit, ognuno approvato: `fix(barge): the name over a reply that says it is JenJo's` (codice + test), `docs(plan): H2 bis manual test and the G2 echo fix plan`, `chore(graph): refresh graphify-out`. Identità noreply di JENJ0 controllata, push solo di `it2/passo-0b`, CI verde.

## Fatto quando

Il test a timeline prima rosso e poi verde; suite, `ruff check`, `ruff format --check` e mypy verdi; gate di replay invariato a 38/0/3; prova H2 bis scritta nel file delle prove e nella stessa modifica in `piani/prove-da-fare.md`; commit approvati e spinti; CI verde. Poi riassunto a JenJo e stop.
