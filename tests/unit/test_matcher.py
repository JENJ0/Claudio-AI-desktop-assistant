"""Router table: >= 5 variants per command, typical Whisper errors, and negatives."""

from pathlib import Path

import pytest

from claudio.router.matcher import CommandMatcher, load_commands

POSITIVE = [
    # open_chrome
    ("Apri Chrome.", "open_chrome"),
    ("Apri il browser", "open_chrome"),
    ("Claudio, apri Google Chrome per favore", "open_chrome"),
    ("Avvia Chrome", "open_chrome"),
    ("apri crome", "open_chrome"),
    ("Apri Krome.", "open_chrome"),
    # open_downloads
    ("Apri la cartella Download", "open_downloads"),
    ("Apri i download.", "open_downloads"),
    ("Apri la cartella dei download", "open_downloads"),
    ("Apri la cartella Downloads", "open_downloads"),
    ("Mostrami i download", "open_downloads"),
    ("apri la cartella donload", "open_downloads"),
    # open_documents
    ("Apri Documenti", "open_documents"),
    ("Apri la cartella Documenti.", "open_documents"),
    ("Apri i documenti", "open_documents"),
    ("Mostrami i documenti", "open_documents"),
    ("Vai su Documenti", "open_documents"),
    ("Apri documento", "open_documents"),
    # open_explorer
    ("Apri Esplora file", "open_explorer"),
    ("Apri Esplora risorse.", "open_explorer"),
    ("Apri esplora files", "open_explorer"),
    ("Apri gestione file", "open_explorer"),
    ("Apri il File Explorer", "open_explorer"),
    # open_notepad
    ("Apri il Blocco note", "open_notepad"),
    ("Apri blocco note.", "open_notepad"),
    ("Apri il blocco notes", "open_notepad"),
    ("Apri Notepad", "open_notepad"),
    ("Avvia il blocco note", "open_notepad"),
    ("Apri il blocconote", "open_notepad"),
    # open_calculator
    ("Apri la calcolatrice", "open_calculator"),
    ("Apri calcolatrice.", "open_calculator"),
    ("Avvia la calcolatrice", "open_calculator"),
    ("Lancia la calcolatrice", "open_calculator"),
    ("Calcolatrice", "open_calculator"),
    ("Apri la calcolatrici", "open_calculator"),
    # volume_up
    ("Alza il volume", "volume_up"),
    ("Volume su", "volume_up"),
    ("Aumenta il volume.", "volume_up"),
    ("Alza il volume di 10", "volume_up"),
    ("Alza l'audio", "volume_up"),
    ("Alza il volum", "volume_up"),
    # volume_down
    ("Abbassa il volume", "volume_down"),
    ("Volume giù", "volume_down"),
    ("Diminuisci il volume.", "volume_down"),
    ("Abbassa l'audio", "volume_down"),
    ("Abbassa il volume di tre", "volume_down"),
    ("Abbasso il volume", "volume_down"),
    # mute
    ("Togli l'audio", "mute"),
    ("Muto", "mute"),
    ("Silenzia", "mute"),
    ("Metti in muto.", "mute"),
    ("Disattiva l'audio", "mute"),
    # tell_time
    ("Che ore sono?", "tell_time"),
    ("Che ora è?", "tell_time"),
    ("Dimmi l'ora", "tell_time"),
    ("Mi dici l'ora?", "tell_time"),
    ("Claudio che ore sono", "tell_time"),
    ("Che ore sono adesso?", "tell_time"),
    # tell_date
    ("Che giorno è oggi?", "tell_date"),
    ("Che giorno è?", "tell_date"),
    ("Quanti ne abbiamo oggi?", "tell_date"),
    ("Dimmi la data", "tell_date"),
    ("Che data è oggi?", "tell_date"),
    # system: pause
    ("Claudio, vai in pausa", "pause"),
    ("Vai a dormire", "pause"),
    ("Mettiti in pausa.", "pause"),
    ("Fai una pausa", "pause"),
    ("Claudio dormi", "pause"),
    # system: resume
    ("Claudio, svegliati!", "resume"),
    ("Svegliati", "resume"),
    ("Sveglia!", "resume"),
    ("Torna attivo", "resume"),
    ("Sei sveglio?", "resume"),
    # system: off ("spegniti", exact phrases only)
    ("Claudio, spegniti.", "off"),
    ("Spegniti pure", "off"),
    ("Puoi spegnerti, per favore?", "off"),
    ("Spengiti", "off"),
    # chat window
    ("Claudio, apri la chat", "open_chat"),
    ("Chat", "open_chat"),
    ("Fammi scrivere", "open_chat"),
    ("Voglio scrivere", "open_chat"),
    ("Mostra la chat", "open_chat"),
    ("Apri la finestra della chat", "open_chat"),
    ("Chiudi la chat", "close_chat"),
    ("Chiudi chat", "close_chat"),
    ("Chiudi la finestra della chat", "close_chat"),
    ("Nascondi la chat", "close_chat"),
    ("Claudio, chiudi la chat per favore", "close_chat"),
    ("Ora spegniti", "off"),
    ("Claudio, chiuditi.", "shutdown"),  # JenJo 25/10
    ("Chiuditi pure", "shutdown"),
    ("Ora chiuditi", "shutdown"),
    ("Puoi chiuderti?", "shutdown"),
    ("Claudio, esci dal programma.", "shutdown"),
    # system: on ("accenditi", exact phrases only)
    ("Claudio, accenditi.", "on"),
    ("Riaccenditi", "on"),
    ("Accenditi pure", "on"),
    ("Puoi accenderti?", "on"),
    ("Torna ad ascoltare", "on"),
    ("Claudio, smetti di ascoltare.", "off"),
]

NEGATIVE = [
    "Scrivi una mail a Marco e poi chattiamo",
    "Che cos'è ChatGPT?",
    "Spegni",  # close to "spegniti", but shutting down needs the exact phrase
    "Spegniti il...",
    "Spegni la luce",
    "Chiudi",
    "Pausa",
    "Prenota un volo per Roma",
    "Apri il frigorifero",
    "Apri Spotify",
    "Che tempo fa domani?",
    "Chiudi Chrome",
    "Raccontami una barzelletta",
    "Apri la porta",
    "Chi è il presidente della Repubblica?",
    "Scrivi una mail a Marco",
    "",
    "Sì",
    "Grazie",
    "Chiuditi di dieci",  # an exact-only command followed by a number
    "Spegniti tre volte",
]


@pytest.mark.parametrize(("text", "command_id"), POSITIVE)
def test_positive(matcher: CommandMatcher, text: str, command_id: str) -> None:
    match = matcher.match(text)
    assert match is not None, f"nessun comando per {text!r}"
    assert match.command.id == command_id


@pytest.mark.parametrize("text", NEGATIVE)
def test_negative(matcher: CommandMatcher, text: str) -> None:
    assert matcher.match(text) is None


def test_every_command_has_five_variants() -> None:
    counts: dict[str, int] = {}
    for _, cid in POSITIVE:
        counts[cid] = counts.get(cid, 0) + 1
    assert all(n >= 5 for n in counts.values()), counts
    assert len(POSITIVE) + len(NEGATIVE) >= 60


def test_number_is_extracted(matcher: CommandMatcher) -> None:
    match = matcher.match("Alza il volume di 10")
    assert match is not None and match.number == 10


def test_exact_match_scores_100(matcher: CommandMatcher) -> None:
    match = matcher.match("Apri Chrome")
    assert match is not None and match.score == 100


def test_duplicate_phrase_across_commands_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text(
        '[[command]]\nid="a"\nphrases=["apri x"]\naction="open_app"\n'
        '[[command]]\nid="b"\nphrases=["Apri la X"]\naction="open_app"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ambigua"):
        CommandMatcher(load_commands(f))


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text('[[command]]\nid="a"\nphrases=["x"]\naction="t"\n' * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicato"):
        load_commands(f)


def test_keywords_gate_fuzzy_matches(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text(
        '[[command]]\nid="t"\nphrases=["che ore sono"]\nkeywords=["Ore"]\naction="tell_time"\n',
        encoding="utf-8",
    )
    (cmd,) = load_commands(f)
    assert cmd.keywords == frozenset({"ore"})
    matcher = CommandMatcher([cmd], threshold=75)
    assert matcher.match("che ora sono") is None  # close, but no keyword
    match = matcher.match("che ore sono?")
    assert match is not None and match.score == 100.0
    assert matcher.match("che ore son") is not None  # fuzzy, with the keyword


@pytest.mark.parametrize(
    ("text", "command_id"),
    [("stop", "stop"), ("play", "media_play_pause"), ("chat", "open_chat")],
)
def test_italian_loanwords_still_match(matcher: CommandMatcher, text: str, command_id: str) -> None:
    match = matcher.match(text)
    assert match is not None and match.command.id == command_id


@pytest.mark.parametrize("text", ["skip", "quit"])
def test_english_only_words_find_nothing(matcher: CommandMatcher, text: str) -> None:
    assert matcher.match(text) is None
