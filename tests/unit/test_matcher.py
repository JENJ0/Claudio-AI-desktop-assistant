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
    # system: shutdown (exact phrases only)
    ("Claudio, spegniti.", "shutdown"),
    ("Spegniti pure", "shutdown"),
    ("Puoi spegnerti, per favore?", "shutdown"),
    ("Spengiti", "shutdown"),
    ("Ora spegniti", "shutdown"),
]

NEGATIVE = [
    "Spegni",  # close to "spegniti", but shutting down needs the exact phrase
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


POSITIVE_EN = [
    ("Open Chrome.", "open_chrome"),
    ("Hey Claudio, open the browser, please", "open_chrome"),
    ("Could you open Google Chrome?", "open_chrome"),
    ("Launch Chrome", "open_chrome"),
    ("Open Crome", "open_chrome"),
    ("Open my downloads", "open_downloads"),
    ("Open the Downloads folder.", "open_downloads"),
    ("Go to downloads", "open_downloads"),
    ("Show my downloads", "open_downloads"),
    ("Open download", "open_downloads"),
    ("Open documents", "open_documents"),
    ("Open the documents folder", "open_documents"),
    ("Open my documents.", "open_documents"),
    ("Go to documents", "open_documents"),
    ("Open document", "open_documents"),
    ("Open File Explorer", "open_explorer"),
    ("Open Explorer.", "open_explorer"),
    ("Open Windows Explorer", "open_explorer"),
    ("Open the file manager", "open_explorer"),
    ("Open my files", "open_explorer"),
    ("Open Notepad.", "open_notepad"),
    ("Open the notepad", "open_notepad"),
    ("Start Notepad", "open_notepad"),
    ("Launch notepad please", "open_notepad"),
    ("Open note pad", "open_notepad"),
    ("Open the calculator", "open_calculator"),
    ("Open calculator.", "open_calculator"),
    ("Start the calculator", "open_calculator"),
    ("Calculator", "open_calculator"),
    ("Open the calculater", "open_calculator"),
    ("Turn up the volume", "volume_up"),
    ("Turn the volume up.", "volume_up"),
    ("Volume up", "volume_up"),
    ("Turn up the volume by 10", "volume_up"),
    ("Louder", "volume_up"),
    ("Turn down the volume", "volume_down"),
    ("Turn the volume down", "volume_down"),
    ("Volume down.", "volume_down"),
    ("Lower the volume three times", "volume_down"),
    ("Quieter", "volume_down"),
    ("Mute", "mute"),
    ("Unmute.", "mute"),
    ("Mute the sound", "mute"),
    ("Turn off the sound", "mute"),
    ("Mute the audio", "mute"),
    ("What time is it?", "tell_time"),
    ("What's the time?", "tell_time"),
    ("Tell me the time", "tell_time"),
    ("Claudio, what time is it now?", "tell_time"),
    ("Do you know what time it is?", "tell_time"),
    ("What day is it today?", "tell_date"),
    ("What day is it?", "tell_date"),
    ("What's the date?", "tell_date"),
    ("What's today's date?", "tell_date"),
    ("Tell me the date", "tell_date"),
    ("Claudio, go to sleep", "pause"),
    ("Pause.", "pause"),
    ("Take a break", "pause"),
    ("Sleep", "pause"),
    ("Go to sleep now", "pause"),
    ("Claudio, wake up!", "resume"),
    ("Wake up", "resume"),
    ("Resume", "resume"),
    ("Are you awake?", "resume"),
    ("Come back", "resume"),
    ("Claudio, shut down.", "shutdown"),
    ("Shut yourself down", "shutdown"),
    ("Turn yourself off, please", "shutdown"),
    ("Switch yourself off", "shutdown"),
    ("Shut down", "shutdown"),
]

NEGATIVE_EN = [
    "Shut down the computer",  # shutting CLAUDIO down needs the exact phrase
    "Turn off the lights",
    "Shut the door",
    "Open Spotify",
    "What's the weather tomorrow?",
    "Book a flight to Rome",
    "Pause the music",
    "Tell me a joke",
    "Thank you",
    "Yes",
]


@pytest.mark.parametrize(("text", "command_id"), POSITIVE_EN)
def test_positive_english(matcher: CommandMatcher, text: str, command_id: str) -> None:
    match = matcher.match(text, "en")
    assert match is not None, f"nessun comando per {text!r}"
    assert match.command.id == command_id and match.language == "en"


@pytest.mark.parametrize("text", NEGATIVE_EN)
def test_negative_english(matcher: CommandMatcher, text: str) -> None:
    assert matcher.match(text, "en") is None


def test_every_english_command_has_five_variants(matcher: CommandMatcher) -> None:
    counts: dict[str, int] = {}
    for _, cid in POSITIVE_EN:
        counts[cid] = counts.get(cid, 0) + 1
    assert set(counts) == {cid for _, cid in POSITIVE}
    assert all(n >= 5 for n in counts.values()), counts
    assert matcher.languages == ["it", "en"]


def test_the_language_of_the_phrase_wins(matcher: CommandMatcher) -> None:
    """Whisper picked the wrong language: the phrase is still found, in its own language."""
    match = matcher.match("open the calculator", "it")
    assert match is not None and match.command.id == "open_calculator" and match.language == "en"
    match = matcher.match("Apri la calcolatrice", "en")
    assert match is not None and match.language == "it"
    match = matcher.match("Turn up the volume by 10", "en")
    assert match is not None and match.number == 10


def test_translations_are_loaded(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text(
        '[[command]]\nid="a"\nphrases=["apri x"]\nphrases_en=["open x"]\naction="open_app"\n'
        'reply="Apro X."\nreply_en="Opening X."\nparams={ already="Già.", already_en="Done." }\n',
        encoding="utf-8",
    )
    (cmd,) = load_commands(f)
    assert cmd.languages == ("it", "en") and cmd.phrases_in("en") == ("open x",)
    assert cmd.reply_in("en") == "Opening X." and cmd.reply_in("de") == "Apro X."
    assert cmd.param_in("already", "en") == "Done." and cmd.param_in("already", "it") == "Già."
