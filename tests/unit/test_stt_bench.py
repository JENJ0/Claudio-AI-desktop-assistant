"""stt_testset.py and stt_bench.py: the test set's files, the scores, the variants, the cut."""

from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from claudio.audio.vad import Endpointer
from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import stt_bench as bench
import stt_testset as testset

SR = 16000


def tone(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_sentences_skip_blank_lines_and_comments(tmp_path) -> None:
    path = tmp_path / "frasi.txt"
    path.write_text("# a comment\nClaudio, apri Chrome.\n\n  Stop.  \n", encoding="utf-8")
    assert testset.read_sentences(path) == ["Claudio, apri Chrome.", "Stop."]


def test_the_shipped_sentences() -> None:
    sentences = testset.read_sentences(testset.SENTENCES)
    assert len(sentences) == 40 and len(set(sentences)) == 40


def test_folder_names_and_what_is_left_to_record(tmp_path) -> None:
    assert testset.folder_name("Microfono (Logi C270 HD WebCam)") == "Microfono_Logi_C270_HD_WebCam"
    assert testset.folder_name("()") == "microfono"
    a, b = tmp_path / "a", tmp_path / "b"
    for folder, done in ((a, 3), (b, 2)):
        folder.mkdir()
        for n in range(1, done + 1):
            (folder / f"{n:02d}.wav").touch()
    assert testset.first_missing([a, b], 5) == 2  # b still lacks the third sentence
    assert testset.first_missing([a], 3) == 3


def test_peak_is_the_loudest_32_ms() -> None:
    quiet_then_loud = np.concatenate((tone(0.5, 0.01), tone(0.1, 0.1)))
    assert testset.peak_dbfs(quiet_then_loud) == pytest.approx(-23.0, abs=0.1)  # 0.1 sine
    as_int16 = (quiet_then_loud * 32768).astype(np.int16)
    assert testset.peak_dbfs(as_int16) == pytest.approx(-23.0, abs=0.1)
    assert testset.peak_dbfs(np.zeros(100, np.int16)) == -120.0


def test_a_recording_mute_or_cut_at_the_end_is_flagged() -> None:
    whole = np.concatenate((silence(0.4), tone(0.6, 0.1), silence(0.3)))
    assert testset.tail_ms(whole) == pytest.approx(300, abs=20)
    assert testset.problem(whole) == ""
    assert testset.problem(np.concatenate((silence(0.4), tone(0.6, 0.1)))) == "tagliata alla fine?"
    assert testset.problem(tone(0.5, 0.0005)) == "muto?"  # -66 dBFS: "Stop." on the headset
    as_int16 = (whole * 32768).astype(np.int16)
    assert testset.tail_ms(as_int16) == testset.tail_ms(whole)
    hum = np.random.default_rng(0).normal(0, 0.012, len(whole))  # about -38 dBFS of noise
    assert testset.tail_ms(whole + hum) == pytest.approx(300, abs=20)


def test_check_lists_the_sentences_to_record_again(tmp_path, capsys) -> None:
    import soundfile as sf

    good = np.concatenate((silence(0.3), tone(0.5, 0.1), silence(0.3)))
    cut, mute = np.concatenate((silence(0.3), tone(0.5, 0.1))), tone(0.4, 1e-4)
    for mic, recordings in (("cuffie", (good, cut, good)), ("webcam", (good, good, mute))):
        (tmp_path / mic).mkdir()
        for n, audio in enumerate(recordings, start=1):
            sf.write(tmp_path / mic / f"{n:02d}.wav", audio, SR, subtype="PCM_16")
    assert testset.check(tmp_path) == [2, 3]
    out = capsys.readouterr().out
    assert "cuffie  02: tagliata alla fine?" in out and "webcam  03: muto?" in out


class FakeRecorder:
    def __init__(self, folder) -> None:
        self.name, self.folder = "C270", folder

    def start(self) -> None:
        pass

    def stop(self) -> np.ndarray:
        return (np.concatenate((silence(0.3), tone(0.5, 0.1), silence(0.3))) * 32768).astype(
            np.int16
        )


def test_record_again_only_the_sentences_asked(tmp_path, monkeypatch) -> None:
    recorder = FakeRecorder(tmp_path)
    for n in (1, 2, 3):
        (tmp_path / f"{n:02d}.wav").write_bytes(b"old")
    answers = iter(["", "", "r", "", "", "", ""])  # 3, then "r" goes back to 3, again 3, then 1
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(testset.time, "sleep", lambda s: None)
    testset.record([recorder], ["uno", "due", "tre"], only=[3, 1])  # type: ignore[list-item]
    assert (tmp_path / "02.wav").read_bytes() == b"old"
    assert (tmp_path / "01.wav").read_bytes() != b"old" != (tmp_path / "03.wav").read_bytes()
    assert next(answers, "finished") == "finished"


def test_words_ignore_case_punctuation_and_spelling() -> None:
    assert bench.words("Ehi Claudio, cos'è un buco nero?") == [
        "ehi",
        "claudio",
        "cos'",
        "è",
        "un",
        "buco",
        "nero",
    ]
    assert bench.words("Hey Claudio! Apri  Chrome.") == bench.words("ehi claudio apri chrome")
    assert bench.words("C’è") == bench.words("c'è")


def test_word_error_rate() -> None:
    assert bench.word_errors(["apri", "la", "chat"], ["apri", "la", "chat"]) == 0
    assert bench.word_errors(["apri", "la", "chat"], ["abri", "la", "chatte"]) == 2
    assert bench.word_errors(["apri", "chrome"], ["apri"]) == 1  # a word lost
    assert bench.word_errors(["stop"], ["stop", "grazie", "mille"]) == 2  # words invented
    assert bench.word_errors([], ["ciao"]) == 1
    references = ["Claudio, apri la chat.", "Stop."]
    assert bench.wer(references, ["Claudio apri la chat", ""]) == pytest.approx(1 / 5)


def test_variants_are_changes_from_live() -> None:
    live = bench.parse_variant("live")
    assert (live.gain, live.prompt, live.beam, live.vad, live.engine) == (
        True,
        "nomi",
        None,
        True,
        "whisper",
    )
    v = bench.parse_variant("live+grezzo+vuoto+beam1+novad+large")
    assert (v.gain, v.prompt, v.beam, v.vad, v.model) == (False, "vuoto", 1, False, "large-v3")
    assert bench.parse_variant("frasi").prompt == "frasi"
    assert live.model == "" and bench.parse_variant("live+turbo").model == "large-v3-turbo"
    assert bench.parse_variant("parakeet").engine == "parakeet"
    with pytest.raises(SystemExit, match="rumore"):
        bench.parse_variant("live+rumore")
    for name in bench.DEFAULT_VARIANTS:
        bench.parse_variant(name)


def test_the_gain_autogain_would_settle_on() -> None:
    def at(dbfs: float) -> np.ndarray:  # a sine whose loudest 32 ms is at `dbfs`
        return (tone(0.2, np.sqrt(2) * 10 ** (dbfs / 20)) * 32768).astype(np.int16)

    assert bench.microphone_gain_db([at(-40), at(-35), at(-45)]) == pytest.approx(20.0, abs=0.2)
    assert bench.microphone_gain_db([at(-5)]) == 0.0
    assert bench.microphone_gain_db([at(-70)]) == bench.MAX_GAIN_DB


def test_silero_cut_reports_a_sentence_ended_too_soon(fake_vad, vad_settings) -> None:
    endpointer = Endpointer(fake_vad, vad_settings, SR)
    whole = np.concatenate((silence(0.3), tone(0.8), silence(0.3)))
    audio, cut = bench.silero_cut(whole, endpointer)
    assert not cut and 1.0 < len(audio) / SR < 1.3  # pre-roll + the voice + ~100 ms
    paused = np.concatenate((silence(0.3), tone(0.8), silence(0.8), tone(0.4)))
    audio, cut = bench.silero_cut(paused, endpointer)
    assert cut and len(audio) / SR < 1.3  # the end of the sentence never reaches Whisper
    audio, cut = bench.silero_cut(silence(1.0), endpointer)
    assert audio.size == 0 and not cut


def test_run_applies_the_gain_and_counts_the_cuts(fake_vad, vad_settings) -> None:
    quiet = (np.concatenate((silence(0.3), tone(0.8, 0.02), silence(0.3))) * 32768).astype(np.int16)
    mic = bench.Microphone("C270", testset.OUT, [1, 2], [quiet, quiet], gain_db=20.0)
    heard: list[float] = []

    def transcribe(audio: np.ndarray, variant: bench.Variant) -> str:
        heard.append(float(np.abs(audio).max()))
        return "apri chrome"

    endpointer = Endpointer(fake_vad, vad_settings, SR)
    score = bench.run(bench.parse_variant("live"), mic, transcribe, endpointer)
    assert score.texts == ["apri chrome"] * 2 and len(score.ms) == 2 and score.cuts == 0
    assert len(heard) == 3  # a warm-up, then the two sentences
    assert heard[-1] == pytest.approx(0.2, abs=0.01)  # +20 dB: loud enough for the fake VAD
    raw = bench.run(bench.parse_variant("live+grezzo+novad"), mic, transcribe, endpointer)
    assert raw.texts == ["apri chrome"] * 2 and heard[-1] == pytest.approx(0.02, abs=0.002)


def test_run_says_why_a_sentence_came_out_empty(fake_vad, vad_settings) -> None:
    voice = (np.concatenate((silence(0.3), tone(0.8, 0.2), silence(0.3))) * 32768).astype(np.int16)
    mute = np.zeros(SR, dtype=np.int16)
    mic = bench.Microphone("C270", testset.OUT, [1, 2, 3], [voice, voice, mute], gain_db=0.0)
    texts = iter(["", "", "Stop.", ""])  # the warm-up, then the sentences

    def transcribe(audio: np.ndarray, variant: bench.Variant) -> str:
        text = next(texts)
        if not text:
            logging.getLogger("claudio.stt.transcriber").info("Trascrizione scartata (%s)", "x")
        return text

    endpointer = Endpointer(fake_vad, vad_settings, SR)
    score = bench.run(bench.parse_variant("live"), mic, transcribe, endpointer)
    assert score.texts == ["", "Stop.", ""]
    assert score.why == {0: "Trascrizione scartata (x)", 2: "Silero non ha sentito voce"}
    assert len(score.ms) == 2  # the time of what was transcribed only
    assert not logging.getLogger("claudio.stt.transcriber").handlers  # taken off again


def test_run_never_transcribes_what_claudio_would_not(fake_vad, vad_settings) -> None:
    whisper = (tone(1.0, 0.0005) * 32768).astype(np.int16)  # -66 dBFS: "Stop." on the headset
    mic = bench.Microphone("Cuffie", testset.OUT, [1], [whisper], gain_db=0.0)
    calls: list[int] = []

    def transcribe(audio: np.ndarray, variant: bench.Variant) -> str:
        calls.append(audio.size)
        return "invented"

    endpointer = Endpointer(fake_vad, vad_settings, SR)
    score = bench.run(bench.parse_variant("live+grezzo+novad"), mic, transcribe, endpointer)
    assert score.texts == [""] and score.why == {0: "sotto -60 dBFS: non trascritta"}
    assert calls == [] and score.ms == []


def test_recupero_goes_on_as_live(matcher) -> None:
    from claudio.intake import Verdict
    from claudio.stt.recovery import Recovered

    assert bench.parse_variant("live+recupero").recover and not bench.parse_variant("live").recover

    class Stt:
        def __init__(self, *listed: str) -> None:
            self.listed = list(listed)

        def candidates(self, audio: np.ndarray) -> list[str]:
            return self.listed

    def second(audio: np.ndarray, text: str) -> Recovered:
        return Recovered("Chiudi la chat.", Verdict.DROP if text == "Pasta." else Verdict.ASK)

    def live(text: str, stt: Stt | None = None) -> str:
        return bench.as_live(silence(0.1), text, stt or Stt(), matcher, second, 5)

    assert live("Apri Chrome.") == "Apri Chrome."  # understood at once
    closed = Stt("Chi lascia atto?", "apri la calcolatrice")
    assert live("Pasta e fagioli", closed) == "apri la calcolatrice"  # the closed list
    assert live("Chi lascia atto?", closed) == "Chiudi la chat."  # a "?": the second opinion
    assert live("Pasta e fagioli") == "Chiudi la chat."  # nothing in the closed list
    assert live("Pasta.") == "Pasta."  # the second opinion drops its proposal: as before
    assert bench.as_live(silence(0.1), "Pasta?", Stt(), matcher, None, 5) == "Pasta?"


def test_report_counts_the_false_commands(matcher, capsys) -> None:
    mic = bench.Microphone("C270", testset.OUT, [1, 2], [], gain_db=0.0)
    references = ["Claudio, apri Chrome.", "Pasta al forno."]
    wrong = bench.Score("live+recupero", ["Claudio, apri Chrome.", "Basta."], [300, 320])
    bench.report(mic, references, [wrong], matcher, details=False)
    row = next(line.split() for line in capsys.readouterr().out.splitlines() if "recupero" in line)
    assert row[2:4] == ["1/1", "1"]  # the command right, and one sentence heard as "basta"


def test_report_scores_words_and_commands(matcher, capsys) -> None:
    mic = bench.Microphone("C270", testset.OUT, [1, 2], [], gain_db=12.0)
    references = ["Claudio, apri Chrome.", "Che tempo fa domani?"]
    good = bench.Score("live", ["Claudio, apri Chrome.", "Che tempo fa domani?"], [300, 320])
    bad = bench.Score("live+vuoto", ["Claudio, abri come.", ""], [250, 260], why={1: "silenzio"})
    bench.report(mic, references, [good, bad], matcher, details=True)
    out = capsys.readouterr().out
    assert "guadagno +12 dB" in out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith("  live")}
    assert rows["live"][1:3] == ["0.0%", "1/1"]
    assert rows["live+vuoto"][1:3] == ["85.7%", "0/1"]  # 2 + 4 words wrong of 7
    assert "01 «Claudio, apri Chrome.»" in out and "'Claudio, abri come.'" in out
    assert "02 «Che tempo fa domani?»" in out and "''  (silenzio)" in out
