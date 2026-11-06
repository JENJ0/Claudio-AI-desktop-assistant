"""label_wavs.py: importing a session, the labels file, and the key loop with fake input."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import label_wavs as lw

SR = 16000


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(160, dtype=np.int16), SR, subtype="PCM_16")


def epoch(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").timestamp()


def make_logs(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    for name in ("2026-11-05_17-00", "2026-11-05_18-00"):
        folder = logs / "sessions" / name
        (folder / "jack").mkdir(parents=True)
        (folder / "console.log").write_text(name, encoding="utf-8")
        (folder / "conversazione.md").write_text("md", encoding="utf-8")
        (folder / "jack" / "001.wav").write_bytes(b"jack")
    write_wav(logs / "audio" / "20261105-170500-100_ok.wav")
    (logs / "audio" / "20261105-170500-100_ok.json").write_text("{}", encoding="utf-8")
    write_wav(logs / "audio" / "rejected" / "20261105-175900-000_no_voice.wav")
    write_wav(logs / "audio" / "20261105-180100-000_ok.wav")  # next session's
    write_wav(logs / "audio" / "20261105-165000-000_ok.wav")  # before this one
    rows = [
        {
            "ts": epoch("20261105-170500"),
            "audio_file": "20261105-170500-100_ok.wav",
            "text": "ciao",
        },
        {
            "ts": epoch("20261105-180100"),
            "audio_file": "20261105-180100-000_ok.wav",
            "text": "dopo",
        },
    ]
    (logs / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return logs


def test_import_session_takes_only_its_own_wavs_and_never_overwrites(tmp_path: Path) -> None:
    logs, corpus = make_logs(tmp_path), tmp_path / "corpus"
    assert lw.import_session(logs, corpus, "2026-11-05_17-00") == 2
    folder = corpus / "2026-11-05_17-00"
    assert sorted(p.name for p in (folder / "audio").glob("*")) == [
        "20261105-170500-100_ok.json",
        "20261105-170500-100_ok.wav",
        "rejected",
    ]
    assert (folder / "audio" / "rejected" / "20261105-175900-000_no_voice.wav").is_file()
    assert (folder / "jack" / "001.wav").read_bytes() == b"jack"
    assert list(lw.load_rows(folder)) == ["20261105-170500-100_ok.wav"]
    (folder / "console.log").write_text("edited", encoding="utf-8")
    assert lw.import_session(logs, corpus, "2026-11-05_17-00") == 0
    assert (folder / "console.log").read_text(encoding="utf-8") == "edited"
    # the last session has no end
    assert lw.import_session(logs, corpus, "2026-11-05_18-00") == 1


def test_propose_prefers_the_text() -> None:
    assert lw.propose({"text": "a", "second_chance": "b"}) == "a"
    assert lw.propose({"second_chance": "b"}) == "b"
    assert lw.propose({"text": "", "second_chance": "b"}) == "b"
    assert lw.propose({}) == ""
    assert lw.propose(None) == ""


def session_with_wavs(tmp_path: Path, count: int = 3, proposals: bool = True) -> Path:
    folder = tmp_path / "corpus" / "s1"
    for i in range(count):
        write_wav(folder / "audio" / f"20261105-17050{i}-000_ok.wav")
    write_wav(folder / "audio" / "rejected" / "20261105-170400-000_no_voice.wav")
    if proposals:  # every WAV gets a proposed text, as `t<n>`
        names = sorted(p.name for p in folder.rglob("*.wav"))
        rows = [{"audio_file": n, "text": f"t{i}"} for i, n in enumerate(names)]
        (folder / "metrics.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
    return folder


def test_pending_skips_what_is_labelled(tmp_path: Path) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    todo = lw.pending(folder, labels)
    assert len(todo) == 4
    assert todo[0].name == "20261105-170400-000_no_voice.wav"  # sorted by name, both folders
    lw.append_label(labels, {"wav": lw.wav_key(folder, todo[0]), "session": "s1"})
    assert lw.pending(folder, labels) == todo[1:]


def test_pending_ignores_a_torn_last_line(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    todo = lw.pending(folder, labels)
    lw.append_label(labels, {"wav": lw.wav_key(folder, todo[0]), "session": "s1"})
    with labels.open("a", encoding="utf-8") as f:
        f.write('{"wav": "audio/xx')  # crash mid-write
    assert lw.pending(folder, labels) == todo[1:]
    assert "incompleta" in capsys.readouterr().out
    lw.append_label(labels, {"wav": lw.wav_key(folder, todo[1]), "session": "s1"})
    assert lw.pending(folder, labels) == todo[2:]


def test_a_session_cannot_change_split(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    assert lw.assign_split(splits, "s1", "dev") == "dev"
    assert lw.assign_split(splits, "s1", None) == "dev"
    assert lw.assign_split(splits, "s1", "dev") == "dev"
    with pytest.raises(SystemExit):
        lw.assign_split(splits, "s1", "test")
    with pytest.raises(SystemExit):
        lw.assign_split(splits, "s2", "bogus")


def test_a_new_session_needs_a_split(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    with pytest.raises(SystemExit):
        lw.assign_split(splits, "s1", None)
    assert not splits.exists()


def keys(sequence: list[str]):
    it = iter(sequence)

    def getch() -> str:
        try:
            return next(it)
        except StopIteration:
            raise KeyboardInterrupt from None

    return getch


def key_reader(typed: list[str], buffered: list[str] | None = None):
    """read_key over a fake console: `buffered` is already waiting, `typed` comes later."""
    waiting, later = list(buffered or []), list(typed)
    pushback: list[str] = []  # like msvcrt: the scan code of a special key, unseen by kbhit

    def kbhit() -> bool:
        return bool(waiting)

    def getwch() -> str:
        if pushback:
            return pushback.pop()
        key = waiting.pop(0) if waiting else later.pop(0)
        if key in ("\x00", "\xe0") and waiting:
            pushback.append(waiting.pop(0))
        return key

    return lw.make_key_reader(kbhit, getwch)


def test_labels_survive_an_interrupted_run(tmp_path: Path) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    todo = lw.pending(folder, labels)
    with pytest.raises(KeyboardInterrupt):  # Ctrl+C arrives as the character \x03
        lw.label_session(
            folder, labels, "dev", key_reader(["\r", "n", "\x03"]), lambda p: None, input
        )
    lines = labels.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["for_claudio"] is True and second["for_claudio"] is False
    assert set(first) == {
        "wav",
        "session",
        "split",
        "proposed",
        "text",
        "for_claudio",
        "labeled_at",
    }
    assert lw.pending(folder, labels) == todo[2:]


def test_read_key_drops_keys_typed_during_playback_and_special_keys() -> None:
    assert key_reader(["q"], buffered=["\r", "\r"])() == "q"
    assert key_reader(["\xe0", "Q", "s"])() == "s"  # PgDn: prefix and scan code both eaten
    assert key_reader(["\x00", ";", "r"])() == "r"  # F1


def test_read_key_drops_a_special_key_pressed_during_playback() -> None:
    # PgDn, Canc and Alt+F7 as msvcrt delivers them: the scan code is a plain letter
    for prefix, scan in (("\xe0", "Q"), ("\xe0", "S"), ("\x00", "n")):
        assert key_reader(["s"], buffered=[prefix, scan])() == "s"


def test_enter_is_refused_when_there_is_nothing_to_confirm(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path, 0, proposals=False), tmp_path / "labels.jsonl"
    lw.label_session(folder, labels, "dev", keys(["\r", "n", "q"]), lambda p: None, input)
    assert "nessun testo" in capsys.readouterr().out
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    assert [(r["text"], r["for_claudio"]) for r in rows] == [("", False)]


def test_an_empty_correction_keeps_the_proposal(tmp_path: Path) -> None:
    folder, labels = session_with_wavs(tmp_path, 0), tmp_path / "labels.jsonl"
    lw.label_session(folder, labels, "dev", keys(["c", "\r", "q"]), lambda p: None, lambda _: "")
    first = json.loads(labels.read_text(encoding="utf-8").splitlines()[0])
    assert first["text"] == first["proposed"] == "t0"


def test_the_metrics_row_is_shown(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path, 0, proposals=False), tmp_path / "labels.jsonl"
    row = {
        "audio_file": "20261105-170400-000_no_voice.wav",
        "text": "ciao",
        "outcome": "ok",
        "route": "local",
        "rejected": None,
        "speech_ms": 1234,
        "reply": "r" * 200,
    }
    (folder / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    lw.label_session(folder, labels, "dev", keys(["q"]), lambda p: None, input)
    out = capsys.readouterr().out
    assert "outcome=ok" in out and "route=local" in out and "speech_ms=1234" in out
    assert "rejected=" not in out and "wake_word" not in out  # null or missing: not shown
    assert "r" * 80 in out and "r" * 81 not in out


def test_a_wav_without_a_metrics_row_prints_no_empty_line(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path, 0, proposals=False), tmp_path / "labels.jsonl"
    lw.label_session(folder, labels, "dev", keys(["q"]), lambda p: None, input)
    assert "riga:" not in capsys.readouterr().out


def test_keys_correct_replay_skip_quit(tmp_path: Path) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    played: list[Path] = []
    lw.label_session(
        folder, labels, "dev", keys(["r", "c", "\r", "s", "q"]), played.append, lambda _: "fixed"
    )
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in rows] == ["fixed"]
    assert len(played) == 4  # first wav twice (r), then the two skipped or quit on
    assert len(lw.pending(folder, labels)) == 3


def test_an_unreadable_wav_is_skipped_and_reported(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path, 1), tmp_path / "labels.jsonl"
    bad = lw.pending(folder, labels)[0]

    def play(path: Path) -> None:
        if path == bad:
            raise RuntimeError("broken")

    lw.label_session(folder, labels, "dev", keys(["\r", "q"]), play, input)
    assert "illeggibile" in capsys.readouterr().out
    assert len(labels.read_text(encoding="utf-8").splitlines()) == 1
    assert bad in lw.pending(folder, labels)


def test_import_refuses_two_runs_in_the_same_minute(tmp_path: Path) -> None:
    logs = make_logs(tmp_path)
    (logs / "sessions" / "2026-11-05_17-00_2").mkdir()
    with pytest.raises(SystemExit) as stop:
        lw.import_session(logs, tmp_path / "corpus", "2026-11-05_17-00")
    assert "2026-11-05_17-00_2" in str(stop.value)
    assert not (tmp_path / "corpus").exists()


def test_import_validates_the_session_name(tmp_path: Path) -> None:
    logs = make_logs(tmp_path)
    for bad in ("banana", "2026-11-05_17-30"):  # malformed; well-formed but no such folder
        with pytest.raises(SystemExit):
            lw.import_session(logs, tmp_path / "corpus", bad)
    assert not (tmp_path / "corpus").exists()


def test_a_leftover_partial_copy_is_not_a_wav_and_is_replaced(tmp_path: Path) -> None:
    logs, corpus = make_logs(tmp_path), tmp_path / "corpus"
    audio = corpus / "2026-11-05_17-00" / "audio"
    audio.mkdir(parents=True)
    (audio / "20261105-170500-100_ok.wav.part").write_bytes(b"half")
    assert lw.pending(corpus / "2026-11-05_17-00", tmp_path / "none.jsonl") == []
    assert lw.import_session(logs, corpus, "2026-11-05_17-00") == 2
    copied = audio / "20261105-170500-100_ok.wav"
    assert copied.read_bytes() == (logs / "audio" / copied.name).read_bytes()


def test_a_label_tail_torn_inside_a_character_does_not_stop_the_run(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    todo = lw.pending(folder, labels)
    lw.append_label(labels, {"wav": lw.wav_key(folder, todo[0]), "session": "s1"})
    with labels.open("ab") as f:
        f.write('{"text": "perché'.encode()[:-1])  # cut inside the two bytes of "é"
    assert lw.pending(folder, labels) == todo[1:]
    assert "incompleta" in capsys.readouterr().out


def test_no_audio_output_is_reported_and_the_wav_is_still_offered(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import sounddevice as sd

    def no_device(*args, **kwargs) -> None:
        raise sd.PortAudioError("no device")

    monkeypatch.setattr(sd, "play", no_device)
    folder, labels = session_with_wavs(tmp_path, 0), tmp_path / "labels.jsonl"
    lw.label_session(folder, labels, "dev", keys(["\r"]), lw.play_wav, input)
    assert "nessuna uscita audio" in capsys.readouterr().out
    assert len(labels.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_the_corpus_is_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "samples/corpus/labels.jsonl"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout


def run_labelling(folder: Path, labels: Path, typed: list[str]) -> list[Path]:
    """Run label_session on `typed` keys; returns the WAVs played, in order."""
    played: list[Path] = []
    lw.label_session(folder, labels, "dev", keys(typed), played.append, input)
    return played


def label_lines(labels: Path) -> list[dict]:
    return [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]


def test_undo_then_relabel_offers_the_same_wav_again(tmp_path: Path) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    played = run_labelling(folder, labels, ["\r", "u", "n", "q"])
    assert played[2] == played[0]  # after the undo: the same WAV, not the next one
    (only,) = label_lines(labels)
    assert only["wav"] == lw.wav_key(folder, played[0]) and only["for_claudio"] is False


def test_undo_at_the_start_says_so_and_touches_nothing(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    todo = lw.pending(folder, labels)
    # an earlier run of this session, and another session: neither is "the last label" of this run
    lw.append_label(labels, {"wav": "audio/other.wav", "session": "s0"})
    lw.append_label(labels, {"wav": lw.wav_key(folder, todo[0]), "session": "s1"})
    before = labels.read_bytes()
    played = run_labelling(folder, labels, ["u", "q"])
    assert labels.read_bytes() == before
    assert played == [todo[1]]  # stays on the current WAV
    assert "nessuna etichetta da annullare" in capsys.readouterr().out


def test_undo_then_crash_resumes_with_that_wav_unlabelled(tmp_path: Path) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    todo = lw.pending(folder, labels)
    with pytest.raises(KeyboardInterrupt):
        run_labelling(folder, labels, ["\r", "\r", "u", "\x03"])
    assert [r["wav"] for r in label_lines(labels)] == [lw.wav_key(folder, todo[0])]
    assert lw.pending(folder, labels) == todo[1:]  # as a restart would see it


def test_two_undos_go_back_one_label_at_a_time(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    todo = lw.pending(folder, labels)
    played = run_labelling(folder, labels, ["\r", "\r", "u", "u", "u", "q"])
    assert played[3] == todo[1] and played[4] == todo[0]  # back through both, in reverse
    assert label_lines(labels) == [] and lw.pending(folder, labels) == todo
    assert capsys.readouterr().out.count("nessuna etichetta da annullare") == 1  # the third u


def test_undo_keeps_the_other_labels_and_the_help_names_the_key(tmp_path: Path, capsys) -> None:
    folder, labels = session_with_wavs(tmp_path), tmp_path / "labels.jsonl"
    todo = lw.pending(folder, labels)
    run_labelling(folder, labels, ["\r", "n", "\r", "u", "q"])
    assert [r["wav"] for r in label_lines(labels)] == [lw.wav_key(folder, w) for w in todo[:2]]
    assert "u=annulla" in capsys.readouterr().out
