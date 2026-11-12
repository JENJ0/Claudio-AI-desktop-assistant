import json

import numpy as np
import soundfile as sf

from claudio.audio.recorder import UtteranceRecorder


def test_writes_wav_and_keeps_only_the_latest(tmp_path) -> None:
    folder = tmp_path / "audio"
    rec = UtteranceRecorder(folder, keep=2)
    assert rec(np.zeros(0, dtype=np.float32), "timeout") is None
    names = [rec(np.full(1600, 0.5, dtype=np.float32), f"ok{i}") for i in range(3)]
    assert all(n is not None and n.endswith(f"_ok{i}.wav") for i, n in enumerate(names))
    assert sorted(p.name for p in folder.glob("*.wav")) == sorted(names[1:])
    data, sr = sf.read(folder / str(names[-1]), dtype="int16")
    assert sr == 16000 and len(data) == 1600 and data[0] == 16383


def test_note_writes_the_context_beside_the_wav(tmp_path) -> None:
    rec = UtteranceRecorder(tmp_path, keep=2)
    name = rec(np.full(1600, 0.5, dtype=np.float32), "ok")
    meta = {"whisper_prompt": "Cartelle: Università.", "gain_db": 12.5}
    rec.note(str(name), meta)
    note = tmp_path / name.replace(".wav", ".json")
    assert json.loads(note.read_text(encoding="utf-8")) == meta
    assert "Università" in note.read_text(encoding="utf-8")  # readable, not ascii-escaped


def test_rotation_removes_the_note_with_its_wav(tmp_path) -> None:
    rec = UtteranceRecorder(tmp_path, keep=2)
    for i in range(3):
        name = rec(np.full(1600, 0.5, dtype=np.float32), f"ok{i}")
        rec.note(str(name), {"gain_db": float(i)})
    assert len(list(tmp_path.glob("*.wav"))) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert {p.stem for p in tmp_path.glob("*.json")} == {p.stem for p in tmp_path.glob("*.wav")}
