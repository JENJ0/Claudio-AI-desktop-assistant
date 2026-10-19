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
