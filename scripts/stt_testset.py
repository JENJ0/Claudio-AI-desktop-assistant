"""Record the speech-to-text test set: every sentence of scripts/stt_frasi_it.txt, from every
microphone at once, as the microphones give it (no gain), for scripts/stt_bench.py.

    uv run python scripts/stt_testset.py               # all the microphones it finds
    uv run python scripts/stt_testset.py --list        # which microphones it finds
    uv run python scripts/stt_testset.py --mics 3 7    # only these (the numbers from --list)
    uv run python scripts/stt_testset.py --controlla   # recordings mute or cut at the end
    uv run python scripts/stt_testset.py --rifai 10 25 # record these sentences again

Close CLAUDIO first: its beeps and its voice would end up in the recordings.
For every sentence: Enter starts, you read the sentence aloud the way you talk to Claudio,
Enter stops (the recording goes on for 0.3 s more, so a word still being finished is kept).
"r" + Enter records the last sentence again, "q" + Enter stops: the next run carries on from
the first sentence missing. A recording that sounds mute, or whose voice runs to its very
end (cut), is flagged at once.

Output: data/stt_testset/<microphone>/NN.wav (16 kHz mono int16, NN = the sentence's line in
frasi.txt), data/stt_testset/<microphone>/microfono.txt (its full name) and
data/stt_testset/frasi.txt (the sentences, so that changing stt_frasi_it.txt later does not
mix up the answers).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from claudio.audio.devices import (
    candidate_inputs,
    describe_device,
    microphone_key,
    stream_settings,
)
from claudio.config import PROJECT_ROOT

SAMPLE_RATE = 16000
BLOCK = 512
POST_ROLL_S = 0.3  # kept after the Enter that stops: on 29/10, cutting 0.25 s off the end
# instead left 23 of 80 recordings with the voice running to their last frame ("Stop." mute)
MUTE_DBFS = -60.0  # quieter than this at its loudest: switched off, muted or not a microphone
CUT_MS = 50  # less quiet than this after the voice: the end of the sentence is missing
SENTENCES = PROJECT_ROOT / "scripts" / "stt_frasi_it.txt"
OUT = PROJECT_ROOT / "data" / "stt_testset"


def read_sentences(path: Path) -> list[str]:
    """One sentence per line; blank lines and "#" comments are skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def folder_name(microphone: str) -> str:
    """A folder name for a microphone's own name: "Microfono (Logi C270 HD WebCam)" ->
    "Microfono_Logi_C270_HD_WebCam"."""
    return re.sub(r"[^\w-]+", "_", microphone).strip("_")[:40] or "microfono"


def peak_dbfs(audio: np.ndarray) -> float:
    """The loudest 32 ms (int16 or float audio), as AutoGain measures a phrase."""
    x = audio.astype(np.float64) / (32768.0 if audio.dtype == np.int16 else 1.0)
    n = len(x) // BLOCK * BLOCK
    if n == 0:
        return -120.0
    mean_square = float(np.max(np.mean(np.square(x[:n].reshape(-1, BLOCK)), axis=1)))
    return round(10 * np.log10(mean_square), 1) if mean_square > 0 else -120.0


def tail_ms(audio: np.ndarray) -> int:
    """The quiet after the voice: ms after the last 10 ms within 30 dB of the loudest and
    10 dB above the room's noise (the C270 without Windows' effects hums at -39 dBFS)."""
    x = audio.astype(np.float64) / (32768.0 if audio.dtype == np.int16 else 1.0)
    n = SAMPLE_RATE // 100
    if len(x) < n:
        return 0
    frames = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10(np.mean(np.square(frames), axis=1) + 1e-12)
    voice = min(max(db.max() - 30, float(np.percentile(db, 10)) + 10), db.max())
    loud = np.nonzero(db >= voice)[0]
    return int((len(db) - 1 - loud[-1]) * 10)


def problem(audio: np.ndarray) -> str:
    """What is wrong with a recording, "" when nothing."""
    if peak_dbfs(audio) < MUTE_DBFS:
        return "muto?"
    if tail_ms(audio) < CUT_MS:
        return "tagliata alla fine?"
    return ""


def first_missing(folders: list[Path], count: int) -> int:
    """The first sentence (0-based) that some microphone has not recorded yet."""
    for n in range(count):
        if any(not (folder / f"{n + 1:02d}.wav").is_file() for folder in folders):
            return n
    return count


class Recorder:
    """One microphone, always open; it keeps the audio only between start() and stop()."""

    def __init__(self, index: int, out: Path) -> None:
        import sounddevice as sd

        self.description = describe_device(index, "input")
        self.name = microphone_key(self.description)
        self.folder = out / folder_name(self.name)
        self.on = False
        self._chunks: list[np.ndarray] = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK,
            channels=1,
            dtype="int16",
            device=index,
            callback=self._callback,
            extra_settings=stream_settings(index),
        )
        self.stream.start()

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if self.on:
            self._chunks.append(indata[:, 0].copy())

    def start(self) -> None:
        self._chunks = []
        self.on = True

    def stop(self) -> np.ndarray:
        self.on = False
        chunks, self._chunks = self._chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    def close(self) -> None:
        self.stream.close()


def open_recorders(indexes: list[int], out: Path) -> list[Recorder]:
    import sounddevice as sd

    recorders: list[Recorder] = []
    for index in indexes:
        try:
            recorder = Recorder(index, out)
        except (sd.PortAudioError, ValueError) as exc:
            print(f"  [{index}] non si apre, salto: {exc}")
            continue
        if any(r.folder == recorder.folder for r in recorders):
            recorder.close()  # the same microphone under another host API
            continue
        recorders.append(recorder)
        print(f"  [{index}] {recorder.description} -> {recorder.folder}")
    return recorders


def keep_sentences(out: Path, sentences: list[str]) -> None:
    """The set's own copy of the sentences: a set started with other ones is not mixed."""
    saved = out / "frasi.txt"
    if saved.is_file() and read_sentences(saved) != sentences:
        raise SystemExit(
            f"{saved} ha frasi diverse da quelle di adesso: usa --out con un'altra cartella"
        )
    out.mkdir(parents=True, exist_ok=True)
    saved.write_text("\n".join(sentences) + "\n", encoding="utf-8")


def record(recorders: list[Recorder], sentences: list[str], only: list[int] | None = None) -> None:
    """Every sentence still missing, or only the sentences `only` (1-based, in that order)."""
    import soundfile as sf

    first = first_missing([r.folder for r in recorders], len(sentences))
    order = [n - 1 for n in only] if only else list(range(first, len(sentences)))
    if not order:
        print("Tutte le frasi sono già registrate. Per rifarne qualcuna: --rifai e i numeri.")
        return
    print(
        f"\n{len(order)} frasi. Invio = inizia, Invio = fine;"
        ' "r" = rifai la precedente, "q" = esci.\n'
    )
    i = 0
    while i < len(order):
        n = order[i]
        answer = input(f"[{n + 1:02d}/{len(sentences)}] «{sentences[n]}»  (Invio e parla) ")
        if answer.strip().lower() == "q":
            return
        if answer.strip().lower() == "r":
            i = max(i - 1, 0)
            continue
        for recorder in recorders:
            recorder.start()
        input("        ... registro: Invio quando hai finito ")
        time.sleep(POST_ROLL_S)
        levels, flagged = [], False
        for recorder in recorders:
            audio = recorder.stop()
            sf.write(recorder.folder / f"{n + 1:02d}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
            wrong = problem(audio)
            flagged = flagged or bool(wrong)
            level = f"{recorder.name[:24]} {peak_dbfs(audio):.0f} dBFS"
            levels.append(f"{level} ({wrong})" if wrong else level)
        print("        " + " | ".join(levels))
        if flagged:
            print('        Se è così, "r" e Invio per rifarla.')
        i += 1
    print("\nFatto: ora uv run python scripts/stt_bench.py")


def check(out: Path) -> list[int]:
    """The sentences (1-based) some microphone recorded mute or cut, each printed with why."""
    import soundfile as sf

    bad: set[int] = set()
    for folder in sorted(p for p in out.iterdir() if p.is_dir()):
        for wav in sorted(folder.glob("[0-9][0-9].wav")):
            wrong = problem(sf.read(wav, dtype="int16")[0])
            if wrong:
                print(f"  {folder.name}  {wav.stem}: {wrong}")
                bad.add(int(wav.stem))
    return sorted(bad)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="elenca i microfoni ed esce")
    parser.add_argument("--mics", type=int, nargs="+", help="i numeri dei microfoni (--list)")
    parser.add_argument("--frasi", type=Path, default=SENTENCES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--controlla", action="store_true", help="registrazioni mute o tagliate")
    parser.add_argument("--rifai", type=int, nargs="+", help="registra di nuovo queste frasi")
    args = parser.parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):  # accents and «» also in a redirected file
        sys.stdout.reconfigure(encoding="utf-8")

    if args.list:
        for index in candidate_inputs():
            print(f"  [{index}] {describe_device(index, 'input')}")
        return
    sentences = read_sentences(args.frasi)
    if args.controlla:
        bad = check(args.out)
        if bad:
            print(f"\nPer rifarle: --rifai {' '.join(str(n) for n in bad)}")
        else:
            print("Nessuna registrazione muta o tagliata.")
        return
    if args.rifai and not all(1 <= n <= len(sentences) for n in args.rifai):
        raise SystemExit(f"--rifai: numeri da 1 a {len(sentences)}")
    indexes = args.mics or candidate_inputs()
    keep_sentences(args.out, sentences)
    print("Microfoni:")
    recorders = open_recorders(indexes, args.out)
    if not recorders:
        raise SystemExit("Nessun microfono si apre: controlla che siano collegati")
    for recorder in recorders:
        recorder.folder.mkdir(parents=True, exist_ok=True)
        (recorder.folder / "microfono.txt").write_text(recorder.description, encoding="utf-8")
    try:
        record(recorders, sentences, args.rifai)
    finally:
        for recorder in recorders:
            recorder.close()


if __name__ == "__main__":
    main()
