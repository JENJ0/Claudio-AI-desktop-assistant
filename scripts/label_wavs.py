"""Label the saved WAVs of a session: listen, then confirm or correct the text with one key.

Copy a session's files from logs/ into samples/corpus/ (nothing is overwritten):
    uv run python scripts/label_wavs.py --session 2026-11-05_17-00 --import
Label it (the first time a session needs --split dev|test, afterwards it cannot change):
    uv run python scripts/label_wavs.py --session 2026-11-05_17-00 --split dev

Keys: Invio = right, for Claudio   n = right, not for Claudio   c = correct the text
      r = listen again   s = skip   u = undo the last label of this run   q = quit
Labels go to samples/corpus/labels.jsonl, one line each, saved at once: Ctrl+C loses nothing.
`text` is meaningful only when `for_claudio` is true (`proposed` always keeps the proposal).
u removes the last label given in this run (one per press) and offers that WAV again.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS = PROJECT_ROOT / "logs"
CORPUS = PROJECT_ROOT / "samples" / "corpus"
SPLITS = ("dev", "test")
SESSION_FORMAT = "%Y-%m-%d_%H-%M"
SESSION_NAME = re.compile(r"\d{4}-\d\d-\d\d_\d\d-\d\d(_\d+)?")
ROW_FIELDS = ("outcome", "wake_word", "route", "rejected", "speech_ms")
REPLY_CHARS = 80
WAV_FORMAT = "%Y%m%d-%H%M%S"


def session_start(name: str) -> datetime:
    """Folder names are `YYYY-MM-DD_HH-MM`, with an optional `_2` for a second run."""
    if not SESSION_NAME.fullmatch(name):
        raise SystemExit(f"Nome di sessione non valido: '{name}' (serve AAAA-MM-GG_HH-MM).")
    try:
        return datetime.strptime(name[:16], SESSION_FORMAT)
    except ValueError:
        raise SystemExit(f"Nome di sessione non valido: '{name}' (data o ora errate).") from None


def wav_time(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.name[:15], WAV_FORMAT)
    except ValueError:
        return None


def copy_new(src: Path, dst: Path) -> bool:
    """Copy a file unless the destination exists. True if it was copied. The copy goes to
    a temporary name first, so an interrupted run never leaves a partial `dst`."""
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".part")
    shutil.copy2(src, partial)
    os.replace(partial, dst)
    return True


def session_window(logs: Path, name: str) -> tuple[datetime, datetime | None]:
    """From the session's start to the next session's start (None: the last one).
    Another folder in the same minute is refused: their WAVs could not be told apart."""
    start = session_start(name)
    others = [
        p.name
        for p in (logs / "sessions").iterdir()
        if p.is_dir() and p.name != name and SESSION_NAME.fullmatch(p.name)
    ]
    for other in others:
        if other[:16] == name[:16]:
            raise SystemExit(f"{name} e {other} partono nello stesso minuto: non importo.")
    later = sorted(s for s in (session_start(o) for o in others) if s > start)
    return start, (later[0] if later else None)


def import_session(logs: Path, corpus: Path, name: str) -> int:
    """Copy the session's console, conversation, Jack, WAVs (with notes) and metrics rows
    into corpus/<name>/. Returns how many WAVs were copied; existing files are kept."""
    source = logs / "sessions" / name
    session_start(name)  # a malformed name stops here, before looking for the folder
    if not source.is_dir():
        raise SystemExit(f"Non esiste la sessione {source}.")
    start, end = session_window(logs, name)
    dest = corpus / name
    for item in source.rglob("*"):
        if item.is_file():
            copy_new(item, dest / item.relative_to(source))
    copied = 0
    audio = logs / "audio"
    for wav in sorted([*audio.glob("*.wav"), *audio.glob("rejected/*.wav")]):
        when = wav_time(wav)
        if when is None or when < start or (end is not None and when >= end):
            continue
        target = dest / "audio" / wav.relative_to(audio)
        copied += copy_new(wav, target)
        note = wav.with_suffix(".json")
        if note.is_file():
            copy_new(note, target.with_suffix(".json"))
    _import_metrics(logs, dest / "metrics.jsonl", start, end)
    return copied


def _import_metrics(logs: Path, target: Path, start: datetime, end: datetime | None) -> None:
    if target.exists():
        return
    kept: list[str] = []
    for file in sorted(logs.glob("metrics.jsonl*"), reverse=True):  # oldest rotation first
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                when = datetime.fromtimestamp(json.loads(line)["ts"])
            except (ValueError, KeyError, TypeError):
                continue
            if when >= start and (end is None or when < end):
                kept.append(line)
    if kept:
        partial = target.with_name(target.name + ".part")
        partial.write_text("\n".join(kept) + "\n", encoding="utf-8")
        os.replace(partial, target)


def load_rows(session_dir: Path) -> dict[str, dict[str, Any]]:
    """audio_file -> metrics row of the session."""
    path = session_dir / "metrics.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("audio_file"):
            rows[row["audio_file"]] = row
    return rows


def propose(row: dict[str, Any] | None) -> str:
    row = row or {}
    return row.get("text") or row.get("second_chance") or ""


def wav_key(session_dir: Path, wav: Path) -> str:
    return wav.relative_to(session_dir).as_posix()


def labelled_keys(session: str, labels_path: Path) -> set[str]:
    done: set[str] = set()
    if not labels_path.is_file():
        return done
    # errors="replace": a tail torn inside a multi-byte character must not stop the run
    text = labels_path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), 1):
        try:
            record = json.loads(line)
        except ValueError:
            print(f"Riga {number} di {labels_path.name} incompleta: la salto (WAV da rifare).")
            continue
        if record.get("session") == session:
            done.add(record.get("wav", ""))
    return done


def pending(session_dir: Path, labels_path: Path) -> list[Path]:
    """The session's WAVs (audio/ and audio/rejected/, by name) with no label yet."""
    done = labelled_keys(session_dir.name, labels_path)
    audio = session_dir / "audio"
    wavs = sorted([*audio.glob("*.wav"), *audio.glob("rejected/*.wav")], key=lambda p: p.name)
    return [w for w in wavs if wav_key(session_dir, w) not in done]


def assign_split(splits_path: Path, session: str, wanted: str | None) -> str:
    """A session's split is chosen once (dev or test) and never changes."""
    if wanted is not None and wanted not in SPLITS:
        raise SystemExit(f"--split deve essere dev o test, non '{wanted}'.")
    splits: dict[str, str] = {}
    if splits_path.is_file():
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
    known = splits.get(session)
    if known is None:
        if wanted is None:
            raise SystemExit("Sessione nuova: serve --split dev|test.")
        splits[session] = wanted
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
        return wanted
    if wanted is not None and wanted != known:
        raise SystemExit(f"La sessione {session} è già '{known}': non si cambia in '{wanted}'.")
    return known


def append_label(labels_path: Path, record: dict[str, Any]) -> None:
    """One complete line, flushed to disk. A torn last line from a crash is closed first."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if labels_path.is_file() and labels_path.stat().st_size:
        with labels_path.open("rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
    with labels_path.open("a", encoding="utf-8") as f:
        f.write(("\n" if needs_newline else "") + json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def remove_label(labels_path: Path, record: dict[str, Any]) -> bool:
    """Drop the last line equal to `record`, leaving every other byte as it was. The file is
    rewritten to a temporary name and swapped in, so a crash leaves the old or the new file,
    never a partial one. False if the line is not there (edited by hand)."""
    line = json.dumps(record, ensure_ascii=False).encode("utf-8")
    lines = labels_path.read_bytes().split(b"\n")  # text mode wrote \r\n on Windows
    found = [i for i, text in enumerate(lines) if text.rstrip(b"\r") == line]
    if not found:
        return False
    del lines[found[-1]]
    partial = labels_path.with_name(labels_path.name + ".part")
    with partial.open("wb") as f:
        f.write(b"\n".join(lines))
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, labels_path)
    return True


def describe_row(row: dict[str, Any] | None) -> str:
    """The metrics fields worth seeing while labelling, when they are not null."""
    row = row or {}
    parts = [f"{k}={row[k]}" for k in ROW_FIELDS if row.get(k) is not None]
    if row.get("reply") is not None:
        parts.append(f"reply={str(row['reply'])[:REPLY_CHARS]}")
    return " ".join(parts)


def make_key_reader(kbhit: Callable[[], bool], getwch: Callable[[], str]) -> Callable[[], str]:
    """One key per call. Keys typed while a WAV played are dropped; Ctrl+C, which getwch
    returns as the character 0x03, raises KeyboardInterrupt; the prefix and scan code of
    special keys (arrows, F-keys, Canc) are consumed instead of acting as keys."""

    def read_key() -> str:
        while kbhit():
            if getwch() in ("\x00", "\xe0"):
                getwch()  # the scan code sits where kbhit cannot see it
        while True:
            key = getwch()
            if key == "\x03":
                raise KeyboardInterrupt
            if key in ("\x00", "\xe0"):
                getwch()
                continue
            return key

    return read_key


def label_session(
    session_dir: Path,
    labels_path: Path,
    split: str,
    getch: Callable[[], str],
    play: Callable[[Path], None],
    read_line: Callable[[str], str],
) -> int:
    """Ask for a key per pending WAV. Returns how many were labelled. Ctrl+C propagates."""
    rows = load_rows(session_dir)
    todo = pending(session_dir, labels_path)
    given: list[tuple[int, dict[str, Any]]] = []  # this run's labels: (index in todo, record)
    index = 0
    while index < len(todo):
        wav = todo[index]
        proposed = propose(rows.get(wav.name))
        text = proposed
        left = len(todo) - index
        print(f"\n[{index + 1}/{len(todo)}, ne restano {left}] {wav_key(session_dir, wav)}")
        if info := describe_row(rows.get(wav.name)):
            print(f"   riga: {info}")
        try:
            play(wav)
        except Exception as error:  # an unreadable file must not stop the session
            print(f"   WAV illeggibile, lo salto ({error})")
            index += 1
            continue
        while True:
            print(f'   testo: "{text}"')
            print("   [Invio=ok  n=non per Claudio  c=correggi  r=riascolta  s=salta]")
            print("   [u=annulla l'ultima etichetta di questa sessione di lavoro  q=esci]")
            key = getch().lower()
            if key == "q":
                return len(given)
            if key == "s":
                index += 1
                break
            if key == "u":
                if not given:
                    print("   nessuna etichetta da annullare in questa sessione di lavoro")
                elif remove_label(labels_path, given[-1][1]):
                    index = given.pop()[0]  # offer that WAV again
                    break
                else:
                    print("   etichetta non trovata in labels.jsonl: non annullo")
            elif key == "r":
                play(wav)
            elif key == "c":
                text = read_line("   testo corretto: ").strip() or text  # bare Enter: keep it
            elif key in ("\r", "\n") and not text:
                print("   nessun testo da confermare: usa c per scriverlo, n se non è per Claudio")
            elif key in ("\r", "\n", "n"):
                record = {
                    "wav": wav_key(session_dir, wav),
                    "session": session_dir.name,
                    "split": split,
                    "proposed": proposed,
                    "text": text,
                    "for_claudio": key != "n",
                    "labeled_at": datetime.now().isoformat(timespec="seconds"),
                }
                append_label(labels_path, record)
                given.append((index, record))
                index += 1
                break
    return len(given)


def play_wav(path: Path) -> None:
    """Play a WAV; the file is read first, so a broken one raises. No output: say so, go on."""
    import sounddevice as sd
    import soundfile as sf

    audio, rate = sf.read(path, dtype="int16")
    try:
        sd.play(audio, rate)
        sd.wait()
    except sd.PortAudioError:
        print("   nessuna uscita audio: leggi il testo e decidi senza ascoltare")


def main() -> None:
    parser = argparse.ArgumentParser(description="Etichetta i WAV di una sessione.")
    parser.add_argument(
        "--session", required=True, help="nome della cartella, es. 2026-11-05_17-00"
    )
    parser.add_argument("--split", choices=SPLITS, help="dev o test (solo la prima volta)")
    parser.add_argument(
        "--import", dest="do_import", action="store_true", help="copia prima da logs/"
    )
    args = parser.parse_args()

    if args.do_import:
        print(
            f"Copiati {import_session(LOGS, CORPUS, args.session)} WAV in {CORPUS / args.session}"
        )
    session_dir = CORPUS / args.session
    if not session_dir.is_dir():
        raise SystemExit(f"{session_dir} non esiste: usa --import.")
    split = assign_split(CORPUS / "splits.json", args.session, args.split)

    import msvcrt

    read_key = make_key_reader(msvcrt.kbhit, msvcrt.getwch)
    try:
        done = label_session(session_dir, CORPUS / "labels.jsonl", split, read_key, play_wav, input)
    except KeyboardInterrupt:
        print("\nInterrotto: le etichette già date sono salvate.")
        sys.exit(130)
    print(f"Etichettati {done} WAV.")


if __name__ == "__main__":
    main()
