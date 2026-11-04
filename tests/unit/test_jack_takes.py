"""Jack's takes checked before they are played (a fake worker: no GPU)."""

import json
import sys
import textwrap
import time

import numpy as np
import pytest

from claudio.tts.jack import JackSynth
from claudio.tts.spoken_text import to_spoken_it
from claudio.tts.takes import TakeStore
from claudio.tts.voice_check import TakeJudge, TakeVerdict

# Same protocol as jack_worker.py. A take is a 120 Hz tone (Jack's usual register) or, for
# the seeds in argv[1], a 300 Hz one (the strained, high Jack); 3 chunks of 0.67 s, with
# argv[3] seconds between them. Every speak request is logged to argv[2].
FAKE_WORKER = textwrap.dedent(
    """
    import json, sys, time
    import numpy as np
    bad = {int(s) for s in sys.argv[1].split(",") if s}
    log, pause = sys.argv[2], float(sys.argv[3])
    out = sys.stdout.buffer
    def emit(header, payload=b""):
        out.write(json.dumps(header).encode() + b"\\n" + payload); out.flush()
    emit({"event": "ready", "sample_rate": 24000, "load_s": 0.1, "vram_mb": 1})
    cancelled = -1
    t = np.arange(16000) / 24000
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "quit":
            break
        if req["op"] == "interrupt":
            cancelled = max(cancelled, req.get("upto", 10**9))
            continue
        rid = req["id"]
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(req) + "\\n")
        if rid <= cancelled:
            emit({"event": "done", "id": rid, "cancelled": True, "first_ms": None})
            continue
        hz = 300.0 if req.get("seed") in bad else 120.0
        pcm = (1300 * np.sin(2 * np.pi * hz * t)).astype("<i2").tobytes()
        for k in range(3):
            if k:
                time.sleep(pause)
            emit({"event": "audio", "id": rid, "bytes": len(pcm), "raw_lufs": -24.0,
                  "raw_clipped": 0}, pcm)
        emit({"event": "done", "id": rid, "cancelled": False, "first_ms": 5, "total_ms": 9,
              "seed": req.get("seed"), "raw_lufs": -24.0, "raw_clipped": 0})
    """
)
TEXT = "Ci guardo, socio, aspetta un attimo."  # 36 characters: 2 s is Jack's usual pace


def pitch(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio.astype(float)))
    return float(np.fft.rfftfreq(len(audio), 1 / 24_000)[np.argmax(spectrum)])


class Jack:
    def __init__(self, tmp_path, bad: str = "", pause: float = 0.0, **kwargs) -> None:
        script = tmp_path / "fake_worker.py"
        script.write_text(FAKE_WORKER, encoding="utf-8")
        self.log = tmp_path / "requests.jsonl"
        command = [sys.executable, str(script), bad, str(self.log), str(pause)]
        kwargs.setdefault("judge", TakeJudge(target_lufs=-24.0, usual_hz=132.0))
        kwargs.setdefault("seed", 7)
        self.synth = JackSynth(command, None, startup_timeout_s=20, **kwargs)
        self.synth.start()

    def seeds(self) -> list[int | None]:
        if not self.log.exists():
            return []
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line).get("seed") for line in lines]


@pytest.fixture
def jacks():
    made: list[Jack] = []
    yield made
    for jack in made:
        jack.synth.close()


def make(jacks, tmp_path, **kwargs) -> Jack:
    jack = Jack(tmp_path, **kwargs)
    jacks.append(jack)
    return jack


def test_a_bad_first_take_is_dropped_before_it_is_heard(tmp_path, jacks, caplog) -> None:
    jack = make(jacks, tmp_path, bad="7")
    with caplog.at_level("INFO", logger="claudio.tts.jack"):
        audio = jack.synth.synthesize(TEXT)
    assert jack.seeds() == [7, 8]
    assert len(audio) == 3 * 16_000 and abs(pitch(audio) - 120) < 5  # only the good take
    assert any("scartata (seme 7)" in r.getMessage() for r in caplog.records)


def test_a_take_dropped_at_the_first_look_stops_the_worker_at_once(tmp_path, jacks) -> None:
    jack = make(jacks, tmp_path, bad="7")
    sent: list[dict] = []
    original = jack.synth._send
    jack.synth._send = lambda message, proc=None: (sent.append(message), original(message, proc))[1]
    jack.synth.synthesize(TEXT)
    assert [m["op"] for m in sent] == ["speak", "interrupt", "speak"]
    assert sent[1]["upto"] == sent[0]["id"] and sent[2]["seed"] == 8


def test_the_opening_piece_is_judged_on_its_first_chunk(tmp_path, jacks) -> None:
    """+0.3-0.5 s only when the first take is bad: a good one is let go after one chunk."""
    jack = make(jacks, tmp_path, pause=1.0)
    started = time.monotonic()
    first = next(iter(jack.synth.stream(TEXT)))
    assert time.monotonic() - started < 0.9  # not held for the whole take (2 s of pauses)
    assert abs(pitch(first) - 120) < 5


def test_the_opening_piece_gets_one_more_take_at_most(tmp_path, jacks, caplog) -> None:
    jack = make(jacks, tmp_path, bad="7,8,9")
    with caplog.at_level("INFO", logger="claudio.tts.jack"):
        audio = jack.synth.synthesize(TEXT)
    assert jack.seeds() == [7, 8]
    assert abs(pitch(audio) - 300) < 5  # the last one is played anyway
    assert any("anche se non buona" in r.getMessage() for r in caplog.records)


def test_a_later_piece_is_judged_whole_while_the_one_before_plays(tmp_path, jacks) -> None:
    jack = make(jacks, tmp_path, bad="7,8")
    jack.synth._horizon = time.monotonic() + 10  # a long piece is still playing
    audio = jack.synth.synthesize(TEXT)
    assert jack.seeds() == [7, 8, 9]
    assert abs(pitch(audio) - 120) < 5


def test_a_held_take_goes_out_before_what_plays_runs_out(tmp_path, jacks) -> None:
    """A slow take is never held into a gap: let go when the piece before is about to end."""
    jack = make(jacks, tmp_path, pause=1.0)  # the whole take needs 2 s
    jack.synth._horizon = time.monotonic() + 1.5
    started = time.monotonic()
    next(iter(jack.synth.stream(TEXT)))
    assert 0.9 < time.monotonic() - started < 1.4  # held, then out 0.4 s before the end


def test_a_take_good_at_first_and_bad_as_a_whole_is_generated_again(tmp_path, jacks) -> None:
    class Judge(TakeJudge):
        """Fine on the early look; only the whole of the first take is bad."""

        def __init__(self) -> None:
            super().__init__()
            self.asked: list[tuple[int, bool, dict]] = []

        def judge(self, audio, rate, spoken, raw=None, complete=True) -> TakeVerdict:
            self.asked.append((len(audio), complete, dict(raw)))
            whole_takes = sum(1 for _, whole, _ in self.asked if whole)
            return TakeVerdict(["finta"] if complete and whole_takes == 1 else [], None)

    judge = Judge()
    jack = make(jacks, tmp_path, bad="7", judge=judge)  # seed 7 is 300 Hz, seed 8 is 120 Hz
    jack.synth._horizon = time.monotonic() + 20  # a later piece: no hurry
    audio = jack.synth.synthesize(TEXT)
    assert jack.seeds() == [7, 8]
    assert len(audio) == 3 * 16_000 and abs(pitch(audio) - 120) < 5  # nothing of the first take
    raw = {"raw_lufs": -24.0, "raw_clipped": 0}  # of the chunk being judged, already there
    assert judge.asked == [(16_000, False, raw), (48_000, True, raw)] * 2


def test_generate_take_gives_one_take_of_that_seed_never_judged(tmp_path, jacks) -> None:
    """scripts/jack_takes.py picks the best of several: a bad one is kept like any other."""
    jack = make(jacks, tmp_path, bad="5")
    jack.synth.rvc = "auto"
    audio, raw = jack.synth.generate_take(TEXT, 5)
    assert len(audio) == 3 * 16_000 and abs(pitch(audio) - 300) < 5
    assert raw == {"raw_lufs": -24.0, "raw_clipped": 0}
    jack.synth.generate_take(TEXT, 6, rvc="sentence")
    requests = [json.loads(line) for line in jack.log.read_text(encoding="utf-8").splitlines()]
    assert [(r["seed"], r["rvc"]) for r in requests] == [(5, "blocks"), (6, "sentence")]
    jack.synth.cancel(upto=jack.synth._next_id + 1)  # the worker answers with no audio
    audio, raw = jack.synth.generate_take(TEXT, 7)
    assert len(audio) == 0 and audio.dtype == np.int16 and raw == {}


def test_with_too_little_time_left_a_take_is_not_held(tmp_path, jacks) -> None:
    jack = make(jacks, tmp_path, bad="7", pause=1.0)
    jack.synth._horizon = time.monotonic() + 0.8  # no time to try another take
    started = time.monotonic()
    first = next(iter(jack.synth.stream(TEXT)))
    assert time.monotonic() - started < 0.5
    assert abs(pitch(first) - 300) < 5 and jack.seeds() == [7]


def test_a_text_keeps_the_seed_of_its_good_take(tmp_path, jacks) -> None:
    takes = TakeStore(tmp_path / "jack_takes.json", "p")
    jack = make(jacks, tmp_path, bad="7", takes=takes)
    jack.synth.synthesize(TEXT)
    assert jack.seeds() == [7, 8]
    jack.synth.synthesize(TEXT)  # said again: its good seed first
    assert jack.seeds() == [7, 8, 8]
    jack.synth.synthesize("Un'altra frase, socio, tutta nuova.")
    assert jack.seeds()[-2:] == [7, 8]  # a new text starts from the configured seed


def test_a_text_whose_take_was_bad_starts_from_another_seed(tmp_path, jacks) -> None:
    """30/10: the same text said again got the same shouted take, every time."""
    takes = TakeStore(tmp_path / "jack_takes.json", "p")
    jack = make(jacks, tmp_path, bad="7,8", takes=takes)
    jack.synth.synthesize(TEXT)  # the opening piece: seeds 7 and 8, both bad, 8 is played
    assert jack.seeds() == [7, 8]
    # Keyed by the text as Jack reads it: "un attimo" is "un àttimo" (stress_it.py).
    spoken = to_spoken_it(TEXT, jack.synth.words)
    assert takes.bad_seeds(takes.key(spoken, jack.synth.rvc)) == {7, 8}
    audio = jack.synth.synthesize(TEXT)
    assert jack.seeds()[2:] == [9] and abs(pitch(audio) - 120) < 5


def test_the_opening_piece_can_be_left_alone(tmp_path, jacks) -> None:
    """jack_first_retries = 0: the first piece is never held back, whatever it sounds like."""
    jack = make(jacks, tmp_path, bad="7", pause=1.0, first_retries=0)
    started = time.monotonic()
    first = next(iter(jack.synth.stream(TEXT)))
    assert time.monotonic() - started < 0.5 and abs(pitch(first) - 300) < 5


def test_a_reply_saved_in_advance_needs_no_worker(tmp_path) -> None:
    takes = TakeStore(tmp_path / "jack_takes.json", "p")
    pcm = (np.sin(np.arange(24_000) / 7) * 2000).astype(np.int16)
    takes.save_audio(takes.key("Apro la chat.", "none"), 5, "Apro la chat.", pcm, 24_000)
    synth = JackSynth(["no-such-worker"], None, rvc="none", takes=takes)  # never started
    audio = synth.synthesize("Apro la chat.")
    assert np.array_equal(audio, pcm) and synth.sample_rate == 24_000


def test_without_a_check_nothing_is_held_and_no_seed_is_sent(tmp_path, jacks) -> None:
    jack = make(jacks, tmp_path, pause=1.0, judge=None)
    started = time.monotonic()
    next(iter(jack.synth.stream(TEXT)))
    assert time.monotonic() - started < 0.5
    jack.synth.close()
    assert jack.seeds() == [None]  # the worker's own --seed, as before


def test_the_seeds_tried_start_with_the_known_good_one() -> None:
    synth = JackSynth(["x"], None, judge=TakeJudge(), seed=7)
    assert synth._seeds(None, 3) == [7, 8, 9]

    class Known:
        def seed(self, key):
            return 8

        def bad_seeds(self, key):
            return {9}

    synth.takes = Known()  # type: ignore[assignment]
    assert synth._seeds("k", 3) == [8, 7, 10]  # 9 was bad for this text
    synth.seed = -1  # random draws, the known one still first
    seeds = synth._seeds("k", 3)
    assert seeds[0] == 8 and len(set(seeds)) == 3
