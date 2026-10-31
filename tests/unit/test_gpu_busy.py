"""IT-1 5.3: Jack's worker with the graphics card full (a game). 30/10: ready in 21 s, or no
answer in 90 s; meanwhile every piece of a reply waited up to 30 s for it."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from claudio.__main__ import jack_wait_s, startup_notices
from claudio.config import load_settings
from claudio.gpu import GpuMemory, has_room, read_gpu
from claudio.tts.jack import JackSynth, WorkerError, WorkerTimeout
from tests.unit.test_jack import FakeFallback, make, started_in_background

# A worker ready after READY_S, then saying 300 samples per sentence (test_jack's protocol).
LATE_WORKER = textwrap.dedent(
    """
    import json, sys, time
    out = sys.stdout.buffer
    def emit(header, payload=b""):
        out.write(json.dumps(header).encode() + b"\\n" + payload); out.flush()
    time.sleep(float(sys.argv[1]))
    emit({"event": "ready", "sample_rate": 24000, "load_s": 0.5, "vram_mb": 2900})
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "quit":
            break
        if req["op"] != "speak":
            continue
        pcm = bytes([1, 0]) * 300
        emit({"event": "audio", "id": req["id"], "bytes": len(pcm)}, pcm)
        emit({"event": "done", "id": req["id"], "cancelled": False, "first_ms": 5})
    """
)


class Smi:
    """A stand-in for nvidia-smi run through Popen."""

    def __init__(self, stdout: str = "", code: int = 0, hang: bool = False) -> None:
        self.stdout, self.code, self.hang = stdout, code, hang
        self.killed = False

    def __call__(self, cmd: list[str], **kwargs: Any) -> Smi:
        assert cmd[0] == "nvidia-smi" and "--query-gpu=memory.free" in cmd[1]
        return self

    def communicate(self, timeout: float) -> tuple[str, None]:
        assert timeout > 0
        if self.hang:
            raise subprocess.TimeoutExpired("nvidia-smi", timeout)
        return self.stdout, None

    @property
    def returncode(self) -> int:
        return self.code

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def synths() -> Any:
    made: list[JackSynth] = []
    yield made
    for synth in made:
        synth.close()


def late(tmp_path: Path, ready_s: float, **kwargs: Any) -> tuple[JackSynth, FakeFallback]:
    script = tmp_path / "late_worker.py"
    script.write_text(LATE_WORKER)
    fallback = FakeFallback()
    synth = JackSynth([sys.executable, str(script), str(ready_s)], fallback, **kwargs)
    return synth, fallback


def test_reads_the_first_cards_free_memory_and_load() -> None:
    # 30/10 21:40: free 4131 + used 11865 + reserved 308 = 16303: "free" is the driver's.
    assert read_gpu(Smi("4131, 16303, 95\n2000, 8000, 0\n")) == GpuMemory(4131, 16303, 95)
    # Utilization not reported (some laptops, vGPU): the memory still counts.
    assert read_gpu(Smi("4131, 16303, [N/A]\n")) == GpuMemory(4131, 16303, None)


@pytest.mark.parametrize("smi", [Smi("", code=9), Smi("[N/A], [N/A], [N/A]\n"), Smi("  \n")])
def test_no_card_or_an_odd_answer_is_none(smi: Smi) -> None:
    assert read_gpu(smi) is None


def test_no_driver_is_none() -> None:
    def popen(cmd: list[str], **kwargs: Any) -> Any:
        raise FileNotFoundError("nvidia-smi")

    assert read_gpu(popen) is None


def test_a_hung_driver_is_given_up_not_waited_for() -> None:
    smi = Smi(hang=True)
    assert read_gpu(smi) is None and smi.killed


def test_room_means_enough_free_or_no_card_at_all() -> None:
    assert has_room(6000, lambda: GpuMemory(7000, 16303)) is True
    assert has_room(6000, lambda: GpuMemory(2000, 16303)) is False
    assert has_room(6000, lambda: None) is True


def test_replies_do_not_wait_for_jack_when_the_card_is_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = load_settings()
    need = settings.tts.jack_min_free_vram_mb
    with caplog.at_level("WARNING", logger="claudio"):
        assert jack_wait_s(settings, GpuMemory(1945, 16303, 98)) == 0.0  # 30/10 15:30
    assert "Scheda video occupata (libera 1945 MB su 16303, calcolo al 98%)" in caplog.text
    assert jack_wait_s(settings, GpuMemory(need, 16303, 98)) is None  # enough: wait
    assert jack_wait_s(settings, None) is None  # no NVIDIA card: as before
    off = settings.model_copy(
        update={"tts": settings.tts.model_copy(update={"jack_min_free_vram_mb": 0})}
    )
    assert jack_wait_s(off, GpuMemory(300, 16303, 98)) is None


def test_the_spoken_notice_says_the_card_was_full_without_blaming_it() -> None:
    settings = load_settings()
    jack = settings.model_copy(update={"tts": settings.tts.model_copy(update={"engine": "jack"})})
    busy = startup_notices(jack, jack_running=False, stt_device="cuda", gpu_busy=True)
    assert busy == [
        "La voce di Jack non è partita: uso quella di riserva.",
        "All'avvio la scheda video era quasi piena.",
        "Il motivo è nel log.",
    ]
    assert startup_notices(jack, jack_running=True, stt_device="cuda", gpu_busy=True) == []


def test_pieces_share_one_wait_from_the_start(tmp_path, synths) -> None:
    """Before: each piece waited first_wait_s anew (three pieces, three waits)."""
    synth, fallback = make(tmp_path, "slow_start", startup_timeout_s=10, first_wait_s=0.5)
    synths.append(synth)
    started_in_background(synth)
    start = time.monotonic()
    synth.synthesize("Prima.")
    synth.synthesize("Seconda.")
    synth.synthesize("Terza.")
    assert time.monotonic() - start < 1.2  # ~0.5 s in all, not 1.5 s
    assert fallback.texts == ["Prima.", "Seconda.", "Terza."]


def test_a_full_card_lets_the_fallback_speak_at_once(tmp_path, synths) -> None:
    synth, fallback = make(tmp_path, "slow_start", startup_timeout_s=10, first_wait_s=30)
    synths.append(synth)
    synth.start_in_background(wait_s=0.0)
    start = time.monotonic()
    synth.synthesize("Ciao.")
    assert time.monotonic() - start < 0.5 and fallback.texts == ["Ciao."]
    assert synth.starting  # Jack keeps loading: later replies use him once he is ready


def test_a_reply_begun_in_the_fallback_voice_ends_in_it(tmp_path, synths) -> None:
    """Review of 5.3: Jack ready halfway through a reply would change the voice mid-reply."""
    synth, fallback = late(tmp_path, 0.5, startup_timeout_s=10, first_wait_s=0)
    synths.append(synth)
    synth.start_in_background(wait_s=0.0)
    synth.begin_reply()
    synth.synthesize("Uno.")  # Jack still loading
    assert synth.wait_started(10) and synth.alive()
    synth.synthesize("Due.")  # same reply: same voice
    assert fallback.texts == ["Uno.", "Due."]
    synth.begin_reply()
    assert len(synth.synthesize("Tre.")) == 300  # the next reply: Jack
    assert fallback.texts == ["Uno.", "Due."]


def test_a_worker_that_timed_out_on_a_full_card_comes_back_once_it_has_room(
    tmp_path, synths
) -> None:
    """Review of 5.3: a start that timed out (the game) was never tried again: Kokoro for
    the rest of the session. Now it is, but only once the card has room."""
    synth, fallback = late(tmp_path, 30, startup_timeout_s=0.3, first_wait_s=0)
    synths.append(synth)
    room = [False]
    synth.can_start = lambda: room[0]
    done = started_in_background(synth)
    assert synth.wait_started(10) and isinstance(done[0], WorkerTimeout)
    synth._restart_at = 0.0  # past the 30 s cooldown
    synth.synthesize("Ancora il gioco.")
    assert not synth._restarting and synth._restart_at > time.monotonic()  # postponed
    synth.command[-1] = "0"  # the game is closed: the worker loads at once
    room[0] = True
    synth._restart_at = 0.0
    synth.synthesize("Gioco chiuso.")  # the fallback says it, the worker restarts meanwhile
    deadline = time.monotonic() + 10
    while not synth.alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    synth.begin_reply()
    assert len(synth.synthesize("Eccomi.")) == 300
    assert fallback.texts == ["Ancora il gioco.", "Gioco chiuso."]


def test_a_broken_install_is_still_not_retried(tmp_path, synths) -> None:
    synth, _ = make(tmp_path, "bad_ready", startup_timeout_s=10, first_wait_s=0)
    synths.append(synth)
    synth.can_start = lambda: True
    done = started_in_background(synth)
    assert synth.wait_started(10)
    assert isinstance(done[0], WorkerError) and not isinstance(done[0], WorkerTimeout)
    assert synth._restart_at == float("inf")
