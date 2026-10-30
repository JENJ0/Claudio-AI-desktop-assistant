"""IT-1 5.3: Jack's worker with the graphics card full (a game). 30/10: ready in 21 s, or no
answer in 90 s; meanwhile every piece of a reply waited up to 30 s for it."""

from __future__ import annotations

import subprocess
import time
from types import SimpleNamespace
from typing import Any

import pytest

from claudio.__main__ import jack_wait_s, startup_notices
from claudio.config import load_settings
from claudio.gpu import GpuMemory, read_gpu
from claudio.tts.jack import JackSynth
from tests.unit.test_jack import make, started_in_background


def answer(stdout: str, code: int = 0) -> Any:
    def run(cmd: list[str], **kwargs: Any) -> Any:
        assert cmd[0] == "nvidia-smi" and kwargs["timeout"] > 0
        return SimpleNamespace(returncode=code, stdout=stdout)

    return run


@pytest.fixture
def synths() -> Any:
    made: list[JackSynth] = []
    yield made
    for synth in made:
        synth.close()


def test_reads_the_first_cards_memory_and_load() -> None:
    gpu = read_gpu(answer("14158, 16303, 93\n2000, 8000, 0\n"))
    assert gpu == GpuMemory(used_mb=14158, total_mb=16303, busy_pct=93)
    assert gpu.free_mb == 2145


@pytest.mark.parametrize(
    "run",
    [
        answer("", code=9),  # no NVIDIA card
        answer("[N/A], [N/A], [N/A]\n"),
        answer("   \n"),
    ],
)
def test_no_card_or_an_odd_answer_is_none(run: Any) -> None:
    assert read_gpu(run) is None


@pytest.mark.parametrize(
    "error", [FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 3)]
)
def test_no_driver_or_a_hung_one_is_none(error: Exception) -> None:
    def run(cmd: list[str], **kwargs: Any) -> Any:
        raise error

    assert read_gpu(run) is None


def test_replies_do_not_wait_for_jack_when_the_card_is_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = load_settings()
    need = settings.tts.jack_min_free_vram_mb
    with caplog.at_level("WARNING", logger="claudio"):
        assert jack_wait_s(settings, GpuMemory(14358, 16303, 98)) == 0.0  # now, 30/10 15:30
    assert "Scheda video occupata (libera 1945 MB su 16303, in uso al 98%)" in caplog.text
    assert jack_wait_s(settings, GpuMemory(16303 - need, 16303, 98)) is None  # enough: wait
    assert jack_wait_s(settings, None) is None  # no NVIDIA card: as before
    off = settings.model_copy(
        update={"tts": settings.tts.model_copy(update={"jack_min_free_vram_mb": 0})}
    )
    assert jack_wait_s(off, GpuMemory(16000, 16303, 98)) is None


def test_the_spoken_notice_names_the_busy_card() -> None:
    settings = load_settings()
    jack = settings.model_copy(update={"tts": settings.tts.model_copy(update={"engine": "jack"})})
    busy = startup_notices(jack, jack_running=False, stt_device="cuda", gpu_busy=True)
    assert busy[0].startswith("La scheda video è occupata")
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
