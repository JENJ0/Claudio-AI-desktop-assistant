import numpy as np

from claudio.audio.vad import Endpointer, EndpointEvent
from claudio.config import VadSettings


def feed(ep: Endpointer, frames: list[np.ndarray]) -> list[EndpointEvent]:
    return [e for f in frames if (e := ep.process(f)) is not EndpointEvent.NONE]


def test_start_then_end_with_preroll(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    events = feed(
        ep, make_frames("silence", 400) + make_frames("speech", 1200) + make_frames("silence", 800)
    )
    assert events == [EndpointEvent.START, EndpointEvent.END]
    audio = ep.audio()
    # pre-roll (300 ms) + speech (1.2 s) + ~100 ms tail kept
    assert 1.4 < len(audio) / 16000 < 1.75
    assert audio.dtype == np.float32
    assert abs(ep.trailing_silence_s - 0.4) < 0.05


def test_timeout_when_nothing_is_said(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    assert feed(ep, make_frames("silence", 3500)) == [EndpointEvent.TIMEOUT]
    assert ep.process(make_frames("speech", 80)[0]) is EndpointEvent.NONE  # finished


def test_short_click_does_not_start(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    click = np.zeros(1280, dtype=np.int16)
    click[:300] = 20000  # < min_speech_ms
    assert feed(ep, [click, *make_frames("silence", 500)]) == []
    assert not ep.started


def test_max_utterance_closes_phrase(fake_vad, make_frames) -> None:
    settings = VadSettings(max_utterance_s=2.0)
    ep = Endpointer(fake_vad, settings)
    assert feed(ep, make_frames("speech", 4000)) == [EndpointEvent.START, EndpointEvent.END]


def test_reset_allows_new_phrase(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    feed(ep, make_frames("speech", 500) + make_frames("silence", 600))
    ep.reset()
    assert fake_vad.resets >= 2
    assert not ep.started and ep.audio().size == 0
    assert feed(ep, make_frames("speech", 500) + make_frames("silence", 600)) == [
        EndpointEvent.START,
        EndpointEvent.END,
    ]
