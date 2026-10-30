import numpy as np

from claudio.audio.vad import LEAD_IN_END_S, Endpointer, EndpointEvent, VoiceMeter
from claudio.config import VadSettings
from tests.conftest import frames as make


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


def test_stats_and_listened_audio(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    silent = {
        "vad_max": 0.0,
        "input_dbfs": -120.0,
        "speech_dbfs": -120.0,
        "clipped": 0.0,
        "voice_ms": 0,
    }
    assert ep.stats() == silent
    feed(ep, make_frames("silence", 400))
    assert ep.stats() == silent
    feed(ep, make_frames("speech", 400))
    stats = ep.stats()
    assert stats["vad_max"] == 0.9
    # tone at half scale is -9.2 dBFS RMS; silent half the time -> -12.2 dBFS
    assert -12.6 < stats["input_dbfs"] < -11.8
    assert -9.6 < stats["speech_dbfs"] < -8.8  # loudest window: the tone itself
    assert abs(stats["voice_ms"] - 400) <= 32  # the tone, not the silence (one VAD window)
    assert len(ep.listened()) == 16000 * 800 // 1000
    ep.reset()
    assert len(ep.listened()) == 0 and ep.stats()["vad_max"] == 0.0


def lead_in(*parts: tuple[str, int]) -> np.ndarray:
    """What the assistant heard before the wake word fired, float32."""
    frames = [f for kind, ms in parts for f in make(kind, ms)]
    return np.concatenate(frames).astype(np.float32) / 32768.0


def phrase_len(ep: Endpointer, lead: np.ndarray | None, frames: list[np.ndarray]) -> int:
    ep.reset(lead_in=lead)
    assert feed(ep, frames) == [EndpointEvent.START, EndpointEvent.END]
    return len(ep.audio())


def test_one_breath_phrase_keeps_the_speech_before_the_wake_word(
    fake_vad, vad_settings, make_frames
) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    rest = make_frames("speech", 600) + make_frames("silence", 600)
    alone = phrase_len(ep, None, rest)
    # "hey claudio che ore" heard before the wake word fired, after a pause: only that is added
    with_lead = phrase_len(ep, lead_in(("silence", 480), ("speech", 400)), rest)
    added = (with_lead - alone) / 16000
    assert 0.4 <= added <= 0.5  # the 400 ms of speech plus one window of margin


def test_the_name_and_the_request_after_the_beep_are_one_phrase(
    fake_vad, vad_settings, make_frames
) -> None:
    """ "Claudio" [the beep] "apri Chrome": the name is in the phrase Whisper gets."""
    ep = Endpointer(fake_vad, vad_settings)
    rest = make_frames("speech", 600) + make_frames("silence", 600)
    alone = phrase_len(ep, None, rest)
    # the wake word fired a little after the name: 400 ms of the pause are in the lead-in
    added = phrase_len(ep, lead_in(("speech", 400), ("silence", 400)), rest) - alone
    assert 0.8 <= added / 16000 <= 0.85  # the name, its pause and one window of margin
    # the request starts 480 ms after the wake word: still the same phrase
    late = make_frames("silence", 480) + rest
    added = phrase_len(ep, lead_in(("speech", 800)), late) - phrase_len(ep, None, late)
    assert 0.8 <= added / 16000 <= 1.3  # the name; the pause was already there


def test_a_request_over_when_the_wake_word_fired_is_heard(
    fake_vad, vad_settings, make_frames
) -> None:
    """30/10 14:06:40: "Claudio, spegniti." said fast, the second chance woke CLAUDIO 350 ms
    after it, then 4 s of silence and a timeout. Now it ends LEAD_IN_END_S after its last
    word, without waiting for more."""
    ep = Endpointer(fake_vad, vad_settings)
    ep.reset(lead_in=lead_in(("silence", 400), ("speech", 800), ("silence", 320)))
    events = feed(ep, make_frames("silence", 1200))
    assert events == [EndpointEvent.START, EndpointEvent.END]
    heard = len(ep.audio()) / 16000
    assert 0.8 <= heard <= 1.0  # the phrase and ~100 ms of its silence
    assert ep.stats()["voice_ms"] >= 768  # its level counts: it is the request's voice
    assert ep.trailing_silence_s >= LEAD_IN_END_S


def test_old_speech_or_a_click_before_the_wake_word_is_not_the_phrase(
    fake_vad, vad_settings, make_frames
) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    late = make_frames("silence", 480) + make_frames("speech", 600) + make_frames("silence", 600)
    alone = phrase_len(ep, None, late)
    # speech over long before the detection: someone else, not this request
    old = lead_in(("speech", 400), ("silence", int(LEAD_IN_END_S * 1000) + 80))
    assert phrase_len(ep, old, late) == alone
    click = np.concatenate((np.zeros(16000), np.full(512, 0.5), np.zeros(4096))).astype(np.float32)
    assert phrase_len(ep, click, late) == alone
    # nothing said at all after it: the usual timeout
    ep.reset(lead_in=old)
    assert feed(ep, make_frames("silence", 3200)) == [EndpointEvent.TIMEOUT]


def test_lead_in_without_pauses_is_kept_whole(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    rest = make_frames("speech", 600) + make_frames("silence", 600)
    alone = phrase_len(ep, None, rest)
    lead = lead_in(("speech", 1440))
    assert len(lead) - 512 < phrase_len(ep, lead, rest) - alone <= len(lead)


def test_voice_meter_counts_the_windows_of_speech(fake_vad, make_frames) -> None:
    meter = VoiceMeter(fake_vad, threshold=0.5)
    audio = np.concatenate(
        make_frames("silence", 1200) + make_frames("speech", 320) + make_frames("silence", 880)
    ).astype(np.float32)
    assert abs(meter(audio / 32768) - 320) <= 32  # one VAD window of margin
    assert meter(np.zeros(40000, dtype=np.float32)) == 0  # silence: Whisper is not asked
    click = np.zeros(40000, dtype=np.float32)
    click[20000:20100] = 0.9
    assert meter(click) <= 64  # one or two windows: under the 96 ms a word needs
    assert fake_vad.resets >= 3  # each stretch is judged on its own


def test_level_counts_speech_only_and_spots_clipping(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    quiet_bump = np.full(1280, 3000, dtype=np.int16)  # audible, but under FakeVad's 0.1
    clipped = make_frames("speech", 400)
    clipped[0] = np.clip(clipped[0].astype(np.int32) * 4, -32768, 32767).astype(np.int16)
    feed(ep, [quiet_bump, *clipped])
    stats = ep.stats()
    assert -9.6 < stats["speech_dbfs"] < 0  # the tone, not the bump
    assert 0.05 < stats["clipped"] < 0.2  # one frame of five, mostly at full scale


def test_extend_waits_for_the_rest_of_the_phrase(fake_vad, vad_settings, make_frames) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    feed(ep, make_frames("speech", 800) + make_frames("silence", 480))
    first = ep.audio()
    ep.extend(first, wait_s=2.0)
    assert ep.extended and not ep.spoke_again
    # A pause longer than end_silence_ms does not close it any more...
    assert feed(ep, make_frames("silence", 1200)) == []
    # ...the voice comes back, and the usual pause ends the whole phrase.
    events = feed(ep, make_frames("speech", 640) + make_frames("silence", 1200))
    assert events == [EndpointEvent.END] and ep.spoke_again
    assert len(ep.audio()) / 16000 > len(first) / 16000 + 1.2 + 0.6
    ep.reset()
    assert not ep.extended and not ep.spoke_again


def test_extend_ends_after_the_wait_when_nothing_more_is_said(
    fake_vad, vad_settings, make_frames
) -> None:
    ep = Endpointer(fake_vad, vad_settings)
    feed(ep, make_frames("speech", 800) + make_frames("silence", 480))
    ep.extend(ep.audio(), wait_s=1.0)
    assert feed(ep, make_frames("silence", 880)) == []
    assert feed(ep, make_frames("silence", 240)) == [EndpointEvent.END]
    assert not ep.spoke_again
