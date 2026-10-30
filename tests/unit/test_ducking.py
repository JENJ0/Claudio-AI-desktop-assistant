"""A reply lowered because a voice talks over it (ducking) is logged with that voice's level."""

import logging

from tests.unit.test_conversation import LongSpeaker, barge, build, speak_over


async def test_lowering_the_reply_is_logged_with_the_voice_level(
    fake_vad, vad_settings, make_frames, matcher, caplog
) -> None:
    """Nobody could tell when ducking fired; its level is the data barge_in_min_dbfs is
    tuned on."""
    speaker = LongSpeaker()
    assistant, _, _, _ = build(
        fake_vad,
        vad_settings,
        matcher,
        ["che ore sono"],
        window_s=0.0,
        speaker=speaker,
        barge_in=barge(vad_settings, ["eh"]),
    )
    with caplog.at_level(logging.INFO, logger="claudio.app"):
        await speak_over(assistant, make_frames, speaker)
    assert speaker.ducks == [0.3, None]
    lowered = [r for r in caplog.records if r.msg == "Abbasso la risposta: voce a %.0f dBFS"]
    assert len(lowered) == 1
    (level,) = lowered[0].args
    assert assistant.barge_in is not None and level >= assistant.barge_in.min_dbfs
