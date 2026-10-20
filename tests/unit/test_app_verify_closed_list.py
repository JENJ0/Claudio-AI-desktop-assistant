"""The assistant with the wake verifier and the closed list switched on."""

from __future__ import annotations

from tests.unit.test_app import FakeResponder, FakeStt, build, run_frames


class Understanding(FakeResponder):
    def understands(self, text: str, language: str = "it") -> bool:
        return "chrome" in text


def setup(fake_vad, vad_settings, text: str, candidates: list[str] | None = None):
    assistant, _, speaker, _ = build(fake_vad, vad_settings, FakeStt(text), {1})
    responder = Understanding()
    assistant.responder = responder
    assistant.verify_wake = True
    calls: list[str] = []
    if candidates is not None:

        def closed_list(audio, language):
            calls.append(language)
            return candidates

        assistant.closed_list = closed_list
    return assistant, speaker, responder, calls


def request(make_frames):
    return make_frames("silence", 80) + make_frames("speech", 800) + make_frames("silence", 800)


async def test_false_wake_ends_in_silence(fake_vad, vad_settings, make_frames) -> None:
    assistant, speaker, responder, _ = setup(fake_vad, vad_settings, "Hey, ciao, come stai?")
    await run_frames(assistant, request(make_frames))
    assert speaker.spoken == [] and speaker.beeps == ["wake"]  # no "non ho capito", no beep
    assert responder.seen == []
    assert assistant.history[-1]["outcome"] == "false_wake"


async def test_name_at_start_is_answered(fake_vad, vad_settings, make_frames) -> None:
    assistant, speaker, _, _ = setup(fake_vad, vad_settings, "Hey Claudio prenota un volo")
    await run_frames(assistant, request(make_frames))
    assert speaker.spoken == ["Questo non lo so ancora fare."]


async def test_command_without_the_name_is_answered(fake_vad, vad_settings, make_frames) -> None:
    assistant, speaker, _, _ = setup(fake_vad, vad_settings, "e apri chrome")
    await run_frames(assistant, request(make_frames))
    assert speaker.spoken == ["Apro Chrome."]


async def test_verifier_off_answers_everything(fake_vad, vad_settings, make_frames) -> None:
    assistant, speaker, _, _ = setup(fake_vad, vad_settings, "Hey, ciao, come stai?")
    assistant.verify_wake = False
    await run_frames(assistant, request(make_frames))
    assert speaker.spoken == ["Questo non lo so ancora fare."]


async def test_closed_list_picks_the_known_command(fake_vad, vad_settings, make_frames) -> None:
    assistant, speaker, responder, calls = setup(
        fake_vad, vad_settings, "Claudio, a prick roam", ["Claudio, a prick roam", "apri chrome"]
    )
    await run_frames(assistant, request(make_frames))
    assert calls == ["it"]
    assert responder.seen == ["apri chrome"] and speaker.spoken == ["Apro Chrome."]
    row = assistant.history[-1]
    assert row["closed_list_from"] == "Claudio, a prick roam" and "closed_list_ms" in row


async def test_closed_list_not_used_when_understood(fake_vad, vad_settings, make_frames) -> None:
    assistant, _, _, calls = setup(fake_vad, vad_settings, "apri chrome", ["x"])
    await run_frames(assistant, request(make_frames))
    assert calls == []


async def test_closed_list_without_a_command_keeps_the_text(
    fake_vad, vad_settings, make_frames
) -> None:
    assistant, _, responder, calls = setup(
        fake_vad, vad_settings, "Claudio prenota un volo", ["Claudio prenoto un volo"]
    )
    await run_frames(assistant, request(make_frames))
    assert calls == ["it"] and responder.seen == ["Claudio prenota un volo"]
    assert "closed_list_from" not in assistant.history[-1]
