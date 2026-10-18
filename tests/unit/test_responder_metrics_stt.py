import json

import pytest

from claudio.metrics import Interaction, MetricsWriter, percentile, summarize
from claudio.router.actions import ActionContext, ActionRunner
from claudio.router.responder import LocalResponder
from claudio.stt.transcriber import is_hallucination


def make_responder(matcher, fail=False) -> LocalResponder:
    def boom(_args):
        raise OSError("no chrome")

    ctx = ActionContext(
        run=boom if fail else (lambda a: None), open_path=lambda p: None, press_key=lambda k: None
    )
    return LocalResponder(matcher, ActionRunner(ctx), "Questo non lo so ancora fare.")


async def collect(stream) -> list[str]:
    return [c async for c in stream]


async def test_responder_known_unknown_and_error(matcher) -> None:
    r = make_responder(matcher)
    assert await collect(r.respond("Apri Chrome.")) == ["Apro Chrome."]
    assert r.last_route["command"] == "open_chrome"
    assert await collect(r.respond("prenota un volo")) == ["Questo non lo so ancora fare."]
    assert r.last_route == {"route": "unknown"}
    r = make_responder(matcher, fail=True)
    assert await collect(r.respond("apri chrome")) == ["Non ci sono riuscito."]
    assert r.last_route["error"] is True


def test_interaction_metrics(tmp_path) -> None:
    times = iter([0.0, 0.5, 2.0, 2.4, 2.7, 2.72, 2.85])
    inter = Interaction(clock=lambda: next(times))
    for stage in (
        "wake",
        "speech_start",
        "speech_end",
        "endpoint",
        "stt_done",
        "first_text",
        "first_audio",
    ):
        inter.mark(stage)
    inter.mark("wake")  # ignored: first occurrence wins
    s = inter.summary()
    assert s["m1_ms"] == 850.0 and s["m2_ms"] == 2850.0
    assert s["vad_wait_ms"] == 400.0 and s["stt_ms"] == 300.0
    row = MetricsWriter(tmp_path / "m.jsonl").write(inter)
    assert json.loads((tmp_path / "m.jsonl").read_text(encoding="utf-8"))["m1_ms"] == row["m1_ms"]


def test_percentile_and_summary() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10, 20], 90) == pytest.approx(19)
    rows = [{"m1_ms": v} for v in (100, 200, 300, None)]
    assert summarize(rows, ("m1_ms",))["m1_ms"] == {"n": 3, "median": 200, "p90": 280}
    with pytest.raises(ValueError):
        percentile([], 50)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sottotitoli creati dalla comunità Amara.org", True),
        ("Grazie per la visione!", True),
        ("...", True),
        ("Apri Chrome.", False),
        ("Grazie", False),
    ],
)
def test_hallucination_filter(text: str, expected: bool) -> None:
    assert is_hallucination(text) is expected


def test_prompt_echo_is_hallucination() -> None:
    prompt = "Claudio, apri Chrome, apri la cartella Download, Documenti, Esplora file."
    assert is_hallucination("apri Chrome, apri la cartella Download, Documenti", prompt)
    assert not is_hallucination("apri Chrome", prompt)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Buon appetito!", True), ("Ciao a tutti.", True), ("Ciao a tutti, apri Chrome", False)],
)
def test_short_hallucinations_only_as_whole_text(text: str, expected: bool) -> None:
    assert is_hallucination(text) is expected


class FakeWhisper:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        seg = type(
            "Seg", (), {"text": " Apri Chrome.", "no_speech_prob": 0.01, "avg_logprob": -0.2}
        )()
        return iter([seg]), None


def make_stt():
    from claudio.config import SttSettings
    from claudio.stt.transcriber import WhisperTranscriber

    stt = WhisperTranscriber.__new__(WhisperTranscriber)
    stt.settings = SttSettings()
    stt._model = FakeWhisper()
    return stt


def test_silence_never_reaches_whisper() -> None:
    import numpy as np

    stt = make_stt()
    assert stt.transcribe(np.zeros(16000, dtype=np.float32)) == ""
    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert stt._model.calls == 0
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    assert stt.transcribe(speech) == "Apri Chrome."
    assert stt._model.calls == 1
