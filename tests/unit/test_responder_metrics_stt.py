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
    return LocalResponder(matcher, ActionRunner(ctx))


async def collect(stream) -> list[str]:
    return [c async for c in stream]


async def test_responder_known_unknown_and_error(matcher) -> None:
    r = make_responder(matcher)
    assert await collect(r.respond("Apri Chrome.")) == ["Apro Chrome."]
    assert r.last_route["command"] == "open_chrome"
    assert await collect(r.respond("prenota un volo")) == ["Questo non lo so ancora fare."]
    assert r.last_route == {"route": "unknown", "language": "it"}
    r = make_responder(matcher, fail=True)
    assert await collect(r.respond("apri chrome")) == ["Non ci sono riuscito."]
    assert r.last_route["error"] is True
    assert await collect(r.respond("open chrome", "en")) == ["I couldn't do it."]


async def test_responder_answers_in_the_language_of_the_phrase(matcher) -> None:
    r = make_responder(matcher)
    assert await collect(r.respond("Open Chrome, please.", "en")) == ["Opening Chrome."]
    assert r.language == "en" and r.last_route["language"] == "en"
    # Whisper thought it was Italian, but the phrase is English: the reply follows the phrase
    assert await collect(r.respond("open the calculator", "it")) == ["Opening the calculator."]
    assert r.language == "en"
    assert await collect(r.respond("book a flight", "en")) == ["I can't do that yet."]
    assert await collect(r.respond("apri la calcolatrice", "en")) == ["Apro la calcolatrice."]
    assert r.language == "it"


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
    def __init__(self, text: str = " Apri Chrome.") -> None:
        self.calls, self.text = 0, text

    def transcribe(self, audio, **kwargs):
        from faster_whisper.transcribe import get_compression_ratio

        self.calls += 1
        self.kwargs = kwargs
        fields = {"text": self.text, "no_speech_prob": 0.01, "avg_logprob": -0.2}
        seg = type("Seg", (), {**fields, "compression_ratio": get_compression_ratio(self.text)})()
        return iter([seg]), None


def make_stt():
    from claudio.config import SttSettings
    from claudio.stt.transcriber import WhisperTranscriber

    stt = WhisperTranscriber.__new__(WhisperTranscriber)
    stt.settings = SttSettings()
    stt.language, stt.language_confidence, stt._encoded, stt._reused = "it", 1.0, None, 0
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


def test_repetition_loops_are_dropped() -> None:
    import numpy as np

    stt = make_stt()
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    stt._model.text = " " + ", ".join(["Ciao"] * 40)
    assert stt.transcribe(speech) == ""
    assert stt._model.kwargs["max_new_tokens"] <= 100  # a loop stops early anyway
    stt._model.text = " No, no, no, apri Chrome."  # a few repeats are fine
    assert stt.transcribe(speech) == "No, no, no, apri Chrome."


def test_whisper_starts_offline_when_downloaded(monkeypatch, tmp_path) -> None:
    import faster_whisper

    from claudio.config import SttSettings
    from claudio.stt.transcriber import WhisperTranscriber

    calls: list[dict] = []

    class Model:
        def __init__(self, name, **kwargs) -> None:
            calls.append({"name": name, **kwargs})
            if kwargs.get("local_files_only") and name == "small":
                raise FileNotFoundError("not downloaded")

        def transcribe(self, audio, **kwargs):
            return iter([]), None

    monkeypatch.setattr(faster_whisper, "WhisperModel", Model)
    stt = WhisperTranscriber(SttSettings(device="cpu", fallback_model="base"), tmp_path)
    assert stt.model_name == "base"
    assert [c.get("local_files_only") for c in calls] == [True]  # no request to the Hub
    calls.clear()
    WhisperTranscriber(SttSettings(device="cpu", fallback_model="small"), tmp_path)
    assert [c.get("local_files_only") for c in calls] == [True, None]  # not cached: download


async def test_paused_responder_answers_only_system_commands(matcher) -> None:
    r = make_responder(matcher)
    assert await collect(r.respond("Claudio, vai in pausa")) == [
        "Vado in pausa. Per riattivarmi chiamami e di' svegliati."
    ]
    assert await collect(r.respond("apri chrome")) == []
    assert r.last_route == {"route": "paused"}
    assert await collect(r.respond("prenota un volo")) == []
    assert await collect(r.respond("svegliati")) == ["Eccomi."]
    assert await collect(r.respond("apri chrome")) == ["Apro Chrome."]


@pytest.mark.parametrize(
    ("probs", "last", "expected"),
    [
        ({"it": 0.7, "en": 0.2, "es": 0.1}, "it", "it"),
        ({"en": 0.8, "it": 0.1}, "it", "en"),
        ({"es": 0.9, "it": 0.08, "en": 0.02}, "en", "it"),  # Spanish is not a choice
        ({"it": 0.45, "en": 0.4}, "en", "en"),  # 53%: unsure, keep the last language
        ({"fr": 1.0}, "en", "en"),
    ],
)
def test_language_is_picked_among_the_configured_ones(probs, last, expected) -> None:
    from claudio.stt.transcriber import pick_language

    assert pick_language(probs, ["it", "en"], last, 0.6)[0] == expected


class FakeFeatures:
    nb_max_frames = 3000

    def __call__(self, audio):
        import numpy as np

        frames = len(audio) // 160 + 1
        return np.tile(np.arange(frames, dtype=np.float32), (2, 1))


class FakeMultilingualWhisper(FakeWhisper):
    """Encodes and detects the language like faster-whisper: transcribe() encodes the first
    30 s window through self.encode."""

    def __init__(self, probs: list[tuple[str, float]], text: str = " Open Chrome.") -> None:
        super().__init__(text)
        self.feature_extractor = FakeFeatures()
        self.encoded = 0
        self.model = type("Ct2", (), {"detect_language": lambda _self, enc: [probs]})()

    def encode(self, features):
        self.encoded += 1
        return ("encoded", features.shape)

    def transcribe(self, audio, **kwargs):
        from faster_whisper.audio import pad_or_trim

        features = self.feature_extractor(audio)
        self.encode(pad_or_trim(features[:, : features.shape[-1] - 1]))
        return super().transcribe(audio, **kwargs)


def test_language_detection_reuses_the_encoder_pass() -> None:
    import numpy as np

    from claudio.config import SttSettings

    stt = make_stt()
    stt.settings = SttSettings(
        languages=["it", "en"], prompts={"it": "Apri Chrome.", "en": "Open Chrome."}
    )
    stt._model = FakeMultilingualWhisper([("<|en|>", 0.8), ("<|it|>", 0.15), ("<|de|>", 0.05)])
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    assert stt.transcribe(speech) == "Open Chrome."
    assert stt.language == "en" and stt._model.kwargs["language"] == "en"
    assert stt._model.kwargs["initial_prompt"] == "Open Chrome."
    assert stt._model.encoded == 1 and stt._reused == 1  # one encoder pass, not two
    assert "encode" not in vars(stt._model)  # Whisper's own encoder is back
    stt._model.model = type("Ct2", (), {"detect_language": lambda s, e: [[("<|it|>", 0.5)]]})()
    stt._model.text = " Apri Chrome."
    assert stt.transcribe(speech) == "Apri Chrome." and stt.language == "it"


def test_one_language_skips_the_detection() -> None:
    import numpy as np

    stt = make_stt()
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    assert stt.transcribe(speech) == "Apri Chrome."
    assert stt._model.kwargs["language"] == "it" and stt._reused == 0


def test_stt_languages_and_prompts() -> None:
    from claudio.config import SttSettings

    s = SttSettings(languages=["it", "en"], initial_prompt="Apri Chrome.", prompts={"en": "Hi."})
    assert s.prompt("it") == "Apri Chrome." and s.prompt("en") == "Hi."
    assert SttSettings(languages=["it", "en"], language="en").languages == ["en"]
    assert is_hallucination("Thanks for watching!") and is_hallucination("you")
