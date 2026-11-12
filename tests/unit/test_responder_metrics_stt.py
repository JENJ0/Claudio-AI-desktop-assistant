import json
import re

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


async def test_claude_answers_in_a_language_with_no_fixed_reply(matcher) -> None:
    """No "non lo so fare" in French, so the fixed reply would be Italian. Claude can write it."""

    async def claude(text: str, language: str):
        yield language

    r = make_responder(matcher)
    r.fallback = claude
    assert await collect(r.respond("quelle heure est-il", "fr")) == ["fr"]
    assert r.language == "fr"


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


async def test_reply_text_and_voice_agree(matcher, tmp_path) -> None:
    """No Italian text read by the English voice, nor English text by the Italian one."""
    from claudio.router.matcher import CommandMatcher, load_commands

    f = tmp_path / "c.toml"
    f.write_text(
        '[[command]]\nid="s"\nphrases=["apri spotify"]\nphrases_en=["open spotify"]\n'
        'action="open_app"\nparams={ app="notepad" }\nreply="Apro Spotify."\n',
        encoding="utf-8",
    )
    ctx = ActionContext(run=lambda a: None, open_path=lambda p: None, press_key=lambda k: None)
    r = LocalResponder(CommandMatcher(load_commands(f)), ActionRunner(ctx))
    assert await collect(r.respond("open spotify", "en")) == ["Apro Spotify."]
    assert r.language == "it"  # no reply_en: the Italian reply, with the Italian voice
    r = LocalResponder(matcher, ActionRunner(ctx), unknown_reply={"it": "Non so."})
    assert await collect(r.respond("book a flight", "en")) == ["Non so."] and r.language == "it"
    r = LocalResponder(matcher, ActionRunner(ctx), voiced={"it"})  # no English voice
    assert await collect(r.respond("open chrome", "en")) == ["Apro Chrome."]
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [  # the webcam's unclear requests (29/10)
        ("Grazie a tutti. Grazie a tutti. Grazie a tutti.", True),
        ("Grazie a tutti, grazie a tutti, grazie a tutti.", True),
        ("grazie a tutti grazie a tutti", True),
        ("Grazie. Grazie a tutti.", True),
        ("Grazie a tutti per aver guardato il video.", True),
        ("Grazie, grazie.", False),  # JenJo thanking
        ("Grazie mille.", False),
        ("Basta, basta!", False),
        ("Grazie a tutti, apri Chrome.", False),
        ("Bye bye.", True),  # itself a repetition
        ("Bye bye, bye bye.", True),
        ("Grazie per aver guardato il video.", True),
        ("Scrivi nel blocco note: grazie per aver guardato il video", False),  # dictation
    ],
)
def test_invented_endings_also_repeated_or_after_a_thanks(text: str, expected: bool) -> None:
    assert is_hallucination(text) is expected


def test_every_end_of_the_shipped_prompt_is_heard_again_without_it() -> None:
    """On unclear audio Whisper repeats the end of its prompt ("Basta, grazie." for "Claudio,
    chiudi il Blocco note", 29/10), and the prompt ends in a command ("poi alza il volume"):
    every end of 2+ words is caught as an echo, so it is transcribed again, never run."""
    from claudio.config import DEFAULT_CONFIG, load_settings
    from claudio.stt.transcriber import ECHO_END_WORDS, is_prompt_echo

    prompt = load_settings(DEFAULT_CONFIG).stt.initial_prompt
    assert prompt
    words = re.split(r"(?<=[.?!])\s+", prompt.strip())[-1].split()
    assert len(words) > ECHO_END_WORDS
    for start in range(len(words) - ECHO_END_WORDS + 1):
        end = " ".join(words[start:])
        assert is_prompt_echo(end, prompt) and is_hallucination(end, prompt), end


def test_a_piece_of_the_prompt_is_transcribed_again_without_it() -> None:
    """ "Poi alza il volume." on unclear audio is the prompt repeated; without the prompt
    Whisper writes what was said. The same request really said stays."""
    import numpy as np

    from claudio.config import SttSettings

    class EchoingWhisper(FakeWhisper):
        def __init__(self, without: str) -> None:
            super().__init__()
            self.without, self.prompts = without, []

        def transcribe(self, audio, **kwargs):
            self.prompts.append(kwargs["initial_prompt"])
            self.text = " Poi alza il volume." if kwargs["initial_prompt"] else self.without
            return super().transcribe(audio, **kwargs)

    stt = make_stt()
    stt.settings = SttSettings(initial_prompt="Ehi Claudio. Apri Chrome, poi alza il volume.")
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    stt._model = EchoingWhisper(" Chiudi il blocco note.")
    assert stt.transcribe(speech) == "Chiudi il blocco note."
    assert stt._model.prompts == ["Ehi Claudio. Apri Chrome, poi alza il volume.", None]
    stt._model = EchoingWhisper(" Poi alza il volume.")  # really said
    assert stt.transcribe(speech) == "Poi alza il volume."
    stt._model = EchoingWhisper(" Grazie a tutti.")
    assert stt.transcribe(speech) == ""  # the second pass is filtered too


def test_last_prompt_is_the_request_prompt_not_the_quick_one() -> None:
    import numpy as np

    from claudio.config import SttSettings

    stt = make_stt()
    stt.vocabulary = lambda language: "Cartelle: Tesi."
    stt.settings = SttSettings(initial_prompt="Apri Chrome.")
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    stt._model.text = " Chiudi il blocco note."  # not a piece of the prompt: one pass
    stt.transcribe(speech)
    assert stt.last_prompt == "Cartelle: Tesi. Apri Chrome."
    stt.transcribe(speech, quick=True)  # the wake word's second look must not replace it
    assert stt.last_prompt == "Cartelle: Tesi. Apri Chrome."
    stt.transcribe(np.zeros(16000, dtype=np.float32))  # silence: no pass at all
    assert stt.last_prompt == ""
    stt._model.text = " Apri Chrome."  # Whisper echoed the prompt: the real pass had none
    stt.transcribe(speech)
    assert stt.last_prompt == ""


def test_an_ordinary_request_takes_one_pass() -> None:
    import numpy as np

    from claudio.config import SttSettings

    stt = make_stt()
    stt.settings = SttSettings(initial_prompt="Apri Chrome, poi alza il volume.")
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    stt._model.text = " Claudio, alza il volume."
    assert stt.transcribe(speech) == "Claudio, alza il volume." and stt._model.calls == 1


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
    stt.language, stt._encoded, stt._reused, stt.last_prompt = "it", None, 0, ""
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


def test_quick_mode_is_a_short_look_with_the_shipped_prompt() -> None:
    """The wake word's second chance: the shipped prompt (30/10: a lone "Claudio" came out
    "Grazie." without it) and few tokens."""
    import numpy as np

    from claudio.config import SttSettings

    stt = make_stt()
    stt.settings = SttSettings(initial_prompt="Hey Claudio.")
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    assert stt.transcribe(speech, quick=True) == "Apri Chrome."
    kwargs = stt._model.kwargs
    assert kwargs["initial_prompt"] == "Hey Claudio." and kwargs["max_new_tokens"] == 24


def test_whisper_always_gets_italian() -> None:
    import numpy as np

    stt = make_stt()
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    assert stt.transcribe(speech) == "Apri Chrome."
    assert stt._model.kwargs["language"] == "it" and stt._reused == 0


def test_english_hallucinations_are_dropped() -> None:
    assert is_hallucination("Thanks for watching!") and is_hallucination("you")


def test_the_names_on_the_pc_come_before_the_sentences() -> None:
    """26/10: "Tesi" heard as "desi": the folder names go in the prompt, never in the quick
    look of the wake word (noise must not become a folder name), which has the sentences."""
    import numpy as np

    from claudio.config import SttSettings

    stt = make_stt()
    stt.settings = SttSettings(initial_prompt="Ehi Claudio, che ore sono?")
    stt.vocabulary = lambda language: "Cartelle: Tesi, PSS." if language == "it" else ""
    speech = (np.sin(np.arange(16000) / 5) * 0.1).astype(np.float32)
    stt._model.text = " Apri la cartella Tesi."
    assert stt.transcribe(speech) == "Apri la cartella Tesi."
    prompt = stt._model.kwargs["initial_prompt"]
    assert prompt == "Cartelle: Tesi, PSS. Ehi Claudio, che ore sono?"
    stt.transcribe(speech, quick=True)
    assert stt._model.kwargs["initial_prompt"] == "Ehi Claudio, che ore sono?"
    stt.vocabulary = None
    assert stt.prompt("it") == "Ehi Claudio, che ore sono?"


def test_from_log_groups_latency_by_path(tmp_path, capsys) -> None:
    import sys

    from claudio.config import PROJECT_ROOT

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import benchmark_latency

    def row(route: str, audible: float | None, m1: float) -> dict:
        return {"outcome": "ok", "route": route, "audible_ms": audible, "m1_ms": m1}

    rows = [
        row("local", 300.0, 250.0),
        row("confirmed", 500.0, 450.0),
        row("claude", 1200.0, 1100.0),
        row("claude", None, 900.0),  # written before audible_ms existed
    ]
    log = tmp_path / "metrics.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    benchmark_latency.from_log(log)
    out = capsys.readouterr().out
    audible = {line.split()[0]: line for line in out.splitlines() if " audible_ms " in line}
    assert "300.0" in audible["locale"]
    assert "500.0" in audible["conferma"]
    assert "n=1" in audible["Claude"] and "1200.0" in audible["Claude"]  # the old row is skipped
    assert any(
        line.split()[:2] == ["Claude", "m1_ms"] and "n=2" in line for line in out.splitlines()
    )
