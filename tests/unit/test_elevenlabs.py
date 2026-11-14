"""ElevenLabsSynth: streaming, fallback hand-over, cancel, masking. No network: MockTransport."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import numpy as np
import pytest

from claudio.tts.elevenlabs import SAMPLE_RATE, ElevenLabsSynth

KEY = "sk_test_SECRET123"


class FakeFallback:
    def __init__(self) -> None:
        self.sample_rate = 22050
        self.calls = 0
        self.cancels = 0
        self.begins = 0

    def synthesize(self, text: str) -> np.ndarray:
        self.calls += 1
        return np.full(10, 7, np.int16)

    def cancel(self) -> None:
        self.cancels += 1

    def begin_reply(self) -> None:
        self.begins += 1


class StreamingFallback(FakeFallback):
    def stream(self, text: str) -> Iterator[np.ndarray]:
        self.calls += 1
        self.sample_rate = 22050
        yield np.full(4, 1, np.int16)
        self.sample_rate = 24000
        yield np.full(4, 2, np.int16)


class Server:
    """Records the requests and answers with `respond(request)`."""

    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.requests: list[httpx2.Request] = []
        self.respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.respond(request)

    def body(self, index: int) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def audio(*chunks: bytes, headers: dict[str, str] | None = None) -> Server:
    def gen() -> Iterator[bytes]:
        yield from chunks

    return Server(lambda request: httpx2.Response(200, content=gen(), headers=headers))


def samples(*values: int) -> bytes:
    return np.array(values, dtype="<i2").tobytes()


def make(server: Server, fallback: Any = None, **kwargs: Any) -> ElevenLabsSynth:
    kwargs.setdefault("lufs", 0.0)
    return ElevenLabsSynth(
        KEY,
        "VOICE",
        "eleven_v4_turbo",
        fallback if fallback is not None else FakeFallback(),
        transport=httpx2.MockTransport(server),
        **kwargs,
    )


def play(synth: ElevenLabsSynth, text: str = "ciao") -> np.ndarray:
    chunks = list(synth.stream(text))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.int16)


def test_streams_the_reply_as_int16_at_24khz() -> None:
    server = audio(samples(1, 2), samples(3), samples(4, 5, 6))
    synth = make(server)
    out = play(synth)
    assert out.dtype == np.int16
    assert out.tolist() == [1, 2, 3, 4, 5, 6]
    assert synth.sample_rate == SAMPLE_RATE == 24000
    request = server.requests[0]
    assert request.url.path == "/v1/text-to-speech/VOICE/stream"
    assert request.url.params["output_format"] == "pcm_24000"
    assert request.headers["xi-api-key"] == KEY
    body = server.body(0)
    assert body["model_id"] == "eleven_v4_turbo"
    assert body["language_code"] == "it"
    assert body["text"] == "ciao"
    assert "voice_settings" not in body and "previous_text" not in body


def test_an_odd_byte_waits_for_the_next_chunk() -> None:
    synth = make(audio(b"\x01", b"\x00\x02", b"\x00\x03"))
    assert play(synth).tolist() == [1, 2]


def test_voice_settings_are_sent_only_when_set() -> None:
    server = audio(samples(1))
    play(make(server, stability=0.4))
    assert server.body(0)["voice_settings"] == {"stability": 0.4}
    server = audio(samples(1))
    play(make(server, similarity=0.8))
    assert server.body(0)["voice_settings"] == {"similarity_boost": 0.8}


def test_previous_text_is_the_piece_before_in_this_reply() -> None:
    server = audio(samples(1))
    synth = make(server)
    play(synth, "prima")
    play(synth, "seconda")
    assert server.body(1)["previous_text"] == "prima"
    synth.begin_reply()
    play(synth, "terza")
    assert "previous_text" not in server.body(2)


def failing(kind: str) -> Server:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if kind == "connect":
            raise httpx2.ConnectError("no route")
        if kind == "timeout":
            raise httpx2.ReadTimeout("slow")
        if kind == "empty":
            return httpx2.Response(200, content=b"")
        return httpx2.Response(int(kind), text="nope")

    return Server(respond)


@pytest.mark.parametrize("kind", ["connect", "timeout", "401", "429", "empty"])
def test_an_error_before_audio_speaks_the_fallback(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    fallback = FakeFallback()
    synth = make(failing(kind), fallback)
    with caplog.at_level(logging.WARNING):
        out = play(synth)
    assert out.tolist() == [7] * 10
    assert synth.sample_rate == 22050
    assert fallback.calls == 1
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_after_a_failure_the_rest_of_the_reply_is_the_fallbacks() -> None:
    server = failing("401")
    fallback = FakeFallback()
    synth = make(server, fallback)
    play(synth, "uno")
    play(synth, "due")
    assert len(server.requests) == 1
    assert fallback.calls == 2
    synth.begin_reply()
    play(synth, "tre")
    assert len(server.requests) == 2


def test_an_error_mid_sentence_stops_without_repeating() -> None:
    def gen() -> Iterator[bytes]:
        yield samples(1, 2)
        raise httpx2.ReadError("cut")

    fallback = FakeFallback()
    server = Server(lambda request: httpx2.Response(200, content=gen()))
    out = play(make(server, fallback))
    assert out.tolist() == [1, 2]
    assert fallback.calls == 0


def test_cancel_stops_the_sentence_and_the_fallback_stays_quiet() -> None:
    fallback = FakeFallback()
    synth = make(audio(samples(1), samples(2), samples(3)), fallback)
    got = []
    for chunk in synth.stream("a"):
        got.append(chunk)
        synth.cancel()
    assert len(got) == 1
    assert fallback.calls == 0

    holder: list[ElevenLabsSynth] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        holder[0].cancel()  # "basta" while waiting for the first byte
        return httpx2.Response(200, content=iter([samples(1)]))

    synth = make(Server(respond), fallback)
    holder.append(synth)
    assert play(synth).size == 0
    assert fallback.calls == 0

    synth2 = make(audio(samples(5)), fallback)
    synth2.cancel()
    assert play(synth2).tolist() == [5]  # a stream after cancel() starts normally


def test_cancel_and_begin_reply_reach_the_fallback() -> None:
    fallback = FakeFallback()
    synth = make(audio(samples(1)), fallback)
    synth.cancel()
    synth.begin_reply()
    assert (fallback.cancels, fallback.begins) == (1, 1)


def test_a_streaming_fallback_sets_the_rate_chunk_by_chunk() -> None:
    synth = make(failing("401"), StreamingFallback())
    rates = []
    for _ in synth.stream("x"):
        rates.append(synth.sample_rate)
    assert rates == [22050, 24000]


def test_the_cost_of_each_request_is_logged_and_kept(caplog: pytest.LogCaptureFixture) -> None:
    synth = make(audio(samples(1), headers={"character-cost": "16"}))
    with caplog.at_level(logging.INFO):
        play(synth)
    assert synth.last_cost == 16
    assert synth.last_first_ms is not None
    assert "16" in caplog.text
    synth = make(audio(samples(1)))
    play(synth)
    assert synth.last_cost is None


def test_the_key_never_reaches_the_log(caplog: pytest.LogCaptureFixture) -> None:
    def leak(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, text=f"bad key {KEY}")

    def connect(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"cannot reach with {KEY}")

    with caplog.at_level(logging.DEBUG):
        play(make(Server(leak)))
        play(make(Server(connect)))
    assert KEY not in caplog.text
    assert "***" in caplog.text


def test_the_leveler_keeps_length_and_type() -> None:
    wave = (np.sin(np.arange(24000) / 10) * 8000).astype("<i2")
    synth = make(audio(wave.tobytes()), lufs=-24.0)
    out = play(synth)
    assert out.shape == wave.shape
    assert out.dtype == np.int16


def test_probe() -> None:
    ok = make(Server(lambda r: httpx2.Response(200, json={})))
    assert ok.probe() is None and ok.problem is None
    denied = make(Server(lambda r: httpx2.Response(401, text="no")))
    assert denied.probe() == "HTTP 401"
    assert denied.problem == "HTTP 401"

    def connect(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"cannot reach with {KEY}")

    down = make(Server(connect))
    result = down.probe()
    assert result is not None and "ConnectError" in result and KEY not in result


def test_an_abandoned_stream_failing_late_does_not_move_the_new_reply_to_the_fallback() -> None:
    gate = threading.Event()
    calls = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        if len(calls) == 1:
            gate.wait(5)
            raise httpx2.ReadTimeout("slow")
        return httpx2.Response(200, content=iter([samples(1)]))

    fallback = FakeFallback()
    synth = make(Server(respond), fallback)
    old: list[np.ndarray] = []
    worker = threading.Thread(target=lambda: old.extend(synth.stream("vecchia")))
    worker.start()
    while not calls:
        time.sleep(0.01)
    synth.cancel()
    synth.begin_reply()
    assert play(synth, "nuova uno").tolist() == [1]
    gate.set()
    worker.join(5)
    assert old == [] and fallback.calls == 0
    assert play(synth, "nuova due").tolist() == [1]
    assert len(calls) == 3


@pytest.mark.parametrize("status", [200, 401])
def test_a_cancel_that_closes_the_response_midway_ends_quietly(status: int) -> None:
    fallback = FakeFallback()
    server = Server(lambda r: httpx2.Response(status, content=iter([samples(1)])))
    synth = make(server, fallback)
    real = synth._is_cancelled
    first = []

    def cancelling(generation: int) -> bool:
        if not first:  # "basta" lands right after the first check, closing the response
            first.append(1)
            synth.cancel()
            return False
        return real(generation)

    synth._is_cancelled = cancelling  # type: ignore[method-assign]
    assert play(synth).size == 0
    assert fallback.calls == 0


# --- configuration and the voice chain (task 2) ---------------------------------------------


def _settings(
    tmp_path, *, key: bool = True, engine: str = "elevenlabs", key_value: str = "sk_test_SECRET123"
):
    from pydantic import SecretStr

    from claudio.config import Settings

    base = Settings(models_dir=tmp_path, tts={"engine": engine})
    return base.model_copy(update={"elevenlabs_api_key": SecretStr(key_value) if key else None})


def _fake_probe(monkeypatch, problem: str | None = None) -> None:
    """The startup probe without the network: sets `.problem` like the real one."""

    def probe(self) -> str | None:
        self.problem = problem
        return problem

    monkeypatch.setattr(ElevenLabsSynth, "probe", probe)


def test_build_synth_chains_elevenlabs_jack_and_the_base(monkeypatch, tmp_path) -> None:
    from claudio import __main__ as entry
    from claudio.tts import jack as jack_module

    base = FakeFallback()
    monkeypatch.setattr(entry, "build_base_synth", lambda settings: (base, "Kokoro im_nicola"))
    monkeypatch.setattr(jack_module.JackSynth, "start", lambda self: None)
    _fake_probe(monkeypatch)
    synth, name = entry.build_synth(_settings(tmp_path))
    assert isinstance(synth, ElevenLabsSynth)
    assert isinstance(synth.fallback, jack_module.JackSynth)
    assert synth.fallback.fallback is base
    assert name == "ElevenLabs (fallback Jack (fallback Kokoro im_nicola))"


def test_build_synth_without_jack_chains_elevenlabs_to_the_base(monkeypatch, tmp_path) -> None:
    from claudio import __main__ as entry
    from claudio.tts import jack as jack_module

    def refuse(self) -> None:
        raise jack_module.WorkerError("no worker")

    base = FakeFallback()
    monkeypatch.setattr(entry, "build_base_synth", lambda settings: (base, "Kokoro im_nicola"))
    monkeypatch.setattr(jack_module.JackSynth, "start", refuse)
    _fake_probe(monkeypatch)
    synth, name = entry.build_synth(_settings(tmp_path))
    assert isinstance(synth, ElevenLabsSynth) and synth.fallback is base
    assert name == "ElevenLabs (fallback Kokoro im_nicola)"


def test_build_synth_without_the_key_keeps_jack(monkeypatch, tmp_path, caplog) -> None:
    from claudio import __main__ as entry
    from claudio.tts import jack as jack_module

    base = FakeFallback()
    monkeypatch.setattr(entry, "build_base_synth", lambda settings: (base, "Kokoro im_nicola"))
    monkeypatch.setattr(jack_module.JackSynth, "start", lambda self: None)
    with caplog.at_level(logging.WARNING):
        synth, name = entry.build_synth(_settings(tmp_path, key=False))
    assert isinstance(synth, jack_module.JackSynth) and synth.fallback is base
    assert name == "Jack (fallback Kokoro im_nicola)"
    assert any("ELEVENLABS_API_KEY" in r.getMessage() for r in caplog.records)
    assert "sk_" not in caplog.text


MALFORMED_KEYS = ["", "sk_abc\r", "sk_abc ", " sk_abc", "sk_abc”"]


@pytest.mark.parametrize("bad", MALFORMED_KEYS)
def test_a_malformed_key_keeps_the_local_voice(tmp_path, caplog, bad: str) -> None:
    from claudio.__main__ import make_elevenlabs

    with caplog.at_level(logging.WARNING):
        assert make_elevenlabs(_settings(tmp_path, key_value=bad), FakeFallback()) is None
    assert any("ELEVENLABS_API_KEY" in r.getMessage() for r in caplog.records)
    assert "sk_abc" not in caplog.text


def test_build_synth_with_a_malformed_key_keeps_jack(monkeypatch, tmp_path) -> None:
    from claudio import __main__ as entry
    from claudio.tts import jack as jack_module

    monkeypatch.setattr(entry, "build_base_synth", lambda settings: (FakeFallback(), "Kokoro"))
    monkeypatch.setattr(jack_module.JackSynth, "start", lambda self: None)
    synth, _ = entry.build_synth(_settings(tmp_path, key_value="sk_abc”"))
    assert isinstance(synth, jack_module.JackSynth)


def test_make_elevenlabs_forwards_the_settings(monkeypatch, tmp_path) -> None:
    from claudio import __main__ as entry
    from claudio.config import Settings
    from claudio.tts import elevenlabs as module

    calls: list[tuple[tuple, dict]] = []

    class Recorder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

    monkeypatch.setattr(module, "ElevenLabsSynth", Recorder)
    base = _settings(tmp_path)
    tts = base.tts.model_copy(
        update={
            "eleven_voice_id": "VOICE",
            "eleven_model": "MODEL",
            "eleven_stability": 0.3,
            "eleven_similarity": 0.6,
            "eleven_timeout_s": 2.5,
            "jack_lufs": -20,
        }
    )
    settings: Settings = base.model_copy(update={"tts": tts})
    fallback = FakeFallback()
    entry.make_elevenlabs(settings, fallback)
    assert calls == [
        (
            ("sk_test_SECRET123", "VOICE", "MODEL", fallback),
            {"stability": 0.3, "similarity": 0.6, "timeout_s": 2.5, "lufs": -20},
        )
    ]


def test_build_synth_probes_the_online_voice_and_keeps_it_on_failure(
    monkeypatch, tmp_path, caplog
) -> None:
    from claudio import __main__ as entry
    from claudio.tts import jack as jack_module

    monkeypatch.setattr(
        entry, "build_base_synth", lambda settings: (FakeFallback(), "Kokoro im_nicola")
    )
    monkeypatch.setattr(jack_module.JackSynth, "start", lambda self: None)
    _fake_probe(monkeypatch, "HTTP 401")
    with caplog.at_level(logging.WARNING):
        synth, _ = entry.build_synth(_settings(tmp_path))
    assert isinstance(synth, ElevenLabsSynth) and synth.problem == "HTTP 401"
    assert any(
        r.levelno == logging.WARNING and "HTTP 401" in r.getMessage() for r in caplog.records
    )
    assert "sk_test_SECRET123" not in caplog.text


def test_build_speaker_groups_by_the_voice_on_top(monkeypatch, tmp_path) -> None:
    import sys
    import types

    from claudio import __main__ as entry
    from claudio.tts import jack as jack_module

    class StubPlayer:
        def __init__(self, sample_rate: int, *args: Any) -> None:
            self.sample_rate = sample_rate

        def start(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules, "claudio.audio.player", types.SimpleNamespace(AudioPlayer=StubPlayer)
    )
    monkeypatch.setattr("claudio.tts.speaker.StreamingSpeaker.warm_up", lambda self: None)
    monkeypatch.setattr(jack_module.JackSynth, "start", lambda self: None)
    monkeypatch.setattr(entry, "build_base_synth", lambda settings: (FakeFallback(), "base"))
    _fake_probe(monkeypatch)
    cases = (
        ("elevenlabs", True, 111),
        ("elevenlabs", False, 170),  # no key: Jack speaks, with his own grouping
        ("jack", True, 170),
        ("kokoro", True, 0),
    )
    for engine, key, expected in cases:
        base = _settings(tmp_path, engine=engine, key=key)
        settings = base.model_copy(
            update={"tts": base.tts.model_copy(update={"eleven_group_chars": 111})}
        )
        speaker = entry.build_speaker(settings, start_player=False)
        assert speaker.group_chars == expected, (engine, key)
