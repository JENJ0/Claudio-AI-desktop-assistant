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
