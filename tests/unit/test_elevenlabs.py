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

from claudio.cost.ledger import VoiceLedger
from claudio.cost.voice import OUT_TEXT, WARN_TEXT, VoiceCredits
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

    base = Settings(
        models_dir=tmp_path,
        data_dir=tmp_path / "data",
        tts={"engine": engine},
        cost={"db_file": tmp_path / "spend.db"},
    )
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
            {
                "stability": 0.3,
                "similarity": 0.6,
                "timeout_s": 2.5,
                "lufs": -20,
                "credits": calls[0][1]["credits"],
                "cache": calls[0][1]["cache"],
                "fixed": calls[0][1]["fixed"],
            },
        )
    ]
    assert isinstance(calls[0][1]["credits"], VoiceCredits)


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


# -- credits ------------------------------------------------------------------------------------

RESET = 2_000_000_000  # an arbitrary future instant; the tests move the clock around it


class Account:
    """A fake ElevenLabs: the subscription endpoint and the TTS endpoint, with a clock."""

    def __init__(
        self,
        used: int,
        *,
        cost: str | None = "24",
        subscription_status: int = 200,
        tts_status: int = 200,
    ) -> None:
        self.used = used
        self.reset = RESET
        self.cost = cost
        self.subscription_status = subscription_status
        self.tts_status = tts_status
        self.now = float(RESET - 1000)
        self.subscription_calls = 0
        self.tts_calls = 0
        self.notices: list[str] = []
        self.server = Server(self._respond)

    def _respond(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v1/user/subscription":
            self.subscription_calls += 1
            if self.subscription_status != 200:
                return httpx2.Response(self.subscription_status)
            return httpx2.Response(
                200,
                json={
                    "character_count": self.used,
                    "character_limit": 40000,
                    "next_character_count_reset_unix": self.reset,
                },
            )
        self.tts_calls += 1
        if self.tts_status != 200:
            return httpx2.Response(self.tts_status, text="denied")
        headers = {"character-cost": self.cost} if self.cost is not None else {}
        return httpx2.Response(200, content=samples(1, 2), headers=headers)

    def synth(self, *, probe: bool = True, fallback: Any = None) -> ElevenLabsSynth:
        credits = VoiceCredits(VoiceLedger(":memory:"), 0.8, clock=lambda: self.now)
        synth = make(self.server, fallback, credits=credits, notify=self.notices.append)
        if probe:
            synth.probe()
        return synth

    def rows(self, synth: ElevenLabsSynth) -> list[tuple[int, int | None]]:
        assert synth.credits is not None
        db = synth.credits._ledger._db
        return db.execute("SELECT characters, credits FROM voice ORDER BY id").fetchall()


def test_probe_reads_the_credits(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    account = Account(13661)
    synth = account.synth()
    assert synth.credits is not None and synth.credits.known
    assert any(synth.credits.summary() in record.getMessage() for record in caplog.records)


def test_a_broken_subscription_body_does_not_change_the_probe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = Server(lambda request: httpx2.Response(200, content=b"not json"))
    credits = VoiceCredits(VoiceLedger(":memory:"))
    synth = make(server, credits=credits)
    assert synth.probe() is None
    assert not credits.known
    assert "non letti" in caplog.text


def test_each_request_is_recorded_and_counted() -> None:
    account = Account(1000)
    synth = account.synth()
    play(synth, "ciao a tutti")
    account.cost = None
    play(synth, "senza costo")
    assert account.rows(synth) == [(12, 24), (11, None)]


def test_crossing_80_percent_notifies_once() -> None:
    account = Account(31990)
    synth = account.synth()
    play(synth, "uno")
    play(synth, "due")
    assert account.notices == [WARN_TEXT]


def test_out_of_credits_speaks_the_fallback_until_the_reset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    account = Account(39990)
    fallback = FakeFallback()
    synth = account.synth(fallback=fallback)
    play(synth, "una")  # the cost 24 takes the account to 100%
    assert len(account.notices) == 1
    assert account.notices[0].startswith(OUT_TEXT.split("{")[0])
    synth.begin_reply()
    calls = account.tts_calls
    assert play(synth, "due").tolist() == [7] * 10
    play(synth, "tre")
    assert account.tts_calls == calls  # no request at all
    assert fallback.calls == 2
    assert caplog.text.count("Crediti ElevenLabs finiti") == 1  # one line, not one per sentence
    account.now = RESET + 60
    account.used = 5
    synth.begin_reply()
    assert play(synth, "dopo il reset").tolist() == [1, 2]
    assert account.subscription_calls == 2
    assert account.tts_calls == calls + 1


def test_a_refused_key_below_100_percent_uses_the_fallback_quietly() -> None:
    account = Account(34800, tts_status=401)  # 87%
    synth = account.synth()
    assert play(synth, "uno").tolist() == [7] * 10
    assert account.notices == []
    account.tts_status = 200
    synth.begin_reply()
    assert play(synth, "due").tolist() == [1, 2]


def test_a_failed_startup_probe_is_retried_after_the_first_success() -> None:
    account = Account(1000, subscription_status=503)
    synth = account.synth()
    assert synth.credits is not None and not synth.credits.known
    account.subscription_status = 200
    play(synth, "uno")
    assert account.subscription_calls == 1  # never inside the stream's bookkeeping
    play(synth, "due")  # the probe runs before the next request
    assert account.subscription_calls == 2 and synth.credits.known
    play(synth, "tre")
    assert account.subscription_calls == 2  # once only


def test_a_402_means_out_of_credits() -> None:
    account = Account(1000, tts_status=402)
    fallback = FakeFallback()
    synth = account.synth(fallback=fallback)
    assert play(synth, "uno").tolist() == [7] * 10
    assert len(account.notices) == 1 and account.notices[0].startswith("Crediti della voce")
    assert synth.credits is not None and synth.credits.exhausted()
    synth.begin_reply()
    assert play(synth, "due").tolist() == [7] * 10
    assert account.tts_calls == 1  # the next reply makes no online request
    assert len(account.notices) == 1
    account.now = RESET + 60  # after the rollover the online voice is back
    account.reset = RESET + 30 * 86400
    account.used = 5
    account.tts_status = 200
    synth.begin_reply()
    assert play(synth, "tre").tolist() == [1, 2]


def test_a_402_without_credits_changes_nothing() -> None:
    server = Server(lambda request: httpx2.Response(402, text="x"))
    fallback = FakeFallback()
    synth = make(server, fallback)
    assert play(synth, "uno").tolist() == [7] * 10
    synth.begin_reply()
    play(synth, "due")
    assert len(server.requests) == 2  # retried: no state


def test_a_402_notice_without_notify_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    account = Account(1000, tts_status=402)
    synth = account.synth()
    synth.notify = None
    play(synth, "uno")
    assert "Avviso crediti (non detto a voce)" in caplog.text


def test_a_real_rollover_brings_the_online_voice_and_the_notices_back() -> None:
    account = Account(39990)
    synth = account.synth()
    play(synth, "una")  # 100%: the notice of the old period
    assert len(account.notices) == 1
    account.now = RESET + 60
    account.reset = RESET + 30 * 86400
    account.used = 31990  # the new period, already near 80%
    calls = account.subscription_calls
    synth.begin_reply()
    assert play(synth, "due").tolist() == [1, 2]  # online again
    assert account.subscription_calls == calls + 1
    assert synth.credits is not None and not synth.credits.stale()
    for text in ("tre", "quattro"):
        synth.begin_reply()
        play(synth, text)
    assert account.subscription_calls == calls + 1  # the re-probes stopped
    assert account.notices[1:] == [WARN_TEXT]  # a notice of the new period is possible again


def test_barge_in_close_bills_once_and_makes_no_call() -> None:
    account = Account(1000, subscription_status=503)  # the one-time retry probe is due
    synth = account.synth()
    account.subscription_status = 200
    stream = synth.stream("ciao")
    next(stream)
    synth.cancel()
    before = (account.subscription_calls, account.tts_calls)
    stream.close()
    assert (account.subscription_calls, account.tts_calls) == before
    assert account.rows(synth) == [(4, 24)]


def test_a_request_cut_in_the_middle_is_billed() -> None:
    def gen() -> Iterator[bytes]:
        yield samples(1, 2)
        raise httpx2.ReadError("cut")

    server = Server(
        lambda request: httpx2.Response(200, content=gen(), headers={"character-cost": "9"})
    )
    credits = VoiceCredits(VoiceLedger(":memory:"))
    synth = make(server, credits=credits)
    assert play(synth, "ciao").tolist() == [1, 2]
    rows = credits._ledger._db.execute("SELECT characters, credits FROM voice").fetchall()
    assert rows == [(4, 9)]


def test_a_cancelled_200_request_is_billed_once() -> None:
    account = Account(1000)
    synth = account.synth()
    stream = synth.stream("ciao")
    next(stream)
    synth.cancel()
    list(stream)
    assert account.rows(synth) == [(4, 24)]
    assert synth.credits is not None and "1024 su" in synth.credits.summary()


def test_a_cancelled_request_still_notifies() -> None:
    account = Account(31990)
    synth = account.synth()
    stream = synth.stream("ciao")
    next(stream)
    synth.cancel()
    list(stream)
    assert account.notices == [WARN_TEXT]


def test_a_request_without_a_200_is_not_billed() -> None:
    account = Account(1000, tts_status=500)
    synth = account.synth()
    play(synth, "ciao")
    assert account.rows(synth) == []


def test_a_completed_request_is_billed_exactly_once() -> None:
    account = Account(1000)
    synth = account.synth()
    play(synth, "ciao")
    assert account.rows(synth) == [(4, 24)]


class BrokenCredits:
    """Every call raises, like SQLite 'database is locked'."""

    known = True

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(f"{name}: database is locked {KEY}")

        return fail


def test_bookkeeping_errors_never_stop_the_voice(caplog: pytest.LogCaptureFixture) -> None:
    server = audio(samples(1, 2))
    synth = make(server, credits=BrokenCredits(), notify=lambda text: None)
    assert synth.probe() is None
    assert play(synth, "ciao").tolist() == [1, 2]
    assert "database is locked" in caplog.text
    assert KEY not in caplog.text


def test_a_failing_notify_never_stops_the_voice() -> None:
    def boom(text: str) -> None:
        raise RuntimeError("no speaker")

    account = Account(39990)
    synth = account.synth()
    synth.notify = boom
    assert play(synth, "ciao").tolist() == [1, 2]


def _cancel_while_answering(status: int) -> tuple[ElevenLabsSynth, FakeFallback]:
    """A barge-in that lands while the POST waits for its first byte."""
    holder: list[ElevenLabsSynth] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        holder[0].cancel()
        return httpx2.Response(status, content=samples(1, 2), headers={"character-cost": "9"})

    fallback = FakeFallback()
    credits = VoiceCredits(VoiceLedger(":memory:"))
    notices: list[str] = []
    synth = make(Server(respond), fallback, credits=credits, notify=notices.append)
    holder.append(synth)
    return synth, fallback


def test_a_cancel_before_the_200_arrives_is_still_billed() -> None:
    synth, fallback = _cancel_while_answering(200)
    assert play(synth, "ciao").size == 0
    assert fallback.calls == 0
    assert synth.credits is not None
    rows = synth.credits._ledger._db.execute("SELECT characters, credits FROM voice").fetchall()
    assert rows == [(4, 9)]


def test_a_cancel_before_a_non_200_is_quiet_and_not_billed() -> None:
    synth, fallback = _cancel_while_answering(500)
    assert play(synth, "ciao").size == 0
    assert fallback.calls == 0
    assert synth.credits is not None
    assert synth.credits._ledger._db.execute("SELECT COUNT(*) FROM voice").fetchone()[0] == 0


def test_make_elevenlabs_counts_credits_in_the_spend_db(tmp_path) -> None:
    import sqlite3

    from claudio.__main__ import make_elevenlabs

    synth = make_elevenlabs(_settings(tmp_path), FakeFallback())
    assert synth is not None and isinstance(synth.credits, VoiceCredits)
    with sqlite3.connect(tmp_path / "spend.db") as db:
        tables = {row[0] for row in db.execute("select name from sqlite_master")}
    assert "voice" in tables


def test_startup_credit_notice_is_claimed_once_from_the_synth(tmp_path) -> None:
    from claudio.__main__ import startup_credit_notice

    class Credits:
        calls = 0

        def startup_notice(self) -> str:
            self.calls += 1
            return "avviso"

    synth = make_elevenlabs_with(tmp_path)
    synth.credits = Credits()
    assert startup_credit_notice(synth) == "avviso"
    assert synth.credits.calls == 1
    assert startup_credit_notice(FakeFallback()) is None


def make_elevenlabs_with(tmp_path):
    from claudio.__main__ import make_elevenlabs

    synth = make_elevenlabs(_settings(tmp_path), FakeFallback())
    assert synth is not None
    return synth


async def test_notify_announces_after_the_reply() -> None:
    import asyncio
    import threading

    from claudio.__main__ import announcer

    told: list[str] = []
    done = asyncio.Event()

    class Assistant:
        async def announce(self, text: str) -> None:
            told.append(text)
            done.set()

    loop = asyncio.get_running_loop()
    notify = announcer(Assistant(), loop)
    thread = threading.Thread(target=notify, args=("dal thread",))
    thread.start()
    thread.join()
    await asyncio.wait_for(done.wait(), 2)
    assert told == ["dal thread"]
    done.clear()
    notify("dal loop")  # from the loop's own thread: returns at once, announces later
    assert told == ["dal thread"]
    await asyncio.wait_for(done.wait(), 2)
    assert told == ["dal thread", "dal loop"]


def test_a_notice_without_notify_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    account = Account(31990)
    synth = account.synth()
    synth.notify = None  # claudio --text and the benchmark leave it unset
    play(synth, "uno")
    assert WARN_TEXT in caplog.text and "non detto a voce" in caplog.text
    assert len(account.rows(synth)) == 1


# --- the cache of fixed replies ---------------------------------------------------------------

FIXED = "Apro Chrome."


def cached_synth(tmp_path, server: Server, fallback: Any = None, **kwargs: Any) -> ElevenLabsSynth:
    from claudio.tts.eleven_cache import TakeCache

    cache = TakeCache(tmp_path / "eleven_cache", "profile")
    return make(server, fallback, cache=cache, fixed=frozenset({FIXED, "Fatto."}), **kwargs)


def wav_files(tmp_path) -> list:
    return sorted((tmp_path / "eleven_cache").glob("*.wav"))


def test_a_fixed_piece_is_generated_once_then_played_from_disk(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)
    account = Account(1000)
    synth = cached_synth(tmp_path, account.server, credits=VoiceCredits(VoiceLedger(":memory:")))
    first = play(synth, FIXED)
    synth.begin_reply()
    second = play(synth, FIXED)
    assert account.tts_calls == 1  # the subscription probe is free and not a synthesis
    assert second.tolist() == first.tolist() == [1, 2]
    assert synth.last_cost is None and synth.last_first_ms is not None
    assert "frase fissa da disco" in caplog.text
    assert len(account.rows(synth)) == 1


def test_a_fixed_piece_goes_online_without_previous_text(tmp_path) -> None:
    server = audio(samples(1, 2))
    synth = cached_synth(tmp_path, server)
    play(synth, "Una frase qualsiasi.")
    play(synth, FIXED)
    assert "previous_text" not in server.body(1)
    play(synth, "Dopo.")
    assert server.body(2)["previous_text"] == FIXED


def test_a_cache_hit_sets_previous_text_for_the_next_online_piece(tmp_path) -> None:
    server = audio(samples(1, 2))
    synth = cached_synth(tmp_path, server)
    play(synth, FIXED)  # generated and saved: request 0
    synth.begin_reply()
    play(synth, FIXED)  # played from disk: no request
    play(synth, "Dopo.")
    assert len(server.requests) == 2
    assert server.body(1)["previous_text"] == FIXED


def test_a_cache_hit_stops_after_cancel(tmp_path) -> None:
    synth = cached_synth(tmp_path, audio(samples(1)))
    synth.cache.save(FIXED, np.ones(SAMPLE_RATE * 2, dtype=np.int16), SAMPLE_RATE)  # 4 slices
    chunks = synth.stream(FIXED)
    assert len(next(chunks)) > 0
    synth.cancel()
    assert list(chunks) == []


def test_other_text_is_never_cached(tmp_path) -> None:
    server = audio(samples(1, 2))
    synth = cached_synth(tmp_path, server)
    play(synth, "Una frase di Claude.")
    play(synth, "Una frase di Claude.")
    assert len(server.requests) == 2
    assert wav_files(tmp_path) == []


def test_a_cut_or_cancelled_generation_is_not_saved(tmp_path) -> None:
    def gen() -> Iterator[bytes]:
        yield samples(1, 2)
        raise httpx2.ReadError("cut")

    cut = Server(lambda request: httpx2.Response(200, content=gen()))
    synth = cached_synth(tmp_path, cut)
    assert play(synth, FIXED).tolist() == [1, 2]
    assert wav_files(tmp_path) == []

    server = audio(samples(1), samples(2), samples(3))
    synth = cached_synth(tmp_path, server)
    for _chunk in synth.stream(FIXED):
        synth.cancel()
    assert wav_files(tmp_path) == []
    synth.begin_reply()
    play(synth, FIXED)
    assert len(server.requests) == 2  # the next time goes online again
    assert len(wav_files(tmp_path)) == 1


def test_a_broken_cache_file_goes_online_again(tmp_path) -> None:
    server = audio(samples(1, 2))
    synth = cached_synth(tmp_path, server)
    play(synth, FIXED)
    (path,) = wav_files(tmp_path)
    path.write_bytes(b"not a wav")
    synth.begin_reply()
    assert play(synth, FIXED).tolist() == [1, 2]
    assert len(server.requests) == 2
    synth.begin_reply()
    assert play(synth, FIXED).tolist() == [1, 2]
    assert len(server.requests) == 2  # rewritten and good


@pytest.mark.parametrize("reason", ["credits", "error"])
def test_the_fallback_reply_never_reads_the_cache(tmp_path, reason: str) -> None:
    account = Account(1000)
    synth = cached_synth(tmp_path, account.server, credits=VoiceCredits(VoiceLedger(":memory:")))
    play(synth, FIXED)  # now in the cache
    synth.begin_reply()
    reads: list[tuple] = []
    real_load = synth.cache.load
    synth.cache.load = lambda *args: reads.append(args) or real_load(*args)
    if reason == "credits":
        account.used = 40000
        synth.probe()
        assert synth.credits is not None and synth.credits.exhausted()
    else:
        account.tts_status = 500
        play(synth, "Prima.")  # error at the start of the reply: the rest is the fallback's
    calls = account.tts_calls
    assert play(synth, FIXED).tolist() == [7] * 10
    assert reads == []
    assert account.tts_calls == calls


def test_make_elevenlabs_has_the_cache(tmp_path) -> None:
    from claudio import __main__ as entry
    from claudio.tts.eleven_cache import TakeCache, eleven_profile

    settings = _settings(tmp_path)
    synth = entry.make_elevenlabs(settings, FakeFallback())
    assert synth is not None
    assert isinstance(synth.cache, TakeCache)
    assert synth.cache.folder == tmp_path / "data" / "eleven_cache"
    assert synth.cache.profile == eleven_profile(settings.tts)
    assert "Sono pronto." in synth.fixed
    assert not (tmp_path / "data" / "eleven_cache").exists()


class Switch:
    """A server that answers with audio or fails, as `mode` says."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.server = Server(self._respond)

    def _respond(self, request: httpx2.Request) -> httpx2.Response:
        if self.mode == "down":
            raise httpx2.ConnectError("no route")
        if self.mode == "cut":

            def gen() -> Iterator[bytes]:
                yield samples(1, 2)
                raise httpx2.ReadError("cut")

            return httpx2.Response(200, content=gen())
        return httpx2.Response(200, content=iter([samples(1, 2)]))


def cached_and_switch(tmp_path):
    switch = Switch()
    fallback = FakeFallback()
    synth = cached_synth(tmp_path, switch.server, fallback)
    play(synth, FIXED)  # in the cache now
    synth.begin_reply()
    return switch, fallback, synth


def spy_on_reads(synth: ElevenLabsSynth) -> list[tuple]:
    reads: list[tuple] = []
    real_load = synth.cache.load
    synth.cache.load = lambda *args: reads.append(args) or real_load(*args)
    return reads


def test_after_a_failure_before_audio_a_fixed_piece_skips_the_cache(tmp_path) -> None:
    switch, _fallback, synth = cached_and_switch(tmp_path)
    reads = spy_on_reads(synth)
    switch.mode = "down"
    play(synth, "Una frase di Claude.")  # fails before audio: the voice is down
    synth.begin_reply()
    requests = len(switch.server.requests)
    assert play(synth, FIXED).tolist() == [7] * 10  # the fallback speaks it, not the cache
    assert reads == []
    assert len(switch.server.requests) == requests + 1  # it went online first


def test_a_success_brings_the_cache_back(tmp_path) -> None:
    switch, _, synth = cached_and_switch(tmp_path)
    switch.mode = "down"
    play(synth, "Una frase di Claude.")
    synth.begin_reply()
    switch.mode = "ok"
    play(synth, FIXED)  # goes online (cache skipped) and gets a 200
    synth.begin_reply()
    requests = len(switch.server.requests)
    assert play(synth, FIXED).tolist() == [1, 2]
    assert len(switch.server.requests) == requests  # from disk again


def test_a_cut_or_cancelled_request_does_not_mark_the_voice_down(tmp_path) -> None:
    switch, _, synth = cached_and_switch(tmp_path)
    switch.mode = "cut"
    play(synth, "Una frase di Claude.")  # cut after audio
    synth.begin_reply()
    for _chunk in synth.stream("Un'altra frase."):
        synth.cancel()  # cancelled after the first chunk
    synth.begin_reply()
    requests = len(switch.server.requests)
    assert play(synth, FIXED).tolist() == [1, 2]
    assert len(switch.server.requests) == requests
