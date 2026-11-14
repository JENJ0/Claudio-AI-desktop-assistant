import json
import sys

import httpx2

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import try_elevenlabs as smoke

KEY = "sk_test_secret_123"


def answering(handler) -> httpx2.Client:
    return httpx2.Client(base_url=smoke.API, transport=httpx2.MockTransport(handler))


def test_an_error_body_never_shows_the_key() -> None:
    with answering(lambda request: httpx2.Response(401, text=f"invalid key {KEY}")) as client:
        attempt = smoke.try_once(client, KEY, "tts", "VOICE", "eleven_v4", "Ciao.")
    assert attempt.status == 401
    assert KEY not in attempt.error
    assert "***" in attempt.error


def test_a_network_error_never_shows_the_key() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"refused for {KEY}", request=request)

    with answering(handler) as client:
        attempt = smoke.try_once(client, KEY, "tts", "VOICE", "eleven_v4", "Ciao.")
    assert attempt.status is None
    assert KEY not in attempt.error
    assert "ConnectError" in attempt.error


def test_an_odd_byte_count_keeps_whole_samples_only() -> None:
    one_second = b"\x01\x00" * smoke.SAMPLE_RATE + b"\x02"
    with answering(lambda request: httpx2.Response(200, content=one_second)) as client:
        attempt = smoke.try_once(client, KEY, "tts", "VOICE", "eleven_flash_v2_5", "Ciao.")
    assert attempt.status == 200
    assert attempt.audio_s == 1.0
    assert attempt.first_byte_ms is not None


def sent(endpoint: str) -> dict:
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(
            path=request.url.path,
            format=request.url.params["output_format"],
            body=json.loads(request.content),
        )
        return httpx2.Response(200, content=b"")

    with answering(handler) as client:
        smoke.try_once(client, KEY, endpoint, "VOICE", "eleven_v4", "Ciao.")
    return seen


def test_the_speech_endpoint_puts_the_voice_in_the_path() -> None:
    assert sent("tts") == {
        "path": "/v1/text-to-speech/VOICE/stream",
        "format": "pcm_24000",
        "body": {"text": "Ciao.", "model_id": "eleven_v4"},
    }


def test_the_dialogue_endpoint_puts_the_voice_in_the_inputs() -> None:
    assert sent("dialogue") == {
        "path": "/v1/text-to-dialogue/stream",
        "format": "pcm_24000",
        "body": {"inputs": [{"text": "Ciao.", "voice_id": "VOICE"}], "model_id": "eleven_v4"},
    }
