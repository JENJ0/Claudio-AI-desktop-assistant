"""ElevenLabs smoke test (G11 step 1): which models answer on which endpoint, how fast
from this PC, and how many credits a character costs. The docs (14/11) offer v4 only on
Text to Dialogue, so each model tries both HTTP endpoints. One sentence per try (about 80
characters, ~500 in all); the account usage is read after each try. The key comes from
.env and is never printed. WAVs go to logs/elevenlabs-smoke/ to listen to.

  uv run python scripts/try_elevenlabs.py
  uv run python scripts/try_elevenlabs.py --models eleven_flash_v2_5 --endpoints tts
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2
from dotenv import dotenv_values

from claudio.config import PROJECT_ROOT

API = "https://api.elevenlabs.io"
VOICE_ID = "K5aUZ2v5ZGQmACb8D757"  # the "JACK" clone, JenJo's pick on 14/11
MODELS = ("eleven_flash_v2_5", "eleven_v4_turbo", "eleven_v4")
ENDPOINTS = ("tts", "dialogue")
SENTENCE = "Ciao, sono Claudio. Questa è una prova della mia nuova voce: dimmi come suona."
SAMPLE_RATE = 24000  # pcm_24000: raw int16 mono
OUT = PROJECT_ROOT / "logs" / "elevenlabs-smoke"
SUBSCRIPTION_FIELDS = (
    "tier",
    "status",
    "character_count",
    "character_limit",
    "next_character_count_reset_unix",
    "can_extend_character_limit",
    "max_credit_limit_extension",
    "current_overage",
    "can_use_instant_voice_cloning",
)
COST_HINTS = ("char", "cost", "credit", "request", "history", "model")


@dataclass
class Attempt:
    endpoint: str
    model: str
    status: int | None = None
    error: str = ""
    first_byte_ms: float | None = None
    total_ms: float | None = None
    pcm: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def audio_s(self) -> float:
        return len(self.pcm) // 2 / SAMPLE_RATE  # an odd last byte is half a sample


def mask(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def request(endpoint: str, voice_id: str, model: str, text: str) -> tuple[str, dict[str, Any]]:
    if endpoint == "tts":
        return f"/v1/text-to-speech/{voice_id}/stream", {"text": text, "model_id": model}
    inputs = [{"text": text, "voice_id": voice_id}]
    return "/v1/text-to-dialogue/stream", {"inputs": inputs, "model_id": model}


def try_once(
    client: httpx2.Client,
    key: str,
    endpoint: str,
    voice_id: str,
    model: str,
    text: str,
    params: dict[str, str] | None = None,
) -> Attempt:
    attempt = Attempt(endpoint, model)
    path, body = request(endpoint, voice_id, model, text)
    start = time.perf_counter()
    try:
        query = {"output_format": "pcm_24000", **(params or {})}
        with client.stream("POST", path, params=query, json=body) as response:
            attempt.status = response.status_code
            attempt.headers = {
                k: v for k, v in response.headers.items() if k.lower() != "set-cookie"
            }
            if response.status_code != 200:
                attempt.error = mask(response.read().decode("utf-8", "replace")[:400], key)
            else:
                chunks = []
                for chunk in response.iter_bytes():
                    if attempt.first_byte_ms is None:
                        attempt.first_byte_ms = (time.perf_counter() - start) * 1000
                    chunks.append(chunk)
                attempt.pcm = b"".join(chunks)
    except httpx2.HTTPError as exc:
        attempt.error = mask(f"{type(exc).__name__}: {exc}", key)
    attempt.total_ms = (time.perf_counter() - start) * 1000
    return attempt


def subscription(client: httpx2.Client, key: str) -> dict[str, Any] | None:
    try:
        response = client.get("/v1/user/subscription")
    except httpx2.HTTPError as exc:
        print(f"abbonamento: {mask(f'{type(exc).__name__}: {exc}', key)}")
        return None
    if response.status_code != 200:
        print(f"abbonamento: HTTP {response.status_code} {mask(response.text[:400], key)}")
        return None
    data: dict[str, Any] = response.json()
    return data


def save_wav(attempt: Attempt, folder: Path) -> Path:
    path = folder / f"{attempt.model}-{attempt.endpoint}.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(attempt.pcm[: len(attempt.pcm) // 2 * 2])
    return path


def show_subscription(data: dict[str, Any]) -> None:
    for name in SUBSCRIPTION_FIELDS:
        value = data.get(name)
        if name == "next_character_count_reset_unix" and value:
            value = f"{value} ({datetime.fromtimestamp(value):%d/%m/%Y %H:%M})"
        print(f"  {name}: {value}")


def show_attempt(attempt: Attempt, spent: int | None, chars: int) -> None:
    where = f"{attempt.model:<18} {attempt.endpoint:<8}"
    if attempt.status != 200 or not attempt.pcm:
        print(f"{where} HTTP {attempt.status}  {attempt.error or 'nessun audio'}")
        return
    per_char = f"{spent / chars:.2f}" if spent is not None else "?"
    print(
        f"{where} HTTP 200  primo byte {attempt.first_byte_ms:5.0f} ms  "
        f"totale {attempt.total_ms:5.0f} ms  audio {attempt.audio_s:4.1f} s  "
        f"scalati {spent if spent is not None else '?'} ({per_char} per carattere)"
    )
    hints = {k: v for k, v in attempt.headers.items() if any(h in k.lower() for h in COST_HINTS)}
    if hints:
        print(f"{'':28}intestazioni: {hints}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--voice", default=VOICE_ID)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--endpoints", nargs="+", choices=ENDPOINTS, default=list(ENDPOINTS))
    parser.add_argument("--timeout", type=float, default=20.0, help="seconds to read")
    args = parser.parse_args()

    key = (dotenv_values(PROJECT_ROOT / ".env").get("ELEVENLABS_API_KEY") or "").strip()
    if not key:
        print("ELEVENLABS_API_KEY manca in .env: segui la guida V0 in piani/IT-2-passo-0b-prove.md")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    timeout = httpx2.Timeout(args.timeout, connect=5.0)
    with httpx2.Client(base_url=API, headers={"xi-api-key": key}, timeout=timeout) as client:
        before = subscription(client, key)
        if before is None:
            return 1
        print("Abbonamento prima della prova:")
        show_subscription(before)
        count = before.get("character_count")
        print(f"\nFrase ({len(SENTENCE)} caratteri): {SENTENCE}\n")
        names: set[str] = set()
        for model in args.models:
            for endpoint in args.endpoints:
                attempt = try_once(client, key, endpoint, args.voice, model, SENTENCE)
                after = subscription(client, key)
                spent = None
                if after is not None and count is not None:
                    spent = after["character_count"] - count
                    count = after["character_count"]
                show_attempt(attempt, spent, len(SENTENCE))
                if attempt.pcm:
                    save_wav(attempt, OUT)
                names.update(attempt.headers)
        print(f"\nNomi delle intestazioni viste: {sorted(names)}")
        print(f"WAV in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
