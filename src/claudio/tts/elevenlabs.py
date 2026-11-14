"""The online voice: ElevenLabs text-to-speech over HTTP, streamed, with a local fallback.

Same interface as JackSynth (stream/synthesize/cancel/begin_reply/sample_rate). When the
network or the key fails before any audio, the rest of the reply is spoken by `fallback`
(one voice per reply, and no timeout wait for every sentence); the next reply tries again."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator, Iterator
from typing import Any

import httpx2
import numpy as np

from claudio.tts.loudness import LevelGuard, to_float

log = logging.getLogger(__name__)

API = "https://api.elevenlabs.io"
SAMPLE_RATE = 24000
ERROR_BODY_CHARS = 200


class _OnlineError(Exception):
    """The online voice answered, but not with audio."""


class ElevenLabsSynth:
    def __init__(
        self,
        key: str,
        voice_id: str,
        model: str,
        fallback: Any,
        *,
        stability: float | None = None,
        similarity: float | None = None,
        timeout_s: float = 3.0,
        lufs: float = -24.0,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self._key = key
        self._voice_id = voice_id
        self._model = model
        self.fallback = fallback
        self._settings = self._voice_settings(stability, similarity)
        self._guard = LevelGuard(SAMPLE_RATE, target=lufs) if lufs != 0 else None
        # One client for the whole life of the synth keeps the TLS connection warm.
        self._client = httpx2.Client(
            base_url=API,
            headers={"xi-api-key": key},
            timeout=httpx2.Timeout(timeout_s),
            transport=transport,
        )
        self.sample_rate = SAMPLE_RATE
        self.problem: str | None = None
        self.last_cost: int | None = None
        self.last_first_ms: float | None = None
        self._previous_text: str | None = None
        self._fallback_reply = False  # an earlier piece of this reply went to the fallback
        self._generation = 0  # cancel() bumps it: a stream older than this one is cancelled
        self._response: httpx2.Response | None = None

    @staticmethod
    def _voice_settings(stability: float | None, similarity: float | None) -> dict[str, float]:
        settings: dict[str, float] = {}
        if stability is not None:
            settings["stability"] = stability
        if similarity is not None:
            settings["similarity_boost"] = similarity
        return settings

    def _mask(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def probe(self) -> str | None:
        """Ask the (free) subscription endpoint: None if the online voice answers."""
        try:
            response = self._client.get("/v1/user/subscription")
        except httpx2.HTTPError as exc:
            self.problem = self._mask(f"{type(exc).__name__}: {exc}")
        else:
            self.problem = None if response.status_code == 200 else f"HTTP {response.status_code}"
        return self.problem

    # -- synth interface --------------------------------------------------------------------

    def cancel(self) -> None:
        """Stop the sentence at the next chunk; the fallback stays quiet."""
        self._generation += 1
        response = self._response
        if response is not None:
            with contextlib.suppress(Exception):
                response.close()
        forward = getattr(self.fallback, "cancel", None)
        if forward is not None:
            forward()

    def begin_reply(self) -> None:
        self._previous_text = None
        self._fallback_reply = False
        forward = getattr(self.fallback, "begin_reply", None)
        if forward is not None:
            forward()

    def synthesize(self, text: str) -> np.ndarray:
        chunks = list(self.stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """int16 chunks as they arrive. An error before any audio leaves the rest of the reply
        to the fallback; one in the middle just ends the sentence (a second voice would repeat)."""
        generation = self._generation
        if not self._fallback_reply:
            failure = yield from self._online(text, generation)
            if failure is None or self._is_cancelled(generation):
                return
            self._fallback_reply = True
            log.warning("ElevenLabs non risponde (%s): uso la voce di ripiego", failure)
        yield from self._fallback_chunks(text, generation)

    # -- online request ---------------------------------------------------------------------

    def _body(self, text: str) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, "model_id": self._model, "language_code": "it"}
        if self._previous_text is not None:
            body["previous_text"] = self._previous_text
        if self._settings:
            body["voice_settings"] = self._settings
        return body

    def _is_cancelled(self, generation: int) -> bool:
        return self._generation != generation

    def _online(self, text: str, generation: int) -> Generator[np.ndarray, None, str | None]:
        """Yields the audio; the return value is None when done (or cut, or cancelled), or the
        reason when nothing was spoken and the fallback must take over."""
        self.last_cost = None
        self.last_first_ms = None
        if self._guard is not None:
            self._guard.begin()
        started = time.perf_counter()
        mine: httpx2.Response | None = None
        spoken = False
        leftover = b""
        path = f"/v1/text-to-speech/{self._voice_id}/stream"
        try:
            with self._client.stream(
                "POST", path, params={"output_format": "pcm_24000"}, json=self._body(text)
            ) as response:
                self._response = response
                mine = response
                if self._is_cancelled(generation):
                    return None
                if response.status_code != 200:
                    detail = self._mask(response.read().decode("utf-8", "replace"))
                    raise _OnlineError(f"HTTP {response.status_code} {detail[:ERROR_BODY_CHARS]}")
                cost = response.headers.get("character-cost")
                self.last_cost = int(cost) if cost and cost.isdigit() else None
                for data in response.iter_bytes():
                    if self._is_cancelled(generation):
                        return None
                    data = leftover + data
                    usable = len(data) // 2 * 2
                    leftover = data[usable:]  # the odd byte goes with the next chunk
                    if not usable:
                        continue
                    if not spoken:
                        self.last_first_ms = (time.perf_counter() - started) * 1000
                        spoken = True
                    self.sample_rate = SAMPLE_RATE
                    yield self._level(np.frombuffer(data[:usable], dtype="<i2"))
                if not spoken:
                    raise _OnlineError("risposta senza audio")
        except (httpx2.HTTPError, httpx2.StreamError, _OnlineError) as exc:
            if self._is_cancelled(generation):
                return None
            reason = self._mask(f"{type(exc).__name__}: {exc}")
            if spoken:  # cut in the middle: no second voice
                log.warning("ElevenLabs interrotto a metà frase (%s)", reason)
                return None
            return reason
        finally:
            if self._response is mine:  # a newer stream may own it by now
                self._response = None
        if self._is_cancelled(generation):
            return None
        self._previous_text = text
        log.info(
            "ElevenLabs: %d caratteri, costo %s, primo chunk %s ms",
            len(text),
            self.last_cost,
            None if self.last_first_ms is None else round(self.last_first_ms),
        )
        return None

    def _level(self, pcm: np.ndarray) -> np.ndarray:
        if self._guard is None:
            return pcm.astype(np.int16)
        leveled = self._guard.process(to_float(pcm))
        return np.clip(np.round(leveled * 32768), -32768, 32767).astype(np.int16)

    # -- fallback ---------------------------------------------------------------------------

    def _fallback_chunks(self, text: str, generation: int) -> Iterator[np.ndarray]:
        stream = getattr(self.fallback, "stream", None)
        if stream is None:
            if not self._is_cancelled(generation):
                audio = self.fallback.synthesize(text)
                self.sample_rate = int(self.fallback.sample_rate)
                yield audio
            return
        for chunk in stream(text):
            if self._is_cancelled(generation):
                return
            self.sample_rate = int(self.fallback.sample_rate)  # it may change between chunks
            yield chunk
