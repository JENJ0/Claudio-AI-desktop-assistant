"""The online voice: ElevenLabs text-to-speech over HTTP, streamed, with a local fallback.

Same interface as JackSynth (stream/synthesize/cancel/begin_reply/sample_rate). When the
network or the key fails before any audio, the rest of the reply is spoken by `fallback`
(one voice per reply, and no timeout wait for every sentence); the next reply tries again."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Generator, Iterator
from typing import Any

import httpx2
import numpy as np

from claudio.cost.voice import VoiceCredits
from claudio.tts.eleven_cache import TakeCache
from claudio.tts.loudness import LevelGuard, to_float

log = logging.getLogger(__name__)

API = "https://api.elevenlabs.io"
SAMPLE_RATE = 24000
ERROR_BODY_CHARS = 200


class _OnlineError(Exception):
    """The online voice answered, but not with audio."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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
        credits: VoiceCredits | None = None,
        notify: Callable[[str], None] | None = None,
        cache: TakeCache | None = None,
        fixed: frozenset[str] = frozenset(),
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
        self.credits = credits
        self.notify = notify  # may be set after construction; called outside any lock
        self._retry_probe = True  # one more read after the first success if usage is unknown
        self._probe_due = False  # set when billing; the probe runs before the next request
        self.cache = cache  # recordings of the fixed pieces, spoken only by this voice
        self.fixed = fixed  # the pieces (stripped) worth keeping
        self._down = False  # a request failed before audio: no cache until one gets a 200
        self._out_logged = False  # one "credits finished" line per period, not per sentence

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
            if self.problem is None:
                self._read_credits(response)
        return self.problem

    def _read_credits(self, response: httpx2.Response) -> None:
        """Bookkeeping must never change the probe: any error here is only logged."""
        if self.credits is None:
            return
        try:
            self.credits.update(response.json())
            log.info(self.credits.summary())
        except Exception as exc:
            log.warning(
                "Crediti ElevenLabs non letti (%s)", self._mask(f"{type(exc).__name__}: {exc}")
            )

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
        if not self._fallback_reply and not self._online_allowed():
            self._fallback_reply = True  # out of credits: quietly, until the reset
        if not self._fallback_reply:
            fixed = self.cache is not None and text.strip() in self.fixed
            saved = self._load_fixed(text) if fixed and not self._down else None
            if saved is not None:
                yield from self._play_saved(text, saved, generation)
                return
            failure = yield from self._online(text, generation, keep=fixed)
            if failure is None or self._is_cancelled(generation):
                return
            self._fallback_reply = True
            log.warning("ElevenLabs non risponde (%s): uso la voce di ripiego", failure)
        yield from self._fallback_chunks(text, generation)

    # -- credits ----------------------------------------------------------------------------

    def _online_allowed(self) -> bool:
        """False when the credits are finished (no request until the monthly reset)."""
        credits = self.credits
        if credits is None:
            return True
        try:
            if self._probe_due or credits.stale():  # free read: after the reset or a failed start
                self._probe_due = False
                self.probe()
            if not credits.exhausted():
                self._out_logged = False
                return True
        except Exception as exc:
            self._bookkeeping_failed(exc)
            return True
        if not self._out_logged:
            self._out_logged = True
            log.info("Crediti ElevenLabs finiti: parla la voce locale fino al reset")
        return False

    def _bill(self, text: str, cost: int | None) -> None:
        """Count a request that produced audio. Errors here are logged, never raised."""
        credits = self.credits
        if credits is None:
            return
        try:
            notice = credits.spend(self._model, len(text), cost)
            if self._retry_probe and not credits.known:
                self._retry_probe = False
                self._probe_due = True  # not here: this may run on the event loop (barge-in)
        except Exception as exc:
            self._bookkeeping_failed(exc)
            return
        self._say(notice)

    def _mark_out(self) -> None:
        """HTTP 402: the credits are out until the reset. Never raises."""
        if self.credits is None:
            return
        try:
            notice = self.credits.mark_out()
        except Exception as exc:
            self._bookkeeping_failed(exc)
            return
        self._say(notice)

    def _say(self, notice: str | None) -> None:
        if not notice:
            return
        if self.notify is None:  # --text, benchmark: the notice is saved as said, so keep a trace
            log.warning("Avviso crediti (non detto a voce): %s", notice)
            return
        try:
            self.notify(notice)
        except Exception as exc:
            self._bookkeeping_failed(exc)

    def _bookkeeping_failed(self, exc: Exception) -> None:
        log.warning(
            "Conteggio crediti ElevenLabs non riuscito (%s)",
            self._mask(f"{type(exc).__name__}: {exc}"),
        )

    # -- online request ---------------------------------------------------------------------

    def _load_fixed(self, text: str) -> np.ndarray | None:
        assert self.cache is not None
        return self.cache.load(text, SAMPLE_RATE)

    def _play_saved(self, text: str, pcm: np.ndarray, generation: int) -> Iterator[np.ndarray]:
        """A fixed piece from disk: no request, no credits."""
        started = time.perf_counter()
        self.last_cost = None
        self.last_first_ms = None
        self.sample_rate = SAMPLE_RATE
        step = SAMPLE_RATE // 2  # half a second at a time, like the online chunks
        for start in range(0, len(pcm), step):
            if self._is_cancelled(generation):
                return
            if self.last_first_ms is None:
                self.last_first_ms = (time.perf_counter() - started) * 1000
            yield pcm[start : start + step]
        if self._is_cancelled(generation):
            return
        self._previous_text = text
        log.info("ElevenLabs: frase fissa da disco (%d caratteri)", len(text))

    def _body(self, text: str, with_previous: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, "model_id": self._model, "language_code": "it"}
        if with_previous and self._previous_text is not None:
            body["previous_text"] = self._previous_text
        if self._settings:
            body["voice_settings"] = self._settings
        return body

    def _is_cancelled(self, generation: int) -> bool:
        return self._generation != generation

    def _online(
        self, text: str, generation: int, keep: bool = False
    ) -> Generator[np.ndarray, None, str | None]:
        """Yields the audio; the return value is None when done (or cut, or cancelled), or the
        reason when nothing was spoken and the fallback must take over. With `keep` (a fixed
        piece) the request has no previous_text and a complete one is saved in the cache."""
        self.last_cost = None
        self.last_first_ms = None
        if self._guard is not None:
            self._guard.begin()
        started = time.perf_counter()
        mine: httpx2.Response | None = None
        spoken = False
        paid = False  # a 200 was received: counted once, however the request ends
        cost: int | None = None
        leftover = b""
        kept: list[np.ndarray] = []
        path = f"/v1/text-to-speech/{self._voice_id}/stream"
        try:
            with self._client.stream(
                "POST",
                path,
                params={"output_format": "pcm_24000"},
                json=self._body(text, with_previous=not keep),
            ) as response:
                self._response = response
                mine = response
                if response.status_code != 200:
                    if self._is_cancelled(generation):
                        return None
                    detail = self._mask(response.read().decode("utf-8", "replace"))
                    raise _OnlineError(
                        f"HTTP {response.status_code} {detail[:ERROR_BODY_CHARS]}",
                        response.status_code,
                    )
                header = response.headers.get("character-cost")
                cost = int(header) if header and header.isdigit() else None
                self.last_cost = cost
                self._down = False
                paid = True  # before the cancel check: a barge-in during the wait is still paid
                if self._is_cancelled(generation):
                    return None
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
                    chunk = self._level(np.frombuffer(data[:usable], dtype="<i2"))
                    if keep:
                        kept.append(chunk)
                    yield chunk
                if not spoken:
                    raise _OnlineError("risposta senza audio")
        except (httpx2.HTTPError, httpx2.StreamError, _OnlineError) as exc:
            if self._is_cancelled(generation):
                return None
            reason = self._mask(f"{type(exc).__name__}: {exc}")
            if isinstance(exc, _OnlineError) and exc.status == 402:
                self._mark_out()  # any 402 counts: no body parsing
            if spoken:  # cut in the middle: no second voice
                log.warning("ElevenLabs interrotto a metà frase (%s)", reason)
                return None
            self._down = True
            return reason
        finally:
            if paid:  # also when cut or cancelled: the budget errs on the safe side
                self._bill(text, cost)
            if self._response is mine:  # a newer stream may own it by now
                self._response = None
        if self._is_cancelled(generation):
            return None
        self._previous_text = text
        if keep and self.cache is not None and kept:  # only here: complete, not cut, not cancelled
            self.cache.save(text, np.concatenate(kept), SAMPLE_RATE)
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
