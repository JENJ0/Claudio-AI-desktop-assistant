"""Claude answers what the local router does not understand, streamed sentence by sentence.

Haiku answers first. What needs reasoning goes to Sonnet: Haiku hands it over with the
`ask_sonnet` tool (after a short "Ci penso un attimo."), or JenJo asks for it ("pensaci bene").

Every call goes through the spending counter: `check()` before (80 % warning said aloud once a
day, nothing sent at 100 %) and `record()` after, also when the reply is cut short by "basta"
(the tokens already generated are billed by Anthropic)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Any

from claudio.brain.prompts import (
    API_ERROR,
    ASK_SONNET_TOOL,
    BAD_KEY,
    BUDGET_BLOCKED,
    BUDGET_WARNING,
    NETWORK_ERROR,
    NO_KEY,
    REFUSED,
    SYSTEM_PROMPT,
    THINKING,
)
from claudio.config import ClaudeSettings, in_language
from claudio.cost.pricing import CacheTtl, Usage
from claudio.cost.tracker import BudgetState, CostTracker
from claudio.i18n import tr
from claudio.router.normalizer import normalize

log = logging.getLogger(__name__)

CACHE_TTL: CacheTtl = "5m"  # the default TTL of `cache_control`: what a cache write costs
CHARS_PER_TOKEN = 3  # estimate for a reply cut short before the API sent its output count
_MARKDOWN = re.compile(r"[*#`]+")  # read aloud as "asterisco": never sent to the voice
WARM_EVERY_S = 15.0  # an idle connection older than this may be closed: open a fresh one
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")  # not strftime: Windows' locale


class ClaudeBrain:
    """Ask Claude, keep a short history, count every token.

    `client` is an `anthropic.AsyncAnthropic` (or a fake in tests); None means no API key."""

    def __init__(
        self,
        client: Any | None,
        tracker: CostTracker,
        settings: ClaudeSettings,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.client = client
        self.tracker = tracker
        self.settings = settings
        self.clock = clock
        self.wall = wall  # local date and time, told to Claude with each request
        # Sonnet, when it has prices (the builder clears it otherwise): None = Haiku only.
        self.smart_model: str | None = settings.smart_model or None
        self._smart_phrases = {
            lang: [normalize(p, drop_fillers=False) for p in phrases if p.strip()]
            for lang, phrases in settings.smart_phrases.items()
        }
        self._history: deque[tuple[str, str]] = deque(maxlen=settings.history_turns or None)
        self._last_turn = -float("inf")
        self._last_contact = -float("inf")  # last time the connection to the API was used
        self._warming: set[asyncio.Task[None]] = set()
        self.last_call: dict[str, Any] = {}  # model, tokens, cost of the last call (metrics)

    def reset(self) -> None:
        self._history.clear()

    def warm_up(self) -> None:
        """JenJo is about to speak (wake word): open the HTTPS connection now, for free (a
        models lookup, no tokens), so the request does not pay the TCP and TLS handshakes."""
        if self.client is None or self.clock() - self._last_contact < WARM_EVERY_S:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._last_contact = self.clock()
        task = loop.create_task(self._warm())
        self._warming.add(task)
        task.add_done_callback(self._warming.discard)

    async def _warm(self) -> None:
        try:
            client: Any = self.client
            await client.models.retrieve(self.settings.model)
        except Exception as exc:  # only a head start: the request itself reports problems
            log.debug("Claude: connessione anticipata non riuscita: %s", exc)

    def wants_smart(self, text: str, language: str) -> bool:
        """JenJo asked for Sonnet in words ("pensaci bene...")."""
        if self.smart_model is None:
            return False
        said = f" {normalize(text, drop_fillers=False)} "
        phrases = self._smart_phrases.get(language, []) + [
            p for lang, ps in self._smart_phrases.items() if lang != language for p in ps
        ]
        return any(f" {p} " in said for p in phrases)

    async def ask(self, text: str, language: str) -> AsyncGenerator[str, None]:
        """Claude's reply to `text`, in `language`, as it arrives. Problems become one short
        spoken sentence: Claudio never goes silent and never crashes because of the API."""
        self.last_call = {}
        if self.client is None:
            log.warning("Claude: chiave API mancante (ANTHROPIC_API_KEY nel file .env)")
            self.last_call = {"claude_error": "no_key"}
            yield in_language(NO_KEY, language)
            return
        status = self.tracker.check()
        if status.state is BudgetState.BLOCKED:
            log.warning("Claude bloccato: budget di oggi finito (%d%%)", status.percent)
            self.last_call = {"claude_error": "budget"}
            yield in_language(BUDGET_BLOCKED, language)
            return
        if status.announce:
            yield in_language(BUDGET_WARNING, language).format(percent=status.percent) + " "
        # aclosing: when "basta" closes this generator, the API stream closes too, at once,
        # and its tokens are recorded now (not whenever the garbage collector gets to it).
        async with contextlib.aclosing(self._answer(text, language)) as stream:
            async for chunk in stream:
                yield chunk

    async def _answer(self, text: str, language: str) -> AsyncGenerator[str, None]:
        messages = self._messages(text, language)
        start = self.clock()
        leg = _Leg()
        smart = self.wants_smart(text, language)
        if not smart:
            async with contextlib.aclosing(
                self._stream(self._params(messages), language, start, leg)
            ) as haiku:
                async for chunk in haiku:
                    yield chunk
            if not leg.handoff or self.smart_model is None:
                self._remember(messages, leg)
                return
            log.info("Haiku passa la richiesta a Sonnet")
            if not "".join(leg.parts).strip():
                yield in_language(THINKING, language) + " "
            if self.tracker.check().state is BudgetState.BLOCKED:
                self.last_call["claude_error"] = "budget"
                yield in_language(BUDGET_BLOCKED, language)
                return
        else:
            log.info("Richiesta di ragionare: chiedo direttamente a Sonnet")
        self.last_call["escalated"] = "tool" if not smart else "asked"
        leg = _Leg()
        async with contextlib.aclosing(
            self._stream(self._smart_params(messages), language, start, leg, "smart_first_ms")
        ) as sonnet:
            async for chunk in sonnet:
                yield chunk
        self._remember(messages, leg)

    def _remember(self, messages: list[dict[str, str]], leg: _Leg) -> None:
        answer = "".join(leg.parts).strip()
        if leg.complete and answer:
            self._history.append((messages[-1]["content"], answer))
            self._last_turn = self.clock()

    def _messages(self, text: str, language: str) -> list[dict[str, str]]:
        if self.clock() - self._last_turn > self.settings.history_reset_s:
            self._history.clear()  # a new conversation: old context costs and confuses
        messages: list[dict[str, str]] = []
        if self.settings.history_turns:
            for question, answer in self._history:
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": f"{self._tag(language)} {text}"})
        return messages

    def _tag(self, language: str) -> str:
        """[it] or, with the time, [it | Thu 2026-10-23 17:15]: after the cached prefix, so
        the time changes nothing that is cached."""
        if not self.settings.tell_time:
            return f"[{language}]"
        now = self.wall()
        return f"[{language} | {_DAYS[now.weekday()]} {now:%Y-%m-%d %H:%M}]"

    def _common(
        self, model: str, max_tokens: int, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }
        if self.settings.cache:
            # Automatic caching: the marker goes on the last block, so the next turn reads
            # this whole conversation from the cache (below 512 tokens nothing is cached).
            params["cache_control"] = {"type": "ephemeral"}
        return params

    def _params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        s = self.settings
        params = self._common(s.model, s.max_tokens, messages)
        params["output_config"] = {"effort": s.effort}
        if not s.thinking:
            params["thinking"] = {"type": "disabled"}
        if s.escalate and self.smart_model is not None:
            params["tools"] = [ASK_SONNET_TOOL]
        return params

    def _smart_params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Sonnet thinks (adaptive thinking, its default: "disabled" is refused on Sonnet 5.5);
        a low effort keeps the wait short."""
        s = self.settings
        assert self.smart_model is not None
        params = self._common(self.smart_model, s.smart_max_tokens, messages)
        params["output_config"] = {"effort": s.smart_effort}
        return params

    async def _stream(
        self,
        params: dict[str, Any],
        language: str,
        start: float,
        leg: _Leg,
        first_key: str = "claude_first_ms",
    ) -> AsyncGenerator[str, None]:
        import anthropic

        model = params["model"]
        stream: Any = None
        final: Any = None
        try:
            client: Any = self.client  # not None: ask() answered already without a key
            async with client.messages.stream(**params) as stream:
                async for chunk in stream.text_stream:
                    clean = _MARKDOWN.sub("", chunk)
                    if first_key not in self.last_call and clean.strip():
                        self.last_call[first_key] = round((self.clock() - start) * 1000)
                    leg.parts.append(clean)
                    if clean:
                        yield clean
                final = await stream.get_final_message()
        except anthropic.AuthenticationError:
            log.error("Claude: chiave API non valida (401)")
            self.last_call["claude_error"] = "auth"
            yield in_language(BAD_KEY, language)
            return
        except anthropic.APIConnectionError as exc:  # also timeouts
            log.error("Claude non raggiungibile: %s", exc)
            self.last_call["claude_error"] = "network"
            yield in_language(NETWORK_ERROR, language)
            return
        except anthropic.APIStatusError as exc:
            log.error("Claude errore %s: %s", exc.status_code, exc.message)
            self.last_call["claude_error"] = exc.status_code
            yield in_language(API_ERROR, language)
            return
        finally:
            self._last_contact = self.clock()
            # Also on "basta" (the speaker cancels this generator): count what was generated.
            if stream is not None:
                self._record(model, final, stream, leg.parts)
        if final is not None and final.stop_reason == "refusal":
            log.info("Claude ha rifiutato la richiesta")
            if not "".join(leg.parts).strip():
                yield in_language(REFUSED, language)
            return
        leg.handoff = final is not None and any(
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == ASK_SONNET_TOOL["name"]
            for block in getattr(final, "content", None) or []
        )
        leg.complete = True

    def _record(self, model: str, final: Any, stream: Any, parts: list[str]) -> None:
        try:
            if final is not None:
                usage = Usage.from_api(final.usage, CACHE_TTL)
                cut = False
            else:
                snapshot = getattr(stream, "current_message_snapshot", None)
                api = Usage.from_api(snapshot.usage, CACHE_TTL) if snapshot else Usage()
                estimate = len("".join(parts)) // CHARS_PER_TOKEN
                usage = Usage(
                    input_tokens=api.input_tokens,
                    output_tokens=max(api.output_tokens, estimate),
                    cache_read_input_tokens=api.cache_read_input_tokens,
                    cache_write_5m_tokens=api.cache_write_5m_tokens,
                    cache_write_1h_tokens=api.cache_write_1h_tokens,
                )
                cut = True
            if usage == Usage():
                return  # nothing reached the API
            cost = self.tracker.record(model, usage, CACHE_TTL)
            log.info(
                "Claude %s: %d in, %d out, cache %d letti / %d scritti (TTL %s)%s",
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_input_tokens,
                usage.cache_creation_input_tokens,
                CACHE_TTL,
                tr(", interrotta (stima)") if cut else "",
            )
            # Haiku then Sonnet: the metrics show the last model and the sum of both calls.
            prev = self.last_call
            self.last_call.update(
                model=model,
                input_tokens=prev.get("input_tokens", 0) + usage.input_tokens,
                output_tokens=prev.get("output_tokens", 0) + usage.output_tokens,
                cache_read=prev.get("cache_read", 0) + usage.cache_read_input_tokens,
                cache_write=prev.get("cache_write", 0) + usage.cache_creation_input_tokens,
                cost=round(prev.get("cost", 0.0) + cost, 6),
            )
        except Exception:
            log.exception("Claude: spesa non registrata")


class _Leg:
    """One API call of an answer: its text, whether it finished, whether Haiku handed over."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.complete = False
        self.handoff = False


def make_client(api_key: str, timeout_s: float, keepalive_s: float = 60.0) -> Any:
    """The real async client: one quick retry, then a spoken error (better than a long wait).
    The connection stays open `keepalive_s` between requests (httpx closes it after 5 s)."""
    import anthropic
    import httpx2

    http = anthropic.DefaultAsyncHttpxClient(
        limits=httpx2.Limits(
            max_connections=10, max_keepalive_connections=2, keepalive_expiry=keepalive_s
        )
    )
    return anthropic.AsyncAnthropic(
        api_key=api_key, timeout=timeout_s, max_retries=1, http_client=http
    )
