"""Claude answers what the local router does not understand, streamed sentence by sentence.

Every call goes through the spending counter: `check()` before (80 % warning said aloud once a
day, nothing sent at 100 %) and `record()` after, also when the reply is cut short by "basta"
(the tokens already generated are billed by Anthropic)."""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from typing import Any

from claudio.brain.prompts import (
    API_ERROR,
    BAD_KEY,
    BUDGET_BLOCKED,
    BUDGET_WARNING,
    NETWORK_ERROR,
    NO_KEY,
    REFUSED,
    SYSTEM_PROMPT,
)
from claudio.config import ClaudeSettings, in_language
from claudio.cost.pricing import CacheTtl, Usage
from claudio.cost.tracker import BudgetState, CostTracker
from claudio.i18n import tr

log = logging.getLogger(__name__)

CACHE_TTL: CacheTtl = "5m"  # what a cache write would cost (caching itself arrives in 2b)
CHARS_PER_TOKEN = 3  # estimate for a reply cut short before the API sent its output count
_MARKDOWN = re.compile(r"[*#`]+")  # read aloud as "asterisco": never sent to the voice


class ClaudeBrain:
    """Ask Claude, keep a short history, count every token.

    `client` is an `anthropic.AsyncAnthropic` (or a fake in tests); None means no API key."""

    def __init__(
        self,
        client: Any | None,
        tracker: CostTracker,
        settings: ClaudeSettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.tracker = tracker
        self.settings = settings
        self.clock = clock
        self._history: deque[tuple[str, str]] = deque(maxlen=settings.history_turns or None)
        self._last_turn = -float("inf")
        self.last_call: dict[str, Any] = {}  # model, tokens, cost of the last call (metrics)

    def reset(self) -> None:
        self._history.clear()

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
        async with contextlib.aclosing(self._stream(text, language)) as stream:
            async for chunk in stream:
                yield chunk

    def _messages(self, text: str, language: str) -> list[dict[str, str]]:
        if self.clock() - self._last_turn > self.settings.history_reset_s:
            self._history.clear()  # a new conversation: old context costs and confuses
        messages: list[dict[str, str]] = []
        if self.settings.history_turns:
            for question, answer in self._history:
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": f"[{language}] {text}"})
        return messages

    def _params(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        s = self.settings
        params: dict[str, Any] = {
            "model": s.model,
            "max_tokens": s.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "output_config": {"effort": s.effort},
        }
        if not s.thinking:
            params["thinking"] = {"type": "disabled"}
        return params

    async def _stream(self, text: str, language: str) -> AsyncGenerator[str, None]:
        import anthropic

        model = self.settings.model
        messages = self._messages(text, language)
        parts: list[str] = []
        stream: Any = None
        final: Any = None
        start = self.clock()
        try:
            client: Any = self.client  # not None: ask() answered already without a key
            async with client.messages.stream(**self._params(messages)) as stream:
                async for chunk in stream.text_stream:
                    clean = _MARKDOWN.sub("", chunk)
                    if not parts:
                        self.last_call["claude_first_ms"] = round((self.clock() - start) * 1000)
                    parts.append(clean)
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
            # Also on "basta" (the speaker cancels this generator): count what was generated.
            if stream is not None:
                self._record(model, final, stream, parts)
        if final is not None and final.stop_reason == "refusal":
            log.info("Claude ha rifiutato la richiesta")
            if not "".join(parts).strip():
                yield in_language(REFUSED, language)
            return
        answer = "".join(parts).strip()
        if answer:
            self._history.append((messages[-1]["content"], answer))
            self._last_turn = self.clock()

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
            self.last_call.update(
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read=usage.cache_read_input_tokens,
                cache_write=usage.cache_creation_input_tokens,
                cost=round(cost, 6),
            )
        except Exception:
            log.exception("Claude: spesa non registrata")


def make_client(api_key: str, timeout_s: float) -> Any:
    """The real async client: one quick retry, then a spoken error (better than a long wait)."""
    import anthropic

    return anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=1)
