"""Claude answers what the local router does not understand, streamed sentence by sentence.

Haiku answers first. What needs reasoning may go to Sonnet, never without JenJo's yes: Haiku
proposes it with the `ask_sonnet` tool and Claudio asks "Vuoi che usi Sonnet?" (yes: Sonnet
answers the question; no: Haiku does), or JenJo asks for it ("pensaci bene").

Every call goes through the spending counter: `check()` before (80 % warning said aloud once a
day, nothing sent at 100 %) and `record()` after, also when the reply is cut short by "basta"
(the tokens already generated are billed by Anthropic).

With the PC tools (Phase 3), Haiku also acts: open, search, read windows, type... Several
tools may come in one reply, up to `[pc] max_rounds` round trips per request. A level 2-3
action stops the loop: Claudio says what it will do and asks; the next request is the answer
(the responder reports `awaiting`, so the assistant listens without the wake word).

v0.3.2: Haiku may also look at the screen (a small screenshot) and run one PowerShell command
per reply, always at level 3. A command on the deny list is refused before JenJo is asked;
the exact command is shown in the chat window (opened if needed), never read aloud. Before a
slow action Claudio says "Ci guardo" when Haiku said nothing yet."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from claudio.brain.prompts import (
    API_ERROR,
    ASK_SONNET_TOOL,
    BAD_KEY,
    BRIDGE,
    BRIDGE_RUN,
    BUDGET_BLOCKED,
    BUDGET_WARNING,
    NETWORK_ERROR,
    NO_KEY,
    OFFER,
    REFUSED,
    system_prompt,
)
from claudio.config import ClaudeSettings, PcSettings, in_language
from claudio.cost.pricing import CacheTtl, Usage
from claudio.cost.tracker import BudgetState, CostTracker
from claudio.i18n import tr
from claudio.router.normalizer import normalize, strip_wake_words
from claudio.safety.confirmation import (
    CANCELLED,
    LEVEL_SAFE,
    NOT_UNDERSTOOD,
    Answer,
    parse_answer,
)

if TYPE_CHECKING:
    from claudio.pc.tools import PcTools, ToolCall

# Tools that may take a while (pc/tools.py SLOW; repeated here: no import of the PC package).
SLOW_TOOLS = frozenset({"search_files", "read_window", "look_at_screen", "run_command"})

log = logging.getLogger(__name__)

CACHE_TTL: CacheTtl = "5m"  # the default TTL of `cache_control`: what a cache write costs
CHARS_PER_TOKEN = 3  # estimate for a reply cut short before the API sent its output count
_MARKDOWN = re.compile(r"[*#`]+")  # read aloud as "asterisco": never sent to the voice
OFFER_TTL_S = 60.0  # "Vuoi che usi Sonnet?" waits this long for its answer
ACTION_TTL_S = 60.0  # and so does "Scrivo ciao. Confermi?"
ACTIONS_NOTE = "[Claudio ran: "  # history note after a reply: "[Claudio ran: open_app ok]"
NEW_REQUEST_WORDS = 4  # an unclear answer longer than this is a new request, not a "no"
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
        tools: PcTools | None = None,
        pc: PcSettings | None = None,
        on_show: Callable[[], object] | None = None,
        personality: str = "",
    ) -> None:
        self.client = client
        self.tracker = tracker
        self.settings = settings
        self.clock = clock
        self.wall = wall  # local date and time, told to Claude with each request
        # Sonnet, when it has prices (the builder clears it otherwise): None = Haiku only.
        self.smart_model: str | None = settings.smart_model or None
        self._smart_phrases = _normalized(settings.smart_phrases)
        self._offer_phrases = _normalized(settings.offer_phrases)
        self._history: deque[tuple[str, str]] = deque(maxlen=settings.history_turns or None)
        self._last_turn = -float("inf")
        self._last_contact = -float("inf")  # last time the connection to the API was used
        self._warming: set[asyncio.Task[None]] = set()
        # Haiku proposed Sonnet and Claudio asked "Vuoi che usi Sonnet?": (messages, language,
        # when). The next request is the answer.
        self._offer: tuple[list[dict[str, Any]], str, float] | None = None
        self.last_call: dict[str, Any] = {}  # model, tokens, cost of the last call (metrics)
        self.tools = tools  # None: Claude answers but cannot act on the PC
        # Built once: the same text at every call is what the cache reads back.
        self.system = system_prompt(tools is not None, personality)
        self.pc = pc or PcSettings()
        self._action: _Action | None = None  # a level 2-3 tool call waiting for its yes
        self.on_show = on_show  # opens the chat window, where the exact command is written

    def reset(self) -> None:
        self._history.clear()

    @property
    def awaiting(self) -> bool:
        """An action asked "Confermi?": the next request is the answer."""
        return self._action is not None and self.clock() - self._action.asked < ACTION_TTL_S

    def cancel_pending(self) -> None:
        """No answer came (silence, "basta", a typed message): the action is not run."""
        if self._action is not None:
            names = ", ".join(c.name for c in self._action.calls)
            log.info("Conferma scaduta: %s non eseguito", names, extra={"route": "tool"})
            self._action = None

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
        return self.smart_model is not None and _says(text, language, self._smart_phrases)

    def wants_offer(self, text: str, language: str) -> bool:
        """JenJo asked for detail ("spiegami in dettaglio..."): Sonnet is proposed at once,
        without a Haiku call (Haiku at low effort often answered these itself)."""
        return self.smart_model is not None and _says(text, language, self._offer_phrases)

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
        start = self.clock()
        action = self._take_action()
        if action is not None and _new_request(text, action.level):
            # Not an answer but another request (said with the wake word after the window
            # closed): the action is dropped and the request answered.
            log.info("Nuova richiesta al posto della conferma: azione annullata")
            self._note_tools(action.calls, "refused")
            action = None
        if action is not None:
            async for chunk in self._confirm(action, text, start):
                yield chunk
            return
        offer = self._take_offer()
        if offer is not None:
            answer = parse_answer(text)
            if answer is not Answer.UNCLEAR:
                # The reply to "Vuoi che usi Sonnet?": the question it was about, not "sì".
                messages, language = offer
                if answer is Answer.YES:
                    log.info("Sonnet approvato")
                    self.last_call["escalated"] = "approved"
                    async for chunk in self._sonnet(messages, language, start):
                        yield chunk
                else:
                    log.info("Sonnet rifiutato: risponde Haiku")
                    async for chunk in self._haiku(messages, language, start, tools=False):
                        yield chunk
                return
        messages = self._messages(text, language)
        if self.wants_smart(text, language):  # JenJo asked for it: that is the approval
            log.info("Richiesta di ragionare: chiedo direttamente a Sonnet")
            self.last_call["escalated"] = "asked"
            async for chunk in self._sonnet(messages, language, start):
                yield chunk
            return
        if self.settings.escalate and self.wants_offer(text, language):
            log.info("Richiesta dettagliata: propongo Sonnet")
            self._offer = (messages, language, self.clock())
            self.last_call["escalated"] = "offered"
            yield in_language(OFFER, language)
            return
        turn = _Turn(messages[-1]["content"])
        async with contextlib.aclosing(
            self._loop(messages, language, start, turn, 0, offer_from=messages)
        ) as haiku:
            async for chunk in haiku:
                yield chunk

    async def _loop(
        self,
        messages: list[dict[str, Any]],
        language: str,
        start: float,
        turn: _Turn,
        round_no: int,
        offer_from: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Haiku, with its tools, until it answers in words: each round runs the level-1
        calls and sends their results back; a level 2-3 call stops here and asks first."""
        while True:
            last = self.tools is None or round_no >= self.pc.max_rounds - 1
            leg = _Leg()
            async with contextlib.aclosing(
                self._stream(self._params(messages, last_round=last), language, start, leg)
            ) as haiku:
                async for chunk in haiku:
                    yield chunk
            turn.parts.extend(leg.parts)
            if leg.handoff and not leg.calls and self.smart_model and offer_from is not None:
                # Never Sonnet without a yes (JenJo, 23/10): ask, keep the question for the answer.
                log.info("Haiku propone Sonnet: chiedo conferma")
                self._offer = (offer_from, language, self.clock())
                self.last_call["escalated"] = "offered"
                yield _space(turn) + in_language(OFFER, language)
                return
            if leg.handoff and leg.calls:  # Sonnet in the middle of an action: not now
                leg.content = [b for b in leg.content if b.get("name") != ASK_SONNET_TOOL["name"]]
            if not leg.complete or not leg.calls or self.tools is None:
                turn.complete = leg.complete
                self._remember(turn)
                return
            messages = [*messages, {"role": "assistant", "content": leg.content}]
            refused = {c.id: why for c in leg.calls if (why := self.tools.refusal(c, leg.calls))}
            active = [c for c in leg.calls if c.id not in refused]
            await self.tools.prepare(active)
            confirm = [c for c in active if self.tools.level(c) > LEVEL_SAFE]
            if confirm:
                level = max(self.tools.level(c) for c in confirm)
                ask = await self.tools.question(confirm, language)
                self._action = _Action(messages, leg.calls, turn, language, level, round_no)
                self._action.refused = refused
                self._action.asked = self.clock()
                self._show(confirm)
                self._note_tools(confirm, "ask")
                log.info(
                    "Conferma richiesta (livello %d): %s",
                    level,
                    ", ".join(c.brief() for c in confirm),
                    extra={"route": "confirm", "level": level},
                )
                said = _space(turn) + ask
                turn.parts.append(said)
                yield said
                return
            slow = any(c.name in SLOW_TOOLS for c in active)
            if self.pc.bridge and slow and not "".join(leg.parts).strip():
                # The trailing space ends the sentence: the voice says it now.
                said = _space(turn) + in_language(BRIDGE, language) + " "
                turn.parts.append(said)
                yield said
            results = await self._run_calls(leg.calls, refused, turn)
            messages = [*messages, {"role": "user", "content": results}]
            round_no += 1

    async def _confirm(self, action: _Action, text: str, start: float) -> AsyncGenerator[str, None]:
        """The answer to "Confermi?": yes runs the calls and Haiku goes on; anything else
        cancels (a TV in the background never confirms an action)."""
        answer = parse_answer(text, action.level)
        self.last_call["confirm"] = answer.value
        if answer is not Answer.YES:
            reply = CANCELLED if answer is Answer.NO else NOT_UNDERSTOOD
            log.info("Conferma %s: azione non eseguita", answer.value, extra={"route": "tool"})
            self._note_tools(action.calls, "refused")
            said = in_language(reply, action.language)
            action.turn.parts.append(" " + said)
            action.turn.complete = True
            self._remember(action.turn)
            yield said
            return
        log.info("Conferma data (livello %d): eseguo", action.level)
        if self.pc.bridge and any(
            c.name in SLOW_TOOLS and c.id not in action.refused for c in action.calls
        ):
            said = in_language(BRIDGE_RUN, action.language) + " "  # a new reply: no lead space
            action.turn.parts.append(_space(action.turn) + said)
            yield said
        results = await self._run_calls(action.calls, action.refused, action.turn)
        messages = [*action.messages, {"role": "user", "content": results}]
        async with contextlib.aclosing(
            self._loop(messages, action.language, start, action.turn, action.round_no + 1)
        ) as haiku:
            async for chunk in haiku:
                yield chunk

    async def _run_calls(
        self,
        calls: list[ToolCall],
        refused: dict[str, str] | None = None,
        turn: _Turn | None = None,
    ) -> list[dict[str, Any]]:
        """One tool_result per call, in order; a refused call is answered without running."""
        assert self.tools is not None
        refused = refused or {}
        results = []
        for call in calls:
            if call.id in refused:
                self._note_tools([call], "refused")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": refused[call.id],
                        "is_error": True,
                    }
                )
                continue
            outcome = await self.tools.run(call)
            self._note_tools([call], "error" if outcome.error else "ok", outcome.ms)
            if turn is not None:
                turn.done.append(f"{call.name} {'failed' if outcome.error else 'ok'}")
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": outcome.content(),
            }
            if outcome.error:
                block["is_error"] = True
            results.append(block)
        return results

    def _note_tools(self, calls: list[ToolCall], outcome: str, ms: int | None = None) -> None:
        """For metrics.jsonl and conversazione.md: what was done, how it went, how long."""
        done = self.last_call.setdefault("tools", [])
        done.extend(f"{c.brief()} {outcome}" for c in calls)
        if ms is not None:
            self.last_call["tool_ms"] = self.last_call.get("tool_ms", 0) + ms

    def _show(self, calls: list[ToolCall]) -> None:
        """A command waits for "confermo": its exact text goes to the chat (with the spoken
        question), and the chat window opens so JenJo can read it."""
        assert self.tools is not None
        shown = self.tools.shown(calls)
        if not shown:
            return
        self.last_call["shown"] = shown
        log.info("Comando proposto: %s", shown, extra={"route": "confirm"})
        if self.on_show is not None:
            try:
                self.on_show()
            except Exception:
                log.exception("Finestra della chat non aperta")

    def _take_action(self) -> _Action | None:
        action, self._action = self._action, None
        if action is None or self.clock() - action.asked > ACTION_TTL_S:
            return None
        return action

    def _take_offer(self) -> tuple[list[dict[str, Any]], str] | None:
        offer, self._offer = self._offer, None
        if offer is None or self.clock() - offer[2] > OFFER_TTL_S:
            return None
        return offer[0], offer[1]

    async def _haiku(
        self, messages: list[dict[str, Any]], language: str, start: float, tools: bool = True
    ) -> AsyncGenerator[str, None]:
        leg = _Leg()
        async with contextlib.aclosing(
            self._stream(self._params(messages, tools), language, start, leg)
        ) as haiku:
            async for chunk in haiku:
                yield chunk
        self._remember(_Turn(messages[-1]["content"], leg.parts, leg.complete))

    async def _sonnet(
        self, messages: list[dict[str, Any]], language: str, start: float
    ) -> AsyncGenerator[str, None]:
        if self.smart_model is None:
            async for chunk in self._haiku(messages, language, start, tools=False):
                yield chunk
            return
        leg = _Leg()
        async with contextlib.aclosing(
            self._stream(self._smart_params(messages), language, start, leg, "smart_first_ms")
        ) as sonnet:
            async for chunk in sonnet:
                yield chunk
        self._remember(_Turn(messages[-1]["content"], leg.parts, leg.complete))

    def _remember(self, turn: _Turn) -> None:
        """Only the words go into the history: tool calls and window text stay out of it
        (they cost tokens at every later turn, and the history is plain text). A short note of
        the actions that really ran is added: without it Haiku, reading only its own words,
        later "admitted" that a command it had run never ran (24/10)."""
        answer = "".join(turn.parts).strip()
        if turn.done:
            answer = f"{answer} {ACTIONS_NOTE}{', '.join(turn.done)}]".strip()
        if turn.complete and answer:
            self._history.append((turn.question, answer))
            self._last_turn = self.clock()

    def _messages(self, text: str, language: str) -> list[dict[str, Any]]:
        if self.clock() - self._last_turn > self.settings.history_reset_s:
            self._history.clear()  # a new conversation: old context costs and confuses
        messages: list[dict[str, Any]] = []
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
        self, model: str, max_tokens: int, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": self.system,
            "messages": messages,
        }
        if self.settings.cache:
            # Automatic caching: the marker goes on the last block, so the next turn reads
            # this whole conversation from the cache (below 512 tokens nothing is cached).
            params["cache_control"] = {"type": "ephemeral"}
        return params

    def _params(
        self, messages: list[dict[str, Any]], tools: bool = True, last_round: bool = False
    ) -> dict[str, Any]:
        """The tool list is always the same (ask_sonnet first, then the PC tools): it is the
        start of the cached prefix. On the last round Haiku must answer in words."""
        s = self.settings
        max_tokens = max(s.max_tokens, self.pc.max_tokens) if self.tools else s.max_tokens
        params = self._common(s.model, max_tokens, messages)
        params["output_config"] = {"effort": s.effort}
        if not s.thinking:
            params["thinking"] = {"type": "disabled"}
        listed: list[dict[str, Any]] = []
        if tools and s.escalate and self.smart_model is not None:
            listed.append(ASK_SONNET_TOOL)
        if tools and self.tools is not None:
            listed.extend(self.tools.definitions())
        if listed:
            params["tools"] = listed
            if last_round and self.tools is not None:
                params["tool_choice"] = {"type": "none"}
        return params

    def _smart_params(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
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
        leg.read(final)
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
    """One API call of an answer: its text, whether it finished, whether Haiku handed over,
    the PC tools it called and its content (sent back with the tool results)."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.complete = False
        self.handoff = False
        self.calls: list[ToolCall] = []
        self.content: list[dict[str, Any]] = []

    def read(self, final: Any) -> None:
        """Calls are kept only when the reply stopped to call them: a reply cut by max_tokens
        may hold a call with half its arguments."""
        from claudio.pc.tools import ToolCall

        for block in getattr(final, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                self.content.append({"type": "text", "text": block.text})
            elif kind == "tool_use":
                args = dict(getattr(block, "input", None) or {})
                self.content.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", ""),
                        "name": block.name,
                        "input": args,
                    }
                )
                if block.name == ASK_SONNET_TOOL["name"]:
                    self.handoff = True
                elif getattr(final, "stop_reason", None) == "tool_use":
                    self.calls.append(ToolCall(getattr(block, "id", ""), block.name, args))


class _Turn:
    """One request across its rounds: the question and everything said in reply."""

    def __init__(self, question: str, parts: list[str] | None = None, complete: bool = False):
        self.question = question
        self.parts = list(parts or [])
        self.complete = complete
        self.done: list[str] = []  # tools that ran in this request, for the history note


class _Action:
    """Tool calls waiting for JenJo's yes, with the conversation to continue from."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        calls: list[ToolCall],
        turn: _Turn,
        language: str,
        level: int,
        round_no: int,
    ) -> None:
        self.messages = messages
        self.calls = calls
        self.turn = turn
        self.language = language
        self.level = level
        self.round_no = round_no
        self.refused: dict[str, str] = {}  # calls answered with an error, never run
        self.asked = -float("inf")


def _new_request(text: str, level: int) -> bool:
    """A phrase that is neither yes nor no and longer than any answer: a new request."""
    if parse_answer(text, level) is not Answer.UNCLEAR:
        return False
    return len(strip_wake_words(normalize(text, drop_fillers=False).split())) > NEW_REQUEST_WORDS


def _space(turn: _Turn) -> str:
    said = "".join(turn.parts)
    return "" if not said or said[-1].isspace() else " "


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


def _normalized(phrases: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        lang: [normalize(p, drop_fillers=False) for p in items if p.strip()]
        for lang, items in phrases.items()
    }


def _says(text: str, language: str, phrases: dict[str, list[str]]) -> bool:
    """One of the phrases is in `text`: those of its language first, then the others."""
    said = f" {normalize(text, drop_fillers=False)} "
    ordered = phrases.get(language, []) + [
        p for lang, ps in phrases.items() if lang != language for p in ps
    ]
    return any(f" {p} " in said for p in ordered)
