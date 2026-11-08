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
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Iterable, Sequence
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from claudio.brain.promises import promises_action
from claudio.brain.prompts import (
    API_ERROR,
    ASK_SONNET_TOOL,
    BAD_KEY,
    BRIDGE,
    BRIDGE_RUN,
    BUDGET_BLOCKED,
    BUDGET_WARNING,
    CUT_AT_LIMIT,
    NETWORK_ERROR,
    NO_KEY,
    NOT_RUN_CUT,
    OFFER,
    REFUSED,
    system_prompt,
)
from claudio.config import ClaudeSettings, PcSettings, in_language
from claudio.cost.pricing import CacheTtl, Usage
from claudio.cost.tracker import BudgetState, CostTracker
from claudio.i18n import tr
from claudio.intake import Hearing
from claudio.memory.prompt import without_recalled
from claudio.protocols import CUT_SHORT
from claudio.router.normalizer import normalize
from claudio.safety.confirmation import (
    ACTION_TTL_S,
    ASK_AGAIN,
    ASK_AGAIN_EXPLICIT,
    ASK_LEVEL_2,
    ASK_LEVEL_3,
    CANCELLED,
    LEVEL_EXPLICIT,
    LEVEL_SAFE,
    NOT_UNDERSTOOD,
    Answer,
    has_question_mark,
    parse_answer,
)
from claudio.tts.sentence_splitter import joined_reply

if TYPE_CHECKING:
    from claudio.memory.prompt import MemoryPrompt
    from claudio.pc.tools import PcTools, ToolCall

# Tools that may take a while (pc/tools.py SLOW; repeated here: no import of the PC package).
SLOW_TOOLS = frozenset({"search_files", "read_window", "look_at_screen", "run_command"})
# Tools that bring outside text in (pc/tools.py OUTSIDE_TEXT, repeated for the same reason).
# open_app, open_path and type_text report the window's text back to Claude.
OUTSIDE_TEXT_TOOLS = frozenset({*SLOW_TOOLS, "list_windows", "open_app", "open_path", "type_text"})

log = logging.getLogger(__name__)

CACHE_TTL: CacheTtl = "5m"  # the default TTL of `cache_control`: what a cache write costs
CHARS_PER_TOKEN = 3  # estimate for a reply cut short before the API sent its output count
_MARKDOWN = re.compile(r"[*#`]+")  # read aloud as "asterisco": never sent to the voice
OFFER_TTL_S = 60.0  # "Vuoi che usi Sonnet?" waits this long for its answer
LOCAL_NOTE = "[Claudio answered: its own command, not Claude]"  # after a local reply
ACTIONS_NOTE = "[Claudio ran: "  # history note after a reply: "[Claudio ran: open_app ok]"
# ... and for a call that never ran: "[Claudio did not run: type_text, cut at the token limit]"
NOT_RUN_NOTE = "[Claudio did not run: "
CUT_AT_LIMIT_WHY = "cut at the token limit"
STOPPED_WHY = "a new request came first"
NOT_CONFIRMED_WHY = "JenJo did not confirm it"
NOT_HEARD_WHY = "the question was not heard"
# The result of a call that a new request stopped: every tool_use still needs one.
STOPPED_RESULT = "Not run: JenJo said something else over the reply."
WARM_EVERY_S = 15.0  # an idle connection older than this may be closed: open a fresh one
RTT_FRESH_S = 30.0  # a warm-up's round trip belongs to a request asked this soon after it
# The bridges, in every language: said aloud, never in the history (Haiku copied them, 05/11).
_BRIDGES = frozenset(b.strip() for b in (*BRIDGE.values(), *BRIDGE_RUN.values()))
# What the history keeps of a reply cut short (read by Claude, never said aloud).
CUT_NOTE = "[cut off: JenJo interrupted here]"
CUT_BEFORE_NOTE = "[no answer: JenJo interrupted before it started]"
# How each note opens. Only the code may write them (`_read_outside` reads them, a safety level
# depends on it), yet the model imitates them (01/11 23:32, with no tool called): _NoteFilter.
_NOTE_OPENINGS = tuple(
    note[: note.index(":") + 1].lower()
    for note in (ACTIONS_NOTE, NOT_RUN_NOTE, CUT_NOTE, CUT_BEFORE_NOTE, LOCAL_NOTE)
)
# Haiku says what it is about to do, then calls the tool, then says what it did (live test of
# 07/11): the first sentences wait until the call is known. Not model facts: not in config.
HOLD_SENTENCES = 2  # held until a sentence after these begins
HOLD_MAX_CHARS = 200  # or until the text is longer: a story does not wait for its end
_SENTENCE_END = re.compile(r"[.!?]+\s+")
# A note ends at "]", at a new line, or at a sentence end followed by a space or "]" (a "." inside
# "report.pdf" is no end). A terminator at the end of a chunk stays undecided: the note goes on.
_NOTE_END = re.compile(r"\]|[.!?]+(?:\]|(?=\s))|\n")
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")  # not strftime: Windows' locale


class StreamEnd(Enum):
    """How one API call ended. `_stream` and `_loop` decide from this alone."""

    DONE = "done"
    CUT_WORDS = "cut_words"  # max_tokens inside the words (or only an ask_sonnet call)
    CUT_CALL = "cut_call"  # max_tokens with a tool_use block: its arguments may be half
    REFUSED = "refused"
    PROMISED = "promised"  # end_turn with tools offered, no call, and words that promise one
    ERROR = "error"  # the request or the stream failed: a fixed sentence was said


def stream_end(
    final: Any, *, tools_offered: bool = False, said: str = "", ran: bool = False
) -> StreamEnd:
    """The end of a finished message. `final` may be None (nothing to read): DONE.
    `said`: the words Claudio spoke; with `tools_offered` they may be a promise without a call
    (`ran`: a tool already ran in this request)."""
    reason = getattr(final, "stop_reason", None)
    if reason == "refusal":
        return StreamEnd.REFUSED
    if reason == "max_tokens":
        return StreamEnd.CUT_CALL if _called_tools(final) else StreamEnd.CUT_WORDS
    if tools_offered and reason == "end_turn" and not _has_tool_use(final):
        return StreamEnd.PROMISED if promises_action(said, ran) else StreamEnd.DONE
    return StreamEnd.DONE


def _has_tool_use(final: Any) -> bool:
    blocks = getattr(final, "content", None) or []
    return any(getattr(block, "type", None) == "tool_use" for block in blocks)


def _called_tools(final: Any) -> list[str]:
    """Names of the tool_use blocks of a message, in order, without ask_sonnet."""
    return [
        block.name
        for block in getattr(final, "content", None) or []
        if getattr(block, "type", None) == "tool_use" and block.name != ASK_SONNET_TOOL["name"]
    ]


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
        memory: MemoryPrompt | None = None,
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
        self._no_offer_phrases = _normalized(settings.no_offer_phrases)
        self._history: deque[tuple[str, str]] = deque(maxlen=settings.history_turns or None)
        self._last_turn = -float("inf")
        self._last_contact = -float("inf")  # last time the connection to the API was used
        self._warming: set[asyncio.Task[None]] = set()
        # The warm-up's round trip for the next request: (ms, new connection, started at).
        self._rtt: tuple[int, bool, float] | None = None
        self._asked_at = -float("inf")  # when the last request went out
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
        # How sure Whisper was of the next request: set before every reply, read once.
        self._hearing = Hearing()
        # What JenJo asked Claudio to remember: a block after the system prompt, cached, and
        # the facts that match a request (None: no long-term memory).
        self.memory = memory
        self._memory_block = ""  # the memories as this request's rounds see them
        # The turn whose words are being spoken: where a cut and a stop are written.
        self._turn: _Turn | None = None

    def reset(self) -> None:
        self._history.clear()

    def heard(self, hearing: Hearing) -> None:
        """Whisper was unsure of the request about to be asked (and what else it heard):
        told to Claude in the request's tag."""
        self._hearing = hearing

    @property
    def awaiting(self) -> bool:
        """An action asked "Confermi?": the next request is the answer."""
        return self._action is not None and self.clock() - self._action.asked < ACTION_TTL_S

    def cancel_pending(self, cut: bool = False) -> None:
        """A request is over (it runs at the end of every one, not only after a cut): the
        action Claude asked "Confermi?" for is not run. The Sonnet offer ("Lo uso?") is dropped
        only when `cut`, JenJo stopped the reply ("basta"): otherwise it waits for the "sì"."""
        if self._action is not None:
            action, self._action = self._action, None
            names = ", ".join(c.name for c in action.calls)
            log.info("Conferma scaduta: %s non eseguito", names, extra={"route": "tool"})
            self._drop(action, NOT_CONFIRMED_WHY)
        if cut and self._offer is not None:
            log.info("Offerta di Sonnet scaduta: interrotta", extra={"route": "tool"})
            self._offer = None

    def note_local(self, question: str, answer: str) -> None:
        """A local command answered `question`: Claude's next request sees the turn, marked as
        not its own. The note holds no ACTIONS_NOTE, so `_read_outside` never reads it."""
        if not self.settings.history_turns:
            return  # no history is sent, and the deque would not be bounded
        self._forget_if_idle()
        self._history.append((question, f"{answer} {LOCAL_NOTE}"))
        self._last_turn = self.clock()

    def stop_acting(self) -> None:
        """A new request was said over this reply. Its stream is still read, for "Dicevo", but
        from now on no call starts and no question is armed; a call already running finishes.
        It is the stop of this turn only: a request asked next has a turn of its own."""
        if self._turn is not None:
            self._turn.stopped = True

    def reply_cut(self, played: Sequence[str], handle: object | None = None) -> object | None:
        """The reply was cut: `played` is what the voice started saying of it, and that is all
        the history keeps of it (also the words still to arrive); the replies of the same turn
        before it stay. A question or an offer whose words were not among them is not waiting
        for an answer. Returns the cut turn: passed back as `handle` later (after "Dicevo",
        when other turns came since), it rewrites that turn and no other."""
        turn = handle if isinstance(handle, _Turn) else self._turn
        if turn is None:
            return None
        before = "".join(turn.parts[: turn.reply_from]).strip()
        now = " ".join(p.strip() for p in played if p.strip() and p.strip() not in _BRIDGES)
        turn.heard = f"{before} {now}".strip()
        self._disarm_unheard(turn, now)
        index = self._index_of(turn.written)
        if index is not None:  # else it is not written yet, or the window dropped it
            self._history[index] = turn.written = (turn.question, _cut_line(turn))
        return turn

    def dropped_action(self, handle: object | None) -> bool:
        """The cut turn `handle` names has tools that never ran, for any reason (stopped, not
        heard, cut at the limit): what it said ends in a promise or a question nobody answers."""
        return isinstance(handle, _Turn) and bool(handle.not_run)

    def _disarm_unheard(self, turn: _Turn, heard: str) -> None:
        if not turn.asked or _said_in(turn.asked, heard):
            return
        action = self._action
        if action is not None and action.turn is turn:
            self._action = None
            self._drop(action, NOT_HEARD_WHY)
        if self._offer is not None and turn is self._turn:  # the offer is this reply's
            log.info("Offerta di Sonnet non sentita: annullata", extra={"route": "tool"})
            self._offer = None

    def _drop(self, action: _Action, why: str) -> None:
        """An armed action will not run: its turn goes into the history with the note."""
        turn = action.turn
        turn.not_run_why = why
        turn.note_not_run(c.name for c in action.calls)
        turn.complete = True
        self._remember(turn)

    def _index_of(self, entry: tuple[str, str] | None) -> int | None:
        """Where `entry` is in the history now (the window may have dropped it)."""
        return next((i for i, e in enumerate(self._history) if e is entry), None)

    def warm_up(self) -> None:
        """JenJo is about to speak (wake word): open the HTTPS connection now, for free (a
        models lookup, no tokens), so the request does not pay the TCP and TLS handshakes."""
        if self.client is None or self.clock() - self._last_contact < WARM_EVERY_S:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        # Idle past the keep-alive, the connection is a new one: then the round trip also
        # pays the handshakes, which the request after it no longer does.
        new = self.clock() - self._last_contact > self.settings.keepalive_s
        self._last_contact = self.clock()
        self._rtt = None  # an older, unused figure is not the next request's
        task = loop.create_task(self._warm(new))
        self._warming.add(task)
        task.add_done_callback(self._warming.discard)

    async def _warm(self, new: bool = False) -> None:
        start = self.clock()
        try:
            client: Any = self.client
            await client.models.retrieve(self.settings.model)
        except Exception as exc:  # only a head start: the request itself reports problems
            log.debug("Claude: connessione anticipata non riuscita: %s", exc)
            return
        # No model runs for it: the network part of the next request's wait (~220 ms on
        # 30/10, against a first token at ~470-610 ms). Not when the request has already
        # gone out meanwhile: it would be reported with the one after.
        if self._asked_at < start:
            self._rtt = (_ms_since(start, self.clock()), new, start)

    def wants_smart(self, text: str, language: str) -> bool:
        """JenJo asked for Sonnet in words ("pensaci bene...")."""
        return self.smart_model is not None and _says(text, language, self._smart_phrases)

    def wants_offer(self, text: str, language: str) -> bool:
        """JenJo asked for detail ("spiegami in dettaglio..."): Sonnet is proposed at once,
        without a Haiku call (Haiku at low effort often answered these itself)."""
        return (
            self.smart_model is not None
            and _says(text, language, self._offer_phrases)
            and not self.never_offer(text, language)
        )

    def never_offer(self, text: str, language: str) -> bool:
        """A story, a joke, a poem: Haiku is good enough, even for a long one (JenJo, 25/10)."""
        return _says(text, language, self._no_offer_phrases)

    async def ask(self, text: str, language: str) -> AsyncGenerator[str, None]:
        """Claude's reply to `text`, in `language`, as it arrives. Problems become one short
        spoken sentence: Claudio never goes silent and never crashes because of the API."""
        self.last_call = {}
        self._asked_at = self.clock()
        rtt, self._rtt = self._rtt, None
        if rtt is not None and self._asked_at - rtt[2] <= RTT_FRESH_S:
            self.last_call["api_rtt_ms"] = rtt[0]
            if rtt[1]:
                self.last_call["api_rtt_new_connection"] = True
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
        reply = self._route(text, language, start)
        async with contextlib.aclosing(reply) as stream:
            async for chunk in stream:
                yield chunk

    def _route(self, text: str, language: str, start: float) -> AsyncGenerator[str, None]:
        """Who answers `text`, not started yet: what waits for an answer comes first (an action,
        then an offer); anything else is a new request."""
        action = self._pending_action(text)
        if action is not None:
            return self._confirm(action, text, start)
        answered = self._answered_offer(text)
        if answered is not None:
            # The reply to "Vuoi che usi Sonnet?": the question it was about, not "sì".
            messages, language, answer = answered
            if answer is Answer.YES:
                log.info("Sonnet approvato")
                self.last_call["escalated"] = "approved"
                return self._sonnet(messages, language, start)
            log.info("Sonnet rifiutato: risponde Haiku")
            return self._haiku(messages, language, start, tools=False)
        messages = self._messages(text, language)
        if self.tools is not None:
            # Text read from a window or a file earlier in this conversation may still be in
            # Claude's own words: a memory it saves then needs JenJo's yes too.
            self.tools.begin_request(self._read_outside())
        if self.wants_smart(text, language):  # JenJo asked for it: that is the approval
            log.info("Richiesta di ragionare: chiedo direttamente a Sonnet")
            self.last_call["escalated"] = "asked"
            return self._sonnet(messages, language, start)
        if self.settings.escalate and self.wants_offer(text, language):
            log.info("Richiesta dettagliata: propongo Sonnet")
            return self._propose(messages, language)
        turn = _Turn(messages[-1]["content"])
        return self._loop(messages, language, start, turn, 0, offer_from=messages)

    def _pending_action(self, text: str) -> _Action | None:
        """The action `text` answers: None when none waits. Whatever is not a clear yes or no
        is asked about once more, then cancelled; a phrase with the name never gets here (the
        responder cancels the action and answers it as a new request)."""
        return self._take_action()

    def repeats_waiting(self, text: str) -> bool:
        """The waiting action is said again as a command ("Claudio, spegni"): a yes."""
        return self._action is not None and self._repeats(self._action, text)

    def _repeats(self, action: _Action, text: str) -> bool:
        """The waiting system command said again ("Claudio, spegni", 02/11: 4 phrases and ~25 s
        to switch off): a yes, where a plain yes is enough (level 3 wants its "confermo"). A
        refusal stays one ("non spegnerti"), a question is no yes ("spegniti?")."""
        assert self.tools is not None
        return (
            action.level < LEVEL_EXPLICIT
            and parse_answer(text, action.level) is Answer.UNCLEAR
            and not has_question_mark(text)
            and self.tools.repeats(action.calls, text, action.language)
        )

    def _answered_offer(self, text: str) -> tuple[list[dict[str, Any]], str, Answer] | None:
        """The offer `text` answers (its messages and language) and the answer: None when none
        waits, it cannot be sent any more or `text` is not a yes or a no."""
        offer = self._take_offer()
        if offer is None:
            return None
        if self.memory is not None and self.memory.forget_pending:
            # A memory was forgotten after "Lo uso?": the question kept the old memories
            # (and a history that may quote them), so it is not sent again.
            log.info("Ricordo dimenticato dopo la proposta di Sonnet: proposta annullata")
            return None
        answer = parse_answer(text)
        if answer is Answer.UNCLEAR:
            return None
        return offer[0], offer[1], answer

    async def _propose(
        self,
        messages: list[dict[str, Any]],
        language: str,
        lead: str = "",
        turn: _Turn | None = None,
    ) -> AsyncGenerator[str, None]:
        """Ask "Vuoi che usi Sonnet?" and keep the question: the next request is the answer."""
        question = in_language(OFFER, language)
        if turn is None:
            turn = self._turn = _Turn(messages[-1]["content"])
        turn.asked = question
        self._offer = (messages, language, self.clock())
        self.last_call["escalated"] = "offered"
        yield lead + question

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
        # Only within this reply: after a yes the turn holds the question, already spoken.
        # True = words the voice really kept (a fragment dropped by a stream retry is not).
        said_here = False
        self._turn = turn
        this_reply = turn.reply_from = len(
            turn.parts
        )  # after a yes the turn holds the question already asked
        streaming: _Leg | None = None  # the leg whose words are not in `turn` yet
        try:
            while True:
                last = round_no >= self.pc.max_rounds - 1
                params = self._params(messages, last_round=last)
                leg = streaming = _Leg(turn)
                async with contextlib.aclosing(self._stream(params, language, start, leg)) as haiku:
                    async for chunk in haiku:
                        yield chunk
                said_here = said_here or bool("".join(leg.parts).strip())
                turn.parts.extend(leg.parts)
                streaming = None
                if turn.stopped and (leg.calls or leg.cut_tools or leg.handoff):
                    self._end_stopped(turn, leg)
                    return
                if leg.end is StreamEnd.CUT_CALL:
                    # At once: a "basta" during the retry finds the right note. It stays until the
                    # call really runs (`_Turn.ran`), also after a "no" or a "basta" before that.
                    turn.note_not_run(leg.cut_tools)
                    cut, leg = leg, _Leg(turn)
                    streaming = leg
                    async with contextlib.aclosing(
                        self._retry_cut(params, cut, leg, turn, language, start, said_here)
                    ) as retry:
                        async for chunk in retry:
                            said_here = said_here or bool(chunk.strip())
                            yield chunk
                    turn.parts.extend(leg.parts)
                    streaming = None
                    if turn.stopped:
                        self._end_stopped(turn, leg)
                        return
                    if not leg.calls:
                        if said := self._end_not_run(turn, language):
                            yield said
                        return
                    leg.held = cut.held  # the words before the cut call wait for its question
                if reply := self._handed_over(leg, offer_from, turn, language, start):
                    async for chunk in reply:
                        yield chunk
                    return
                if leg.handoff and not leg.calls and leg.held.strip():
                    # Nobody proposes Sonnet (after a yes): the words are all there is to say.
                    said, leg.held = leg.held, ""
                    self._mark_word({}, self.clock(), start, "claude_first_ms")
                    turn.parts.append(said)
                    yield said
                if not leg.complete or not leg.calls or self.tools is None:
                    turn.complete = leg.complete
                    self._remember(turn)
                    return
                messages = [*messages, {"role": "assistant", "content": leg.content}]
                refused = {
                    c.id: why for c in leg.calls if (why := self.tools.refusal(c, leg.calls))
                }
                active = [c for c in leg.calls if c.id not in refused]
                await self.tools.prepare(active)
                confirm = [c for c in active if self.tools.level(c) > LEVEL_SAFE]
                # Nothing said in this reply before the question: it reads the action out.
                read_out = not leg.held.strip() and not "".join(turn.parts[this_reply:]).strip()
                ask = (
                    await self.tools.question(confirm, language, read_out=read_out)
                    if confirm
                    else ""
                )
                if turn.stopped:  # it came during those two waits
                    self._end_stopped(turn, leg)
                    return
                if confirm:
                    yield self._ask_first(
                        ask, messages, leg, confirm, refused, turn, language, round_no
                    )
                    return
                if leg.held:  # a free action says only what it did, after it
                    log.debug("Parole prima dell'azione trattenute e scartate: %d", len(leg.held))
                if said := self._before_tools(turn, leg, active, said_here, language):
                    said_here = True
                    yield said
                alone = not turn.done and not leg.handoff  # nothing else in this request
                results, own = await self._run_calls(leg.calls, language, refused, turn)
                if own is not None and alone:
                    if said := self._end_with(turn, own):
                        yield said
                    return
                messages = [*messages, {"role": "user", "content": results}]
                round_no += 1
        except (GeneratorExit, asyncio.CancelledError):
            if streaming is not None:
                turn.parts.extend(streaming.parts)
            self._remember_cut(turn)  # "continua la storia" must know what was cut
            raise

    async def _retry_cut(
        self,
        params: dict[str, Any],
        cut: _Leg,
        leg: _Leg,
        turn: _Turn,
        language: str,
        start: float,
        said_here: bool,
    ) -> AsyncGenerator[str, None]:
        """Ask again, once, for the first tool of the `cut` reply: same request, more
        tokens, that tool forced, the model's own words thrown away. Fills `leg`; says
        a sentence only to cover the wait or for an error (the caller says the rest)."""
        if self.settings.model not in self.pc.retry_models:
            self.last_call["cut_retry"] = "skipped"  # Sonnet and Opus answer 400 to tool_choice
            return
        if said_here:
            gap = _space(turn)  # ends the sentence: it is said while the retry runs
            turn.parts.append(gap)
            if gap:
                yield gap
        elif self.pc.bridge:
            yield in_language(BRIDGE, language) + " "  # not in turn.parts: Haiku would copy it
        retry = {
            **params,
            "max_tokens": self.pc.retry_max_tokens,
            "tool_choice": {"type": "tool", "name": cut.cut_tools[0]},
            "timeout": self.pc.retry_timeout_s,
        }
        async with contextlib.aclosing(
            self._stream(retry, language, start, leg, silent=True)
        ) as stream:
            async for chunk in stream:
                yield chunk
        if leg.end in (StreamEnd.ERROR, StreamEnd.REFUSED):
            self.last_call["cut_retry"] = "error"
        elif leg.calls:
            self.last_call["cut_retry"] = "ok"
        else:
            self.last_call["cut_retry"] = "cut"

    def _end_not_run(self, turn: _Turn, language: str) -> str:
        """The retry gave no whole call: say so (an error already said its own sentence), keep
        the note and end the reply."""
        self.last_call.setdefault("tools", []).extend(f"{n}() not_run_cut" for n in turn.not_run)
        said = ""
        if self.last_call.get("cut_retry") != "error":
            said = _space(turn) + in_language(NOT_RUN_CUT, language)
            turn.parts.append(said)
        turn.complete = True
        self._remember(turn)
        return said

    def _end_stopped(self, turn: _Turn, leg: _Leg) -> None:
        """`stop_acting` came: what `leg` called does not run, nothing is armed, the reply ends."""
        self._not_started(turn, leg.calls, leg.cut_tools)
        self._remember_cut(turn)

    def _not_started(
        self, turn: _Turn, calls: list[ToolCall], cut_tools: Sequence[str] = ()
    ) -> None:
        names = [*(c.name for c in calls), *cut_tools]
        if not names:
            return
        turn.not_run_why = STOPPED_WHY
        turn.note_not_run(names)
        self._note_tools(calls, "not_run_stopped")
        log.info(
            "Nuova richiesta sopra la risposta: %s non eseguito",
            ", ".join(names),
            extra={"route": "tool"},
        )

    def _handed_over(
        self,
        leg: _Leg,
        offer_from: list[dict[str, Any]] | None,
        turn: _Turn,
        language: str,
        start: float,
    ) -> AsyncGenerator[str, None] | None:
        """Who goes on, not started yet, when Haiku asked for Sonnet and called nothing else:
        None when nobody does and its reply ends as it is."""
        if not leg.handoff or leg.calls or offer_from is None:
            return None
        # JenJo's words only: a "storia" in a recalled memory is not a story asked for
        if self.never_offer(without_recalled(offer_from[-1]["content"]), language):
            # Haiku asked for Sonnet for a story: it tells it itself, no question.
            log.info("Sonnet chiesto per una storia: risponde Haiku")
            self.last_call["escalated"] = "skipped"
            return self._haiku(offer_from, language, start, tools=False)
        if self.smart_model:
            # Never Sonnet without a yes (JenJo, 23/10): ask, keep the question.
            log.info("Haiku propone Sonnet: chiedo conferma")
            return self._propose(offer_from, language, _space(turn), turn)
        return None

    def _ask_first(
        self,
        ask: str,
        messages: list[dict[str, Any]],
        leg: _Leg,
        confirm: list[ToolCall],
        refused: dict[str, str],
        turn: _Turn,
        language: str,
        round_no: int,
    ) -> str:
        """Level 2-3 calls wait for JenJo's yes: arms `ask` and returns what to say."""
        assert self.tools is not None
        level = max(self.tools.level(c) for c in confirm)
        self._action = _Action(messages, leg.calls, turn, language, level, round_no)
        self._action.refused = refused
        self._action.handoff = leg.handoff
        self._action.asked = self.clock()
        self._show(confirm)
        self._note_tools(confirm, "ask")
        log.info(
            "Conferma richiesta (livello %d): %s",
            level,
            ", ".join(c.brief() for c in confirm),
            extra={"route": "confirm", "level": level},
        )
        said = _space(turn)
        if held := leg.held.strip():  # "Chiudo Chrome." was held: it is said with its question
            said += held + " "
        said += ask
        turn.parts.append(said)
        turn.asked = ask
        return said

    def _before_tools(
        self, turn: _Turn, leg: _Leg, active: list[ToolCall], said_here: bool, language: str
    ) -> str:
        """What to say before the tools run ("" = nothing): "Ci guardo" before a slow action
        when Haiku said nothing, else the end of the sentence it said."""
        slow = any(c.name in SLOW_TOOLS for c in active)
        if self.pc.bridge and slow and not said_here:
            # The trailing space ends the sentence: the voice says it now. Not in turn.parts:
            # in the history Haiku copied it (05/11 17:04: "Ci guardo", and no action).
            return in_language(BRIDGE, language) + " "
        # The space ends the sentence: it is said while the tools run. Without it
        # the splitter held "Ci guardo, socio." for the next round's text (30/10
        # 14:07: the first word came 2.1 s later, after three calls), and 25/10:
        # "La apro.Aperta" was read without a pause.
        said = _space(turn) if said_here else ""
        turn.parts.append(said)
        return said

    async def _confirm(self, action: _Action, text: str, start: float) -> AsyncGenerator[str, None]:
        """The answer to "Confermi?": yes runs the calls and Haiku goes on; anything else
        cancels (a TV in the background never confirms an action)."""
        self._turn = action.turn
        answer = Answer.YES if self._repeats(action, text) else parse_answer(text, action.level)
        self.last_call["confirm"] = answer.value
        if answer is Answer.UNCLEAR and not action.asked_again:
            yield await self._ask_again(action)
            return
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
            # Not in turn.parts, like BRIDGE: Haiku would copy it. Only the space that ends the
            # question's sentence, or the next words stick to it ("confermo.Chrome").
            action.turn.parts.append(_space(action.turn))
            yield in_language(BRIDGE_RUN, action.language) + " "
        alone = not action.turn.done and not action.handoff
        try:
            results, own = await self._run_calls(
                action.calls, action.language, action.refused, action.turn
            )
        except asyncio.CancelledError:  # outside `_loop`: its note must not be lost
            self._remember_cut(action.turn)
            raise
        if own is not None and alone:  # "spegniti" confirmed: its reply, no call to Claude
            if said := self._end_with(action.turn, own, new_reply=True):
                yield said
            return
        messages = [*action.messages, {"role": "user", "content": results}]
        async with contextlib.aclosing(
            self._loop(messages, action.language, start, action.turn, action.round_no + 1)
        ) as haiku:
            async for chunk in haiku:
                yield chunk

    async def _ask_again(self, action: _Action) -> str:
        """The answer was not heard: the question is asked once more, for a full ACTION_TTL_S,
        with the action read out (the first one may have been the brief "Confermi?")."""
        assert self.tools is not None
        self.last_call.update(confirm="asked_again", asked_again=True)
        asked = [
            c
            for c in action.calls
            if c.id not in action.refused and self.tools.level(c) > LEVEL_SAFE
        ]
        what = await self.tools.describe_calls(asked, action.language)
        if what:
            again = ASK_AGAIN_EXPLICIT if action.level >= LEVEL_EXPLICIT else ASK_AGAIN
            said = in_language(again, action.language).format(what=what)
        else:
            plain = ASK_LEVEL_3 if action.level >= LEVEL_EXPLICIT else ASK_LEVEL_2
            said = in_language(plain, action.language)
        action.asked_again = True
        action.asked = self.clock()
        self._action = action
        turn = action.turn
        turn.reply_from = len(turn.parts)
        turn.parts.append(" " + said)
        turn.asked = said
        log.info("Conferma non sentita: chiedo ancora", extra={"route": "confirm"})
        return said

    async def _run_calls(
        self, calls: list[ToolCall], language: str, refused: dict[str, str], turn: _Turn
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One tool_result per call, in order; a refused call is answered without running.
        Also Claudio's own words for the round when every call ran and has them (only
        claudio_command): None when Claude must read the results."""
        assert self.tools is not None
        results = []
        own: list[str] | None = []
        for call in calls:
            if call.id in refused:
                own = None
                self._note_tools([call], "refused")
                results.append(_failed_result(call, refused[call.id]))
                continue
            if turn.stopped:  # it came while the call before this one was running
                own = None
                self._not_started(turn, [call])
                results.append(_failed_result(call, STOPPED_RESULT))
                continue
            try:
                outcome = await self.tools.run(call, language)
            except asyncio.CancelledError:  # its thread goes on: the action may well be done
                self._note_tools([call], "interrupted")
                turn.ran(call.name, "interrupted")
                raise
            if outcome.error or outcome.spoken is None:
                own = None
            elif own is not None:
                own.append(outcome.spoken)
            self._note_tools([call], "error" if outcome.error else "ok", outcome.ms)
            turn.ran(call.name, "failed" if outcome.error else "ok")
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": outcome.content(),
            }
            if outcome.error:
                block["is_error"] = True
            results.append(block)
        return results, (" ".join(w for w in own if w) if own is not None else None)

    def _end_with(self, turn: _Turn, words: str, new_reply: bool = False) -> str:
        """Only Claudio's own commands ran, marked `last` by Claude, and nothing else in this
        request (no earlier round, no Sonnet): their reply ends it, without a second call
        whose only job was "Fatto, socio" (IT-1 5.3: +0.6-1 s and one more call).
        Returns what to say ("" = nothing: the media keys are silent, as with the router);
        a `new_reply` (after "sì") starts without the space that joins it to the turn."""
        if not words:
            said = ""
        else:
            gap = _space(turn)
            turn.parts.append(gap + words)
            said = words if new_reply else gap + words
        turn.complete = True
        self._remember(turn)
        return said

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
        turn = self._turn = _Turn(messages[-1]["content"])
        leg = _Leg(turn)
        try:
            async with contextlib.aclosing(
                self._stream(self._params(messages, tools), language, start, leg)
            ) as haiku:
                async for chunk in haiku:
                    yield chunk
        except (GeneratorExit, asyncio.CancelledError):
            turn.parts.extend(leg.parts)
            self._remember_cut(turn)
            raise
        turn.parts.extend(leg.parts)
        turn.complete = leg.complete
        self._remember(turn)

    async def _sonnet(
        self, messages: list[dict[str, Any]], language: str, start: float
    ) -> AsyncGenerator[str, None]:
        turn = self._turn = _Turn(messages[-1]["content"])
        leg = _Leg(turn)
        try:
            async with contextlib.aclosing(
                self._stream(self._smart_params(messages), language, start, leg, "smart_first_ms")
            ) as sonnet:
                async for chunk in sonnet:
                    yield chunk
        except (GeneratorExit, asyncio.CancelledError):
            turn.parts.extend(leg.parts)
            self._remember_cut(turn)
            raise
        turn.parts.extend(leg.parts)
        turn.complete = leg.complete
        self._remember(turn)

    def _read_outside(self) -> bool:
        """The conversation sent with the next request ran a tool that read outside text."""
        if not self.settings.history_turns:
            return False
        notes = (
            a.rsplit(ACTIONS_NOTE, 1)[1].split("]", 1)[0]  # the code's own note is last
            for _, a in self._history
            if ACTIONS_NOTE in a
        )
        return any(f"{name} " in note for note in notes for name in OUTSIDE_TEXT_TOOLS)

    def _remember(self, turn: _Turn) -> None:
        """Only the words go into the history: tool calls and window text stay out of it
        (they cost tokens at every later turn, and the history is plain text). A short note of
        the actions that really ran is added: without it Haiku, reading only its own words,
        later "admitted" that a command it had run never ran (24/10)."""
        if turn.heard is not None:  # cut meanwhile: what was heard, never the words after it
            self._remember_cut(turn)
            return
        answer = f"{''.join(turn.parts).strip()} {_notes(turn)}".strip()
        if turn.complete and answer:
            self._write(turn, answer)

    def _remember_cut(self, turn: _Turn) -> None:
        """A reply JenJo cut short ("basta", or a new request over it) stays in the history
        with what was said and a note, so "continua la storia" or "ripeti la storia" make
        sense to Claude (before, only finished replies were kept and the story was lost)."""
        self._write(turn, _cut_line(turn))

    def _write(self, turn: _Turn, answer: str) -> None:
        """The turn's line: a turn written before (cut, then answered) is replaced, not doubled."""
        entry = (turn.question, answer)
        index = self._index_of(turn.written)
        if index is None:
            self._history.append(entry)
        else:
            self._history[index] = entry
        turn.written = entry
        self._last_turn = self.clock()

    def _forget_if_idle(self) -> None:
        if self.clock() - self._last_turn > self.settings.history_reset_s:
            self._history.clear()  # a new conversation: old context costs and confuses

    def _messages(self, text: str, language: str) -> list[dict[str, Any]]:
        self._forget_if_idle()
        if self.memory is not None and self.memory.forgotten():
            # JenJo had something forgotten: the conversation may still quote it. Its cache
            # was to be written again anyway (the memory block changed).
            self._history.clear()
        messages: list[dict[str, Any]] = []
        if self.settings.history_turns:
            for question, answer in self._history:
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
        hearing, self._hearing = self._hearing, Hearing()
        tag = self._tag(language, hearing)
        if self.memory is not None:
            # One snapshot for every round of this request: a memory Claude saves with its
            # tool shows from the next request, so the rounds read each other's cache.
            self._memory_block = self.memory.block()
            # Facts already in the history are not sent again, if the history is sent.
            sent = (q for q, _ in self._history) if self.settings.history_turns else ()
            recalled = self.memory.relevant(text, sent)
            if recalled:  # after the cached prefix, like the time: it changes nothing cached
                tag = f"{tag} {recalled}"
        messages.append({"role": "user", "content": f"{tag} {text}"})
        return messages

    def _tag(self, language: str, hearing: Hearing | None = None) -> str:
        """[it] or, with the time, [it | Thu 2026-10-23 17:15]: after the cached prefix, so
        the time changes nothing that is cached. When Whisper was unsure: [... | unsure:
        "apri la cartella Tesi" / "apri la cartella tesi?"] (~10-25 tokens, only then)."""
        fields = [language]
        if self.settings.tell_time:
            now = self.wall()
            fields.append(f"{_DAYS[now.weekday()]} {now:%Y-%m-%d %H:%M}")
        if hearing is not None and hearing.unsure:
            others = " / ".join(f'"{_quoted(a)}"' for a in hearing.alternatives)
            fields.append(f"unsure: {others}" if others else "unsure")
        return f"[{' | '.join(fields)}]"

    def _common(
        self, model: str, max_tokens: int, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        memory = self._memory_block
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": f"{self.system.rstrip()}\n\n{memory}" if memory else self.system,
            "messages": messages,
        }
        if self.settings.cache:
            # Two markers. One fixed on the system prompt: tools + system (~5,000 tokens) are
            # always read from the cache, even when the history window drops its oldest
            # question and the conversation prefix changes (26/10: from the 7th request on
            # every request wrote the whole prefix again, 0.0007 $ instead of 0.0002 $).
            # Plus automatic caching on the last block, so the next turn reads this whole
            # conversation too (below 512 tokens nothing is cached).
            params["system"] = [
                {"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}
            ]
            if memory:
                # A third marker on the memories: they change only when JenJo says
                # "ricordati" or "dimentica", and then the system prompt is still read back.
                params["system"].append(
                    {"type": "text", "text": memory, "cache_control": {"type": "ephemeral"}}
                )
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
        silent: bool = False,
    ) -> AsyncGenerator[str, None]:
        """One API call. `silent`: the model's words are read but not said nor kept (the
        retry of a cut call); the fixed error sentences are still said."""
        import anthropic
        import httpx2

        model = params["model"]
        # No PC tools, no tools in the call (Sonnet) or tool_choice none (last round): no action
        # can be promised, and no call can follow the first words.
        offered = (
            self.tools is not None
            and "tools" in params
            and params.get("tool_choice") != {"type": "none"}
        )
        for attempt in (1, 2):
            # The voice dropped the broken sentence at CUT_SHORT: so does the history.
            if attempt > 1:
                heard = joined_reply([*leg.parts, CUT_SHORT])
                leg.parts[:] = [heard + " "] if heard else []
            before = len(leg.parts)
            hold = _Hold() if offered else None  # a broken attempt's held words are dropped
            leg.held = ""
            stream: Any = None
            final: Any = None
            sent = self.clock()
            # Per call (IT-1 5.2): request -> first token (the API sends its HTTP headers with
            # it: measured equal on 30/10, so this counts tool-only calls too), -> first word
            # to say, -> end. The network alone is the warm-up's round trip (api_rtt_ms).
            timing: dict[str, int] = {}
            try:
                client: Any = self.client  # not None: ask() answered already without a key
                async with client.messages.stream(**params) as stream:
                    timing["ttft_ms"] = _ms_since(sent, self.clock())
                    # aclosing: a "basta" at the yield closes this first, then the API stream.
                    async with contextlib.aclosing(_said_text(stream.text_stream)) as words:
                        async for clean in words:
                            if silent:
                                continue
                            if hold is not None:
                                clean = hold.feed(clean)
                                leg.held = hold.held
                            if clean.strip():
                                self._mark_word(timing, sent, start, first_key)
                            if clean:
                                leg.parts.append(clean)
                                yield clean
                    final = await stream.get_final_message()
                    if leg.held and not _has_tool_use(final):
                        # No call follows: what waited is said now, before any closing sentence.
                        said, leg.held = leg.held, ""
                        if said.strip():
                            self._mark_word(timing, sent, start, first_key)
                        leg.parts.append(said)
                        yield said
            except anthropic.AuthenticationError:
                log.error("Claude: chiave API non valida (401)")
                self.last_call["claude_error"] = "auth"
                leg.end = StreamEnd.ERROR
                yield in_language(BAD_KEY, language)
                return
            # Also timeouts. DecodingError: the body arrived corrupted, the reply is just as lost.
            # OSError: a raw socket error, not wrapped (ssl.SSLError: 5 of 240 calls, A/B of 07/11).
            except (
                anthropic.APIConnectionError,
                httpx2.TransportError,
                httpx2.DecodingError,
                OSError,
            ) as exc:
                if self._retry_broken(exc, stream, attempt, silent, leg.turn.stopped):
                    yield CUT_SHORT
                    continue
                log.error("Claude non raggiungibile: %s", exc)
                self.last_call["claude_error"] = "network"
                leg.end = StreamEnd.ERROR
                # The SDK wraps these only at request time; raw, they broke a reply under way. Its
                # words and the tools that ran stay in the history (no cut note: JenJo did not
                # interrupt), and the voice never says "Attra" before this sentence. Also when the
                # request itself failed, in a later round: the note of what ran must not be lost
                # (a first round with nothing said or run stores nothing anyway: `_remember`).
                leg.complete = True
                yield CUT_SHORT
                yield _space(leg) + in_language(NETWORK_ERROR, language)
                return
            except anthropic.APIStatusError as exc:
                log.error("Claude errore %s: %s", exc.status_code, exc.message)
                # An error event inside a 200 stream: its kind is in the body, the status is a 200.
                self.last_call["claude_error"] = _event_error(exc) or exc.status_code
                leg.end = StreamEnd.ERROR
                # Also an error event during a reply (an overloaded API): as a lost connection.
                leg.complete = True
                yield CUT_SHORT
                yield _space(leg) + in_language(API_ERROR, language)
                return
            # JSONDecodeError, UnicodeDecodeError and jiter's errors are ValueErrors.
            except ValueError as exc:
                if self._retry_broken(exc, stream, attempt, silent, leg.turn.stopped):
                    yield CUT_SHORT
                    continue
                log.exception("Claude: evento illeggibile nello stream: %s", exc)
                self.last_call["claude_error"] = "bad_stream"
                leg.end = StreamEnd.ERROR
                leg.complete = True  # as a lost connection: the words and the tools stay
                yield CUT_SHORT
                yield _space(leg) + in_language(NETWORK_ERROR, language)
                return
            finally:
                self._last_contact = self.clock()
                # Also on "basta" (the speaker cancels this generator): count what was generated.
                if stream is not None:
                    timing["total_ms"] = _ms_since(sent, self.clock())
                    self._record(model, final, stream, [*leg.parts[before:], leg.held], timing)
            break  # this attempt ended: no retry
        leg.end = end = stream_end(
            final,
            tools_offered=offered,
            said="".join(leg.parts),
            ran=any(t.endswith(" ok") for t in self.last_call.get("tools", [])),
        )
        if end is StreamEnd.REFUSED:
            log.info("Claude ha rifiutato la richiesta")
            if not "".join(leg.parts).strip():
                yield in_language(REFUSED, language)
            return
        if end is StreamEnd.CUT_CALL:
            # 17:12:48 on 05/11: a long type_text was cut and Claudio said it had written it.
            # No closing sentence: the loop retries the call or says it did not run.
            log.warning("Claude: chiamata tagliata al limite di %d token", params["max_tokens"])
            self.last_call["cut"] = "tool_use"
        elif end is StreamEnd.PROMISED:
            # 17:04:15 on 05/11: "dovrei aprire la cartella AMD", no call. Only logged, no text.
            log.info("Promessa senza azione", extra={"route": "tool"})
            self.last_call["promised"] = True
        elif end is StreamEnd.CUT_WORDS and not silent:
            # 30/10: a story stopped inside "Attraverso" and the voice said "Attra". The
            # marker drops the unfinished sentence; the closing one stays in the history too,
            # so a later "continua" makes sense to Claude.
            log.warning("Claude: risposta tagliata al limite di %d token", params["max_tokens"])
            self.last_call["cut"] = "max_tokens"
            said = " " + in_language(CUT_AT_LIMIT, language)
            leg.parts.append(said)
            yield CUT_SHORT
            yield said
        leg.read(final)
        leg.complete = True

    def _mark_word(self, timing: dict[str, int], sent: float, start: float, first_key: str) -> None:
        """The first word that is really said: the first of this call and of the request."""
        if "word_ms" not in timing:
            timing["word_ms"] = _ms_since(sent, self.clock())
        if first_key not in self.last_call:
            self.last_call[first_key] = round((self.clock() - start) * 1000)

    def _retry_broken(
        self, exc: BaseException, stream: Any, attempt: int, silent: bool, stopped: bool
    ) -> bool:
        """07/11: ~2% of calls broke mid-reply ([SSL] record layer failure) and the request was
        lost. Retry once, only if the reply had begun (before it the SDK retried already) and
        nothing says to stop: the call of a broken round never started, so nothing runs twice."""
        if stream is None or attempt > 1 or silent or stopped:
            return False
        log.warning("Claude: stream interrotto, riprovo una volta: %s", exc)
        self.last_call["stream_retry"] = True
        return True

    def _record(
        self,
        model: str,
        final: Any,
        stream: Any,
        parts: list[str],
        timing: dict[str, int] | None = None,
    ) -> None:
        timing = timing or {}
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
                "Claude %s: %d in, %d out, cache %d letti / %d scritti (TTL %s), "
                "primo token %s ms, prima parola %s ms, fine %s ms%s",
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_input_tokens,
                usage.cache_creation_input_tokens,
                CACHE_TTL,
                timing.get("ttft_ms", "-"),
                timing.get("word_ms", "-"),  # "-": the call only asked for tools
                timing.get("total_ms", "-"),
                tr(", interrotta (stima)") if cut else "",
                extra={"claude_call": {"model": model, **timing}},
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
                # Every call of the request, in order: a tool round adds one.
                claude_calls=[*prev.get("claude_calls", []), {"model": model, **timing}],
            )
        except Exception:
            log.exception("Claude: spesa non registrata")


class _Leg:
    """One API call of an answer: its text, whether it finished, whether Haiku handed over,
    the PC tools it called and its content (sent back with the tool results)."""

    def __init__(self, turn: _Turn) -> None:
        self.turn = turn  # whose stop it reads
        self.parts: list[str] = []
        self.complete = False
        self.handoff = False
        self.calls: list[ToolCall] = []
        self.content: list[dict[str, Any]] = []
        self.end = StreamEnd.DONE
        self.held = ""  # words not said yet: a call may follow (`_Hold`)
        self.cut_tools: list[str] = []  # tool_use blocks cut at max_tokens (CUT_CALL), in order

    def read(self, final: Any) -> None:
        """Calls are kept only when the reply stopped to call them: a reply cut by max_tokens
        may hold a call with half its arguments."""
        from claudio.pc.tools import ToolCall

        if self.end is StreamEnd.CUT_CALL:
            self.cut_tools = _called_tools(final)
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
        if self.handoff and self.calls:  # Sonnet in the middle of an action: not now
            self.content = [b for b in self.content if b.get("name") != ASK_SONNET_TOOL["name"]]


class _Turn:
    """One request across its rounds: the question and everything said in reply."""

    def __init__(self, question: str, parts: list[str] | None = None, complete: bool = False):
        self.question = question
        self.parts = list(parts or [])
        self.complete = complete
        self.done: list[str] = []  # tools that ran in this request, for the history note
        self.not_run: list[str] = []  # tools that never ran, for the history note
        # shortcut: one reason for the whole note, so a cut call that a later stop also names gets
        # the stop's reason; upgrade to a reason per tool if G2's turn state keeps this note.
        self.not_run_why = CUT_AT_LIMIT_WHY
        self.stopped = False  # a new request came over this reply: no call starts, no question
        self.heard: str | None = None  # what the voice started saying, once cut (None: not cut)
        self.written: tuple[str, str] | None = None  # its line in the history, to rewrite
        self.reply_from = 0  # where the reply being spoken starts in `parts`: a "sì" begins one
        self.asked = ""  # the question it ended on, if any: "Confermi?", "Lo uso?"

    def note_not_run(self, names: Iterable[str]) -> None:
        # shortcut: two calls of one tool share one note, lost when one of them runs; upgrade when
        # a real session shows that note missing.
        self.not_run.extend(n for n in dict.fromkeys(names) if n not in self.not_run)

    def ran(self, name: str, how: str) -> None:
        """A tool started: in the note of what ran, out of the note of what never did."""
        self.done.append(f"{name} {how}")
        if name in self.not_run:
            self.not_run.remove(name)


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
        self.asked_again = False  # the answer was not heard once: the second question is the last
        self.handoff = False  # Haiku also asked for Sonnet: Claude goes on after the calls


class _Hold:
    """Keeps the first sentences of a reply until it is known that no tool call follows: feed()
    returns what may be said now (all of it, once the third sentence begins or the text is
    longer than HOLD_MAX_CHARS), `held` what still waits."""

    def __init__(self) -> None:
        self.held = ""
        self._flowing = False

    def feed(self, chunk: str) -> str:
        if self._flowing:
            return chunk
        self.held += chunk
        ends = list(_SENTENCE_END.finditer(self.held))
        third_begins = len(ends) >= HOLD_SENTENCES and bool(
            self.held[ends[HOLD_SENTENCES - 1].end() :].strip()
        )
        if not third_begins and len(self.held) <= HOLD_MAX_CHARS:
            return ""
        self._flowing = True
        said, self.held = self.held, ""
        return said


class _NoteFilter:
    """Takes out of the model's text the notes only the code may write, chunk by chunk: feed()
    returns what may be said now, flush() what is left at the end. Text flows through at once;
    only a "[" that can still become a note's opening waits."""

    def __init__(self) -> None:
        self._held = ""  # from that "[" on
        self._in_note = False  # the opening matched: dropped up to its end

    def feed(self, chunk: str) -> str:
        said = ""
        rest, self._held = self._held + chunk, ""
        while rest:
            if self._in_note:
                end = _NOTE_END.search(rest)
                self._in_note = end is None
                rest = "" if end is None else rest[end.end() :]
                continue
            before, bracket, after = rest.partition("[")
            said += before
            rest = bracket + after
            lowered = rest.lower()
            if lowered.startswith(_NOTE_OPENINGS):
                self._in_note = True
            elif bracket and any(opening.startswith(lowered) for opening in _NOTE_OPENINGS):
                self._held = rest
                break
            else:  # no "[", or not a note after all: "[1]", "[nota]"
                said += bracket
                rest = after
        return said

    def flush(self) -> str:
        """The reply is over: a note's whole opening was dropped by feed(), so what still waits
        is a part of one ("[Cla"), and that is text."""
        return self._held


async def _said_text(chunks: AsyncIterable[str]) -> AsyncGenerator[str, None]:
    """The model's text as the voice and the history get it: no markdown, no note. Compared
    without markdown, as it would be said ("[**Claudio ran**: ..."). Never an empty chunk."""
    notes = _NoteFilter()
    async for chunk in chunks:
        if said := notes.feed(_MARKDOWN.sub("", chunk)):
            yield said
    if held := notes.flush():
        yield held


def _event_error(exc: Any) -> str:
    """The kind of an error event ("overloaded_error"), "" for any other status error."""
    body = exc.body if exc.status_code == 200 else None
    error = body.get("error") if isinstance(body, dict) else None
    kind = error.get("type") if isinstance(error, dict) else None
    return kind if isinstance(kind, str) else ""


def _notes(turn: _Turn) -> str:
    """The history notes of what the code did: what never ran first, then what ran."""
    notes = []
    if turn.not_run:
        notes.append(f"{NOT_RUN_NOTE}{', '.join(turn.not_run)}, {turn.not_run_why}]")
    if turn.done:
        notes.append(f"{ACTIONS_NOTE}{', '.join(turn.done)}]")
    return " ".join(notes)


def _cut_line(turn: _Turn) -> str:
    """The history line of a cut turn: what was heard (or, uncut-by-the-voice, what was read),
    the cut note right after it, then the notes. All of it heard is no cut."""
    read = "".join(turn.parts)
    said = read.strip() if turn.heard is None else turn.heard
    note = CUT_NOTE if said else CUT_BEFORE_NOTE
    if turn.heard and turn.complete and " ".join(said.split()) == " ".join(read.split()):
        note = ""  # nothing was left unheard
    return " ".join(part for part in (said, note, _notes(turn)) if part)


def _said_in(question: str, heard: str) -> bool:
    """The last sentence of `question` is among the words `heard`."""
    last = re.split(r"(?<=[.!?])\s+", question.strip())[-1]
    return f" {normalize(last, drop_fillers=False)} " in f" {normalize(heard, drop_fillers=False)} "


def _failed_result(call: ToolCall, why: str) -> dict[str, Any]:
    """The tool_result of a call that did not run."""
    return {"type": "tool_result", "tool_use_id": call.id, "content": why, "is_error": True}


def _ms_since(start: float, now: float) -> int:
    return round((now - start) * 1000)


def _quoted(text: str) -> str:
    return text.replace('"', "'").strip()


def _space(turn: _Turn | _Leg) -> str:
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
