"""Claude guesses what JenJo said when Whisper's phrase was about to be dropped (and, with
[guess] precall, when Whisper was unsure of a request no command understands: PRECALL).

A false wake, a phrase in the conversation window that did not look addressed to Claudio, or
noise Whisper was unsure of: dropped for free, unless it sounds like a command ("Prossima."
after a song, "Claudio spegni ti", "Audio alsa il volu"). Then Haiku gets the phrase, Whisper's
other guesses, how it was heard (wake word, loudness, confidence, the reply before) and the
closest commands, and answers in JSON: what JenJo most likely said, how likely, which command.
intake.verdict() turns that into act / ask / drop.

One short call (~1,100 tokens in, ~40 out: ~0.00013 $ with Haiku 5.5, 1-1.5 s on JenJo's
line, 29/10: mostly the network, the length of the answer barely counts), counted by the
spending counter like every other. No answer in time, no key, budget spent: None, and the
phrase is dropped as before."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from claudio.brain.prompts import GUESS_PROMPT, PRECALL_PROMPT
from claudio.config import DEFAULT_LANGUAGE, ClaudeSettings, GuessSettings
from claudio.cost.pricing import CacheTtl, Usage
from claudio.cost.tracker import BudgetState, CostTracker
from claudio.intake import BARGE_IN, CONVERSATION, PRECALL, Doubt, Guess
from claudio.router.matcher import Command, Near

log = logging.getLogger(__name__)

CACHE_TTL: CacheTtl = "5m"  # nothing is cached (rare calls): only for the ledger's column
PHRASES_SHOWN = 3  # phrases of each command in the prompt
SAID_BEFORE_CHARS = 160  # of Claudio's reply before the phrase
WHY_CHARS = 80
SURE_LOGPROB = -0.2  # Whisper above this: "sure" (as [stt] unsure_logprob)
LOST_LOGPROB = -0.8  # below this: "very unsure" (as [noise] avg_logprob)

# Fixed schema: a schema that changes from call to call would be compiled again each time.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {  # in this order: the reason first, then what it leads to
        "why": {"type": "string"},
        "command": {"type": "string"},
        "said": {"type": "string"},
        "probability": {"type": "number"},
    },
    "required": ["why", "command", "said", "probability"],
    "additionalProperties": False,
}

# Why the phrase was about to be dropped (intake.Decision.reason), as Claude is told.
REASONS = {
    "name_not_heard": "the wake word fired, but the name was not in the phrase",
    "not_a_request": "it came after Claudio's reply without the name and did not look "
    "addressed to Claudio",
    "unsure": "recognition was unsure of it and it is short, like noise",
}


def catalog(commands: Sequence[Command], language: str = DEFAULT_LANGUAGE) -> str:
    """One line per command: "- volume_up: alza il volume / volume su / aumenta il volume"."""
    lines = []
    for cmd in commands:
        phrases = cmd.phrases_in(language)[:PHRASES_SHOWN] or cmd.phrases[:PHRASES_SHOWN]
        lines.append(f"- {cmd.id}: {' / '.join(phrases)}")
    return "\n".join(lines)


def describe(doubt: Doubt, near: Sequence[Near]) -> str:
    """The user message: the phrase and everything known about how it was heard."""
    lines = [f"Heard: {_quote(doubt.text)}"]
    if doubt.alternatives:
        lines.append("Other hearings: " + " / ".join(_quote(a) for a in doubt.alternatives))
    if doubt.reason == PRECALL:
        lines.append("Recognition was unsure of it and no local command understood it.")
    else:
        lines.append(f"About to be ignored because {REASONS.get(doubt.reason, doubt.reason)}.")
    lines.append(f"How: {_how(doubt)}.")
    lines.append(f"Name at the start: {'yes' if doubt.name_heard else 'no'}.")
    if doubt.nearby:
        lines.append(f"Names on the PC: {' '.join(doubt.nearby.split())}")
    if doubt.last_command:
        lines.append(f"Last local command: {doubt.last_command}")
    if doubt.said_before.strip():
        before = " ".join(doubt.said_before.split())
        if len(before) > SAID_BEFORE_CHARS:
            before = before[:SAID_BEFORE_CHARS].rsplit(" ", 1)[0] + "..."
        lines.append(f"Claudio's reply just before: {_quote(before)}")
    lines.append(
        "Closest commands: "
        + ", ".join(f"{n.command.id} {_quote(n.phrase)} ({n.score:.0f})" for n in near)
    )
    return "\n".join(lines)


def _how(doubt: Doubt) -> str:
    if doubt.wake_word == CONVERSATION:
        parts = ["in the conversation window after Claudio's reply, no wake word needed"]
    elif doubt.wake_word == BARGE_IN:
        parts = ["said over Claudio's reply"]
    elif doubt.wake_word:
        parts = [f'after the wake word "{doubt.wake_word}" (score {doubt.wake_score:.2f})']
    else:
        parts = ["no wake word"]
    if doubt.close is True:
        parts.append("loud voice, close to the microphone")
    elif doubt.close is False:
        parts.append("quiet voice, far from the microphone")
    if doubt.logprob is not None:
        if doubt.logprob >= SURE_LOGPROB:
            parts.append("recognition sure of the words")
        elif doubt.logprob >= LOST_LOGPROB:
            parts.append("recognition unsure of the words")
        else:
            parts.append("recognition very unsure of the words")
    return "; ".join(parts)


def _quote(text: str) -> str:
    return '"' + " ".join(text.replace('"', "'").split()) + '"'


def parse(text: str, known: set[str]) -> Guess | None:
    """Claude's JSON as a Guess: the probability kept within 0-1, a command id that is not
    one of Claudio's dropped. None when the JSON is broken (a reply cut by max_tokens)."""
    try:
        data = json.loads(text)
        said = " ".join(str(data["said"]).split())
        probability = min(max(float(data["probability"]), 0.0), 1.0)
        command = str(data.get("command") or "").strip()
        why = " ".join(str(data.get("why") or "").split())[:WHY_CHARS]
    except (ValueError, TypeError, KeyError):
        log.warning("Ipotesi di Claude illeggibile: %r", text[:200])
        return None
    return Guess(said, probability, command if command in known else "", why)


class Guesser:
    """Asks Haiku what JenJo most likely said (intake.Doubt). `client` is an
    `anthropic.AsyncAnthropic` (or a fake in tests); None means no API key."""

    def __init__(
        self,
        client: Any | None,
        tracker: CostTracker,
        claude: ClaudeSettings,
        settings: GuessSettings,
        commands: Sequence[Command],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.tracker = tracker
        self.model = settings.model
        self.settings = settings
        self.clock = clock
        # Frozen for the run: the same text at every call.
        listed = catalog(commands)
        self.system = GUESS_PROMPT.format(commands=listed)
        self.precall_system = PRECALL_PROMPT.format(commands=listed)
        self.known = {cmd.id for cmd in commands}
        self.last_call: dict[str, Any] = {}  # for metrics.jsonl: time, tokens, cost

    async def __call__(self, doubt: Doubt, near: Sequence[Near]) -> Guess | None:
        self.last_call = {}
        if self.client is None:
            return None
        if self.tracker.check().state is BudgetState.BLOCKED:
            log.info("Budget di oggi finito: non chiedo a Claude cosa hai detto")
            return None
        params = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            "system": self.precall_system if doubt.reason == PRECALL else self.system,
            "messages": [{"role": "user", "content": describe(doubt, near)}],
            "thinking": {"type": "disabled"},
            "output_config": {
                "effort": "low",
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
        }
        start = self.clock()
        try:
            client: Any = self.client
            response = await asyncio.wait_for(
                client.messages.create(**params), self.settings.timeout_s
            )
        except TimeoutError:
            log.warning("Claude non ha risposto in %.1f s: frase scartata", self.settings.timeout_s)
            self.last_call = {"guess_error": "timeout"}
            return None
        except Exception as exc:  # network, key, API: the phrase is only dropped, as before
            log.warning("Ipotesi di Claude non riuscita: %s", exc)
            self.last_call = {"guess_error": type(exc).__name__}
            return None
        self.last_call["guess_ms"] = round((self.clock() - start) * 1000)
        self._record(response)
        if getattr(response, "stop_reason", None) == "refusal":
            return None
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
        return parse(text, self.known)

    def _record(self, response: Any) -> None:
        try:
            usage = Usage.from_api(response.usage, CACHE_TTL)
            if usage == Usage():
                return
            cost = self.tracker.record(self.model, usage, CACHE_TTL)
            self.last_call.update(
                guess_model=self.model,
                guess_in=usage.input_tokens + usage.cache_read_input_tokens,
                guess_out=usage.output_tokens,
                guess_cost=round(cost, 6),
            )
        except Exception:
            log.exception("Ipotesi di Claude: spesa non registrata")
