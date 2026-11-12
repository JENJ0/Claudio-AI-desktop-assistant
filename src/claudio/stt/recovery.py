"""A short phrase no command understands may be a command Whisper misheard: the headset's gate
clips the first consonants ("Scaniti." for "Spegniti.", "Chi ce l'ha fatta?" for "Chiudi la
chat.": JenJo's test of 30/10), and the closed list cannot help when the right words are not in
Whisper's beam. Here Whisper hears the phrase again with the commands as its prompt; the command
it proposes is kept only when the sound alone, with no prompt, finds it about as likely as the
first hearing or likelier. Local and free: ~0.2 s, only on those phrases (IT-1, 2.1).

The phrase JenJo says over Claudio's reply, once put back together, can go through the same
recovery: Assistant._recover_command()."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from claudio.config import RecoverySettings
from claudio.intake import Verdict, asked_or_done, words
from claudio.router.matcher import Command, CommandMatcher
from claudio.router.normalizer import only_wake_words
from claudio.system import ON, RESUME
from claudio.wakeword.verifier import after_clipped_name

NAME = "Claudio"  # the command prompt starts the way JenJo does
# What the system commands do that changes nothing while Claudio is on and awake, the only
# time the recovery runs: there "Svegliati." and "Accenditi." are what Whisper writes on
# unclear audio told about the commands (the 4 wrong proposals of 8 on JenJo's recordings,
# 30/10), never a request. They stay out of the prompt and are never proposed.
IDLE_WHEN_AWAKE = frozenset({RESUME, ON})
_NAME_FIRST = re.compile(rf"^\W*(?:(?:ehi|hey|hei|ei)\W+)?{NAME}\b\W*", re.IGNORECASE)


@dataclass(frozen=True)
class Fit:
    """How likely Whisper finds a text on some audio with no prompt: the log-probability of
    its word pieces and of the end of the text, in all and per piece."""

    total: float
    mean: float


NO_FIT = Fit(float("-inf"), float("-inf"))  # nothing to score (silence, an empty text)


@dataclass(frozen=True)
class SecondHearing:
    """What Whisper heard when told about the commands, and how likely it finds any text on
    the same audio with no prompt (sharing the encoder pass of the transcription)."""

    text: str
    fit: Callable[[Sequence[str]], list[Fit]]


class Rescorer(Protocol):
    """What the recovery needs from Whisper (stt/transcriber.py)."""

    def listen_again(self, audio: np.ndarray, prompt: str, language: str) -> SecondHearing:
        """One more transcription, with `prompt` in place of the usual one."""
        ...


@dataclass(frozen=True)
class Recovered:
    """What the recovery made of a phrase: the command as Whisper wrote it ("Spegniti."), what
    to do with it and the figures for metrics.jsonl (recovery_*)."""

    text: str
    verdict: Verdict
    info: dict[str, Any] = field(default_factory=dict)


def without_name(text: str) -> str:
    """The request without the name in front ("Claudio Chiuditi." -> "Chiuditi."), written
    as Whisper wrote it: what "Hai detto: ...?" repeats."""
    rest = _NAME_FIRST.sub("", text, count=1)
    return rest if rest.strip() else text


def recoverable(command: Command) -> bool:
    """A command worth proposing while Claudio is on and awake."""
    return not (command.action == "system" and command.params.get("do") in IDLE_WHEN_AWAKE)


def command_prompt(commands: Sequence[Command]) -> str:
    """ "Claudio, Apri chrome. Apri la cartella download. ...": the first phrase of every
    command worth proposing, as if JenJo had said them one after the other."""
    firsts = [cmd.phrases[0] for cmd in commands if recoverable(cmd)]
    return f"{NAME}, " + " ".join(f"{p[:1].upper()}{p[1:]}." for p in firsts)


def judge(first: Fit, proposal: Fit, action: str, settings: RecoverySettings) -> Verdict:
    """ACT when the proposal sounds clearly likelier than the first hearing, ASK when about as
    likely, DROP when less likely or heard badly in itself. Pausing, switching Claudio off and
    closing it are always asked, "basta" never (intake.asked_or_done)."""
    if not (math.isfinite(first.total) and math.isfinite(proposal.total)):
        return Verdict.DROP  # one of the two could not be scored: nothing to compare
    gap = proposal.total - first.total
    if proposal.mean < settings.min_mean or gap < settings.ask_gap:
        return Verdict.DROP
    return asked_or_done(gap >= settings.act_gap, action)


class CommandRecovery:
    """Whisper's second opinion on a short phrase no command understands."""

    def __init__(
        self,
        whisper: Rescorer,
        matcher: CommandMatcher,
        commands: Sequence[Command],
        settings: RecoverySettings,
    ) -> None:
        self.whisper = whisper
        self.matcher = matcher
        self.settings = settings
        self._prompt = command_prompt(commands)

    def wants(self, text: str, language: str) -> bool:
        """Worth a try: a short phrase (a question too) that no command understands."""
        return (
            self.settings.enabled
            and bool(text.strip())
            and not only_wake_words(text)
            and words(text) <= self.settings.max_words
            and self.matcher.match(text, language) is None
        )

    def __call__(self, audio: np.ndarray, text: str, language: str) -> Recovered | None:
        """None when the phrase is not worth a try. Whisper encodes the audio once, transcribes
        it with the commands' prompt and, when that is a command, scores both texts: call it
        from a worker thread, holding the lock of every Whisper use."""
        if not self.wants(text, language):
            return None
        heard = self.whisper.listen_again(audio, self._prompt, language)
        proposal = heard.text
        match = self.matcher.match(proposal, language) if proposal else None
        if match is None or not recoverable(match.command):
            # Whisper heard no command worth proposing, even when told about them.
            return Recovered(proposal, Verdict.DROP, {"recovery": proposal})
        first, fit = heard.fit([text, proposal])
        verdict = judge(first, fit, match.command.action, self.settings)
        info = {
            "recovery": proposal,
            "recovery_command": match.command.id,
            "recovery_fit": [round(first.total, 2), round(fit.total, 2)],
            "recovery_mean": round(fit.mean, 2),
            "recovery_verdict": verdict.value,
        }
        return Recovered(self._said(proposal, match.command, language), verdict, info)

    def _said(self, proposal: str, command: Command, language: str) -> str:
        """The proposal as "Hai detto: ...?" repeats it: without the name in front, a clipped
        one too ("L'audio metti in pausa la musica." -> "metti in pausa la musica")."""
        said = without_name(proposal)
        rest = after_clipped_name(said)
        match = self.matcher.match(rest, language) if rest else None
        return rest if match is not None and match.command.id == command.id else said
