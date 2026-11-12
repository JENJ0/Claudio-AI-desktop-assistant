"""Claudio's own commands as one tool for Claude ("claudio_command").

A command Whisper garbled beyond the matcher ("Claudio, ok, la chata", "Audio alsa il volu")
reaches Claude, which works out what JenJo meant, but had no tool for it: the volume, the
media keys, the chat window, the spending, pausing or switching Claudio off are local
actions. This tool runs them through the same ActionRunner as the router. Claude has its
own tools for opening apps, folders and windows; the time and date come with each request;
"basta", "ripeti" and "svegliati" never need Claude.

The command's reply is the router's. When Claude marks the call
`last` (the command is the whole request), Claudio says that reply itself and Claude is not
called again just to say "Fatto" (IT-1: 0.6-1 s and one call less). Unmarked, Claude gets
the reply as the result and goes on: a request with more in it ("abbassa il volume e dimmi
che ore sono") is never cut short."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rapidfuzz import fuzz

from claudio.router.actions import ActionRunner
from claudio.router.matcher import Command, Match
from claudio.router.normalizer import normalize
from claudio.safety.confirmation import LEVEL_CONFIRM, YES_WORDS

NAME = "claudio_command"
ACTIONS = frozenset({"media_key", "chat", "system", "tell_spend"})
NOT_OFFERED = frozenset({"resume"})  # params "do": Claude is never asked while paused
MAX_NUMBER = 50  # as ActionRunner: one volume press is 2 %


class UnknownCommand(ValueError):
    """Claude named a command the tool does not offer."""


class CommandTool:
    """The definition Claude gets, the level of a call and the run."""

    def __init__(
        self, commands: Sequence[Command], actions: ActionRunner, threshold: float = 85
    ) -> None:
        self.actions = actions
        self.threshold = threshold  # the matcher's: how close counts as said
        self.commands = {
            cmd.id: cmd
            for cmd in commands
            if cmd.action in ACTIONS and cmd.params.get("do") not in NOT_OFFERED
        }
        listed = "; ".join(f"{cmd.id} ({cmd.phrases[0]})" for cmd in self.commands.values())
        self.definition: dict[str, Any] = {
            "name": NAME,
            "description": (
                "Run one of Claudio's own commands: the volume, the music and video keys, "
                "Claudio's chat window, today's AI spending, pausing Claudio or switching it "
                "off. Use it whenever JenJo asks for one of these, also when the words came "
                f"out misheard. Commands: {listed}. number: only when JenJo said how many "
                "(volume steps of 2 percent, songs to skip). last: true only when this "
                "command is all JenJo asked for: Claudio then confirms it aloud himself and "
                "you get no result back, so write nothing before or after it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": list(self.commands)},
                    "number": {"type": "integer", "minimum": 1, "maximum": MAX_NUMBER},
                    "last": {"type": "boolean"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        }

    def command(self, args: Mapping[str, Any]) -> Command:
        cmd = self.commands.get(str(args.get("command", "")))
        if cmd is None:
            raise UnknownCommand(f"unknown command: {args.get('command')!r}")
        return cmd

    def level(self, args: Mapping[str, Any]) -> int:
        """The command's own level; pausing and switching off are confirmed too: Claude only
        guessed them from words the matcher could not read."""
        try:
            cmd = self.command(args)
        except UnknownCommand:
            return 1  # nothing will run: the result reports the error
        return max(cmd.level, LEVEL_CONFIRM) if cmd.action == "system" else cmd.level

    def phrase(self, args: Mapping[str, Any]) -> str:
        """How JenJo would say the command, for the confirmation question."""
        try:
            cmd = self.command(args)
        except UnknownCommand:
            return str(args.get("command", ""))
        return cmd.phrases[0]

    def said_again(self, args: Mapping[str, Any], text: str) -> bool:
        """`text` is this command once more: close to one of its own phrases ("spegni" is 85.7
        from "spegniti", and no phrase of the matcher's). One yes word may open it ("sì,
        spegniti")."""
        try:
            cmd = self.command(args)
        except UnknownCommand:
            return False
        words = normalize(text).split()
        heard = " ".join(words[1:] if words and words[0] in YES_WORDS else words)
        return any(fuzz.ratio(heard, normalize(phrase)) >= self.threshold for phrase in cmd.phrases)

    def run(self, args: Mapping[str, Any]) -> str:
        """Run it; the command's reply, as the router would say it ("" for
        the media keys, silent as with the router)."""
        cmd = self.command(args)
        number = args.get("number")
        if number is not None:
            number = max(1, min(int(number), MAX_NUMBER))
        return self.actions.execute(Match(cmd, 100.0, number))
