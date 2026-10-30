"""Claudio's own commands as one tool for Claude ("claudio_command").

A command Whisper garbled beyond the matcher ("Claudio, ok, la chata", "Audio alsa il volu")
reaches Claude, which works out what JenJo meant, but had no tool for it: the volume, the
media keys, the chat window, the spending, pausing or switching Claudio off are local
actions. This tool runs them through the same ActionRunner as the router. Claude has its
own tools for opening apps, folders and windows; the time and date come with each request;
"basta", "ripeti" and "svegliati" never need Claude."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from claudio.config import DEFAULT_LANGUAGE
from claudio.router.actions import ActionRunner
from claudio.router.matcher import Command, Match
from claudio.safety.confirmation import LEVEL_CONFIRM

NAME = "claudio_command"
ACTIONS = frozenset({"media_key", "chat", "system", "tell_spend"})
NOT_OFFERED = frozenset({"resume"})  # params "do": Claude is never asked while paused
MAX_NUMBER = 50  # as ActionRunner: one volume press is 2 %
RESULT_LANGUAGE = "en"  # a tool result is data for Claude, not spoken


class UnknownCommand(ValueError):
    """Claude named a command the tool does not offer."""


class CommandTool:
    """The definition Claude gets, the level of a call and the run."""

    def __init__(self, commands: Sequence[Command], actions: ActionRunner) -> None:
        self.actions = actions
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
                "(volume steps of 2 percent, songs to skip)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": list(self.commands)},
                    "number": {"type": "integer", "minimum": 1, "maximum": MAX_NUMBER},
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

    def phrase(self, args: Mapping[str, Any], language: str = DEFAULT_LANGUAGE) -> str:
        """How JenJo would say the command, for the confirmation question."""
        try:
            cmd = self.command(args)
        except UnknownCommand:
            return str(args.get("command", ""))
        return (cmd.phrases_in(language) or cmd.phrases)[0]

    def run(self, args: Mapping[str, Any]) -> str:
        """Run it; the command's reply (or "Done.") as the tool result."""
        cmd = self.command(args)
        number = args.get("number")
        if number is not None:
            number = max(1, min(int(number), MAX_NUMBER))
        reply = self.actions.execute(Match(cmd, 100.0, number, DEFAULT_LANGUAGE), RESULT_LANGUAGE)
        return reply or "Done."
