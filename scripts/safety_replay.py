"""Replays safety takes through today's real confirmation path, shared by the final safety test
(tests/integration/test_safety_audio.py) and the effects A/B (research_notes/.../chain_ab.py).

Each take answers a real local question ("Confermi?" for close_window, level 2): gain, Endpointer,
Whisper with the live prompt, the confidence floor, the second question. No API, no real action
(Recorder, FakeDesktop, tripwire). Importing this module loads no model and opens no microphone.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

import replay_gate as g  # scripts/ is on sys.path for the callers
from claudio.audio.capture import apply_gain
from claudio.config import PROJECT_ROOT, load_settings
from claudio.router.actions import VK_F4, VK_MENU

GAIN_DB = 14.0  # the headset's gain in the live path
OPENING = "2026-11-05_17-00/audio/20261105-171603-648_ok.wav"  # asks "Confermi?" for close_window
PROMPT_NOTES = PROJECT_ROOT / "samples" / "corpus" / "2026-11-06_12-48" / "audio"


def live_prompt() -> str:
    """The Whisper prompt of the live notes with the most words (names on the PC), frozen."""
    prompts = [
        json.loads(n.read_text(encoding="utf-8")).get("whisper_prompt") or ""
        for n in PROMPT_NOTES.glob("*.json")
    ]
    if not prompts or not max(prompts, key=len):
        raise FileNotFoundError(f"nessun prompt dal vivo in {PROMPT_NOTES}")
    return max(prompts, key=len)


def phrase_of(wav: Path) -> str:
    return wav.stem.rsplit("__", 1)[0]


def second_answers(wavs: list[Path]) -> dict[Path, Path]:
    """The recording heard if the first one does not settle the question: the next take of the
    same phrase; a trap with one take gets the next trap of its group. Never the same WAV."""
    by_phrase: dict[tuple[str, str], list[Path]] = collections.defaultdict(list)
    for w in wavs:
        by_phrase[(w.parent.name, phrase_of(w))].append(w)
    singles: dict[str, list[Path]] = collections.defaultdict(list)
    for (group, _), takes in by_phrase.items():
        if len(takes) == 1:
            singles[group].append(takes[0])
    chosen = {}
    for w in wavs:
        takes = by_phrase[(w.parent.name, phrase_of(w))]
        pool = takes if len(takes) > 1 else singles[w.parent.name]
        chosen[w] = pool[(pool.index(w) + 1) % len(pool)]
    return chosen


def amplified(wavs: list[Path], folder: Path) -> dict[Path, Path]:
    """Each take with the live gain, like the live path hears it."""
    copies = {}
    for w in wavs:
        audio, rate = sf.read(w, dtype="int16")
        assert rate == 16000, f"{w.name}: serve 16 kHz"
        copy = folder / w.parent.name / w.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        sf.write(copy, np.asarray(apply_gain(audio, 10 ** (GAIN_DB / 20))), rate, subtype="PCM_16")
        copies[w] = copy
    return copies


@contextlib.contextmanager
def open_harness(work: Path) -> Iterator[g.Harness]:
    """The gate's harness on the real models, behind the tripwire; closed on exit."""
    import pytest

    settings = load_settings()
    g.require_personality(settings)
    import ctranslate2  # noqa: F401
    import faster_whisper  # noqa: F401
    import onnxruntime  # noqa: F401
    import openwakeword  # noqa: F401

    with pytest.MonkeyPatch.context() as patch:  # the libraries first, then the tripwire
        g.install_tripwire(patch.setattr)
        client = g.CassetteClient(g.CASSETTES)
        built = g.build_harness(settings, work, wake=g.GateWake(), client=client)
        g.require_recorder(built)
        try:
            yield built
        finally:
            built.tracker.ledger.close()


@dataclasses.dataclass
class Result:
    ran: bool  # the waiting close_window ran (Alt+F4), after the first or the second answer
    first_settled: bool  # the first answer ran the action
    asked_claude: bool  # a request reached Claude: the scene went beyond the question
    heard: list[str]
    stuck: bool
    error: str = ""  # the scene raised: its type and message, the take counts as no answer

    @property
    def fails_a_non_yes(self) -> bool:
        """For a non-yes take: it ran the action, or errored (never counted as safe)."""
        return self.ran or bool(self.error)


def error_result(exc: Exception) -> Result:
    return Result(False, False, False, [], False, f"{type(exc).__name__}: {exc}")


def settled(harness: g.Harness) -> bool:
    """The question is over: its action ran, or the answer was final (a refusal, an unclear one
    after the second question). Nothing more is fed: a later clip would be a new request."""
    if (VK_MENU, VK_F4) in harness.recorder.combos:
        return True
    return any(
        r.get("route") in ("confirmed", "refused") for r in list(harness.assistant.history)[1:]
    )


def run_scene(harness: g.Harness, first: Path, second: Path, prompt: str) -> Result:
    entries = [g.Entry(OPENING, {"woke": True}, scene="s")]
    entries += [g.Entry(str(w), {"woke": True}, scene="s") for w in (first, second)]
    harness.fresh()
    before = g.take_mark(harness)
    misses = len(harness.client.misses)
    planned = g.plan_scene(harness, entries, g.CORPUS)
    for n in (1, 2):  # the live prompt, as the opening clip had it
        planned[n] = dataclasses.replace(
            planned[n], clip=dataclasses.replace(planned[n].clip, prompt=prompt)
        )
    rans = asyncio.run(
        g._replay_scene(harness, planned, before, stop_when=lambda: settled(harness))
    )
    rows = list(harness.assistant.history)
    assert rows and rows[0].get("route") == "confirm", (
        "la scena non ha fatto la domanda «Confermi?»"
    )
    ran = (VK_MENU, VK_F4) in harness.recorder.combos
    return Result(
        ran,
        len(rows) > 1 and rows[1].get("route") == "confirmed",
        len(harness.client.misses) > misses,
        [str(r.get("text", "")) for r in rows[1:]],
        any(r.stuck for r in rans),
    )


def run_scene_guarded(harness: g.Harness, first: Path, second: Path, prompt: str) -> Result:
    """run_scene, but a scene that raises is that scene's result and the run goes on."""
    try:
        return run_scene(harness, first, second, prompt)
    except Exception as exc:  # one scene failing must not void the others
        return error_result(exc)


def old_new_summary(results: list[tuple[Path, Result]], takes_of: dict[str, int]) -> str:
    """Non-yes takes that ran or errored, split: phrases with 2 takes (old) and traps with 1."""
    counts = {2: [0, 0], 1: [0, 0]}
    for wav, result in results:
        kind = counts[takes_of[phrase_of(wav)]]
        kind[0] += result.fails_a_non_yes
        kind[1] += 1
    return (
        "Non-sì che hanno eseguito o sono andati in errore: "
        f"vecchie {counts[2][0]}/{counts[2][1]}, nuove {counts[1][0]}/{counts[1][1]}"
    )


Items = list[tuple[Path, Result]]


def error_lines(results: Items) -> list[str]:
    return [f"  {w.parent.name}/{w.stem}: {r.error}" for w, r in results if r.error]


def report_lines(
    groups: dict[str, Items],
    non_yes: Items,
    yes: Items,
    question: Items,
    takes_of: dict[str, int],
) -> list[str]:
    """The printed report. A non-yes that errored counts as a failure, never as safe."""
    non_yes_names = {w.parent.name for w, _ in non_yes}
    lines = [
        f"Non-sì che hanno eseguito o sono andati in errore: "
        f"{sum(r.fails_a_non_yes for _, r in non_yes)}/{len(non_yes)}",
        old_new_summary(non_yes, takes_of),
        f"Sì che hanno eseguito: {sum(r.ran for _, r in yes)}/{len(yes)}",
        f"Sì al primo colpo: {sum(r.first_settled for _, r in yes)}/{len(yes)}",
        f"«Sì?» che hanno eseguito: {sum(r.ran for _, r in question)}/{len(question)}"
        " (senza criterio)",
    ]
    errors = error_lines([item for items in groups.values() for item in items])
    if errors:
        lines += [f"\nScene andate in errore ({len(errors)}):", *errors]
    lines.append("\nPer gruppo (eseguito o errore per i non-sì / eseguito per i sì, su prese):")
    for name, items in groups.items():
        failed = sum(r.fails_a_non_yes if name in non_yes_names else r.ran for _, r in items)
        lines.append(f"  {name:<9} {failed}/{len(items)}")
    lines.append("\nPer frase (stesso conteggio, su prese):")
    per_phrase = collections.defaultdict(list)
    for name, items in groups.items():
        for w, r in items:
            counted = r.fails_a_non_yes if name in non_yes_names else r.ran
            per_phrase[(name, phrase_of(w))].append(counted)
    for (group, phrase), counted in sorted(per_phrase.items()):
        lines.append(f"  {group:<9} {phrase:<34} {sum(counted)}/{len(counted)}")
    return lines
