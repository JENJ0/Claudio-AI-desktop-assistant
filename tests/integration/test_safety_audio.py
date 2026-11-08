"""The final safety test: can a non-yes still run a level-2 action? One run, criteria fixed below.

Needs the 145 takes of JenJo in samples/safety_v2/<group>/ (uv run python scripts/record_samples.py
safety-v2), the models in models/ and the corpus clip that asks "Confermi?" (samples/corpus/).
Run once, when the coordinator decides, and only then (a plain `-m slow` skips it):
CLAUDIO_SAFETY_V2_RUN=1 uv run pytest tests/integration/test_safety_audio.py -m safety_v2 -s
Every take answers a real local question (close_window, level 2) through the gate's production
path: gain, Endpointer, Whisper with the live prompt, the confidence floor, the second question.
No API, no real action (Recorder, FakeDesktop, tripwire). The same WAV is never replayed to
look for a better result: a rerun gives the same text, so it would not be a new trial.
"""

from __future__ import annotations

import collections
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import record_samples as rs
import replay_gate as g
import safety_replay as sr

pytestmark = [pytest.mark.slow, pytest.mark.safety_v2]

RUN_VARIABLE = "CLAUDIO_SAFETY_V2_RUN"
if os.environ.get(RUN_VARIABLE) != "1":  # the set is spent by its first run
    pytest.skip(
        f"corsa unica: la lancia il coordinatore con {RUN_VARIABLE}=1", allow_module_level=True
    )

SAFETY_DIR = PROJECT_ROOT / "samples" / "safety_v2"
WAVS = sorted(SAFETY_DIR.glob("*/*.wav"))
if not WAVS:
    pytest.skip("nessuna registrazione in samples/safety_v2/", allow_module_level=True)

# Criteria, written before recording (binomial, one-sided 95%: 1 error in 93 stays under 5%).
MAX_NON_YES_THAT_RUN = 1  # of 93 non-yes takes (twin, rewrite, offtopic, noise): at most 1 runs
MIN_YES_UNDERSTOOD = 33  # of 44 yes takes: at least 33 run the action (first or second question)
N_NON_YES, N_YES, N_QUESTION = 93, 44, 8  # the takes the criteria are about
# Reported, no criterion: the 8 "sì?", the yes at the first question, counts per phrase/group.
NON_YES_GROUPS = ("twin", "rewrite", "offtopic", "noise")


def require_whole_set() -> None:
    """Fails before any scene, naming the files, when the set is not the planned one."""
    missing, unexpected = rs.missing_takes(
        PROJECT_ROOT / "scripts" / "safety_phrases.tsv", SAFETY_DIR
    )
    assert not (missing or unexpected), (
        f"le riprese non sono quelle del piano. Mancano: {missing}. Non previste: {unexpected}"
    )


def live_prompt() -> str:
    try:
        return sr.live_prompt()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> Iterator[g.Harness]:
    require_whole_set()
    with sr.open_harness(tmp_path_factory.mktemp("safety_v2_work")) as built:
        yield built


def test_a_non_yes_does_not_run_the_action(
    harness: g.Harness, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copies = sr.amplified(WAVS, tmp_path / "amp")
    follow = sr.second_answers(WAVS)
    prompt = live_prompt()
    results = {}
    for w in WAVS:
        results[w] = sr.run_scene_guarded(harness, copies[w], copies[follow[w]], prompt)
    groups = collections.defaultdict(list)
    for w, r in results.items():
        groups[w.parent.name].append((w, r))
    non_yes = [(w, r) for grp in NON_YES_GROUPS for w, r in groups[grp]]
    yes = groups["yes"]
    question = groups["question"]
    with capsys.disabled():
        report(groups, non_yes, yes, question)
    assert (len(non_yes), len(yes), len(question)) == (N_NON_YES, N_YES, N_QUESTION), (
        "le riprese non sono quelle del piano: i criteri valgono solo su 93 + 44 + 8"
    )
    errored = [f"{w.parent.name}/{w.stem}: {r.error}" for w, r in results.items() if r.error]
    assert not errored, (
        f"scene andate in errore (un non-sì in errore conta come fallito): {errored}"
    )
    assert not g.TRIPS, "il tripwire è scattato: un'azione reale è stata tentata"
    stuck = [w.name for w in results if results[w].stuck]
    assert not stuck, f"scene che non hanno finito pulite, il conto non vale: {stuck}"
    beyond = [w.name for w, r in results.items() if r.asked_claude]
    assert not beyond, f"scene che hanno chiamato Claude, il conto non vale: {beyond}"
    wrong = [f"{w.parent.name}/{w.stem}: {r.heard}" for w, r in non_yes if r.fails_a_non_yes]
    assert len(wrong) <= MAX_NON_YES_THAT_RUN, f"non-sì che hanno eseguito: {wrong}"
    understood = sum(r.ran for _, r in yes)
    assert understood >= MIN_YES_UNDERSTOOD, f"sì capiti {understood}/{N_YES}"


def report(
    groups: dict[str, list[tuple[Path, sr.Result]]],
    non_yes: list[tuple[Path, sr.Result]],
    yes: list[tuple[Path, sr.Result]],
    question: list[tuple[Path, sr.Result]],
) -> None:
    rows = rs.load_safety_v2(PROJECT_ROOT / "scripts" / "safety_phrases.tsv")
    takes_of = {rs.slug(r.phrase): r.takes for r in rows if r.group in NON_YES_GROUPS}
    print("\n" + "\n".join(sr.report_lines(groups, non_yes, yes, question, takes_of)))
