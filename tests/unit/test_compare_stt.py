"""compare_stt.py: which requests it reads, how it scores engines, with fake engines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from claudio.config import PROJECT_ROOT, load_settings
from claudio.router.matcher import CommandMatcher, load_commands

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compare_stt as cs

SR = 16000


@pytest.fixture
def matcher() -> CommandMatcher:
    settings = load_settings()
    commands = load_commands(settings.resolve(settings.router.commands_file))
    return CommandMatcher(commands, settings.router.fuzzy_threshold)


def write_log(tmp_path: Path) -> tuple[Path, Path]:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for i, (outcome, prewake) in enumerate([("ok", 1500), ("near_miss", 1500), ("ok", 2000)]):
        name = f"req{i}.wav"
        sf.write(audio_dir / name, np.full(SR * 3, 0.1 * (i + 1), np.float32), SR)
        rows.append(
            {"audio_file": name, "prewake_ms": prewake, "outcome": outcome, "text": f"t{i}"}
        )
    rows.append({"audio_file": "gone.wav", "prewake_ms": 1500, "outcome": "ok"})  # deleted WAV
    rows.append({"outcome": "ok"})  # no audio saved
    log = tmp_path / "metrics.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return audio_dir, log


def test_reads_requests_as_whisper_got_them(tmp_path: Path) -> None:
    requests = cs.load_requests(*write_log(tmp_path))
    assert [r.name for r in requests] == ["req0", "req2"]  # near misses are not requests
    assert requests[0].live_text == "t0"
    assert len(requests[0].audio) == SR * 3  # 1.5 s before the wake word: the whole WAV
    assert len(requests[1].audio) == round(SR * 2.5)  # 2 s before it: the first 0.5 s cut


def test_no_requests_says_so(tmp_path: Path) -> None:
    (tmp_path / "metrics.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="Nessuna richiesta"):
        cs.load_requests(tmp_path, tmp_path / "metrics.jsonl")
    with pytest.raises(SystemExit, match="Manca"):
        cs.load_requests(tmp_path, tmp_path / "missing.jsonl")


def test_scores_and_verifier(matcher: CommandMatcher, capsys: pytest.CaptureFixture) -> None:
    audio = np.zeros(SR, np.float32)
    requests = [cs.Request(f"r{i}", audio, "") for i in range(3)]
    good = iter(["", "Hey Claudio apri Chrome", "Che ore sono", "Hey, ciao, come stai?"])
    bad = iter(["", "cue the calculator", "Che ore sono", "Hey, ciao, come stai?"])
    results = [
        cs.run("turbo", lambda _a: next(good), requests, matcher),
        cs.run("lista", lambda _a: next(bad), requests, matcher),
    ]
    assert results[0].commands[0] != "-" and results[0].commands[2] == "-"
    cs.report(requests, results, details=True)
    out = capsys.readouterr().out
    assert "Capisce di più: turbo" in out
    assert "scarterebbe 1/3" in out  # only "Hey, ciao, come stai?"
    assert out.count("scarta\n") == 1


def test_frase_is_turbo_with_the_natural_prompt(
    matcher: CommandMatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool, str | None]] = []
    monkeypatch.setattr(
        cs,
        "whisper_engine",
        lambda settings, model, closed, m, prompt=None: calls.append((model, closed, prompt)),
    )
    cs.build_engine("frase", None, matcher)  # type: ignore[arg-type]
    cs.build_engine("turbo", None, matcher)  # type: ignore[arg-type]
    cs.build_engine("frase", None, matcher, "Ciao Claudio.")  # type: ignore[arg-type]
    assert calls == [
        ("large-v3-turbo", False, cs.NATURAL_PROMPT),
        ("large-v3-turbo", False, None),
        ("large-v3-turbo", False, "Ciao Claudio."),
    ]
