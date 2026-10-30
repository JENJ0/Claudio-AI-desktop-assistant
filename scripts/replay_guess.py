"""What would Claude guess for the phrases CLAUDIO dropped? (IT-0, [guess] in claudio.toml)

Replays the dropped phrases of logs/metrics.jsonl (false wakes, phrases in the conversation
window that did not look addressed to CLAUDIO, noise) through the same steps as the live
assistant: how close each is to a command, then, for those close enough, Haiku's guess and
what CLAUDIO would do with it (act / ask / drop). Nothing is run on the PC.

    uv run python scripts/replay_guess.py --dry          # closeness only: free, no API call
    uv run python scripts/replay_guess.py                # with Haiku: ~0.0001 $ per phrase
    uv run python scripts/replay_guess.py --phrase "Spenditi." --phrase "Audio Bust."

The calls are counted in the spending ledger like every other.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from claudio.config import Settings, load_settings
from claudio.intake import CONVERSATION, Doubt
from claudio.router.normalizer import heard_wake_word
from claudio.router.responder import LocalResponder
from claudio.wakeword.verifier import after_clipped_name, clipped_name

# metrics.jsonl outcome -> intake.Decision.reason, as the live assistant has it.
REASONS = {"false_wake": "name_not_heard", "conversation_end": "not_a_request", "noise": "unsure"}


def dropped(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("outcome") in REASONS and row.get("text") and row.get("noise") != "quiet":
            rows.append(row)
    return rows


def doubt_of(row: dict, settings: Settings) -> Doubt:
    text = str(row["text"])
    dbfs = row.get("speech_dbfs")
    return Doubt(
        text=text,
        language=str(row.get("stt_language") or row.get("language") or "it"),
        reason=REASONS[row["outcome"]],
        wake_word=str(row.get("wake_word") or ""),
        wake_score=float(row.get("wake_score") or 0.0),
        name_heard=heard_wake_word(text) or clipped_name(text),
        close=None if dbfs is None else dbfs >= settings.noise.answer_min_dbfs,
        logprob=row.get("stt_logprob"),
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry", action="store_true", help="closeness only, no API call")
    parser.add_argument("--phrase", action="append", default=[], help="a phrase to try")
    parser.add_argument("--log", type=Path, default=Path("logs/metrics.jsonl"))
    args = parser.parse_args()

    settings = load_settings()
    if args.phrase:
        rows = [
            {"text": p, "outcome": "conversation_end", "wake_word": CONVERSATION}
            for p in args.phrase
        ]
    else:
        rows = dropped(args.log)
    if not rows:
        print("Nessuna frase scartata in logs/metrics.jsonl.")
        return 0

    from claudio.__main__ import build_responder

    responder: LocalResponder = build_responder(settings)
    floor = settings.guess.near_floor
    spent = 0.0
    for row in rows:
        doubt = doubt_of(row, settings)
        texts = [doubt.text]
        texts += [rest for t in texts if (rest := after_clipped_name(t))]
        near = responder.matcher.nearest(texts, doubt.language, settings.guess.candidates)
        best = f"{near[0].command.id} {near[0].score:.0f}" if near else "-"
        line = f"{doubt.text[:40]:42} {doubt.reason:15} vicino: {best:22}"
        if not near or near[0].score < floor or args.dry:
            print(line + ("" if args.dry else "  (nessuna chiamata)"))
            continue
        recovery = await responder.guess(doubt)
        responder.cancel_pending(silence=True)  # an ASK arms a question: not here
        if recovery is None:
            print(line + "  (nessuna chiamata)")
            continue
        info = recovery.info
        spent += float(info.get("guess_cost", 0.0))
        print(
            f"{line}  -> {info.get('guess_verdict', 'drop'):4} {info.get('guess', '')!r} "
            f"p={info.get('guess_p', '-')} {info.get('guess_ms', '-')} ms"
        )
    if not args.dry:
        print(f"Spesa totale: {spent:.5f} $")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
