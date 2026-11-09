"""What would the Haiku pre-call do with the requests Whisper was unsure of? (G3c Task 4)

Replays the requests of logs/metrics.jsonl that Whisper doubted (stt_unsure) and no command
understood, through the real wiring: the responder's guess (the same decision as live), then,
for a reading Claudio would act on, the router, with every key, program and folder going to a
recorder (nothing is pressed or opened). The Haiku calls are real and counted by a throwaway
spending ledger in a temporary folder (data/spend.db is not touched); `--brake` stops the run.

    uv run python scripts/replay_precall.py --dry          # which requests, no API call
    uv run python scripts/replay_precall.py --runs 3 --brake 0.20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from claudio.app import NEARBY_CHARS
from claudio.config import Settings, load_settings
from claudio.intake import PRECALL, Doubt, Verdict
from claudio.router.normalizer import heard_wake_word
from claudio.router.responder import LocalResponder
from claudio.wakeword.verifier import clipped_name


def uncertain(path: Path) -> list[dict[str, Any]]:
    """The requests Whisper doubted, in the order of the log."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("stt_unsure") and row.get("text"):
                rows.append(row)
    return rows


def doubt_of(row: dict[str, Any], settings: Settings, nearby: str, last_command: str) -> Doubt:
    text = str(row["text"])
    dbfs = row.get("speech_dbfs")
    return Doubt(
        text=text,
        language=str(row.get("stt_language") or row.get("language") or "it"),
        reason=PRECALL,
        alternatives=tuple(row.get("stt_alternatives") or ()),
        wake_word=str(row.get("wake_word") or ""),
        wake_score=float(row.get("wake_score") or 0.0),
        name_heard=heard_wake_word(text) or clipped_name(text),
        close=None if dbfs is None else dbfs >= settings.noise.answer_min_dbfs,
        logprob=row.get("stt_logprob"),
        nearby=nearby[:NEARBY_CHARS],
        last_command=last_command,
    )


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))] if ordered else 0.0


async def replay_once(
    responder: LocalResponder, rows: list[dict[str, Any]], settings: Settings, nearby: str, rec: Any
) -> list[dict[str, Any]]:
    out = []
    last_command = ""
    for row in rows:
        doubt = doubt_of(row, settings, nearby, last_command)
        last_command = str(row.get("command") or "")
        start = time.perf_counter()
        recovery = await responder.guess(doubt)
        wall_ms = (time.perf_counter() - start) * 1000
        if recovery is None:
            out.append({"text": doubt.text, "verdict": "no_call", "ms": 0.0, "cost": 0.0})
            continue
        item: dict[str, Any] = {
            "text": doubt.text,
            "alternatives": list(doubt.alternatives),
            "verdict": recovery.verdict.value,
            "reading": recovery.said,
            "p": recovery.info.get("guess_p"),
            "ms": wall_ms,
            "cost": float(recovery.info.get("guess_cost", 0.0)),
            "error": recovery.info.get("guess_error"),
        }
        before = (len(rec.keys), len(rec.combos), len(rec.ran), len(rec.opened))
        if recovery.verdict is Verdict.ACT:  # what the router does with the reading
            reply = "".join([c async for c in responder.respond(recovery.said, doubt.language)])
            item["reply"] = reply[:80]
            item["asked_confirm"] = responder.awaiting
            item["effects"] = (
                len(rec.keys) - before[0],
                len(rec.combos) - before[1],
                len(rec.ran) - before[2],
                len(rec.opened) - before[3],
            )
        item["asked_question"] = recovery.verdict is Verdict.ASK
        responder.cancel_pending()  # nothing stays armed between rows
        out.append(item)
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry", action="store_true", help="list the requests, no API call")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--brake", type=float, default=0.20, help="stop at this total spend ($)")
    parser.add_argument("--log", type=Path, default=Path("logs/metrics.jsonl"))
    args = parser.parse_args()

    settings = load_settings()
    settings.guess.precall = True
    settings.memory.enabled = False  # the memory database is not opened
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from tests.unit.test_actions import Recorder
        from tests.unit.test_pc_tools import FakeDesktop

        from claudio.__main__ import build_responder, build_vocabulary
        from claudio.cost.pricing import load_pricing
        from claudio.cost.tracker import open_tracker

        tracker = open_tracker(
            load_pricing(settings.resolve(settings.cost.pricing_file)),
            Path(tmp) / "spend.db",
            args.brake,  # blocked at 100 %: the brake is the budget
            settings.cost.warn_at,
        )
        rec = Recorder()
        responder = build_responder(
            settings, rec.ctx(), tracker=tracker, desktop=FakeDesktop(), home=Path(tmp)
        )
        vocabulary = build_vocabulary(settings)
        vocabulary.read_now()
        nearby = vocabulary.text("it")

        rows = [r for r in uncertain(args.log) if not responder.understands(r["text"], "it")]
        print(f"{len(rows)} richieste incerte che nessun comando capisce")
        if args.dry:
            for row in rows:
                print(f"  {row['text'][:60]!r} {row.get('stt_alternatives')}")
            return 0
        all_items: list[dict[str, Any]] = []
        for run in range(1, args.runs + 1):
            items = await replay_once(responder, rows, settings, nearby, rec)
            all_items += items
            for item in items:
                print(
                    f"[{run}] {item['text'][:38]:40} {item['verdict']:7} "
                    f"{str(item.get('reading', ''))[:34]!r:36} p={item.get('p')} "
                    f"{item['ms']:.0f} ms {item.get('effects', '')} "
                    f"{'CONFERMI' if item.get('asked_confirm') else ''}"
                )
            print(f"-- giro {run}: speso {tracker.spent_today():.5f} $")
            if tracker.spent_today() >= args.brake:
                print("Freno raggiunto: mi fermo.")
                break
        calls = [i for i in all_items if i["verdict"] != "no_call"]
        times = [i["ms"] for i in calls]
        summary = {
            "rows": len(rows),
            "calls": len(calls),
            "act": sum(i["verdict"] == "act" for i in calls),
            "ask": sum(i["verdict"] == "ask" for i in calls),
            "drop": sum(i["verdict"] == "drop" for i in calls),
            "readings_for_claude": sum(
                i["verdict"] == "drop" and bool(i["reading"]) for i in calls
            ),
            "confirm_questions": sum(bool(i.get("asked_confirm")) for i in calls),
            "level1_effects": sum(1 for i in calls if any(i.get("effects", ()))),
            "p50_ms": round(statistics.median(times)) if times else 0,
            "p90_ms": round(percentile(times, 0.9)),
            "spent": round(tracker.spent_today(), 5),
        }
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
