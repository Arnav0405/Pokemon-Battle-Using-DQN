"""Non-overlapping block win rates for a run, to describe a trajectory without smoothing artefacts.

Usage: python spikes/004-budget-scaling/blocks.py runs/<run> [block_episodes]

Prints the per-block win rate (with the block's 95% binomial half-width) plus where the run crossed
60/70/80% for the first time on a 200-episode rolling mean, so a write-up can say *when* progress
happened instead of eyeballing a curve.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 1000


def main() -> int:
    run_dir = Path(sys.argv[1])
    rows = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outcomes = [1.0 if row.get("win") else 0.0 for row in rows]
    total = len(outcomes)
    print(f"{run_dir}: {total} episodes, block = {BLOCK}")
    print(f"{'block':>6} {'episodes':>9} {'win rate':>9} {'±95%':>6}")
    for start in range(0, total, BLOCK):
        block = outcomes[start : start + BLOCK]
        if not block:
            continue
        rate = 100.0 * sum(block) / len(block)
        half = 196.0 * math.sqrt(0.25 / len(block))
        print(f"{start // BLOCK:>6} {f'{start + 1}-{start + len(block)}':>9} {rate:>8.1f}% {half:>5.1f}")

    # first crossing of each threshold on a 200-episode rolling mean
    window = 200
    rolling = [
        sum(outcomes[index : index + window]) / window
        for index in range(0, total - window + 1)
    ]
    for threshold in (0.6, 0.7, 0.8):
        crossing = next((i + window for i, value in enumerate(rolling) if value >= threshold), None)
        print(
            f"first 200-episode window at/above {threshold:.0%}: "
            + (f"episode {crossing}" if crossing else "never")
        )
    tail = outcomes[-window:]
    print(f"final {window} episodes: {100.0 * sum(tail) / len(tail):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
