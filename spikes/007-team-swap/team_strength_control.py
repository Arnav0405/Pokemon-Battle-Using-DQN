"""Spike 007c -- control: how much of the team-swap drop is *team strength*, not transfer failure?

transfer_check.py measured the frozen policy at 90 % on the canon mirror and 0.5 % when the
*opponent* fields ALT (same agent, same agent team). Two very different stories fit that number:

  (a) the policy cannot handle a novel opponent  -> a transfer problem, or
  (b) the ALT team is simply much stronger into canon -> a matchup problem, and the agent was
      never in a fair fight.

This separates them the clean way: play the *same scripted heuristic* on both teams against itself.
If heuristic(ALT) crushes heuristic(CANON), the drop is mostly (b) -- the report's "we outplayed the
baseline" claim depends on the mirror, and no amount of retraining fixes a matchup.

    python spikes/007-team-swap/team_strength_control.py --battles 200
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from poke_env.player import Player, SimpleHeuristicsPlayer  # noqa: E402
from poke_env.ps_client import AccountConfiguration  # noqa: E402

from lab_ca.teams import BATTLE_FORMAT, CANON_TEAM  # noqa: E402
from transfer_check import ALT_TEAM  # noqa: E402


def heuristic(name: str, team: str, log_level: int = 40) -> Player:
    return SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
        log_level=log_level,
        account_configuration=AccountConfiguration(name, None),
    )


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (100 * (c - h), 100 * (c + h))


async def pairing(label: str, p1: Player, p2: Player, n: int) -> None:
    await p1.battle_against(p2, n_battles=n)
    wins, games = p1.n_won_battles, p1.n_finished_battles
    lo, hi = wilson(wins, games)
    print(f"  {label:26s} {p1.username:9s} {wins:3d}/{games:3d} = "
          f"{100 * wins / max(games, 1):5.1f} %  [{lo:.1f}, {hi:.1f}]")
    p1.reset_battles()
    p2.reset_battles()


async def main_async(n: int) -> None:
    await pairing("heuristic ALT vs CANON", heuristic("ctlalta", ALT_TEAM),
                  heuristic("ctlcanb", CANON_TEAM), n)
    await pairing("heuristic CANON vs CANON", heuristic("ctlcan1", CANON_TEAM),
                  heuristic("ctlcan2", CANON_TEAM), n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=200)
    args = ap.parse_args()
    print(f"scripted heuristic vs itself, {args.battles} battles per pairing")
    asyncio.run(main_async(args.battles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
