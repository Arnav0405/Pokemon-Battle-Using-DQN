"""Spike 007 -- if the team changes, does the frozen policy still work, and how much does it lose?

The observation (lab_ca.env.embed_battle) carries **no species or move identity**: every feature is
a scalar derived on the fly -- base power, type effectiveness against the *current* opponent's
types, HP fractions, status one-hots, boosts. So a different team lands in the same 37-dim space
and the same Discrete(18) action space, and the checkpoint loads and plays with no code change.

What that costs in win rate is an empirical question, and this measures it against the report's
87.5 % (freeze-policy eval vs the scripted heuristic, canon mirror team).

Run (server up):

    python spikes/007-team-swap/transfer_check.py --battles 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from lab_ca.evaluate import load_agent, run_battles  # noqa: E402
from lab_ca.env import make_env  # noqa: E402
from lab_ca.teams import CANON_TEAM, pack_team, validate_team  # noqa: E402
from lab_ca.train import OutcomeWrapper  # noqa: E402

# A deliberately different roster: different species, a different type spread, and two mons whose
# only move-setup tool is a stat-boost move (invisible to the embedding -- base power 0).
ALT_TEAM = """Roaring Moon @ Booster Energy
Ability: Protosynthesis
Level: 100
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dragon Dance
- Crunch
- Acrobatics
- Earthquake

Iron Treads @ Leftovers
Ability: Quark Drive
Level: 100
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Earthquake
- Stealth Rock
- Rapid Spin
- Body Press

Skeledirge @ Heavy-Duty Boots
Ability: Unaware
Level: 100
EVs: 252 HP / 4 SpA / 252 SpD
Calm Nature
- Torch Song
- Shadow Ball
- Slack Off
- Will-O-Wisp

Meowscarada @ Choice Band
Ability: Protean
Level: 100
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Flower Trick
- Knock Off
- U-turn
- Triple Axel

Clodsire @ Leftovers
Ability: Unaware
Level: 100
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Earthquake
- Recover
- Toxic
- Spikes

Dondozo @ Leftovers
Ability: Unaware
Level: 100
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Wave Crash
- Body Press
- Rest
- Sleep Talk
"""

CANON_BASELINE_PCT = 87.5  # runs/p3_400k/eval.json, freeze-policy vs heuristic, canon mirror


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (100 * (c - h), 100 * (c + h))


def evaluate(agent_team: str, opponent_team: str, battles: int, checkpoint: str, label: str,
             tag: str) -> Any:
    # Explicit, unique account names per config: with DummyVecEnv every config runs in THIS process,
    # and poke-env's generated name repeats, so the server answers |nametaken| and the challenge
    # never starts ("Agent is not challenging"). One name pair per config avoids that.
    from poke_env.ps_client import AccountConfiguration

    def factory() -> Any:
        return Monitor(
            OutcomeWrapper(
                make_env(
                    "heuristic",
                    agent_team=agent_team,
                    opponent_team=opponent_team,
                    account_configuration1=AccountConfiguration(f"labca{tag}a", None),
                    account_configuration2=AccountConfiguration(f"labca{tag}b", None),
                )
            )
        )

    vec_env = DummyVecEnv([factory])
    try:
        model = load_agent(Path(checkpoint), vec_env, "cpu")
        outcomes = run_battles(vec_env, model=model, agent="dqn", opponent="heuristic",
                               battles=battles, turn_delay=0.0)
    finally:
        vec_env.close()
    wins = sum(1 for o in outcomes if o["won"])
    n = len(outcomes)
    lo, hi = wilson(wins, n)
    print(f"  {label:34s} {wins:3d}/{n:3d} = {100*wins/max(n,1):5.1f} %  "
          f"[{lo:.1f}, {hi:.1f}]")
    return wins, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/p3_400k/model_final.zip")
    ap.add_argument("--battles", type=int, default=200)
    args = ap.parse_args()

    ok, msg = validate_team(ALT_TEAM)
    print(f"ALT_TEAM legal for gen9customgame: {ok}")
    if not ok:
        print(msg)
        return 1
    print(f"packed: {pack_team(ALT_TEAM)[:70]}...\n")

    print(f"frozen checkpoint: {args.checkpoint}   (canon-mirror reference: "
          f"{CANON_BASELINE_PCT} %)\n")
    print(f"agent team                     win rate vs scripted heuristic ({args.battles} battles)")
    evaluate(CANON_TEAM, CANON_TEAM, args.battles, args.checkpoint, "canon     vs canon  (as shipped)", "c1")
    evaluate(ALT_TEAM,   CANON_TEAM, args.battles, args.checkpoint, "ALT       vs canon  (agent swapped)", "c2")
    evaluate(CANON_TEAM, ALT_TEAM,   args.battles, args.checkpoint, "canon     vs ALT    (opponent swapped)", "c3")
    evaluate(ALT_TEAM,   ALT_TEAM,   args.battles, args.checkpoint, "ALT       vs ALT    (both swapped)", "c4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
