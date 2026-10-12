"""Spike 006 -- can a single battle be watched live, with no browser and no client build?

Question: plan.md leaves "live spectate path verified" unchecked -- the checkout is the server
package only (`showdown/` has no client directory) and the page at the server root is Showdown's
stub, whose only statement is a `document.location.replace(... + '.psim.us')`. So the spectate link
`evaluate.py` prints lands on the public client, which knows nothing about a local battle.

This probe tests the substitute: a live terminal view. poke-env tracks the battle's protocol
messages in `AbstractBattle._replay_data` **as they arrive**, so diffing that list each step gives
the same lines the real client renders -- turn numbers, moves, damage, faints -- in real time.

Run (server must already be up):

    python spikes/006-live-demo/live_battle.py --checkpoint runs/p3_400k/model_final.zip --delay 0.4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lab_ca.evaluate import (  # noqa: E402
    DEMO_AGENT_USERNAME,
    DEMO_OPPONENT_USERNAME,
    build_vec_env,
    choose_actions,
    load_agent,
)

# Protocol line types worth showing a human. Everything else (|request| JSON, timers, chat) is
# noise in a demo -- |request| in particular is a full battle-state JSON blob per turn.
SHOW = {
    "turn", "move", "switch", "drag", "faint",
    "-damage", "-heal", "-status", "-curestatus", "-boost", "-unboost", "-setboost",
    "-supereffective", "-resisted", "-crit", "-miss", "-immune", "-fail", "-activate",
    "-start", "-end", "-sidestart", "-sideend", "-weather", "-fieldstart", "-prepare",
    "cant", "win", "tie",
}


def find_battle(vec_env: Any) -> Optional[Any]:
    """The in-flight battle object, walking the same wrapper chain `_live_battle_tag` uses."""
    env: Any = vec_env.envs[0] if hasattr(vec_env, "envs") else None
    for _ in range(8):
        if env is None:
            return None
        if getattr(env, "battle1", None) is not None:
            return env.battle1
        env = getattr(env, "env", None)
    return None


def render(line: str) -> str:
    """A protocol line with its type stripped, pipes turned into spaces."""
    parts = line.split("|")
    return "  ".join(p for p in parts[1:] if p).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/p3_400k/model_final.zip")
    ap.add_argument("--opponent", default="heuristic")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between turns (readability)")
    ap.add_argument("--max-turns", type=int, default=300)
    args = ap.parse_args()

    vec_env = build_vec_env(
        args.opponent, 1,
        agent_username=DEMO_AGENT_USERNAME, opponent_username=DEMO_OPPONENT_USERNAME,
    )
    infos: List[Dict[str, Any]] = [{}]
    rng = np.random.default_rng()
    try:
        model = load_agent(Path(args.checkpoint), vec_env, "cpu")
        obs = vec_env.reset()
        seen = 0
        turns = 0
        done = False
        print(f"[live] agent={DEMO_AGENT_USERNAME}  opponent={DEMO_OPPONENT_USERNAME} "
              f"({args.opponent})  checkpoint={args.checkpoint}")
        while not done and turns < args.max_turns:
            obs, _r, dones, infos = vec_env.step(choose_actions(model, obs, "dqn", rng))
            turns += 1
            battle = find_battle(vec_env)
            if battle is not None:
                data: List[List[str]] = list(battle._replay_data)  # type: ignore[attr-defined]
                for split in data[seen:]:
                    if len(split) > 1 and split[1] in SHOW:
                        marker = ">>>" if split[1] == "turn" else "   "
                        print(f"{marker} {render('|'.join(split))}")
                seen = len(data)
            done = bool(dones[0])
            if args.delay:
                time.sleep(args.delay)
        for info in infos:
            if "episode" in info:
                print(f"[live] done: win={info.get('win')} outcome={info.get('outcome')} "
                      f"turns={turns}")
    finally:
        vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
