"""Spike 007b -- check the mechanism behind the team-swap numbers.

`transfer_check.py` says a policy that wins 90 % of canon-mirror battles drops to 0.5 % when the
*opponent* fields a different team (same agent, same agent team). Before calling that "team swap
ruins transfer", verify what is actually happening inside those battles: field the swapped team and
print the protocol, so the loss can be read (real losses? forfeits? team-preview mismatch?).

    python spikes/007-team-swap/mechanism_check.py --battles 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from poke_env.ps_client import AccountConfiguration  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from lab_ca.env import make_env  # noqa: E402
from lab_ca.evaluate import choose_actions, load_agent  # noqa: E402
from lab_ca.teams import CANON_TEAM  # noqa: E402
from lab_ca.train import OutcomeWrapper  # noqa: E402
from transfer_check import ALT_TEAM  # noqa: E402

SHOW = {"turn", "move", "switch", "faint", "-damage", "-heal", "-status", "win", "tie",
        "-sidestart", "cant", "-activate", "-start", "-end", "-boost", "-unboost"}


def find_battle(vec_env: Any) -> Optional[Any]:
    env: Any = vec_env.envs[0] if hasattr(vec_env, "envs") else None
    for _ in range(8):
        if env is None:
            return None
        if getattr(env, "battle1", None) is not None:
            return env.battle1
        env = getattr(env, "env", None)
    return None


def run(agent_team: str, opponent_team: str, label: str, battles: int,
        checkpoint: str = "runs/p3_400k/model_final.zip") -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = repo_root / ckpt  # keep working when invoked from the spike directory
    def factory() -> Any:
        return Monitor(
            OutcomeWrapper(
                make_env(
                    "heuristic", agent_team=agent_team, opponent_team=opponent_team,
                    account_configuration1=AccountConfiguration("labcamechA", None),
                    account_configuration2=AccountConfiguration("labcamechB", None),
                )
            )
        )

    vec_env = DummyVecEnv([factory])
    try:
        model = load_agent(ckpt, vec_env, "cpu")
        obs = vec_env.reset()
        print(f"\n===== {label} =====")
        seen, turns, done = 0, 0, False
        while not done and turns < 400:
            obs, _r, dones, infos = vec_env.step(
                choose_actions(model, obs, "dqn", __import__("numpy").random.default_rng())
            )
            turns += 1
            battle = find_battle(vec_env)
            if battle is not None:
                data: List[List[str]] = list(battle._replay_data)  # type: ignore[attr-defined]
                for split in data[seen:]:
                    if len(split) > 1 and split[1] in SHOW:
                        print("   ", "  ".join(p for p in split[1:] if p).strip())
                seen = len(data)
            done = bool(dones[0])
        for info in infos:
            if "episode" in info:
                print(f"  -> win={info.get('win')} outcome={info.get('outcome')} steps={turns}")
    finally:
        vec_env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=1)
    args = ap.parse_args()
    run(CANON_TEAM, ALT_TEAM, "agent CANON  vs  heuristic ALT (the 0.5% config)", args.battles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
