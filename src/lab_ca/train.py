"""Phase 2 -- DQN training loop and the per-episode metric log.

The Showdown server must already be running in its own terminal
(``bash scripts/showdown_start.sh``); this script does not spawn it.

    python -m lab_ca.train --timesteps 2000 --opponent heuristic
    python -m lab_ca.train --timesteps 2000 2>&1 | tee runs/<run>/train.log

Writes, under ``runs/<run>/``:

    episodes.jsonl    one row per finished episode -- the metric contract (spec §6)
    episodes.csv      the same rows, CSV mirror
    run_config.json   every hyperparameter, seed, package and git version behind the run
    model_*.zip       SB3 checkpoints (periodic) and ``model_final.zip``

SB3's own logger is deliberately left off: the report contract is ``episodes.jsonl``, and its
tensorboard/csv output would only duplicate it.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from lab_ca.env import make_env
from lab_ca.opponent import OPPONENT_KINDS
from lab_ca.policy import MaskedDQN, MaskedDQNPolicy

# Contract fields (spec §6), in write order; `outcome` is an extra so a truncation is not
# silently reported as a loss.
EPISODE_FIELDS = (
    "episode",
    "cumulative_reward",
    "steps",
    "win",
    "outcome",
    "epsilon",
    "timestamp",
    "opponent",
)


class OutcomeWrapper(gym.Wrapper):
    """Adds the battle outcome to ``info`` when the episode ends.

    poke-env's info dict is empty and the win flag lives on the ``Battle`` object, while the
    metrics contract is per episode. Surfacing it here keeps the callback out of the
    environment's internals.
    """

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            battle = self.env.env.battle1  # SingleAgentWrapper -> PokeEnv -> our side's battle
            won = None if battle is None else battle.won
            info["win"] = bool(won)
            info["outcome"] = "win" if won else ("loss" if won is False else "unfinished")
        return obs, reward, terminated, truncated, info


class EpisodeMetricsCallback(BaseCallback):
    """Appends one row per finished episode to ``episodes.jsonl`` and ``episodes.csv``."""

    def __init__(self, run_dir: Path, opponent: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.opponent = opponent
        self.episodes = 0
        self.jsonl_path = self.run_dir / "episodes.jsonl"
        self.csv_path = self.run_dir / "episodes.csv"

    def _on_training_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=EPISODE_FIELDS).writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:  # Monitor fills this in only when an episode ends
                continue
            self.episodes += 1
            row = {
                "episode": self.episodes,
                "cumulative_reward": round(float(info["episode"]["r"]), 6),
                "steps": int(info["episode"]["l"]),
                "win": int(bool(info.get("win", False))),
                "outcome": info.get("outcome", "unknown"),
                "epsilon": round(float(self.model.exploration_rate), 6),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "opponent": self.opponent,
            }
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=EPISODE_FIELDS).writerow(row)
        return True


@dataclass
class TrainingConfig:
    """Everything that shapes a run -- dumped to ``run_config.json`` for the report."""

    timesteps: int = 2_000
    opponent: str = "heuristic"
    seed: int = 0
    run_name: str = ""
    checkpoint_every_steps: int = 1_000
    device: str = "cpu"

    # SB3 DQN hyperparameters. Spec §4: "start from SB3 defaults; tune only if curves are
    # poor" -- these are exactly the SB3 2.9 defaults, written out so the run is documented.
    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    net_arch: tuple = (64, 64)  # Q-network hidden layers; SB3's default for DQN

    def updates_per_env_step(self) -> float:
        """Gradient updates performed per environment step -- the knob this project tuned."""
        return self.gradient_steps / max(1, self.train_freq)


def make_run_dir(runs_root: Path, run_name: str) -> Path:
    """Create ``runs/<run_name>`` (never reusing an existing one)."""
    run_dir = Path(runs_root) / run_name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = Path(runs_root) / f"{run_name}_{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def make_vec_env(opponent: str, seed: int) -> DummyVecEnv:
    """One env, monitor-wrapped so episode returns/steps reach the callback.

    The env is reused across episodes (SB3 resets it on termination), which is the path the
    Phase-1 tests exercise.
    """

    def _factory() -> gym.Env:
        return Monitor(OutcomeWrapper(make_env(opponent)))

    vec_env = DummyVecEnv([_factory])
    vec_env.seed(seed)
    return vec_env


def _run_config_dump(config: TrainingConfig, run_dir: Path) -> Dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # not a repo / git missing: the run is still reproducible enough
            return "unknown"

    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def train(config: TrainingConfig, runs_root: Path = Path("runs")) -> Path:
    """Train one agent; returns the run directory that holds the artifacts."""
    if config.opponent not in OPPONENT_KINDS:
        raise ValueError(f"unknown opponent {config.opponent!r}; expected {OPPONENT_KINDS}")

    name = config.run_name or f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{config.opponent}"
    run_dir = make_run_dir(runs_root, name)
    (run_dir / "run_config.json").write_text(
        json.dumps(_run_config_dump(config, run_dir), indent=2), encoding="utf-8"
    )

    vec_env = make_vec_env(config.opponent, config.seed)
    model = MaskedDQN(
        MaskedDQNPolicy,
        vec_env,
        seed=config.seed,
        device=config.device,
        verbose=1,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        policy_kwargs={"net_arch": list(config.net_arch)},
    )

    metrics = EpisodeMetricsCallback(run_dir, config.opponent, verbose=1)
    checkpoints = CheckpointCallback(
        save_freq=max(1, config.checkpoint_every_steps),
        save_path=str(run_dir),
        name_prefix="model",
    )
    try:
        model.learn(
            total_timesteps=config.timesteps,
            callback=CallbackList([metrics, checkpoints]),
            progress_bar=False,
        )
        model.save(run_dir / "model_final")
    finally:
        vec_env.close()

    print(f"[train] {metrics.episodes} episodes logged to {run_dir / 'episodes.jsonl'}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", type=int, default=2_000, help="env steps (= turns)")
    parser.add_argument("--opponent", choices=OPPONENT_KINDS, default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="", help="runs/<run-name> (default: timestamp)")
    parser.add_argument("--checkpoint-every", type=int, default=1_000, help="steps")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", default="cpu")
    # Hyperparameter overrides. Defaults are the SB3 ones; the Phase-3 comparison runs set these.
    parser.add_argument("--net-arch", default="64,64", help="Q-network hidden layers, e.g. 256,256")
    parser.add_argument("--train-freq", type=int, default=4, help="env steps between updates")
    parser.add_argument("--gradient-steps", type=int, default=1, help="updates per train_freq block")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-interval", type=int, default=10_000, help="env steps")
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--exploration-final-eps", type=float, default=0.02)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        timesteps=args.timesteps,
        opponent=args.opponent,
        seed=args.seed,
        run_name=args.run_name,
        checkpoint_every_steps=args.checkpoint_every,
        device=args.device,
        net_arch=tuple(int(part) for part in args.net_arch.split(",") if part.strip()),
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
    )
    train(config, runs_root=Path(args.runs_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
