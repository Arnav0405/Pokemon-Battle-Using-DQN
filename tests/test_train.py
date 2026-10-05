"""Training-loop integration: a short real run against the live server writes the contract.

Slow-ish (it plays real battles), so it stays small: enough steps to finish a few episodes and
cross two checkpoint boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_ca import train as train_module
from lab_ca.train import EPISODE_FIELDS, TrainingConfig, train

TRAIN_STEPS = 300


def test_cli_flags_reach_the_training_config(monkeypatch) -> None:
    """The comparison runs are only interpretable if the overrides land in the config dump.

    ``run_config.json`` is written from this dataclass, so this is the path that makes an
    "A vs B, one knob changed" claim checkable after the fact.
    """
    captured = {}

    def fake_train(config, runs_root=Path("runs")):
        captured["config"] = config
        return runs_root

    monkeypatch.setattr(train_module, "train", fake_train)
    exit_code = train_module.main([
        "--timesteps", "10", "--run-name", "x",
        # the SB3 baseline, so this also covers the documented way to reproduce arm A
        "--train-freq", "4", "--gradient-steps", "1", "--batch-size", "32",
        "--target-update-interval", "10000", "--net-arch", "128,128",
        "--exploration-fraction", "0.4", "--exploration-final-eps", "0.1",
    ])
    assert exit_code == 0
    config = captured["config"]
    assert config.updates_per_env_step() == 0.25  # arm A: 1 update per 4 env steps
    assert config.batch_size == 32
    assert config.target_update_interval == 10_000
    assert config.net_arch == (128, 128)
    assert config.exploration_fraction == 0.4
    assert config.exploration_final_eps == 0.1


def test_default_config_is_the_adopted_arm_b() -> None:
    """The default must stay the config the sweep chose, or new runs silently fall back to arm A."""
    default = TrainingConfig()
    assert default.updates_per_env_step() == 4.0  # 16x SB3's 0.25
    assert default.batch_size == 128
    assert default.target_update_interval == 1_000
    assert default.net_arch == (64, 64)
    # exploration was left alone on purpose (it was never the binding constraint)
    assert default.exploration_fraction == 0.1
    assert default.exploration_final_eps == 0.02


@pytest.mark.integration
def test_short_training_run_writes_the_contract(server: None, tmp_path: Path) -> None:
    run_dir = train(
        TrainingConfig(
            timesteps=TRAIN_STEPS,
            opponent="heuristic",
            seed=0,
            run_name="test_run",
            checkpoint_every_steps=200,
        ),
        runs_root=tmp_path,
    )

    rows = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows, "training finished without logging a single episode"
    assert all(set(row) == set(EPISODE_FIELDS) for row in rows)
    assert all(row["win"] in (0, 1) for row in rows)
    assert all(row["steps"] > 0 for row in rows)
    assert all(row["opponent"] == "heuristic" for row in rows)
    assert all(0.0 <= row["epsilon"] <= 1.0 for row in rows)

    # the shaped reward must actually vary: Phase 2's "not flat at zero" check
    rewards = [row["cumulative_reward"] for row in rows]
    assert any(abs(reward) > 0 for reward in rewards), "rewards are all zero"
    assert len(set(rewards)) > 1, "rewards are constant"

    assert list(run_dir.glob("model_*.zip")), "no checkpoint was written"
    assert (run_dir / "model_final.zip").exists()
    assert (run_dir / "episodes.csv").exists()
    assert (run_dir / "run_config.json").exists()

    csv_lines = (run_dir / "episodes.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].split(",") == list(EPISODE_FIELDS)
    assert len(csv_lines) == len(rows) + 1, "CSV mirror and JSONL disagree on episode count"

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["config"]
    # the default path is the adopted arm B, and it must be recorded as such
    assert config["train_freq"] == 1
    assert config["gradient_steps"] == 4
    assert config["batch_size"] == 128
    assert config["target_update_interval"] == 1000
    assert config["net_arch"] == [64, 64]
