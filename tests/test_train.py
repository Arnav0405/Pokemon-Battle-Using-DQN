"""Training-loop integration: a short real run against the live server writes the contract.

Slow-ish (it plays real battles), so it stays small: enough steps to finish a few episodes and
cross two checkpoint boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_ca.train import EPISODE_FIELDS, TrainingConfig, train

TRAIN_STEPS = 300


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
