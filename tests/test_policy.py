"""Masked DQN: no illegal action ever leaves the policy, the exploration, or the replay buffer.

Offline -- no Showdown server: a stub dict-observation env with a moving legality mask stands in
for the battle environment, and it raises if it is ever asked to execute an illegal action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from lab_ca.policy import MASK_KEY, MaskedDQN, MaskedDQNPolicy
from lab_ca.train import EPISODE_FIELDS, EpisodeMetricsCallback

N_ACTIONS = 4
EPISODE_STEPS = 6


class StubBattleEnv(gym.Env):
    """Dict-observation stub whose legal actions flip every step; raises on an illegal step."""

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-1.0, 1.0, (3,), np.float32),
                MASK_KEY: spaces.Box(0, 1, (N_ACTIONS,), np.int8),
            }
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.executed: List[int] = []
        self.mask = np.array([1, 0, 1, 0], dtype=np.int8)
        self._step = 0

    def _obs(self) -> Dict[str, Any]:
        return {"observation": np.zeros(3, np.float32), MASK_KEY: self.mask.copy()}

    def reset(self, *, seed: Any = None, options: Any = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._step = 0
        self.mask = np.array([1, 0, 1, 0], dtype=np.int8)
        return self._obs(), {}

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action = int(action)
        assert self.mask[action], f"env was asked to execute illegal action {action}"
        self.executed.append(action)
        self._step += 1
        self.mask = 1 - self.mask  # a different legal set every turn
        terminated = self._step >= EPISODE_STEPS
        info = {"win": True, "outcome": "win"} if terminated else {}
        return self._obs(), 0.1, terminated, False, info


def _env_and_model(**kwargs: Any) -> Tuple[StubBattleEnv, DummyVecEnv, MaskedDQN]:
    env = StubBattleEnv()
    vec_env = DummyVecEnv([lambda: env])
    model = MaskedDQN(MaskedDQNPolicy, vec_env, seed=0, device="cpu", **kwargs)
    return env, vec_env, model


def _as_int(action: Any) -> int:
    return int(np.asarray(action).reshape(-1)[0])


def test_greedy_action_is_legal_under_a_moving_mask() -> None:
    env, _, model = _env_and_model()
    for _ in range(25):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert env.mask[_as_int(action)], "greedy action was illegal"


def test_epsilon_greedy_exploration_is_always_legal() -> None:
    """exploration_rate pinned at 1.0: every action here comes from the random branch."""
    env, _, model = _env_and_model(
        exploration_initial_eps=1.0, exploration_final_eps=1.0, exploration_fraction=1.0
    )
    for _ in range(50):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=False)
        assert env.mask[_as_int(action)], "exploration landed on an illegal action"


def test_learning_neither_executes_nor_stores_an_illegal_action() -> None:
    """The stub env raises on an illegal step, and the buffer is checked afterwards."""
    env, _, model = _env_and_model(
        exploration_initial_eps=1.0,
        exploration_final_eps=0.5,
        exploration_fraction=0.5,
        learning_starts=10,
        buffer_size=200,
        batch_size=8,
    )
    model.learn(total_timesteps=120)

    assert len(env.executed) > 0, "the env never ran"
    buffer = model.replay_buffer
    size = buffer.size()
    # raw storage keeps the vec env's batch dim: (size, 1, n_actions) / (size, 1, 1)
    stored = np.asarray(buffer.actions[:size]).reshape(size, -1)[:, 0]
    masks = np.asarray(buffer.observations[MASK_KEY][:size]).reshape(size, -1).astype(bool)
    assert masks[np.arange(size), stored].all(), "replay buffer holds an illegal action"
    assert len(env.executed) == model.num_timesteps


def test_both_q_networks_report_minus_inf_for_illegal_actions() -> None:
    """Online *and* target net -- the target one is what the TD target's max reads."""
    _, _, model = _env_and_model()
    obs = {
        "observation": th.zeros(1, 3),
        MASK_KEY: th.tensor([[1, 0, 1, 0]], dtype=th.int8),
    }
    for name, net in (("q_net", model.q_net), ("q_net_target", model.q_net_target)):
        q_values = net(obs)
        assert th.isfinite(q_values[0, 0]) and th.isfinite(q_values[0, 2]), name
        assert th.isinf(q_values[0, 1]) and q_values[0, 1] < 0, name
        assert th.isinf(q_values[0, 3]) and q_values[0, 3] < 0, name


def test_mask_applies_row_by_row_in_a_batch() -> None:
    """A batched mask must mask its own row, not broadcast across the batch."""
    _, _, model = _env_and_model()
    obs = {
        "observation": th.zeros(2, 3),
        MASK_KEY: th.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=th.int8),
    }
    q_values = model.q_net(obs)
    assert q_values.shape == (2, N_ACTIONS)
    assert th.isfinite(q_values[0, :2]).all() and th.isinf(q_values[0, 2:]).all()
    assert th.isinf(q_values[1, :2]).all() and th.isfinite(q_values[1, 2:]).all()


def test_episode_metrics_callback_writes_the_contract(tmp_path: Path) -> None:
    env = StubBattleEnv()
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    run_dir = tmp_path / "run"
    model = MaskedDQN(
        MaskedDQNPolicy,
        vec_env,
        seed=0,
        device="cpu",
        learning_starts=10,
        buffer_size=200,
        batch_size=8,
    )
    model.learn(total_timesteps=40, callback=EpisodeMetricsCallback(run_dir, "stub"))

    rows = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows, "no episode rows were written"
    assert set(rows[0]) == set(EPISODE_FIELDS)
    for row in rows:
        assert row["steps"] == EPISODE_STEPS
        assert row["opponent"] == "stub"
        assert row["win"] in (0, 1)
        assert row["outcome"] == "win"
        assert 0.0 <= row["epsilon"] <= 1.0
        assert abs(row["cumulative_reward"] - EPISODE_STEPS * 0.1) < 1e-6
    header = (run_dir / "episodes.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(EPISODE_FIELDS)
