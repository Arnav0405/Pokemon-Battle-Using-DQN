"""Environment wrapper: observation shape/bounds, action mask, shaped reward, one episode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lab_ca.env import N_FEATURES, OBS_HIGH, OBS_LOW, FEATURE_NAMES, make_env

ACTION_SPACE_SIZE = 26  # gen9: 6 switches + 4 moves x (plain + 4 gimmicks)
MAX_STEPS = 500
EPISODE_LOG = Path("runs/phase1_smoke/episodes.jsonl")


def _masked_action(mask: np.ndarray, rng: np.random.Generator) -> np.int64:
    """Pick uniformly among legal actions -- illegal ones are exactly what we must avoid."""
    legal = np.flatnonzero(mask)
    assert legal.size > 0, "action mask has no legal action"
    return np.int64(rng.choice(legal))


def _append_episode_log(
    *, opponent: str, steps: int, total_reward: float, win: bool, per_step: list[float]
) -> None:
    """Append one episode record to ``runs/`` (Phase 2 grows this into the metric contract)."""
    EPISODE_LOG.parent.mkdir(parents=True, exist_ok=True)
    episode = 1
    if EPISODE_LOG.exists():
        episode += len(EPISODE_LOG.read_text(encoding="utf-8").splitlines())
    record = {
        "episode": episode,
        "opponent": opponent,
        "steps": steps,
        "cumulative_reward": round(total_reward, 6),
        "win": win,
        "final_reward": round(per_step[-1], 6),
        "per_step_reward": [round(r, 6) for r in per_step],
    }
    with EPISODE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_feature_layout_is_consistent() -> None:
    assert len(FEATURE_NAMES) == N_FEATURES == len(set(FEATURE_NAMES))
    # 4 move slots x (base power, effectiveness) + 6 team HP + opp HP + 2 x 6 status + 2 x 5 boosts
    assert N_FEATURES == 4 * 2 + 6 + 1 + 12 + 10


@pytest.mark.integration
def test_observation_and_action_spaces(server: None) -> None:
    env = make_env("heuristic")
    try:
        obs, _ = env.reset()
        assert set(obs) == {"observation", "action_mask"}, "wrapper must surface the mask"

        vector, mask = obs["observation"], obs["action_mask"]
        assert vector.shape == (N_FEATURES,)
        assert vector.dtype == np.float32
        assert np.all(vector >= OBS_LOW) and np.all(vector <= OBS_HIGH), vector
        assert np.all(np.isfinite(vector))

        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert set(np.unique(mask)) <= {0, 1}
        assert mask.sum() >= 1

        assert env.observation_space["observation"].shape == (N_FEATURES,)
        assert env.observation_space["action_mask"].shape == (ACTION_SPACE_SIZE,)
        assert env.action_space.n == ACTION_SPACE_SIZE
    finally:
        env.close()


@pytest.mark.integration
def test_plain_int_action_is_accepted(server: None) -> None:
    """poke-env calls `action.item()` internally; a plain int must not crash the env."""
    env = make_env("heuristic")
    try:
        obs, _ = env.reset()
        legal = np.flatnonzero(obs["action_mask"])
        action = int(legal[0])  # deliberately a Python int, not np.int64
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(reward)
        assert obs["observation"].shape == (N_FEATURES,)
    finally:
        env.close()


@pytest.mark.integration
def test_one_full_episode_against_the_heuristic(server: None) -> None:
    """End-to-end: reset -> masked-random steps -> terminal, with sane rewards throughout."""
    env = make_env("heuristic")
    rng = np.random.default_rng(0)
    total_reward = 0.0
    per_step: list[float] = []
    try:
        obs, _ = env.reset()
        for _ in range(MAX_STEPS):
            action = _masked_action(obs["action_mask"], rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            per_step.append(float(reward))
            total_reward += float(reward)
            assert np.isfinite(reward)
            assert obs["observation"].shape == (N_FEATURES,)
            assert obs["action_mask"].sum() >= 1
            if terminated or truncated:
                break
        else:
            pytest.fail(f"episode did not finish within {MAX_STEPS} steps")

        battle = env.env.battle1
        assert battle is not None and battle.finished
        assert battle.won in (True, False)

        print(
            f"episode: steps={len(per_step)} total_reward={total_reward:.3f} "
            f"won={battle.won} final_reward={per_step[-1]:.3f}"
        )
        _append_episode_log(
            opponent="heuristic",
            steps=len(per_step),
            total_reward=total_reward,
            win=bool(battle.won),
            per_step=per_step,
        )
        # the terminal step carries the victory/defeat term, so it must not be a rounding error
        assert abs(per_step[-1]) > 10.0, f"terminal reward too small: {per_step[-1]}"
        # and a shaped episode should not be silent on the intermediate steps
        assert any(abs(r) > 0 for r in per_step[:-1]), "no intermediate reward signal"
    finally:
        env.close()


@pytest.mark.integration
def test_random_baseline_opponent_also_runs_a_full_episode(server: None) -> None:
    """Same mechanics with the random baseline -- both opponents must drive the far side."""
    env = make_env("random")
    rng = np.random.default_rng(1)
    try:
        obs, _ = env.reset()
        for _ in range(MAX_STEPS):
            obs, _, terminated, truncated, _ = env.step(_masked_action(obs["action_mask"], rng))
            if terminated or truncated:
                break
        else:
            pytest.fail(f"episode did not finish within {MAX_STEPS} steps")
        assert env.env.battle1 is not None and env.env.battle1.finished

        # the Phase-2 training loop reuses one env across episodes, so a second reset
        # must hand back a fresh, correctly-masked observation
        obs, _ = env.reset()
        assert obs["observation"].shape == (N_FEATURES,)
        assert obs["action_mask"].sum() >= 1
        assert env.env.battle1 is not None and not env.env.battle1.finished
    finally:
        env.close()
