"""Action masking for SB3's DQN -- the Phase-2 masking decision.

SB3 core `DQN` has no masking and sb3-contrib ships no `MaskableDQN` (only `MaskablePPO`), so
we keep the algorithm the spec pins and mask in the two places that matter:

* **The Q-networks.** Both the online and the target network are `MaskedQNetwork`, so illegal
  actions score ``-inf``. That covers the greedy action (`QNetwork._predict` -> `argmax`) *and*
  the TD target in `DQN.train`, which does
  ``q_net_target(next_obs).max(dim=1)`` -- the place where an unmasked DQN quietly learns an
  optimistic value for a move it can never play.
* **The exploration.** SB3's epsilon-greedy branch (`DQN.predict`) and its warm-up phase
  (`OffPolicyAlgorithm._sample_action`) both sample uniformly over the *whole* action space.
  `MaskedDQN` snaps those draws onto a legal action, so the replay buffer only ever holds
  actions the environment actually executed (and therefore what the update learns from).

The mask travels in the observation dict as ``"action_mask"`` -- poke-env's environment puts it
there next to ``"observation"``. At least one action is always legal (asserted in the env tests),
so ``argmax`` over an all-but-one ``-inf`` row still returns a legal action.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy, QNetwork

MASK_KEY = "action_mask"


class MaskedQNetwork(QNetwork):
    """``QNetwork`` that reports ``-inf`` for every illegal action."""

    def forward(self, obs: Any) -> th.Tensor:
        q_values = super().forward(obs)
        mask = self._legal_mask(obs)
        if mask.numel() != q_values.numel():
            raise ValueError(
                f"action mask has {mask.numel()} entries but there are {q_values.numel()} "
                f"q-values (shapes {tuple(mask.shape)} vs {tuple(q_values.shape)})"
            )
        # reshape rather than rely on broadcasting: a mask with an extra trailing dim would
        # otherwise broadcast silently into the wrong rows instead of failing
        return q_values.masked_fill(~mask.reshape(q_values.shape), -th.inf)

    @staticmethod
    def _legal_mask(obs: Any) -> th.Tensor:
        mask = obs[MASK_KEY]
        if not th.is_tensor(mask):
            mask = th.as_tensor(mask)
        return mask.to(dtype=th.bool)


class MaskedDQNPolicy(MultiInputPolicy):
    """`MultiInputPolicy` (dict observations) whose Q-networks mask illegal actions."""

    def make_q_net(self) -> QNetwork:
        # mirrors DQNPolicy.make_q_net, which is what builds both q_net and q_net_target
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)


class MaskedDQN(DQN):
    """`DQN` whose exploration cannot land on an illegal action."""

    def predict(
        self,
        observation: Any,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        actions, state = super().predict(observation, state, episode_start, deterministic)
        return self._snap_to_legal(actions, observation), state

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Any = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        actions, buffer_actions = super()._sample_action(learning_starts, action_noise, n_envs)
        observation = self._last_obs
        assert observation is not None, "learn() must be running before sampling actions"
        return self._snap_to_legal(actions, observation), buffer_actions

    @staticmethod
    def _snap_to_legal(actions: np.ndarray, observation: Any) -> np.ndarray:
        """Replace any illegally sampled action with a uniformly drawn legal one."""
        actions = np.asarray(actions)
        masks = np.asarray(observation[MASK_KEY]).astype(bool, copy=False)
        if masks.ndim == 1:
            masks = masks[None, :]

        flat = actions.reshape(-1)
        if masks.shape[0] == 1 and len(flat) > 1:
            masks = np.repeat(masks, len(flat), axis=0)

        for index, action in enumerate(flat):
            row = masks[index]
            if row[int(action)]:
                continue
            flat[index] = np.random.choice(np.flatnonzero(row))
        return flat.reshape(actions.shape)
