"""Singles environment over the local Pokémon Showdown server.

Subclasses poke-env's ``SinglesEnv`` (a two-agent PettingZoo env) and defines the two things
it leaves abstract:

  * ``embed_battle`` -- the fixed-length float observation. poke-env 0.16.1 ships **no**
    default embedding, so this is ours: move slots, team HP, statuses and boosts. The
    environment wraps the returned ``Box`` space into
    ``{"observation": <vector>, "action_mask": <0/1 vector>}`` itself, so the legality mask
    reaches the learner for free.
  * ``calc_reward`` -- shaped reward (spec §4): HP/status differential per step, plus a
    victory term an order of magnitude larger than a typical per-step delta.

``make_env`` returns a single-agent gymnasium env (``SingleAgentWrapper``) with a scripted
opponent on the far side. One battle = one episode; one turn = one step.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces
from poke_env.battle import AbstractBattle, Status
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player.battle_order import (
    BattleOrder,
    ForfeitBattleOrder,
    _EmptyBattleOrder,
)

from lab_ca.opponent import make_opponent
from lab_ca.teams import BATTLE_FORMAT, CANON_TEAM, MOVES_PER_MON, TEAM_SIZE

# Action layout (see SinglesEnv.action_to_order): 0-5 switch to team slot i, 6-9 use our
# known move i, 10+ are gimmick variants of those same moves. We embed the plain slots.
_STATUSES = (Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX)
_BOOST_STATS = ("atk", "def", "spa", "spd", "spe")

FEATURE_NAMES: List[str] = (
    [f"move{slot}_base_power" for slot in range(MOVES_PER_MON)]
    + [f"move{slot}_effectiveness" for slot in range(MOVES_PER_MON)]
    + [f"team{slot}_hp_fraction" for slot in range(TEAM_SIZE)]
    + ["opp_active_hp_fraction"]
    + [f"our_status_{status.name.lower()}" for status in _STATUSES]
    + [f"opp_status_{status.name.lower()}" for status in _STATUSES]
    + [f"our_boost_{stat}" for stat in _BOOST_STATS]
    + [f"opp_boost_{stat}" for stat in _BOOST_STATS]
)
N_FEATURES = len(FEATURE_NAMES)

# Bounds cover every term: base power /100 in [-1, 2.5] (or -1 for "not selectable"),
# effectiveness in [0, 4], fractions in [0, 1], one-hots in {0, 1}, boosts/6 in [-1, 1].
OBS_LOW, OBS_HIGH = -1.0, 4.0

# Shaped-reward weights (spec §4). HP differential and status changes give a dense signal;
# victory_value is ~2-3 orders of magnitude above a typical per-step delta, so the terminal
# outcome dominates while never being the only feedback.
FAINTED_VALUE = 2.0
HP_VALUE = 1.0
STATUS_VALUE = 0.5
VICTORY_VALUE = 30.0

_TYPE_CHARTS: Dict[int, Dict[str, Dict[str, float]]] = {}


def _drain_nowait(queue: Any) -> None:
    """Empty an ``asyncio.Queue`` without ever waiting.

    Safe to call from the queue's own loop thread: it never awaits, so nothing can interleave.
    """
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


def _type_chart(gen: int) -> Dict[str, Dict[str, float]]:
    """Cached type chart for a generation (GenData loads it from disk otherwise)."""
    if gen not in _TYPE_CHARTS:
        _TYPE_CHARTS[gen] = GenData.from_gen(gen).type_chart
    return _TYPE_CHARTS[gen]


class PokemonSinglesEnv(SinglesEnv):
    """1v1 singles with our embedding and shaped reward."""

    # How long close() will wait for the in-flight challenge before cancelling it.
    CLOSE_TIMEOUT_S = 15.0
    # How long each queue drain may take before we give up on it (it never blocks on data).
    DRAIN_TIMEOUT_S = 5.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Assigning raw Boxes is enough: PokeEnv.__setattr__ wraps each one into a Dict space
        # with an "action_mask" entry of the right size.
        self.observation_spaces = {
            agent: spaces.Box(
                low=OBS_LOW, high=OBS_HIGH, shape=(N_FEATURES,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

    @staticmethod
    def action_to_order(
        action: np.int64, battle: AbstractBattle, fake: bool = False, strict: bool = True
    ) -> BattleOrder:
        """Coerce the action to ``np.int64`` before delegating.

        poke-env's converter calls ``action.item()`` internally, so a plain Python ``int``
        raises ``AttributeError: 'int' object has no attribute 'item'`` from deep inside the
        library. Accepting both keeps ``env.step(6)`` from blowing up in a confusing way.
        """
        return SinglesEnv.action_to_order(
            np.int64(action), battle, fake=fake, strict=strict
        )

    def close(self, force: bool = True, wait: bool = True) -> None:
        """``PokeEnv.close``, rebuilt so teardown can never block forever.

        The base implementation has three edges that each hang the calling thread against a
        live server:

        1. It drains the players' queues with ``while not q.empty(): q.get()``, and
           ``_AsyncQueue.get()`` has no timeout -- if the event loop's own consumer takes the
           item first, that ``get()`` waits forever.
        2. Those queues are ``maxsize=1``, so a forfeit queued while a stale order is still
           pending is *silently dropped* (``put_nowait`` raises ``QueueFull`` inside a loop
           callback), the battle keeps running, and the wait in (3) never returns.
        3. ``self._challenge_task.result()`` itself has no timeout.

        So this version drains safely first (making room for the forfeit, and not racing its
        consumer), queues the same orders the base class does, waits a bounded window, and
        clears state without ever performing a blocking queue read.
        """
        self._drain_player_queues()  # loop-side, bounded: frees the cap-1 slots, no race

        if force and self.battle1 is not None and not self.battle1.finished:
            assert self.battle2 is not None
            if self.agent1_to_move:
                self.agent1_to_move = False
                self.agent1.order_queue.put(ForfeitBattleOrder())
                if self.agent2_to_move:
                    self.agent2_to_move = False
                    self.agent2.order_queue.put(_EmptyBattleOrder())
            else:
                self.agent2_to_move = False
                self.agent2.order_queue.put(ForfeitBattleOrder())

        task = self._challenge_task
        if wait and task is not None and not task.done():
            try:
                task.result(timeout=self.CLOSE_TIMEOUT_S)
            except cf.TimeoutError:
                self.agent1.logger.warning(
                    "challenge still running after %.0fs during close(); cancelling it",
                    self.CLOSE_TIMEOUT_S,
                )
                task.cancel()
            except Exception as exc:  # the challenge itself failed: report it, don't hide it
                self.agent1.logger.warning("challenge task raised during close(): %r", exc)

        if self._challenge_task is None or self._challenge_task.done():
            try:
                self.reset_battles()
            except OSError as exc:  # battle never finished (cancelled above): drop bookkeeping
                self.agent1.logger.warning(
                    "battle still open at close(), clearing bookkeeping: %s", exc
                )
                self.agent1._battles = {}
                self.agent2._battles = {}
        self._challenge_task = None
        self.battle1 = None
        self.battle2 = None
        self.agent1.battle = None
        self.agent2.battle = None
        self._drain_player_queues()  # replaces the base class's four blocking drain loops

    def reset_battles(self) -> None:
        """``PokeEnv.reset_battles``, with the same guarded drain."""
        self._drain_player_queues()
        super().reset_battles()

    def _drain_player_queues(self) -> None:
        """Empty both players' queues without ever blocking.

        ``PokeEnv`` drains them with ``while not q.empty(): q.get()``, but ``_AsyncQueue.get()``
        has no timeout: if the event loop's consumer wins the item between the ``empty()`` check
        and the ``get()``, that call waits forever. Draining as a single loop-side coroutine
        (which never awaits) closes that window, so the library's own loops afterwards always
        find empty queues -- and the cap-1 slots are free for the next put.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for agent in (self.agent1, self.agent2):
            for async_queue in (agent.order_queue, agent.battle_queue):
                if current_loop is async_queue._loop:
                    # already on that loop: atomic, because this never awaits
                    _drain_nowait(async_queue.queue)
                    continue

                async def _drain(queue: Any = async_queue.queue) -> None:
                    _drain_nowait(queue)

                try:
                    asyncio.run_coroutine_threadsafe(_drain(), async_queue._loop).result(
                        timeout=self.DRAIN_TIMEOUT_S
                    )
                except Exception:  # teardown must never raise, whatever the loop is doing
                    pass

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        feats: List[float] = []
        active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        # Move slots, aligned with action indices 6..9. The action space indexes the
        # *known* moves (battle.active_pokemon.moves), so we must embed in that same order --
        # embedding battle.available_moves instead would silently misalign features with
        # actions whenever a move is disabled or out of PP.
        known_moves = list(active.moves.values())[:MOVES_PER_MON] if active else []
        available_ids = {move.id for move in battle.available_moves}
        for slot in range(MOVES_PER_MON):
            if slot < len(known_moves):
                move = known_moves[slot]
                legal = move.id in available_ids
                # -1 marks "known but not currently selectable" (disabled / out of PP).
                feats.append(min(move.base_power, 250) / 100.0 if legal else -1.0)
                if legal and opp_active is not None:
                    feats.append(
                        float(
                            move.type.damage_multiplier(
                                opp_active.type_1,
                                opp_active.type_2,
                                type_chart=_type_chart(battle.gen),
                            )
                        )
                    )
                else:
                    feats.append(0.0)
            else:
                feats.extend([-1.0, 0.0])

        # Our team HP fractions, in team order -- the same order switch actions 0..5 index.
        team = list(battle.team.values())
        for slot in range(TEAM_SIZE):
            feats.append(float(team[slot].current_hp_fraction) if slot < len(team) else 0.0)

        # Opponent's active HP. Unrevealed/absent -> 0.0.
        feats.append(float(opp_active.current_hp_fraction) if opp_active is not None else 0.0)

        for mon in (active, opp_active):
            boosts = mon.boosts if mon is not None else {}
            feats.extend(1.0 if mon is not None and mon.status == status else 0.0 for status in _STATUSES)
            feats.extend(boosts.get(stat, 0) / 6.0 for stat in _BOOST_STATS)

        if len(feats) != N_FEATURES:  # pragma: no cover - guards feature-list edits
            raise AssertionError(f"embedding produced {len(feats)} features, expected {N_FEATURES}")
        return np.asarray(feats, dtype=np.float32)

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Shaped reward: state-value difference (HP, faints, status, victory)."""
        return float(
            self.reward_computing_helper(
                battle,
                fainted_value=FAINTED_VALUE,
                hp_value=HP_VALUE,
                status_value=STATUS_VALUE,
                victory_value=VICTORY_VALUE,
            )
        )


def make_env(
    opponent: str = "heuristic",
    *,
    battle_format: str = BATTLE_FORMAT,
    agent_team: str = CANON_TEAM,
    opponent_team: str = CANON_TEAM,
    **kwargs: Any,
) -> SingleAgentWrapper:
    """Build a single-agent env: our learner on side 1, a scripted baseline on side 2.

    Both sides default to the same fixed roster (mirror match), so beating a baseline means
    outplaying it rather than out-teaming it. Passing a different ``opponent_team`` makes the
    battle asymmetric (each side then sends its own team).
    """
    kwargs.setdefault("open_timeout", None)  # never time out on connect; battles can be slow to start
    env = PokemonSinglesEnv(
        battle_format=battle_format, team=agent_team, **kwargs
    )
    if opponent_team != agent_team:
        env.agent2.update_team(opponent_team)
    return SingleAgentWrapper(env, make_opponent(opponent, battle_format=battle_format, team=opponent_team))
