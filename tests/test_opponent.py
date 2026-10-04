"""Baseline opponents: construction, legality of their decisions, and ordering.

The ordering check is the "baselines before training" gate: the type-aware heuristic must
actually outplay the random baseline, otherwise a "beats the heuristic" claim later would
mean nothing.
"""

from __future__ import annotations

import asyncio

import pytest
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from lab_ca.env import make_env
from lab_ca.opponent import OPPONENT_KINDS, make_opponent
from lab_ca.teams import BATTLE_FORMAT, CANON_TEAM

HEAD_TO_HEAD_BATTLES = 8


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_opponent("nope")


def test_baselines_are_the_expected_classes_with_a_team() -> None:
    random_opp = make_opponent("random")
    heuristic_opp = make_opponent("heuristic")

    assert isinstance(random_opp, RandomPlayer)
    assert isinstance(heuristic_opp, SimpleHeuristicsPlayer)
    assert not isinstance(random_opp, SimpleHeuristicsPlayer)

    for opp in (random_opp, heuristic_opp):
        assert opp._team is not None, "baseline must carry a fixed team"
        assert opp._team.yield_team()  # packed, non-empty
        # the in-process baselines never open a socket: their login event stays unset
        assert not opp.ps_client.logged_in.is_set(), "in-process baselines must not connect"


@pytest.mark.integration
def test_heuristic_returns_a_legal_order_on_a_live_battle(server: None) -> None:
    env = make_env("heuristic")
    try:
        env.reset()
        far_side = env.env.battle2
        assert far_side is not None
        order = env.opponent.choose_move(far_side)
        valid = [str(o) for o in far_side.valid_orders]
        if valid:  # empty on the very first request window
            assert str(order) in valid, f"{order} not legal in {valid}"
    finally:
        env.close()


@pytest.mark.integration
def test_heuristic_beats_random_over_a_short_series(server: None) -> None:
    """Both baselines on the fixed roster: the type-aware one must come out ahead."""
    heuristic = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, team=CANON_TEAM)
    random_player = RandomPlayer(battle_format=BATTLE_FORMAT, team=CANON_TEAM)

    async def play() -> None:
        await heuristic.battle_against(random_player, n_battles=HEAD_TO_HEAD_BATTLES)

    asyncio.run(play())
    wins = heuristic.n_won_battles
    losses = heuristic.n_lost_battles
    print(f"heuristic {wins} - {losses} random (of {HEAD_TO_HEAD_BATTLES})")
    assert wins + losses == HEAD_TO_HEAD_BATTLES
    assert wins > losses, f"heuristic went {wins}-{losses} vs random: baseline is not usable"
