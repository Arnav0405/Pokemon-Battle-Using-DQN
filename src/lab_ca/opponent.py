"""Scripted baseline opponents, each on a fixed canon-style team.

Two baselines, per the design spec §5:
  * ``random``    -- poke-env's ``RandomPlayer``: uniform pick among legal actions.
  * ``heuristic`` -- poke-env's ``SimpleHeuristicsPlayer``, the type-aware baseline that is
    the primary training opponent. It scores moves by expected damage and type
    effectiveness, so "beats random -> competitive with the heuristic" is a meaningful claim.

``make_opponent`` returns a player that does **not** connect to the Showdown server
(``start_listening=False``): ``SingleAgentWrapper`` drives it in-process by calling
``choose_move`` on the far-side battle. Its ``team`` is therefore documentation only -- the
far side's real team comes from the environment (see ``env.make_env``).
"""

from __future__ import annotations

from typing import Any

from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer

from lab_ca.teams import BATTLE_FORMAT, CANON_TEAM

OPPONENT_KINDS = ("random", "heuristic")


def make_opponent(
    kind: str = "heuristic",
    *,
    battle_format: str = BATTLE_FORMAT,
    team: str = CANON_TEAM,
    **kwargs: Any,
) -> Player:
    """Build one of the scripted baselines, loaded with a fixed team.

    :param kind: ``"random"`` or ``"heuristic"``.
    :param battle_format: Format the opponent is configured for (must match the env).
    :param team: Fixed roster this baseline is associated with (import or packed text).
    :param kwargs: Forwarded to the player (e.g. log_level).
    """
    if kind not in OPPONENT_KINDS:
        raise ValueError(f"unknown opponent {kind!r}; expected one of {OPPONENT_KINDS}")

    kwargs.setdefault("start_listening", False)
    kwargs.setdefault("log_level", 40)  # errors only: baselines are chatty in training loops

    cls = RandomPlayer if kind == "random" else SimpleHeuristicsPlayer
    return cls(battle_format=battle_format, team=team, **kwargs)
