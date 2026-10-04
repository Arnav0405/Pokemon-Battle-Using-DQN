"""Team definitions: shape, round-trip, and validation by the Showdown engine itself."""

from __future__ import annotations

import pytest

from lab_ca.teams import (
    BATTLE_FORMAT,
    CANON_TEAM,
    CANON_TEAM_PACKED,
    MOVES_PER_MON,
    TEAM_SIZE,
    pack_team,
    parse_team,
    showdown_dir,
    validate_team,
)


def test_roster_has_six_mons_with_four_moves_each() -> None:
    mons = parse_team(CANON_TEAM)
    assert len(mons) == TEAM_SIZE
    assert all(len(mon.moves) == MOVES_PER_MON for mon in mons)
    # the Showdown parser keeps the species in `nickname`; `species` is filled in at pack time
    assert all(mon.nickname for mon in mons)
    assert all(mon.item and mon.ability for mon in mons)
    assert [mon.level for mon in mons] == [100] * TEAM_SIZE


def test_team_is_packed_and_round_trips() -> None:
    assert pack_team(CANON_TEAM) == CANON_TEAM_PACKED
    assert len(parse_team(CANON_TEAM_PACKED)) == TEAM_SIZE
    # packed form is what actually goes on the wire: no newlines, six ']' separators
    assert "\n" not in CANON_TEAM_PACKED
    assert CANON_TEAM_PACKED.count("]") == TEAM_SIZE - 1


@pytest.mark.integration
def test_engine_accepts_team_for_battle_format() -> None:
    """The engine's own validator accepts the roster for our battle format."""
    ok, message = validate_team(CANON_TEAM, BATTLE_FORMAT)
    assert ok, f"engine rejected the roster for {BATTLE_FORMAT}: {message}"


@pytest.mark.integration
def test_engine_accepts_team_as_standard_play() -> None:
    """The roster is 'canon-style': legal in normal Gen-9 singles, not just Custom Game."""
    if not (showdown_dir() / "dist").is_dir():
        pytest.skip("no built Showdown checkout")
    ok, message = validate_team(CANON_TEAM, "gen9ou")
    assert ok, f"engine rejected the roster for gen9ou: {message}"
