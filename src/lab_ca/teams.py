"""Fixed canon-style teams, in Showdown's human-readable import format.

`pack_team()` turns an imported team into the packed string poke-env expects on the wire,
and `validate_team()` asks the *local Showdown engine itself* whether a team is legal for a
format -- no hand-rolled legality rules, so this doubles as the Phase 1 team check.

The roster is deliberately a plain, recognizable Gen-9 set of six (all OU-legal in standard
play, no gimmick-dependent win conditions) so that a battle's outcome depends on play rather
than on team strength. Both sides of every training battle use it (mirror match), which makes
policy quality the only variable between our agent and a baseline -- see env.make_env for how
to give the far side a different roster if that is ever wanted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from poke_env.teambuilder import ConstantTeambuilder, Teambuilder, TeambuilderPokemon

BATTLE_FORMAT = "gen9customgame"
"""Format both sides are played in: fixed teams are supplied, singles, level 100."""

TEAM_SIZE = 6
MOVES_PER_MON = 4

CANON_TEAM = """Great Tusk @ Leftovers
Ability: Protosynthesis
Level: 100
Tera Type: Ground
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Earthquake
- Rapid Spin
- Ice Spinner
- Stealth Rock

Gholdengo @ Air Balloon
Ability: Good as Gold
Level: 100
Tera Type: Steel
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Recover
- Nasty Plot

Dragapult @ Choice Specs
Ability: Infiltrator
Level: 100
Tera Type: Dragon
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Draco Meteor
- Shadow Ball
- U-turn
- Fire Blast

Kingambit @ Leftovers
Ability: Supreme Overlord
Level: 100
Tera Type: Dark
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance

Garganacl @ Leftovers
Ability: Purifying Salt
Level: 100
Tera Type: Rock
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Salt Cure
- Recover
- Protect
- Iron Defense

Iron Valiant @ Booster Energy
Ability: Quark Drive
Level: 100
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Moonblast
- Thunderbolt
- Aura Sphere
- Calm Mind
"""

# The one roster used by the agent and by both baselines (they mirror each other).
CANON_TEAM_PACKED = ConstantTeambuilder(CANON_TEAM).packed_team

# An alternative agent roster (Kanto-leaning six), for showing a different team against the same
# opponent. Note what the embedding can and cannot see: Dragon Dance / Swords Dance / Calm Mind /
# Soft-Boiled / Thunder Wave / Trick / Stealth Rock / Roost all have base power 0, so their move slots carry no identity -- see
# lab_ca.env.embed_battle and spikes/007-team-swap/CONCLUSION.md. A frozen policy trained on the
# canon mirror is not expected to play this team well; that is a measurement, not a bug.
MIXED_TEAM = """Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Level: 100
Tera Type: Normal
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Extreme Speed
- Earthquake
- Ice Spinner

Clefable @ Leftovers
Ability: Magic Guard
Level: 100
Tera Type: Steel
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Moonblast
- Soft-Boiled
- Calm Mind
- Thunder Wave

Kingambit @ Leftovers
Ability: Supreme Overlord
Level: 100
Tera Type: Dark
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance

Gholdengo @ Choice Scarf
Ability: Good as Gold
Level: 100
Tera Type: Steel
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Focus Blast
- Trick

Zapdos @ Heavy-Duty Boots
Ability: Static
Level: 100
Tera Type: Flying
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Hurricane
- Volt Switch
- Roost

Great Tusk @ Leftovers
Ability: Protosynthesis
Level: 100
Tera Type: Ground
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Earthquake
- Rapid Spin
- Ice Spinner
- Stealth Rock
"""

# Registered team names, so a CLI flag can say "mixed" instead of echoing a whole team string.
TEAM_REGISTRY = {"canon": CANON_TEAM, "mixed": MIXED_TEAM}


def load_team(spec: str) -> str:
    """Resolve a team spec to team text: a registry name, a file path, or inline team text.

    Lets ``--agent-team``/``--opponent-team`` accept ``canon``, ``mixed``, ``/path/to/team.txt``,
    or a team pasted straight onto the command line.
    """
    if spec in TEAM_REGISTRY:
        return TEAM_REGISTRY[spec]
    candidate = Path(spec)
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return spec



def pack_team(team: str) -> str:
    """Return the packed form of a team (accepts import or packed text)."""
    return ConstantTeambuilder(team).packed_team


def parse_team(team: str) -> List[TeambuilderPokemon]:
    """Parse a team (import or packed text) into TeambuilderPokemon objects."""
    if "|" in team:
        return Teambuilder.parse_packed_team(team)
    return Teambuilder.parse_showdown_team(team)


def showdown_dir() -> Path:
    """Path to the local pokemon-showdown checkout (repo_root/showdown)."""
    return Path(__file__).resolve().parents[2] / "showdown"


def validate_team(team: str = CANON_TEAM, fmt: str = BATTLE_FORMAT) -> tuple[bool, str]:
    """Ask the local Showdown engine to validate `team` for `fmt`.

    Returns (ok, message). The team is piped to `node pokemon-showdown validate-team <fmt>`,
    which exits 0 when the team is legal and prints the reasons when it is not.
    """
    checkout = showdown_dir()
    if not (checkout / "pokemon-showdown").is_file() or not (checkout / "dist").is_dir():
        return False, f"no built Showdown checkout at {checkout} (run scripts/showdown_setup.sh)"

    import_text = team if "|" not in team else Teambuilder.join_team(Teambuilder.parse_packed_team(team))
    proc = subprocess.run(
        ["node", "pokemon-showdown", "validate-team", fmt, "--skip-build"],
        cwd=checkout,
        input=import_text,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output
