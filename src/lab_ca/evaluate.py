"""Phase 4 -- frozen-policy evaluation, per the plan's two passes.

The win rates in ``runs/<run>/episodes.jsonl`` are **training-time** numbers: a moving policy at
``epsilon = 0.02``, measured while it was still learning. They are not the report's evaluation
numbers. This module runs the *frozen* greedy masked policy for a fixed number of battles against
each scripted baseline and reports win rates with Wilson intervals, so the report can state a
number a reader can check.

Pass 1 -- metrics (parallel, unwatched):

    python -m lab_ca.evaluate --checkpoint runs/p3_400k/model_final.zip --battles 200
    python -m lab_ca.evaluate --agent random --out runs/eval_random_floor.json --battles 200

Pass 2 -- watchable demo (one battle at a time, replays saved locally):

    python -m lab_ca.evaluate --checkpoint runs/p3_400k/model_final.zip --demo --demo-battles 3

Watch a fight turn by turn as it happens (this deployment has no browser client -- see
``spikes/006-live-demo/CONCLUSION.md`` -- so the terminal is the live view):

    python -m lab_ca.evaluate --checkpoint runs/p3_400k/model_final.zip --demo --demo-battles 2 \\
        --demo-live --demo-delay 0.4 --agent-team mixed --opponent-team canon

Both passes accept ``--agent-team``/``--opponent-team`` (a registered name from
``lab_ca.teams.TEAM_REGISTRY``, a path to a team file, or inline team text).

Summary across evaluated checkpoints (table + figure):

    python -m lab_ca.evaluate --summarize runs/p3_400k/eval.json runs/p3b_100k/eval.json \\
        --out runs/phase4_eval

Parallelism, precisely: ``PokeEnv`` drives exactly **one** battle (agent1 vs agent2) and builds both
players with ``max_concurrent_battles=1``, so ``--concurrent 8`` means eight independent
environments inside a ``SubprocVecEnv`` -- eight battles live at once -- not one environment with
eight battles in it. Each env therefore gets its own randomly generated accounts; the demo, which
runs one env, overrides them with recognizable names so a spectate link and a replay title read
like something.

Replays are written by poke-env from the protocol messages it already tracks
(``AbstractBattle.save_replay`` builds the HTML locally), so ``save_replays`` needs no upload to a
public replay service and works against this project's ``--no-security`` server.
"""

from __future__ import annotations

import argparse
import functools
import glob
import html
import json
import math
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from lab_ca.env import make_env
from lab_ca.opponent import OPPONENT_KINDS
from lab_ca.policy import MaskedDQN, MaskedQNetwork
from lab_ca.teams import CANON_TEAM, TEAM_REGISTRY, load_team
from lab_ca.train import OutcomeWrapper

AGENT_KINDS = ("dqn", "random")

# A battle that runs longer than this is stalled (real battles here average ~26 turns); bail out
# with fewer battles recorded rather than hanging the evaluation on a dead connection.
MAX_TURNS_PER_BATTLE = 300

# Demo identities: recognizable in a spectate link and in the replay's title.
#
# The opponent's handle reads like a trainer because that is what the replay is showing a viewer --
# a fight against a named opponent. It is still a scripted baseline, and the write-up must keep
# saying so: a trainer-flavoured handle is presentation, not evidence of an in-game AI.
# Overridable per run with --demo-opponent-name; the two handles must not be prefixes of each other,
# because replays are attributed to a side by which handle their filename starts with.
DEMO_AGENT_USERNAME = "lab_ca_dqn"
DEMO_OPPONENT_USERNAME = "RivalBlue"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, in percent.

    Preferred to the normal approximation at the extremes this evaluation lives in: beating the
    random baseline lands at or near 100%, where the naive interval collapses to a zero-width band
    and prints a fake +/-0.0. ``total == 0`` returns the maximally uninformative (0, 100).
    """
    if total <= 0:
        return 0.0, 100.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    # rounded at source: an interval reported to more than two decimals is noise, and unrounded
    # arithmetic gave 99.99999999999999 for a perfect score instead of 100
    return (
        round(max(0.0, centre - half) * 100.0, 2),
        round(min(1.0, centre + half) * 100.0, 2),
    )


def summarize_outcomes(outcomes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Win rate (over decided battles), Wilson interval, counts and battle-length stats.

    Unfinished battles are reported separately and excluded from the win-rate denominator: a
    truncated battle is not evidence of a loss, and folding it in would understate the agent
    exactly when something is flaky.
    """
    total = len(outcomes)
    unfinished = sum(1 for outcome in outcomes if outcome.get("outcome") == "unfinished")
    wins = sum(1 for outcome in outcomes if outcome.get("won"))
    decided = total - unfinished
    low, high = wilson_interval(wins, decided)
    # means over *decided* battles only: one stuck 300-turn battle would otherwise dominate both
    # statistics, and the stuck battle is already visible as `unfinished`
    steps = [int(outcome.get("steps", 0)) for outcome in outcomes if outcome.get("outcome") != "unfinished"]
    return {
        "battles": total,
        "decided": decided,
        "wins": wins,
        "losses": decided - wins,
        "unfinished": unfinished,
        "win_rate_pct": (100.0 * wins / decided) if decided else None,
        "win_rate_ci95_pct": [round(low, 2), round(high, 2)],
        "steps_mean": (sum(steps) / len(steps)) if steps else None,
        "steps_max": max(steps) if steps else None,
        "battles_detail": [dict(outcome) for outcome in outcomes],
    }


def _wrapped_env(
    opponent: str,
    *,
    save_replays: Any = False,
    agent_username: Optional[str] = None,
    opponent_username: Optional[str] = None,
    agent_team: str = CANON_TEAM,
    opponent_team: str = CANON_TEAM,
) -> gym.Env:
    """One monitored, outcome-annotated evaluation env.

    Module-level (not a closure) because ``SubprocVecEnv`` pickles this callable into the worker
    processes.
    """
    kwargs: Dict[str, Any] = {
        "save_replays": save_replays,
        "agent_team": agent_team,
        "opponent_team": opponent_team,
    }
    if agent_username:
        kwargs["account_configuration1"] = AccountConfiguration(agent_username, None)
    if opponent_username:
        kwargs["account_configuration2"] = AccountConfiguration(opponent_username, None)
    return Monitor(OutcomeWrapper(make_env(opponent, **kwargs)))


def build_vec_env(
    opponent: str,
    concurrent: int = 1,
    *,
    save_replays: Any = False,
    agent_username: Optional[str] = None,
    opponent_username: Optional[str] = None,
    seed: int = 0,
    agent_team: str = CANON_TEAM,
    opponent_team: str = CANON_TEAM,
) -> VecEnv:
    """``concurrent`` independent single-battle envs (see the module docstring)."""
    if opponent not in OPPONENT_KINDS:
        raise ValueError(f"unknown opponent {opponent!r}; expected {OPPONENT_KINDS}")
    if concurrent < 1:
        raise ValueError("concurrent must be >= 1")
    factory = functools.partial(
        _wrapped_env,
        opponent,
        save_replays=save_replays,
        agent_username=agent_username,
        opponent_username=opponent_username,
        agent_team=agent_team,
        opponent_team=opponent_team,
    )
    vec_env: VecEnv = (
        SubprocVecEnv([factory] * concurrent) if concurrent > 1 else DummyVecEnv([factory])
    )
    vec_env.seed(seed)
    return vec_env


def load_agent(checkpoint: Path, vec_env: VecEnv, device: str = "cpu") -> MaskedDQN:
    """Load a checkpoint as a frozen masked agent.

    Two properties the evaluation depends on are asserted rather than assumed: the loaded object is
    our ``MaskedDQN`` (so the epsilon branch cannot execute an illegal action) and its online Q-net
    is a ``MaskedQNetwork`` (so the greedy action is masked). ``exploration_rate`` is zeroed as well
    so even a non-deterministic ``predict`` call would be greedy.
    """
    model = MaskedDQN.load(str(checkpoint), env=vec_env, device=device)
    if not isinstance(model, MaskedDQN):
        raise TypeError(f"loaded {type(model).__name__}, expected MaskedDQN (masking would be off)")
    if not isinstance(model.policy.q_net, MaskedQNetwork):
        raise TypeError("loaded Q-network is not MaskedQNetwork: illegal actions could be chosen")
    model.exploration_rate = 0.0
    return model


def choose_actions(
    model: Optional[MaskedDQN], obs: Any, agent: str, rng: np.random.Generator
) -> np.ndarray:
    """Greedy masked actions for the agent, or uniformly legal actions for the random floor."""
    if agent == "random":
        masks = np.atleast_2d(np.asarray(obs["action_mask"]).astype(bool))
        return np.array([rng.choice(np.flatnonzero(row)) for row in masks])
    if agent != "dqn":
        raise ValueError(f"unknown agent {agent!r}; expected {AGENT_KINDS}")
    assert model is not None, "agent 'dqn' needs a loaded model"
    actions, _ = model.predict(obs, deterministic=True)
    return actions


def run_battles(
    vec_env: VecEnv,
    *,
    model: Optional[MaskedDQN] = None,
    agent: str = "dqn",
    opponent: str = "heuristic",
    battles: int = 1,
    rng: Optional[np.random.Generator] = None,
    turn_delay: float = 0.0,
    on_step: Optional[Callable[[VecEnv, int], None]] = None,
    record_tags: bool = False,
) -> List[Dict[str, Any]]:
    """Play until ``battles`` episodes have finished; return one record per finished battle.

    With ``concurrent > 1`` the last step can finish more than the wanted number of battles, so the
    list is truncated to exactly ``battles`` and the extras are dropped (they are reported by
    neither the counts nor the rate).

    ``record_tags`` also records which battle each finished episode belonged to. The demo needs that
    to join a replay file to its outcome; everything else leaves it off, since it costs an attribute
    walk per step.
    """
    rng = rng or np.random.default_rng()
    outcomes: List[Dict[str, Any]] = []
    obs = vec_env.reset()
    steps = 0
    limit = max(1, battles) * MAX_TURNS_PER_BATTLE
    while len(outcomes) < battles and steps < limit:
        # read the in-flight battle *before* the step: that is the battle this step may finish (after
        # it, the environment may already have started the next one)
        tag = _live_battle_tag(vec_env) if record_tags else None
        obs, _rewards, _dones, infos = vec_env.step(choose_actions(model, obs, agent, rng))
        steps += 1
        if turn_delay:
            time.sleep(turn_delay)
        if on_step is not None:
            on_step(vec_env, steps)
        for info in infos:
            if "episode" not in info:  # Monitor fills this only when an episode ends
                continue
            outcomes.append(
                {
                    "opponent": opponent,
                    "won": bool(info.get("win", False)),
                    "outcome": info.get("outcome", "unknown"),
                    "steps": int(info["episode"]["l"]),
                    "tag": tag,
                }
            )
            if len(outcomes) >= battles:
                break
    return outcomes[:battles]


def _live_battle_tag(vec_env: VecEnv) -> Optional[str]:
    """Best-effort tag of the battle in flight, for the spectate link (never raises)."""
    try:
        env: Any = vec_env.envs[0] if hasattr(vec_env, "envs") else None
        for _ in range(6):
            if env is None:
                return None
            battle = getattr(env, "battle1", None)
            if battle is not None:
                tag = getattr(battle, "battle_tag", None)
                return str(tag) if tag else None
            env = getattr(env, "env", None)
    except Exception:
        return None
    return None


def spectate_url(battle_tag: str) -> str:
    """The local client's URL for a battle room, derived from the server configuration."""
    room = battle_tag[1:] if battle_tag.startswith(">") else battle_tag
    room = room[len("battle-") :] if room.startswith("battle-") else room
    http_base = LocalhostServerConfiguration.websocket_url.split("/showdown")[0]
    http_base = http_base.replace("ws://", "http://").replace("wss://", "https://")
    return f"{http_base}/{room}"


# Protocol line types worth showing a human. ``|request|`` is deliberately absent: it is a full
# battle-state JSON blob sent every turn, and it is most of the stream's bytes.
LIVE_LINE_TYPES = frozenset({
    "turn", "move", "switch", "drag", "faint",
    "-damage", "-heal", "-status", "-curestatus", "-boost", "-unboost", "-setboost",
    "-supereffective", "-resisted", "-crit", "-miss", "-immune", "-fail", "-activate",
    "-start", "-end", "-sidestart", "-sideend", "-weather", "-fieldstart", "-prepare",
    "cant", "win", "tie",
})


def find_battle(vec_env: VecEnv) -> Optional[Any]:
    """The in-flight battle object, walking the same wrapper chain ``_live_battle_tag`` uses."""
    env: Any = vec_env.envs[0] if hasattr(vec_env, "envs") else None
    for _ in range(8):
        if env is None:
            return None
        battle = getattr(env, "battle1", None)
        if battle is not None:
            return battle
        env = getattr(env, "env", None)
    return None


def print_live_lines(battle: Optional[Any], seen: int) -> int:
    """Print protocol lines this battle has received since ``seen``; return the new total.

    ``AbstractBattle._replay_data`` is where poke-env appends every message as it arrives -- the same
    list ``save_replay`` renders from -- so this is a genuine real-time view of the battle with no
    client and no extra connection. It is a private attribute, hence ``getattr``: a poke-env bump
    that renames it degrades to silence rather than crashing a demo.
    """
    data = getattr(battle, "_replay_data", None)
    if data is None:
        return seen
    for split in list(data)[seen:]:
        if len(split) > 1 and split[1] in LIVE_LINE_TYPES:
            prefix = ">>>" if split[1] == "turn" else "   "
            print(f"{prefix} " + "  ".join(part for part in split[1:] if part).strip())
    return len(data)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # not a repo / git missing: the evaluation is still reproducible enough
        return "unknown"


def _source_run_provenance(checkpoint: Path) -> Dict[str, Any]:
    """The training run a checkpoint came from, if its ``run_config.json`` is on disk."""
    run_config = checkpoint.parent / "run_config.json"
    if not run_config.exists():
        return {}
    payload = json.loads(run_config.read_text(encoding="utf-8"))
    return {
        "source_run": checkpoint.parent.name,
        "source_config": payload.get("config"),
        "source_git_commit": payload.get("git_commit"),
    }


def _checkpoint_steps(checkpoint: Path) -> Optional[int]:
    """Step count parsed from the checkpoint filename (``model_final`` has none)."""
    stem = checkpoint.stem
    if stem == "model_final":
        return None
    parts = [part for part in stem.split("_") if part.isdigit()]
    return int(parts[-1]) if parts else None


def write_payload(path: Path, updates: Dict[str, Any]) -> Path:
    """Merge ``updates`` into an existing ``eval.json`` so Pass 2 cannot clobber Pass 1."""
    path = Path(path)
    payload: Dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    payload["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@dataclass
class EvalConfig:
    """Everything that shapes one evaluation; dumped into ``eval.json``.

    The parser takes its defaults from this dataclass (one source of truth -- two copies of a
    default is how the adopted training config silently failed to reach the CLI).
    """

    checkpoint: str = ""
    agent: str = "dqn"
    opponents: tuple = OPPONENT_KINDS
    battles: int = 200
    concurrent: int = 8
    seed: int = 0
    device: str = "cpu"
    out: str = ""
    demo: bool = False
    demo_battles: int = 3
    demo_opponents: tuple = ("heuristic",)
    demo_delay: float = 0.0
    demo_live: bool = False
    demo_dir: str = ""
    demo_opponent_name: str = DEMO_OPPONENT_USERNAME
    agent_team: str = "canon"
    opponent_team: str = "canon"

    def resolved_demo_opponent_name(self) -> str:
        """The opponent's demo handle, guarded where a bad value would silently corrupt artifacts.

        Sides are attributed by ``filename.startswith(DEMO_AGENT_USERNAME)`` and the replay filename
        is ``"<handle> - <tag>.html"``, so a handle that starts with the agent's, contains a space,
        or contains ``" - "`` mislabels or unparses every replay instead of failing.
        """
        name = self.demo_opponent_name or DEMO_OPPONENT_USERNAME
        if not name or any(char.isspace() for char in name):
            raise ValueError("--demo-opponent-name must be a single token with no spaces")
        if " - " in name:
            raise ValueError("--demo-opponent-name must not contain ' - ' (the replay filename split)")
        if name.startswith(DEMO_AGENT_USERNAME):
            raise ValueError(
                f"--demo-opponent-name must not start with the agent handle {DEMO_AGENT_USERNAME!r}"
                " -- that prefix is how a replay is attributed to a side"
            )
        return name

    def resolved_demo_dir(self) -> Path:
        """Where the demo's replays go: ``--demo-dir``, else ``demo/`` next to the checkpoint.

        Overridable because a second demo with a different team would otherwise drop its replays
        into the directory the report already links.
        """
        if self.demo_dir:
            return Path(self.demo_dir)
        return Path(self.checkpoint).parent / "demo"

    def resolved_agent_team(self) -> str:
        """The learner's team text (``agent_team`` as a registry name, path, or inline text)."""
        return load_team(self.agent_team)

    def resolved_opponent_team(self) -> str:
        """The baseline's team text."""
        return load_team(self.opponent_team)

    def team_provenance(self) -> Dict[str, str]:
        """Team specs for the payload: names stay readable, inline text is recorded as such."""
        return {
            "agent": self.agent_team if self.agent_team in TEAM_REGISTRY else "inline/custom",
            "opponent": self.opponent_team if self.opponent_team in TEAM_REGISTRY else "inline/custom",
            "agent_team_text": self.resolved_agent_team(),
            "opponent_team_text": self.resolved_opponent_team(),
        }

    def resolved_out(self) -> Path:
        """Where ``eval.json`` goes: next to the checkpoint, or an explicit ``--out``."""
        if self.out:
            return Path(self.out)
        if not self.checkpoint:
            raise ValueError("give --checkpoint or --out so the results have somewhere to go")
        return Path(self.checkpoint).parent / "eval.json"


def evaluate_checkpoint(config: EvalConfig) -> Path:
    """Pass 1: greedy masked policy vs every requested baseline; write ``eval.json``."""
    checkpoint = Path(config.checkpoint) if config.checkpoint else None
    out_path = config.resolved_out()
    rng = np.random.default_rng(config.seed)
    results: Dict[str, Any] = {}

    for opponent in config.opponents:
        vec_env = build_vec_env(
            opponent, config.concurrent, seed=config.seed,
            agent_team=config.resolved_agent_team(),
            opponent_team=config.resolved_opponent_team(),
        )
        try:
            model = None
            if config.agent == "dqn":
                assert checkpoint is not None, "agent 'dqn' needs --checkpoint"
                model = load_agent(checkpoint, vec_env, config.device)
            started = time.time()
            outcomes = run_battles(
                vec_env, model=model, agent=config.agent, opponent=opponent,
                battles=config.battles, rng=rng,
            )
            elapsed = time.time() - started
        finally:
            vec_env.close()
        summary = summarize_outcomes(outcomes)
        summary["wall_clock_s"] = round(elapsed, 2)
        summary["battles_per_s"] = round(len(outcomes) / elapsed, 2) if elapsed > 0 else None
        results[opponent] = summary
        rate = summary["win_rate_pct"]
        low, high = summary["win_rate_ci95_pct"]
        print(
            f"[eval] {opponent:<9} {summary['wins']}/{summary['decided']} decided"
            f" = {rate:.1f}% (95% CI {low:.1f}-{high:.1f})"
            f" in {elapsed:.1f}s"
            + (f", unfinished {summary['unfinished']}" if summary["unfinished"] else "")
        )

    payload: Dict[str, Any] = {
        "evaluated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_steps": _checkpoint_steps(checkpoint) if checkpoint else None,
        "agent": config.agent,
        "config": asdict(config),
        "results": results,
        "teams": config.team_provenance(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    payload.update(_source_run_provenance(checkpoint) if checkpoint else {})
    write_payload(out_path, payload)
    print(f"[eval] wrote {out_path}")
    return out_path


REPLAY_NAME = re.compile(r"^(?P<side>.+?) - (?P<tag>battle-.+)\.html$")
REPLAY_WIN_LINE = re.compile(r"\|win\|([^\s<\"\\]+)")
REPLAY_LOG_DATA = re.compile(
    r'<script type="text/plain" class="battle-log-data">(?P<log>.*?)</script>', re.DOTALL
)


def demo_out_path(config: "EvalConfig", demo_dir: Path) -> Path:
    """Where the demo payload goes: the run's ``eval.json``, unless ``--out`` overrides it.

    ``eval.json`` is where the plan wants ``demo_replays`` recorded, and ``write_payload`` merges, so
    the demo adds its keys next to the pass-1 ``results`` instead of replacing them.
    """
    return config.resolved_out()


def group_replays(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    """Replay files grouped by battle tag -- both sides of a battle write their own file."""
    grouped: Dict[str, List[Path]] = {}
    for path in paths:
        match = REPLAY_NAME.match(path.name)
        grouped.setdefault(match.group("tag") if match else path.stem, []).append(path)
    return grouped


def replay_result(path: Path) -> Dict[str, Any]:
    """What the replay file itself says: the ``|win|<username>`` line, and whether anyone forfeited.

    The file is self-describing, so the demo reports the result the artifact states instead of one
    inferred from the order files happened to be written in. The first version zipped a sorted file
    list against the outcome list, which quietly attached a win to a different battle's replay.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    winner = REPLAY_WIN_LINE.search(text)
    # the username runs to the end of the line in the embedded log, so trailing layout whitespace
    # would otherwise land in eval.json as part of the winner's name
    return {
        "winner": winner.group(1).strip() if winner else None,
        "forfeit": "|forfeit" in text,
    }


def write_log_view(replay_path: Path, out_path: Path) -> Optional[Path]:
    """Write an offline transcript of a replay, or ``None`` if it carries no log.

    poke-env embeds the whole protocol log in the replay, but renders it with
    ``play.pokemonshowdown.com/js/replay-embed.js`` -- so *viewing* a replay reaches the public
    client. This checkout ships no client at all (the page at the server root is a stub that
    redirects to the psim.us client), which makes the transcript the watchable artifact: one
    self-contained page, no network.
    """
    match = REPLAY_LOG_DATA.search(replay_path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    lines = "".join(f"<div>{html.escape(line)}</div>" for line in match.group("log").splitlines())
    out_path.write_text(
        "<!DOCTYPE html><meta charset='utf-8'>"
        f"<title>{html.escape(replay_path.stem)}</title>"
        "<style>body{font:13px/1.6 ui-monospace,Consolas,monospace;margin:24px;max-width:1000px}"
        "h1{font-size:15px;font-family:system-ui,sans-serif}</style>"
        f"<h1>{html.escape(replay_path.stem)}</h1>{lines}\n",
        encoding="utf-8",
    )
    return out_path


def run_demo(config: EvalConfig) -> Path:
    """Pass 2: one battle at a time, replays saved locally, live spectate links printed."""
    checkpoint = Path(config.checkpoint)
    demo_dir = config.resolved_demo_dir()
    demo_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    opponent_handle = config.resolved_demo_opponent_name()
    replays: List[Dict[str, Any]] = []
    announced: set = set()
    live_seen = 0

    def _announce(vec_env: VecEnv, step: int) -> None:
        nonlocal live_seen
        tag = _live_battle_tag(vec_env)
        if tag and tag not in announced:
            announced.add(tag)
            print(f"[demo] battle live: {tag}\n[demo] spectate: {spectate_url(tag)}")
            live_seen = 0  # a new battle: restart the line stream from its first message
        if config.demo_live:
            live_seen = print_live_lines(find_battle(vec_env), live_seen)

    for opponent in config.demo_opponents:
        vec_env = build_vec_env(
            opponent, 1, save_replays=str(demo_dir),
            agent_username=DEMO_AGENT_USERNAME, opponent_username=opponent_handle,
            seed=config.seed,
            agent_team=config.resolved_agent_team(),
            opponent_team=config.resolved_opponent_team(),
        )
        before = set(demo_dir.glob("*.html"))
        try:
            model = load_agent(checkpoint, vec_env, config.device) if config.agent == "dqn" else None
            outcomes = run_battles(
                vec_env, model=model, agent=config.agent, opponent=opponent,
                battles=config.demo_battles, rng=rng, turn_delay=config.demo_delay,
                on_step=_announce, record_tags=True,
            )
        finally:
            vec_env.close()
        written = sorted(set(demo_dir.glob("*.html")) - before)
        by_tag = {outcome["tag"]: outcome for outcome in outcomes if outcome.get("tag")}
        grouped = group_replays(written)
        for tag, paths in sorted(grouped.items()):
            outcome = by_tag.get(tag)
            for path in paths:
                record = {
                    "path": str(path),
                    "battle": tag,
                    "side": "agent" if path.name.startswith(DEMO_AGENT_USERNAME) else "opponent",
                    "opponent": opponent,
                    "counted": outcome is not None,
                    "won": outcome.get("won") if outcome else None,
                    "steps": outcome.get("steps") if outcome else None,
                    **replay_result(path),
                }
                if record["counted"] and record["side"] == "agent":
                    transcript = write_log_view(path, demo_dir / f"{tag}-transcript.html")
                    if transcript:
                        record["transcript"] = str(transcript)
                replays.append(record)
        counted = len(by_tag)
        stray = len(grouped) - counted
        # the environment accepts a fresh challenge the moment a battle ends, so one last battle can
        # arrive between the final counted battle and vec_env.close() -- it ends in a forfeit. Say so
        # instead of passing its replay off as one of the requested games.
        print(
            f"[demo] {opponent}: {counted} of {config.demo_battles} requested battle(s) counted, "
            f"{len(written)} replay file(s) in {demo_dir}"
            + (f" ({stray} unrequested battle(s) the env started as it closed)" if stray else "")
        )

    demo_out = demo_out_path(config, demo_dir)
    if not demo_out.exists():
        print(
            f"[demo] note: {demo_out} does not exist yet -- the demo keys are written there and a "
            f"later pass-1 evaluation merges into the same file"
        )
    counted_battles = {entry["battle"] for entry in replays if entry["counted"]}
    all_battles = {entry["battle"] for entry in replays}
    return write_payload(
        demo_out,
        {
            "demo": {
                "battles": {opponent: config.demo_battles for opponent in config.demo_opponents},
                "delay_s": config.demo_delay,
                "directory": str(demo_dir),
                "teams": config.team_provenance(),
                "handles": {"agent": DEMO_AGENT_USERNAME, "opponent": opponent_handle},
                "counted_battles": len(counted_battles),
                "unrequested_battles": len(all_battles - counted_battles),
                "replays": replays,
                # the demo runs at its own revision: the top-level provenance belongs to the pass-1
                # results, so the demo section carries its own instead of inheriting that claim
                "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
            },
            # what the report links: the agent's own copy of every counted battle
            "demo_replays": [
                entry["path"] for entry in replays if entry["counted"] and entry["side"] == "agent"
            ],
        },
    )


def _expand_eval_paths(patterns: Sequence[str]) -> List[Path]:
    """``eval.json`` paths from explicit files, directories, or globs."""
    found: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            candidate = path / "eval.json"
            if candidate.exists():
                found.append(candidate)
            continue
        if path.exists():
            found.append(path)
            continue
        found.extend(Path(match) for match in glob.glob(pattern))
    unique: List[Path] = []
    for path in found:
        if path not in unique:
            unique.append(path)
    return unique


def summarize_evals(eval_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """One row per (eval file, opponent): the table and the figure are both drawn from this."""
    rows: List[Dict[str, Any]] = []
    for path in eval_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        results = payload.get("results") or {}
        for opponent, result in sorted(results.items()):
            rows.append(
                {
                    # a run dir is the honest label; for a loose file like runs/eval_random_floor.json
                    # the parent directory is just "runs", so fall back to the file's own stem
                    "run": payload.get("source_run")
                    or (
                        Path(path).parent.name
                        if Path(path).parent.name not in ("runs", ".")
                        else Path(path).stem
                    ),
                    "agent": payload.get("agent"),
                    "opponent": opponent,
                    "battles": result.get("decided"),
                    "unfinished": result.get("unfinished"),
                    "win_rate_pct": result.get("win_rate_pct"),
                    "ci_low": (result.get("win_rate_ci95_pct") or [None, None])[0],
                    "ci_high": (result.get("win_rate_ci95_pct") or [None, None])[1],
                    "steps_mean": result.get("steps_mean"),
                    "eval_path": str(path),
                }
            )
    return rows


def write_summary(rows: Sequence[Dict[str, Any]], out_stem: Path, *, figure: bool = True) -> Dict[str, Path]:
    """Markdown table (+ optional bar chart with Wilson intervals) for the report."""
    out_stem = Path(out_stem)
    written: Dict[str, Path] = {}

    lines = [
        "# Phase 4 -- frozen-policy evaluation",
        "",
        "Greedy masked policy (epsilon disabled) vs the scripted baselines on the fixed roster.",
        "50% is parity. Intervals are 95% Wilson score intervals over decided battles.",
        "",
        "| run | agent | opponent | battles | win rate | 95% CI | mean turns |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        rate = row["win_rate_pct"]
        rate_text = "n/a" if rate is None else f"{rate:.1f}%"
        ci = (
            "n/a"
            if row["ci_low"] is None
            else f"{row['ci_low']:.1f}-{row['ci_high']:.1f}"
        )
        steps = row["steps_mean"]
        lines.append(
            f"| {row['run']} | {row['agent']} | {row['opponent']} | {row['battles']} | "
            f"{rate_text} | {ci} | {'n/a' if steps is None else f'{steps:.1f}'} |"
        )
    written["markdown"] = out_stem.with_suffix(".md")
    written["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    if figure:
        written.update(_summary_figure(rows, out_stem))
    return written


def _figure_caption(frame: Any) -> Dict[str, str]:
    """The title and subtitle shared by the seaborn and plotly twins.

    The title names the metric and the opponents because a bare "Frozen-policy evaluation" leaves a
    reader to reconstruct what is plotted from the legend.
    """
    opponents = ", ".join(sorted(frame["opponent"].unique()))
    counts = sorted({count for count in frame["battles"].dropna().unique()})
    n_note = f"n = {counts[0]:.0f} decided battles per bar" if len(counts) == 1 else "n varies per bar"
    return {
        "title": f"Frozen-policy evaluation — greedy masked policy vs {opponents}",
        "subtitle": (
            f"bars: win rate · whiskers: 95% Wilson interval · dashed line: 50% parity · {n_note}"
        ),
    }


def eval_summary_figure(rows: Sequence[Dict[str, Any]]) -> Tuple[Any, Any, Any]:
    """The seaborn panel, factored out as ``(figure, axis, frame)`` so tests can measure the layout.

    Layout claims about a figure are cheap to make and expensive to be wrong about, so the legend is
    required to clear the bars and the parity label by measurement (see tests/test_evaluate.py).
    """
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt

    frame = pd.DataFrame(rows)
    frame = frame[frame["win_rate_pct"].notna()]
    frame["run_opponent"] = frame["run"] + " · " + frame["opponent"]
    frame["error_low"] = frame["win_rate_pct"] - frame["ci_low"]
    frame["error_high"] = frame["ci_high"] - frame["win_rate_pct"]

    sns.set_theme(style="whitegrid", context="talk")
    runs = sorted(frame["run"].unique())
    opponents = sorted(frame["opponent"].unique())
    figure, axis = plt.subplots(figsize=(max(8.0, 1.5 * len(runs)), 6.8))
    palette = sns.color_palette("deep", n_colors=max(2, len(opponents)))
    colours = {opponent: palette[index] for index, opponent in enumerate(opponents)}
    # tick labels carry the run only: the opponent is what the colour means, and repeating it in
    # every label both crowds the axis and duplicates the legend
    width = 0.8 / max(1, len(opponents))
    for index, opponent in enumerate(opponents):
        group = frame[frame["opponent"] == opponent]
        positions = np.array([runs.index(run) for run in group["run"]]) - 0.4 + width * (index + 0.5)
        bars = axis.bar(
            positions, group["win_rate_pct"], width=width, color=colours[opponent], label=opponent,
            yerr=[group["error_low"], group["error_high"]], capsize=4,
            error_kw={"linewidth": 1.4},
        )
        for bar, rate, high in zip(bars, group["win_rate_pct"], group["ci_high"]):
            # the value is on the bar because a near-zero result (the random floor vs the heuristic)
            # is otherwise an invisible sliver that reads as a missing bar rather than a measured 1%
            axis.annotate(
                f"{rate:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, max(high, rate)),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9,
                color="0.25",
            )
    axis.set_xticks(range(len(runs)), runs, rotation=30, ha="right")
    axis.axhline(50.0, color="0.35", linestyle=(0, (4, 3)), linewidth=1.3)
    axis.annotate(
        "parity (50%)", xy=(0.995, 50.0), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points", ha="right", va="bottom", fontsize=10, color="0.2",
    )
    axis.set_ylabel("Win rate (%)")
    axis.set_xlabel("run")
    axis.set_ylim(0.0, 105.0)
    axis.tick_params(axis="x")
    caption = _figure_caption(frame)
    axis.set_title(f"{caption['title']}\n{caption['subtitle']}", fontsize=13)
    # The legend lives in a band reserved at the bottom of the figure, and below the axes rather
    # than inside them: every bar is anchored at zero, so any position inside the axes covers some
    # bar, and above the axes it collides with the title (both measured, see the layout test).
    # tight_layout(rect=...) is what reserves the band; a legend placed outside the axes is invisible
    # to plain tight_layout().
    figure.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, title="opponent", loc="lower left", bbox_to_anchor=(0.01, 0.0),
        ncol=len(opponents), frameon=False,
    )
    return figure, axis, frame


def _summary_figure(rows: Sequence[Dict[str, Any]], out_stem: Path) -> Dict[str, Path]:
    """Grouped bars with Wilson error bars -- seaborn PNG plus the plotly HTML twin."""
    import plotly.graph_objects as go
    from matplotlib import pyplot as plt

    if not rows:
        return {}
    figure, _axis, frame = eval_summary_figure(rows)
    caption = _figure_caption(frame)
    written: Dict[str, Path] = {}
    written["png"] = out_stem.with_suffix(".png")
    figure.savefig(written["png"], dpi=150, bbox_inches="tight")
    plt.close(figure)

    plotly_figure = go.Figure()
    for opponent, group in frame.groupby("opponent", sort=True):
        plotly_figure.add_trace(
            go.Bar(
                x=group["run"], y=group["win_rate_pct"], name=opponent,
                text=[f"{rate:.1f}%" for rate in group["win_rate_pct"]],
                textposition="outside",
                error_y=dict(
                    type="data", symmetric=False,
                    array=group["error_high"].tolist(), arrayminus=group["error_low"].tolist(),
                ),
                hovertemplate="%{x} vs " + opponent + "<br>%{y:.1f}%<extra></extra>",
            )
        )
    plotly_figure.add_hline(y=50.0, line=dict(color="#808080", width=1.4, dash="dash"),
                            annotation_text="parity (50%)")
    plotly_figure.update_layout(
        template="plotly_white", barmode="group", height=640,
        title=dict(
            text=f"{caption['title']}<br><sub>{caption['subtitle']}</sub>", x=0.02,
        ),
        yaxis=dict(title="Win rate (%)", range=[0, 105]),
        xaxis=dict(title="run", tickangle=-30),
        legend=dict(title="opponent", orientation="h", yanchor="bottom", y=-0.32, xanchor="left", x=0),
    )
    written["html"] = out_stem.with_suffix(".html")
    plotly_figure.write_html(written["html"], include_plotlyjs=True)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    defaults = EvalConfig()
    parser.add_argument("--checkpoint", default=defaults.checkpoint,
                        help="model_*.zip from a training run (not needed with --agent random)")
    parser.add_argument("--agent", choices=AGENT_KINDS, default=defaults.agent,
                        help="'random' = uniform legal actions, for a floor row in the report")
    parser.add_argument("--opponents", default=",".join(defaults.opponents),
                        help="comma-separated baselines to evaluate against")
    parser.add_argument("--battles", type=int, default=defaults.battles,
                        help="finished battles per opponent")
    parser.add_argument("--concurrent", type=int, default=defaults.concurrent,
                        help="parallel single-battle envs (PokeEnv drives one battle each)")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--out", default=defaults.out, help="eval.json path (default: next to checkpoint)")
    parser.add_argument("--demo", action="store_true", default=defaults.demo,
                        help="Pass 2: a few watchable battles, replays saved locally")
    parser.add_argument("--demo-battles", type=int, default=defaults.demo_battles)
    parser.add_argument("--demo-opponents", default=",".join(defaults.demo_opponents))
    parser.add_argument("--demo-delay", type=float, default=defaults.demo_delay,
                        help="seconds per turn, so a battle can actually be watched")
    parser.add_argument("--demo-live", action="store_true", default=defaults.demo_live,
                        help="with --demo: print each turn as it happens (pair with --demo-delay)")
    parser.add_argument("--demo-dir", default=defaults.demo_dir,
                        help="where the demo writes replays (default: demo/ next to the checkpoint)")
    parser.add_argument("--demo-opponent-name", default=defaults.demo_opponent_name,
                        help=f"the opponent's handle in the demo (default: {DEMO_OPPONENT_USERNAME})")
    parser.add_argument("--agent-team", default=defaults.agent_team,
                        help="learner's team: a registered name (canon, mixed), a file path, or "
                             "inline team text")
    parser.add_argument("--opponent-team", default=defaults.opponent_team,
                        help="baseline's team; same accepted forms as --agent-team")
    parser.add_argument("--summarize", nargs="*", default=None,
                        help="eval.json files/dirs/globs to tabulate and plot instead of evaluating")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.summarize is not None:
        paths = _expand_eval_paths(args.summarize or ["runs/*/eval.json"])
        if not paths:
            print("[eval] no eval.json found to summarize")
            return 1
        rows = summarize_evals(paths)
        stem = Path(args.out) if args.out else Path("runs/phase4_eval")
        written = write_summary(rows, stem)
        for kind, path in written.items():
            print(f"[eval] wrote {kind}: {path}")
        return 0

    config = EvalConfig(
        checkpoint=args.checkpoint,
        agent=args.agent,
        opponents=tuple(part.strip() for part in args.opponents.split(",") if part.strip()),
        battles=args.battles,
        concurrent=args.concurrent,
        seed=args.seed,
        device=args.device,
        out=args.out,
        demo=args.demo,
        demo_battles=args.demo_battles,
        demo_opponents=tuple(part.strip() for part in args.demo_opponents.split(",") if part.strip()),
        demo_delay=args.demo_delay,
        demo_live=args.demo_live,
        demo_dir=args.demo_dir,
        demo_opponent_name=args.demo_opponent_name,
        agent_team=args.agent_team,
        opponent_team=args.opponent_team,
    )
    if config.agent == "dqn" and not config.checkpoint:
        print("[eval] --checkpoint is required unless --agent random (or --summarize)")
        return 2
    if config.demo:
        run_demo(config)
    else:
        evaluate_checkpoint(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
