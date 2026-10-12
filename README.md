# lab_ca — DQN agent for 1v1 Pokémon Showdown battles

A reinforcement-learning agent (Stable-Baselines3 DQN, with action masking) that learns to win
1v1 singles battles on a **local** Pokémon Showdown server, plus the logged metrics and graphs
an external report is written from. The repo produces **data and graphs**, not report prose.

## Setup

```bash
# 1. Python deps. This repo is a member of the l_lab uv workspace: sync from the WORKSPACE root
#    with --all-packages (running plain `uv sync` inside lab_ca prunes the parent's deps).
cd ../            # l_lab/
uv sync --all-packages

# 2. Showdown server (Node). Clones + builds + patches localhost-only binding.
cd lab_ca
bash scripts/showdown_setup.sh      # once per machine: clone, npm install, node build
```

Node 22+, Python 3.13 and `uv` are expected on `PATH`.

## Running a training run

The server runs in **its own terminal** — it needs a TTY and `train.py` never spawns it:

```bash
# terminal 1
bash scripts/showdown_start.sh                 # add --skip-build for a faster start

# terminal 2
uv run python -m lab_ca.train --timesteps 2000 --opponent heuristic
```

Useful flags: `--opponent {heuristic,random}`, `--seed`, `--run-name`, `--checkpoint-every`,
`--runs-root`. A 2000-step run takes ~10 s on a local server, so longer runs are cheap: at the
adopted config ~110 steps/s, i.e. `--timesteps 100000` ≈ 15 min.

**The defaults are the adopted Phase-3 config** (arm B of the seed sweep — 4 gradient updates per
env step instead of SB3's 0.25, batch 128, target net refreshed every 1 000 steps). It beat the SB3
defaults by +22.2 pp on the final-200 win rate, with complete separation across 3 seeds each
(`runs/sweep_notes.md`). Hyperparameter flags (`--train-freq`, `--gradient-steps`, `--batch-size`,
`--target-update-interval`, `--net-arch`, `--learning-rate`, `--gamma`, `--buffer-size`,
`--exploration-fraction`, `--exploration-final-eps`) all default to that config; to reproduce the
SB3 baseline instead:

```bash
uv run python -m lab_ca.train --timesteps 100000 --train-freq 4 --gradient-steps 1 \
  --batch-size 32 --target-update-interval 10000
```

Each run writes to `runs/<run-name>/`:

| File | Contents |
|------|----------|
| `episodes.jsonl` | one row per finished episode — the metric contract (`episode, cumulative_reward, steps, win, outcome, epsilon, timestamp, opponent`) |
| `episodes.csv` | the same rows as CSV |
| `run_config.json` | every hyperparameter, seed, git commit and library version behind the run |
| `model_*.zip` | SB3 checkpoints, plus `model_final.zip` |
| `train.log` | stdout of the run (SB3's progress tables) when you `tee` it yourself |

`runs/` is gitignored: artifacts stay on disk for the report author, they are not committed.

## Graphs and metrics

`plots.py` renders from a run's `episodes.jsonl` (seaborn for the static PNG the report needs,
plotly for a self-contained interactive HTML), and writes `metrics.json`:

```bash
uv run python -m lab_ca.plots --run runs/<run>                        # three-panel curve check
uv run python -m lab_ca.plots --run runs/<run> --only success         # one panel
uv run python -m lab_ca.plots --compare runs/<a> runs/<b>             # overlay runs (A/B read)
uv run python -m lab_ca.plots --aggregate runs/<a> runs/<b> runs/<c>  # group by config, average seeds
```

Convergence is the first episode after which the 200-episode rolling win rate stays within `2σ` of
its final value (band floor ±5 pp), and `metrics.json` also carries the late-trend slope so
"plateaued" and "still climbing slowly" are distinguishable. `--aggregate` groups runs by the config
recorded in their `run_config.json` and shades the min–max envelope across seeds.

## Watching a fight

This deployment has no browser client (the checkout is the server package only), so the live view is
the terminal: `--demo-live` prints each turn as it happens, and `--demo-delay` sets the pace. Replays
and standalone transcript pages are written alongside, offline-viewable.

```bash
VP="C:/Users/Arnav Waghdhare/Desktop/Arnav20/Coding/Python/l_lab/.venv/Scripts/python.exe"

# live: your team vs the scripted baseline, turn by turn
"$VP" -m lab_ca.evaluate --checkpoint runs/p3_400k/model_final.zip \
  --demo --demo-battles 3 --demo-live --demo-delay 0.6 \
  --agent-team mixed --opponent-team canon \
  --demo-dir runs/agent_mixed_demo --out runs/agent_mixed_demo/eval.json
```

`--agent-team` / `--opponent-team` take a registered name (`canon`, `mixed` — see
`lab_ca.teams.TEAM_REGISTRY`), a path to a team file, or inline team text. `--demo-dir` keeps a
second demo from dropping replays into the directory the report already links.
`--demo-opponent-name` sets the opponent's handle in the demo (default `RivalBlue`) — presentation
only: the opponent is a scripted baseline, and the write-up should keep saying so.

Two things to expect when the team changes: a policy trained on the canon mirror does **not**
transfer (measured 90 % -> 0.5 %, see `spikes/007-team-swap/CONCLUSION.md`), and status/setup moves
are invisible to the observation because `embed_battle` encodes moves by base power and type only.
Retraining is the fix for the first; the second needs move identity in the observation.

## Tests

```bash
uv run python -m pytest -q                      # everything (~15 s with the server up)
uv run python -m pytest -q -m "not integration" # offline only: masking + logging logic
```

Tests marked `integration` need the local server; they skip themselves when it is not reachable.
`pytest-timeout` is configured so a stalled battle fails with a traceback instead of hanging.

## Layout

```
lab_ca/
├── showdown/            # local pokemon-showdown server clone (gitignored)
├── scripts/             # server setup/start, config patch
├── src/lab_ca/
│   ├── teams.py         # canon Gen-9 roster (packed-team format) + validation
│   ├── env.py           # SinglesEnv subclass: embed_battle, shaped calc_reward, safe teardown
│   ├── opponent.py      # scripted baselines (random, type-aware heuristic) on fixed teams
│   ├── policy.py        # action masking for SB3 DQN (masked Q-networks + exploration)
│   ├── train.py         # DQN training loop, per-episode metric logging, checkpoints
│   └── plots.py         # curves/metrics.json: seaborn PNG, plotly HTML, compare/aggregate modes
├── tests/               # offline unit tests + server-backed integration tests
├── spikes/              # throwaway probes with their measured conclusions (update cost, figure
│                        # audit, sweep stats, budget scaling, report-asset audit)
├── runs/                # training artifacts (gitignored)
└── report_assets/       # final graphs + metrics.json for the report (Phase 5)
```

## How the pieces fit

One battle = one episode; one turn = one step. `train.py` builds a `SingleAgentWrapper` (our
learner on side 1, a scripted baseline on side 2) around `PokemonSinglesEnv`, and hands it to
SB3's DQN. The observation is a dict — a 37-float state vector plus the legality mask — and
masking is enforced in two places: illegal actions score `-inf` in both Q-networks (so the greedy
action *and* the TD target's `max` are masked), and any illegally sampled exploratory action is
snapped onto a legal one before it reaches the replay buffer (`policy.py`).
