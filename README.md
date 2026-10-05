# lab_ca — DQN agent for 1v1 Pokémon Showdown battles

A reinforcement-learning agent (Stable-Baselines3 DQN, with action masking) that learns to win
1v1 singles battles on a **local** Pokémon Showdown server, plus the logged metrics and graphs
an external report is written from. The repo produces **data and graphs**, not report prose.

Design spec: [`docs/superpowers/specs/2026-10-04-pokemon-rl-battle-agent-design.md`](docs/superpowers/specs/2026-10-04-pokemon-rl-battle-agent-design.md)
Execution plan / phase status: [`plan.md`](plan.md)
poke-env 0.16.1 library edges we hit and worked around: [`docs/poke-env-0.16.1-gotchas.md`](docs/poke-env-0.16.1-gotchas.md)

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
├── spikes/              # throwaway probes with their measured conclusions (update cost, figure audit)
├── runs/                # training artifacts (gitignored)
└── report_assets/       # final graphs + metrics.json for the report (Phase 5)
```

## How the pieces fit

One battle = one episode; one turn = one step. `train.py` builds a `SingleAgentWrapper` (our
learner on side 1, a scripted baseline on side 2) around `PokemonSinglesEnv`, and hands it to
SB3's DQN. The observation is a dict — a 33-float state vector plus the legality mask — and
masking is enforced in two places: illegal actions score `-inf` in both Q-networks (so the greedy
action *and* the TD target's `max` are masked), and any illegally sampled exploratory action is
snapped onto a legal one before it reaches the replay buffer (`policy.py`).
