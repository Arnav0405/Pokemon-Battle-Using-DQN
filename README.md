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

**The server must already be running** — the demo is a poke-env client connecting to
`ws://localhost:8000/showdown/websocket`, and neither `evaluate.py` nor `train.py` spawns anything.
Started cold, it fails with a connection error before the first battle. So: one terminal for

```bash
bash scripts/showdown_start.sh --skip-build    # omit --skip-build for the full node build (~30 s)
```

(right after it prints `Worker 1 now listening on 0.0.0.0:8000`), then a second terminal for

```bash
VP="C:/Users/Arnav Waghdhare/Desktop/Arnav20/Coding/Python/l_lab/.venv/Scripts/python.exe"

# live: your team vs the scripted baseline, turn by turn
"$VP" -m lab_ca.evaluate --checkpoint runs/p3_400k/model_final.zip \
  --demo --demo-battles 3 --demo-live --demo-delay 0.6 \
  --agent-team mixed --opponent-team canon \
  --demo-dir runs/agent_mixed_demo --out runs/agent_mixed_demo/eval.json
```

The demo's "N of M requested battle(s) counted" line is the authoritative count: PokeEnv starts one
extra battle while tearing its env down, so the replay directory normally holds one stray file more
than you asked for.

`--agent-team` / `--opponent-team` take a registered name (`canon`, `mixed` — see
`lab_ca.teams.TEAM_REGISTRY`), a path to a team file, or inline team text. `--demo-dir` keeps a
second demo from dropping replays into the directory the report already links.
`--demo-opponent-name` sets the opponent's handle in the demo (default `RivalBlue`) — presentation
only: the opponent is a scripted baseline, and the write-up should keep saying so.

Two things to expect when the team changes: a policy trained on the canon mirror does **not**
transfer (measured 90 % -> 0.5 %, see `spikes/007-team-swap/CONCLUSION.md`), and status/setup moves
are invisible to the observation because `embed_battle` encodes moves by base power and type only.
Retraining is the fix for the first; the second needs move identity in the observation.

## Observation and action space

Both numbers below are the ones to quote; `spikes/007-team-swap/CONCLUSION.md` has the derivation.

- **Observation: 37 floats**, built by `FEATURE_NAMES` in `env.py` — 4 move base powers
  (`min(bp, 250)/100`, `-1` when the slot is not selectable), 4 move type effectivenesses vs the
  opponent's active, 6 own team HP fractions, the opponent active's HP fraction, 6 + 6 status
  one-hots, 5 + 5 boost values (`/6`). Handed to the learner as
  `{"observation": <37-vector>, "action_mask": <0/1 vector>}`.
  *The 33-feature figure in `plan.md` §Phase 0, `docs/superpowers/specs/…-design.md` and
  `HANDOFF.md` §2.4/§3.3 is stale — the code has been 37 since Phase 1, and
  `tests/test_env.py` pins the arithmetic (`N_FEATURES == 4*2 + 6 + 1 + 12 + 10`).*
- **Action: a declared `Discrete(26)`** (0–5 switch to team slot *i*, 6–9 move slot *i*,
  10–13 mega, 14–17 z-move, 18–21 dynamax, 22–25 terastallize), of which **only 0–17 are ever
  legal** in `gen9customgame`. The gimmick tail exists because it is baked into poke-env's
  `SinglesEnv` converter and is masked to illegal in every state we see. So "26" is the space and
  "18" is the reachable set — `HANDOFF.md` §2.4 says 26 and spike 007 says 18; both are right about
  different things.
- **The embedding carries no species and no move identity.** Every feature is derived on the fly, so
  a team swap lands in the same space with different values and the checkpoint still loads. The cost
  is that all 0-BP moves collapse to one value (Recover and Protect are both Normal-type, 0 BP ⇒
  permanently indistinguishable), which is why the agent cannot learn "Recover when low, Protect to
  scout" and can only learn a slot-order preference.

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
│   ├── evaluate.py      # frozen-policy evaluation: win rates, Wilson CIs, demo mode
│   └── plots.py         # curves/metrics.json: seaborn PNG, plotly HTML, compare/aggregate modes
├── tests/               # offline unit tests + server-backed integration tests
├── spikes/              # throwaway probes with their measured conclusions (update cost, figure
│                        # audit, sweep stats, budget scaling, report-asset audit, live demo,
│                        # team-swap transfer cost)
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

## References

The five papers the design decisions actually rest on. Each line says what it justifies here, so
the report can cite them where they apply rather than as a general reading list.

1. **Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 2015.**
   The DQN algorithm this repo uses, and the source of the defaults arm A reproduces (1M-transition
   replay buffer, target network, ε-greedy decay). Arm A *is* this baseline; arm B keeps the
   algorithm and changes only the update budget.
2. **Huang & Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms,"
   FLAIRS 2022** ([arXiv:2006.14171](https://arxiv.org/abs/2006.14171)). The theoretical cover for
   `policy.py`: masking is a state-dependent differentiable transform whose update is a valid policy
   gradient, and it scales as the invalid-action space grows while *invalid-action penalties do not*
   — penalties sometimes never reach the first reward. That failure mode is what the ~2 % early win
   rate against the type-aware opponent would look like if illegal moves were priced instead of
   masked. Also the source of the trap we avoid: naive masking (sample masked, backprop unmasked)
   explodes PPO-KL.
3. **Fedus et al., "Revisiting Fundamentals of Experience Replay," ICML 2020**
   ([arXiv:2007.06700](https://arxiv.org/html/2007.06700v1)). Defines the replay ratio as gradient
   updates per environment transition — the metric that separates our arms. DQN's canonical 0.25
   (which SB3 ships) is a historical default rather than a tuned value, and replay ratio and capacity
   are separable axes. Arm B raises the ratio to 4 for +22.2 pp; arm C raises capacity instead and
   gains nothing.
4. **Nikishin et al., "The Primacy Bias in Deep Reinforcement Learning," ICML 2022**
   ([arXiv:2205.07802](https://arxiv.org/abs/2205.07802)). The best available explanation for arm C's
   51 pp cross-seed spread and unchanged mean: agents overfit to early interactions, replay amplifies
   the effect, and a **higher replay ratio makes it worse** — exactly the C configuration. Their fix
   is cheap and applies directly (periodically re-initialize the last layers, keep the buffer), which
   makes it the first experiment to run if the project is resumed.
5. **Cobbe et al., "Quantifying Generalization in Reinforcement Learning," ICML 2019, with the
   HackAtari follow-up ("Deep RL Agents are not even close to Human Intelligence,"
   [arXiv:2505.21731](https://arxiv.org/abs/2505.21731)).** The brittleness result spike 007 measures
   locally. Cobbe et al. show agents overfit even to very large training sets; HackAtari shows they
   collapse on *simplified* variants, and that the failure is shortcut learning — "the right actions
   for the wrong reasons." Our 90 % → 0.5 % on a team swap, with team strength worth only ~20 pp to a
   scripted player, is the same pathology with a control that rules out the matchup explanation.

Papers that extend these, if the work continues rather than stopping at the report: **Metamon**
([arXiv:2504.04395](https://arxiv.org/pdf/2504.04395v1), offline RL with transformers over tokenized
battle logs — the escape from the hand-crafted 37-vector), **PokéLLMon** (arXiv:2402.01118, an LLM
policy at human parity on Showdown), the **PokéAgent Challenge**
([arXiv:2603.15563](https://arxiv.org/html/2603.15563v1), a 4M-battle dataset plus a
Bradley–Terry rating harness for external validity), **SPEQ**
([arXiv:2501.08669](https://arxiv.org/html/2501.08669v1), high-UTD performance at 40–99 % fewer
gradient updates — the direct answer to arm B's 189 → 99 fps cost), and **"When Does Opponent
Modeling Pay Off?"** ([arXiv:2610.18229](https://arxiv.org/abs/2610.18229), opponent-intent inference
fused into a dueling double DQN, which removes the degenerate-seed collapse plain DQN shows).
