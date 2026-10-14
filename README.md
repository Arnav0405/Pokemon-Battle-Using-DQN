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

Papers and course reports that implement RL agents in **Pokémon Showdown / Pokémon battle
environments specifically** — not general RL literature. Ordered by how directly each one maps onto a
decision in this repo. Two of them (goldmark #1 and #2) are the same task as this project, run at a
different scale, and are the closest thing to a direct precedent for it.

1. **Grigsby, Xie, Sasek, Zheng & Zhu, "Human-Level Competitive Pokémon via Scalable Offline
   Reinforcement Learning with Transformers," *Reinforcement Learning Journal* 6:2685–2719, 2025**
   ([arXiv:2504.04395](https://arxiv.org/pdf/2504.04395v1), code: Metamon). The strongest
   peer-reviewed result on this exact environment. Offline RL over a dataset of human Showdown
   battles, with a transformer over tokenized battle state/action/reward; agents reach the top 10 %
   of ladder players. **Why it matters here:** their observation is a tokenized battle log rather than
   a hand-built vector, which is the documented escape from our 37-feature embedding and its
   move-slot aliasing (spike 007). They also evaluate through **poke-env** against the same kind of
   scripted baselines we use (`Random`, `MaxBP`, `SimpleHeuristicsPlayer`), so their heuristic
   benchmark table is directly comparable with our numbers.
2. **Kumar, Sinha et al., "AI Learns to Battle Pokémon"** ([ankitsinha.net/ai-learns](https://ankitsinha.net/ai-learns/)).
   Trains **both DQN and PPO** in a Pokémon battle environment and reports **DQN 77.5 % / PPO 73.5 %**
   win rate against the hardest rule-based opponent. **Why it matters here:** this is the closest
   published head-to-head for our algorithm choice, and our 87.5 % frozen-policy figure sits in the
   same regime. Their central ablation matches our reward design: with **victory reward only, DQN
   barely beats random play; adding HP/faint terms transforms DQN's performance**. That is the
   empirical justification for `calc_reward`'s `HP_VALUE=1.0` / `FAINTED_VALUE=2.0` / shaped
   `VICTORY_VALUE=30.0` split rather than a terminal-only reward.
3. **Zhang, "Deep Reinforcement Learning for Pokémon Battling"** (Purdue CS587 report,
   [kevin-ys-zhang.github.io](https://kevin-ys-zhang.github.io/files/CS587_Project_Report.pdf)).
   Implements REINFORCE, DQN, WoLF-GIGA and A2C in PyTorch against poke-env, and finds **DQN the
   strongest of the four** — ~90 % against `RandomPlayer` where an earlier Q-learning agent
   (Kalose et al.) reached ~70 %. Their state embedding is almost exactly our first two feature
   groups: 4 base powers scaled by 1/100 with **-1 for status moves**, then 4 type effectivenesses.
   That independently corroborates both our action space and the fact that a compact hand-built
   embedding is sufficient for a strong result — and their write-up shows the same weakness we found,
   in that the -1 encoding is what loses move identity.
4. **Kalose, Kaya & Kim, "Optimal Battle Strategy in Pokémon using Reinforcement Learning"**
   (Stanford AA228 course report, [web.stanford.edu](https://web.stanford.edu/class/aa228/reports/2018/final151.pdf)).
   **Tabular Q-learning plus softmax exploration** via poke-env against Pokémon Showdown. Softmax
   exploration beat ε-greedy (60 % → 65 % → 70 % against random over 5 k/20 k games). **Why it matters
   here:** this is the tabular, single-agent ancestor of our setup, and it establishes the baseline
   our DQN has to beat — it is also the reference the DQN work above is measured against. Useful in
   the report as the "why a neural Q-function rather than a table" argument: their state space had to
   be kept deliberately small precisely because tabular Q-values do not scale.
5. **Pleines, Addis, Rubinstein, Zimmer, Preuss & Whidden, "Pokémon Red via Reinforcement Learning,"
   IEEE Conference on Games (CoG) 2025** ([project site](https://drubinstein.github.io/pokerl/)).
   **PPO** over a PyBoy-based Gym environment, <10M parameters, no pretraining, beating the game's
   early-mid content. **Why it matters here:** it is the reference implementation for reward design in
   a Pokémon environment, and its documented finding is that **reward bonuses get exploited in
   unintended ways when tuned poorly** — a real reward-hacking failure, not a hypothetical. That is
   the risk our shaped reward carries (HANDOFF §4.5 states it as "the policy partly optimizes the
   shaping terms"), and this paper is the evidence for taking it seriously, together with
   `reward_computing_helper`'s use as our shaping function.

Also in this environment, for the report's "related work / why not X" section: **PokéLLMon**
([arXiv:2402.01118](https://arxiv.org/html/2402.01118v1), an LLM policy at human parity on Showdown),
**PokéChamp** ([arXiv:2503.04094](https://arxiv.org/html/2503.04094v1), LLM-powered minimax, 84 % vs
the strongest rule-based bot, plus a 3M-game human battle dataset), the **PokéAgent Challenge**
([arXiv:2603.15563](https://arxiv.org/html/2603.15563v1), standardized RL baselines and a
Bradley–Terry leaderboard for external validity), and **Wang's MIT thesis** (PPO + MCTS + self-play,
1693 Elo, rank 8 on the gen4randombattle ladder).
