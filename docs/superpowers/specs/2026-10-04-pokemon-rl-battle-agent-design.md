# Pokémon RL Battle Agent — Design Spec

**Date:** 2026-10-04
**Status:** Approved design; Phase 0 complete (see §10 for API revisions)
**Deliverable:** A trained DQN agent that plays 1v1 Pokémon Showdown battles, plus the logged metrics and graphs needed for an external report author. The repo produces **data + graphs only**, not the report prose.

---

## 1. Goal & Success Criteria

Build a reinforcement-learning agent that learns to win 1v1 Pokémon battles in a local Pokémon Showdown environment, and emit the quantitative results a report needs.

**Success looks like:**
- Win rate climbs from ~baseline (random play) to meaningfully above it over training.
- Three graphs produced from logged training data:
  - Reward vs. Episode
  - Success Rate vs. Episode
  - Steps per Episode vs. Episode
- A metrics summary: cumulative reward, average reward, success rate, steps/episode, convergence point, performance improvement.
- All artifacts land in `report_assets/` for the report author.

**Explicitly out of scope:**
- Writing the report itself.
- Competitive ladder / human play.
- Replicating literal in-game trainer AI (see §4).
- Doubles/triples battles, team-building search, multi-gen support.

---

## 2. Tech Stack & Environment

| Component | Choice | Notes |
|-----------|--------|-------|
| Battle engine | `smogon/pokemon-showdown` (Node) | Local server, `--no-security`; v0.11.11 pinned in Phase 0 |
| Python bridge | `poke-env` | Two-agent `SinglesEnv` + `SingleAgentWrapper`, state embedding, action masks |
| RL library | Stable-Baselines3 `DQN` (MlpPolicy) | Black-box, off-the-shelf; masking needs a small custom policy (§4) |
| Language / env mgmt | Python 3.13, `uv` | `poke-env==0.16.1`; venv is the `l_lab` uv-workspace venv (`../.venv`), not in-repo |
| Server runtime | Node.js | v22.23.3 confirmed present |
| Training compute | Local CPU; Colab T4 fallback | battles are server-bound, not NN-bound |

**Pinned decisions:**
- Algorithm: **DQN** (neural, not tabular).
- Reward: **shaped** (HP differential + terminal win/loss bonus).
- Opponent: **both** a Random baseline and a type-aware Heuristic baseline, each on a **fixed canon-style team**. Train against Heuristic; evaluate against both.
- Action masking: the legality mask rides along in the environment observation dict (`action_mask`); DQN gets a small custom policy that masks Q-values with it, so both exploration and the TD target ignore illegal actions. (Decided in Phase 0: sb3-contrib ships `MaskablePPO` but no `MaskableDQN`.)

---

## 3. Architecture & Components

```
lab_ca/
├── showdown/                  # cloned pokemon-showdown server (Node), --no-security
├── src/lab_ca/
│   ├── teams.py               # canon team definitions (packed-team format)
│   ├── env.py                 # SinglesEnv subclass: embed_battle() + calc_reward() overrides
│   ├── opponent.py            # scripted baseline opponents w/ fixed teams
│   ├── train.py               # SB3 DQN training loop + per-episode metric logging
│   ├── evaluate.py            # trained-agent evaluation → success metrics
│   └── plots.py               # logs → graphs + metrics.json
├── runs/                      # training logs (JSONL/CSV) + saved model checkpoints
└── report_assets/             # final PNGs + metrics.json for the report author
```

**Data flow:**
`train.py` builds a `SinglesEnv` and wraps it in `SingleAgentWrapper` with the opponent `Player` (the wrapper drives that opponent in-process via its `choose_move`) → the env communicates over websocket with the local Showdown server → SB3 DQN steps the wrapper → each finished battle appends a row to a log → `plots.py` renders graphs from the log.

Each unit:
- `teams.py` — pure data; returns packed-team strings. No logic to test beyond format validity.
- `env.py` — owns state embedding and reward shaping. Testable via a mock/short battle.
- `opponent.py` — thin subclass of poke-env baselines with a team attached.
- `train.py` — orchestration; owns the metric logging contract.
- `evaluate.py` — loads a checkpoint, runs N battles, aggregates.
- `plots.py` — pure function of the logs; fully testable offline.

---

## 4. RL Setup

- **Env:** `SinglesEnv` (a **two-agent** `PokeEnv` — two account configurations, one battle between them), `battle_format='gen9customgame'` (or the appropriate fixed-team singles format) so both teams are exactly controlled. Single-agent training wraps it: `SingleAgentWrapper(env, opponent)`. **One battle = one episode; each turn = one step.**
- **State (`embed_battle`):** poke-env 0.16.1 provides **no default embedding** — `embed_battle(battle)` is abstract and must be written, returning a fixed-length float vector (HP fractions, active Pokémon stats/types, available moves, status, boosts, opponent side). Observations are dicts: `{"observation": <vector>, "action_mask": <mask>}`, with a `Box` observation space declared per agent. Start modest (~12–40 features) and widen only on evidence.
- **Action space:** `Discrete(26)` for gen9 — 0–5 switch, 6–9 move, 10–13 mega, 14–17 z-move, 18–21 dynamax, 22–25 terastallize. Legality comes from `SinglesEnv.get_action_mask(battle)`. **Masking is the single biggest lever for clean convergence and smooth graphs**, applied to the Q-values (see §2 pinned decisions) so illegal actions can neither be taken nor become the TD target.
- **Reward (`calc_reward`) — shaped:** `reward_computing_helper(battle, fainted_value=…, hp_value=…, status_value=…, victory_value=…)` is the ready-made helper; tune its weights to the scheme below:
  - Per-step: `+ (opponent HP fraction lost) − (own HP fraction lost)`, scaled.
  - Terminal: `+1` win / `−1` loss (scaled larger than per-step terms).
  - Rationale: shaped rewards converge faster and give smooth, near-monotonic reward-vs-episode curves, which the report needs on a short timeline.
- **Algorithm:** SB3 `DQN` with `MlpPolicy`, target network, replay buffer, epsilon-greedy decay. Start from SB3 defaults; tune only if curves are poor.

---

## 5. Opponent

- poke-env baseline `Player` subclasses, each loaded with a **fixed canon-style team** via packed-team string (e.g. an Elite-Four-style roster), so every training battle sees the same team. `RandomPlayer` and `SimpleHeuristicsPlayer` (type-aware) are the ready-made baselines.
- **Heuristic (type-aware) baseline** = primary training opponent. **Random baseline** = easy reference point.
- Evaluate the final agent against **both** for a "beats random → competitive with/beats heuristic" success narrative.

**Note for the report:** this is honestly described as *"a scripted, type-aware opponent using a fixed team,"* not the literal in-game Gym Leader / Elite Four AI. Team composition is faithful; the decision logic is a stronger scripted heuristic. This is a deliberate scope call (the literal NPC AI is high-effort/low-payoff).

---

## 6. Metrics & Graphs (Deliverable Contract)

Logged per episode to `runs/*.jsonl` (+ CSV mirror): `episode, cumulative_reward, steps, win(0/1), epsilon, timestamp, opponent`.

`plots.py` reads logs and produces into `report_assets/`:
1. **Reward vs. Episode** — raw + moving-average smoothed overlay.
2. **Success Rate vs. Episode** — rolling win % over last N episodes.
3. **Steps per Episode vs. Episode** — raw + smoothed.

Plus `metrics.json`:
- cumulative reward (total), average reward/episode
- overall success rate, final-window success rate
- average steps/episode
- **convergence episode** (first episode where smoothed win-rate enters its plateau band)
- **performance improvement** (early-window win % → late-window win %)

These map 1:1 onto the report's required sections.

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| poke-env / Showdown version drift on Windows | Setup failure | Pin versions; get one local battle running **first** (Phase 0) before any RL code — done, both pinned |
| Training too slow to show a curve in time | Weak graphs | Parallel battles via `max_concurrent_battles` (Phase 0 measured 0.20 s/battle at 8× vs 1.37 s at 1×); Colab T4 fallback; shaped reward speeds convergence |
| Unstable / flat learning curve | Bad graphs | Action masking + shaped reward; tune LR / buffer / epsilon-decay only if needed |
| State embedding too large/noisy | Slow/unstable training | Start with a small hand-written embedding (~12–40 features); widen only on evidence |
| Server lifecycle (no TTY → orphaned worker; `bindaddress` ignored, binds `0.0.0.0`) | Stale port, LAN exposure while training | Start it from an interactive terminal and leave it running; never spawned by `train.py`; kill orphan workers via `netstat`/`taskkill` |

**Compute note:** the training bottleneck is Showdown's battle simulation, not the small MLP — so GPU gives less speedup than parallel battles. Prefer `max_concurrent_battles` before reaching for Colab.

---

## 8. Phases

**Phase 0 — Environment smoke test (spike, throwaway) — DONE**
Install toolchain, clone + launch Showdown with `--no-security`, run two built-in `RandomPlayer`s for 1 battle. Goal: prove the server + poke-env bridge works on this machine before building anything. Result: PASS (24 battles at 8× concurrency, 0.20 s/battle). Evidence and gotchas: `spikes/000-environment-smoke/README.md`.

**Phase 1 — Environment wrapper**
`teams.py` (a canon fixed team), `env.py` (`embed_battle` + shaped `calc_reward`, action mask surfaced in the observation dict), `opponent.py` (Random + Heuristic with fixed team), all wired through `SingleAgentWrapper`. Verify a single training-style episode runs and returns sane reward/observation shapes.

**Phase 2 — Training + logging**
`train.py`: SB3 DQN loop (with the masked Q-value policy from §2), per-episode metric logging contract, periodic checkpoints to `runs/`. Train against the Heuristic opponent.

**Phase 3 — First learning curves**
Generate initial graphs from a partial training run; confirm reward rises and win-rate climbs; adjust hyperparameters only if curves are flat/noisy.

**Phase 4 — Evaluation**
`evaluate.py`: load best/final checkpoint, run N battles vs both Random and Heuristic, aggregate success metrics.

**Phase 5 — Report assets**
`plots.py` final run → all three graphs + `metrics.json` into `report_assets/`. Sanity-check graphs (correct axes, smoothed + raw, no truncation). Cleanup pass.

---

## 9. Open Items / Assumptions

- Default reward-shaping scale and win-bonus magnitude chosen in `env.py`; tunable, documented inline.
- "Canon-style team" = a recognizable fixed roster; exact 6 Pokémon chosen in Phase 1.
- Convergence defined as smoothed win-rate plateau (exact band/threshold set in `plots.py`).
- If a specific literal in-game trainer AI is required for grading, that is a scope change to be raised explicitly.

---

## 10. Revision log

**2026-10-04 — Phase 0 API reconciliation.** The §3/§4 API references were written against an older `poke-env` and were corrected to the installed 0.16.1:

| Original | Corrected to |
|---|---|
| `embed()` override | `embed_battle(battle)` — abstract; no default embedding exists |
| `SinglesEnv` treated as single-agent | two-agent `PokeEnv`; single-agent training via `SingleAgentWrapper(env, opponent)` |
| "Enable poke-env action masking" | `SinglesEnv.get_action_mask(battle)`, delivered in the observation dict as `action_mask` |
| Python 3.11, venv in repo | Python 3.13, `l_lab` uv-workspace venv (`../.venv`) |
| "Start with the default embedding" (risk table) | no default exists — start with a small hand-written embedding |

Also added: the masking decision (custom masked DQN policy), the measured concurrency throughput in the risk table, and the server-lifecycle risk discovered in Phase 0.
