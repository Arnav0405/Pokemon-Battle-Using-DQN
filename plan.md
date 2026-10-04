# Pokémon RL Battle Agent — Execution Plan

Spec: `docs/superpowers/specs/2026-10-04-pokemon-rl-battle-agent-design.md`

Rule: tick a box only when its verification artifact exists on disk. Commit at phase boundaries only.

## Phase 0 — Environment smoke test (spike, throwaway) — DONE

- [x] Toolchain installed: `showdown/` cloned (v0.11.11, `9fb3a5b`) + `npm install` + `node build`; `poke-env==0.16.1` pinned in `pyproject.toml`, installed into the `l_lab` uv-workspace venv (`../.venv`) on Python 3.13.7 — artifact: `scripts/showdown_setup.sh`, `pyproject.toml`
- [x] Showdown server answers an HTTP probe on `127.0.0.1:8000` with `--no-security` — artifact: `scripts/showdown_start.sh` (verified: `http_code=200`), `scripts/patch_showdown_config.js`
- [x] Two `RandomPlayer`s finish real battles over the local bridge, incl. concurrency scaling (1×: 1.37 s/battle, 3×: 0.53, 8×: 0.20) — artifact: `spikes/000-environment-smoke/smoke_battle.py` output
- [x] Spike verdict + gotchas recorded (server parent dies without a TTY, orphan workers, `bindaddress` ignored, poke-env API deltas) — artifact: `spikes/000-environment-smoke/README.md`

## Spec deltas found in Phase 0 (spec §3/§4 predate the installed poke-env)

| Spec says | Installed poke-env 0.16.1 |
|---|---|
| `embed()` override | `embed_battle(battle)` — abstract, **no default embedding** exists |
| — | ready-made `reward_computing_helper(battle, fainted_value, hp_value, status_value, victory_value)` |
| `SinglesEnv` is single-agent | two-agent `PokeEnv`; wrap with `SingleAgentWrapper(env, opponent)` + `Monitor` |
| "poke-env action masking" | `SinglesEnv.get_action_mask(battle)`; env observations are dicts with `"observation"` + `"action_mask"` |
| Python 3.11 | `pyproject.toml` declares `>=3.13`; running 3.13.7 |
| "state: poke-env's fixed-length float vector — use as-is" | must be written by hand |

## Masking decision (settled 2026-10-04)

SB3 core `DQN` has no action masking and sb3-contrib has **no `MaskableDQN`** (only `MaskablePPO`).
**Decision: keep DQN** and add a small custom policy that masks the Q-values with the `action_mask`
carried in the observation dict — applied to the online and target nets, so it masks both the greedy
action and the TD target's `max`. Implement in Phase 2 (`train.py`); fall back to `MaskablePPO` only
if the custom policy fights SB3.

Spec §2/§4/§10 updated to match poke-env 0.16.1 (embed_battle, two-agent env + `SingleAgentWrapper`,
`get_action_mask`, Python 3.13).

## Phase 1 — Environment wrapper

- [ ] `src/lab_ca/teams.py`: fixed canon-style team + packed-team validity check — artifact: green `tests/test_teams.py`
- [ ] `src/lab_ca/env.py`: `SinglesEnv` subclass (`embed_battle` + shaped `calc_reward` via `reward_computing_helper`, action mask exposed in the observation dict) + `opponent.py` (`RandomOpponent` / `HeuristicOpponent` with fixed teams) — artifact: `tests/test_env.py` shows sane observation shapes, reward range and mask correctness
- [ ] One full training-style episode runs end-to-end against the Heuristic opponent (`SingleAgentWrapper`: reset → step → done) — artifact: episode log in `tests/`

## Phase 2 — Training + logging

- [ ] `src/lab_ca/train.py`: SB3 `DQN` (MlpPolicy) loop with the masking decision above, checkpoints to `runs/` — artifact: `runs/<run>/model_*.zip`
- [ ] Per-episode metric logging contract (`episode, cumulative_reward, steps, win, epsilon, timestamp, opponent`) — artifact: `runs/<run>/episodes.jsonl` + CSV mirror
- [ ] Shaped-reward curves look sane on a short run (not flat at zero) — artifact: partial-run log

## Phase 3 — First learning curves

- [ ] Initial graphs from a partial run; reward rises, win rate climbs — artifact: `runs/<run>/curve_check.png`
- [ ] Hyperparameters adjusted only if curves are flat/noisy; changes noted — artifact: note in run dir

## Phase 4 — Evaluation

- [ ] `src/lab_ca/evaluate.py`: load checkpoint, N battles vs Random and vs Heuristic — artifact: `runs/<run>/eval.json`
- [ ] Evaluation win rates clearly above the random baseline — artifact: `runs/<run>/eval.json`

## Phase 5 — Report assets

- [ ] `src/lab_ca/plots.py` renders all three graphs + `metrics.json` — artifact: `report_assets/*.png`, `report_assets/metrics.json`
- [ ] Graph sanity pass (axes labelled, raw + smoothed, no truncated series) — artifact: review note
- [ ] Cleanup pass: delete `spikes/000-environment-smoke/`, README documents reproducing a run — artifact: final commit
