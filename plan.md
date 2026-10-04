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

## Phase 1 — Environment wrapper — DONE

- [x] `src/lab_ca/teams.py`: fixed canon team + packed-team validity check — artifact: green `tests/test_teams.py` (4 tests: roster shape, pack/parse round-trip, engine accepts the packed team for both `gen9customgame` and `gen9ou`)
- [x] `src/lab_ca/env.py`: `SinglesEnv` subclass (`embed_battle` 33-feature vector + shaped `calc_reward` via `reward_computing_helper`, action mask in the observation dict) + `src/lab_ca/opponent.py` (`Random`/`Heuristic` baselines, each on a fixed team) — artifact: `tests/test_env.py` (shape, bounds, finiteness, mask legality, reward range, plain-`int` action coercion, env reuse across episodes) + `tests/test_opponent.py`
- [x] One full training-style episode end-to-end against the Heuristic opponent (`SingleAgentWrapper`: reset → step → done) — artifact: `runs/phase1_smoke/episodes.jsonl` (26 steps, terminal reward −32.2, `win=false`; masked-random play loses to the type-aware baseline, as expected pre-training)
- [x] (added during the phase) teardown hardening so a failed episode cannot wedge the process: `close()`/`reset_battles()` no longer touch poke-env's racy cap-1 queue drains — artifact: `docs/poke-env-0.16.1-gotchas.md`, 13/13 tests green in ~15 s
- Baseline sanity: the type-aware heuristic beats the random baseline 8–0 over 8 battles — artifact: `tests/test_opponent.py::test_heuristic_beats_random_over_a_short_series`

## Phase 2 — Training + logging — DONE

- [x] `src/lab_ca/train.py`: SB3 `DQN` loop with the masking decision above, checkpoints to `runs/` — artifact: `runs/p2_smoke/model_1000_steps.zip`, `model_2000_steps.zip`, `model_final.zip`
- [x] Per-episode metric logging contract (`episode, cumulative_reward, steps, win, epsilon, timestamp, opponent`) — artifact: `runs/p2_smoke/episodes.jsonl` + `episodes.csv` (70 rows, CSV header matches the contract, counts agree)
- [x] Shaped-reward curves look sane on a short run (not flat at zero) — artifact: `runs/p2_smoke/train.log` (`ep_rew_mean -41.5`, `ep_len_mean 28.2`, 454 updates in 10 s / 189 steps·s⁻¹) and the log itself: rewards −48.0…+35.6, 64 distinct values over 70 episodes, 0 at zero, 1 win
- [x] (added) masking as its own unit, `src/lab_ca/policy.py`, so it is testable without a server — artifact: `tests/test_policy.py` (6 offline tests: greedy legality under a moving mask, epsilon-greedy legality, no illegal action stored in the replay buffer, −inf on both Q-nets, row-wise batch masking, log contract)
- [x] (added) `tests/test_train.py` runs a real 300-step training run and asserts the artifacts — artifact: same file, `13 → 20` tests green in ~22 s

### Phase 2 deltas vs spec §3/§4

| Spec says | What we did |
|---|---|
| `MlpPolicy` | `MultiInputPolicy`: the observation is a **dict** (state vector + `action_mask`), which `MlpPolicy` cannot consume |
| masking "in `train.py`" | in `policy.py` as `MaskedQNetwork` / `MaskedDQNPolicy` / `MaskedDQN`, imported by `train.py` — same phase, but unit-testable without the battle env |
| `Monitor` | from `stable_baselines3.common.monitor`: gymnasium 1.3.0 no longer exports `gymnasium.wrappers.Monitor` |
| — | one `DummyVecEnv([Monitor(OutcomeWrapper(make_env(...)))])`; `OutcomeWrapper` puts `win`/`outcome` into `info` because poke-env's info dict is empty |
| — | SB3's own logger is off (no tensorboard); the report contract is our `episodes.jsonl`/`episodes.csv` |

Verified along the way: 70 episodes ran on **one reused env** (no teardown stalls, no
`EnvironmentError` between episodes), and the env never received an illegal action — with
`strict=True` that would have raised immediately.

## Phase 3 — First learning curves — DONE

- [x] Initial graphs from a partial run; reward rises, win rate climbs — artifact: `runs/p3_100k/curve_check.png` (+ `curve_check.html`, interactive) and `metrics.json`
- [x] Hyperparameters adjusted only if curves are flat/noisy; changes noted — artifact: `runs/p3_100k/notes.md` — **nothing was tuned**: SB3 2.9 defaults kept, because the curves are neither flat nor degenerate (pre-run worry that `target_update_interval=10000` would leave the target net stale did not materialise: ~10 syncs over 100 k steps)
- Result: 100 k steps / 3650 episodes. Win rate 2.0 % over the first 100 episodes → 53.0 % over the last 100 → **57.5 % in the final 200** against a parity of 50 %; average reward per episode −41.6 → ≈ +7; 0 truncated battles. **Convergence "not reached"** (rolling win rate still drifting up at episode 3650), noted in `notes.md` as the reason Phase 5 may want a longer final run (~9 min per 100 k steps)
- [x] `src/lab_ca/plots.py` written here (Phase 5 reuses it): `compute_metrics` (spec §6, incl. the convergence rule) plus the three curves rendered twice — seaborn → PNG, plotly → interactive HTML; `--only reward|success|steps` and `--out-dir` for Phase 5's `report_assets`
- [x] Figure QA by a second vision model (opencode `fledge-alpha-free`), two passes per Arnav's review rule — first pass caught a missing per-episode trace and unreadable raw data (both fixed), second pass: report-ready

## Phase 4 — Evaluation

Two passes, because parallel metrics and watchable battles pull in opposite directions (you can't spectate 8 battles at once):

**Pass 1 — metrics (parallel, unwatched)**
- [ ] `src/lab_ca/evaluate.py`: load checkpoint, run N battles vs Random and vs Heuristic at `max_concurrent_battles=8`, aggregate win rates — artifact: `runs/<run>/eval.json`
- [ ] Evaluation win rates clearly above the random baseline — artifact: `runs/<run>/eval.json`

**Pass 2 — watchable demo (sequential, recorded locally)**
- [ ] Demo mode in `evaluate.py` (`--demo`): a handful of battles at `max_concurrent_battles=1` so exactly one battle is live at a time, `save_replays` enabled, recognizable agent username — artifact: `runs/<run>/demo/` replay files saved **locally** (no upload to the public replay service)
- [ ] Live spectate path verified: with the demo running, the single in-progress battle is watchable at `http://localhost:8000` — artifact: note confirming spectate works against the local server
- [ ] Local replay capture works against the `--no-security` server (verify whether the replay endpoint saves locally; if the hosted-upload path is unreachable, fall back to saving the battle log + a screen capture instead) — artifact: saved local replay/log file in `runs/<run>/demo/`
- [ ] Replay file paths recorded into `eval.json` so the report links a few concrete example games — artifact: `runs/<run>/eval.json` gains a `demo_replays` list

## Phase 5 — Report assets

- [ ] `src/lab_ca/plots.py` renders all three graphs + `metrics.json` — artifact: `report_assets/*.png`, `report_assets/metrics.json`
  (module already written in Phase 3: `--out-dir report_assets` is all this needs; consider also
  emitting the three curves as separate figures via `--only reward|success|steps`, plus their
  `.html` twins)
- [ ] Graph sanity pass (axes labelled, raw + smoothed, no truncated series) — artifact: review note
- [ ] Cleanup pass: delete `spikes/000-environment-smoke/`, README documents reproducing a run — artifact: final commit
