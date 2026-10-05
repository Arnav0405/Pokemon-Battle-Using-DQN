# Pokémon RL Battle Agent — Execution Plan

Spec: `docs/superpowers/specs/2026-10-04-pokemon-rl-battle-agent-design.md`

Rule: tick a box only when its verification artifact exists on disk. Commit at phase boundaries only.

## Phase 0 — Environment smoke test (spike, throwaway) — DONE

- [x] Toolchain installed: `showdown/` cloned (v0.11.11, `9fb3a5b`) + `npm install` + `node build`; `poke-env==0.16.1` pinned in `pyproject.toml`, installed into the `l_lab` uv-workspace venv (`../.venv`) on Python 3.13.7 — artifact: `scripts/showdown_setup.sh`, `pyproject.toml`
- [x] Showdown server answers an HTTP probe on `127.0.0.1:8000` with `--no-security` — artifact: `scripts/showdown_start.sh` (verified: `http_code=200`), `scripts/patch_showdown_config.js`
- [x] Two `RandomPlayer`s finish real battles over the local bridge, incl. concurrency scaling (1×: 1.37 s/battle, 3×: 0.53, 8×: 0.20) — evidence: `docs/poke-env-0.16.1-gotchas.md` (the probe itself, `spikes/000-environment-smoke/smoke_battle.py`, was deleted in the Phase-5 cleanup; these numbers are reproducible with two `RandomPlayer`s against a running server)
- [x] Spike verdict + gotchas recorded (server parent dies without a TTY, orphan workers, `bindaddress` ignored, poke-env API deltas) — artifact: `docs/poke-env-0.16.1-gotchas.md` § "Server process and toolchain" (moved there from the deleted spike README in the Phase-5 cleanup)

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
- Result: 100 k steps / 3650 episodes. Win rate 2.0 % over the first 100 episodes → 53.0 % over the last 100 → **57.5 % in the final 200** against a parity of 50 %; average reward per episode −41.6 → ≈ +7; 0 truncated battles. **Convergence "not reached"**
- [x] `src/lab_ca/plots.py` written here (Phase 5 reuses it): `compute_metrics` (spec §6, incl. the convergence rule) plus the three curves rendered twice — seaborn → PNG, plotly → interactive HTML; `--only reward|success|steps` and `--out-dir` for Phase 5's `report_assets`
- [x] Figure QA by a second vision model (opencode `fledge-alpha-free`), two passes per Arnav's review rule — first pass caught a missing per-episode trace and unreadable raw data (both fixed), second pass: report-ready

### Phase 3, tier 0 — the convergence test was measuring its own noise (Arnav asked for this)

First read of the curve ("looks like convergence is hard even by 200 k steps") did not survive
arithmetic, so the measurement was fixed before any architecture change:

- The old rule tested a **50-episode** rolling rate against a fixed **±5 pp** band. At a 60 % win
  rate that window's binomial σ is 6.9 pp, so the band sat at 0.72 σ: an agent that *had*
  converged would leave it routinely. Now `CONVERGENCE_WINDOW = 200` (σ ≈ 3.5 pp) and the band is
  `2 σ` of the observed final rate (floor ±5 pp) → "converged" means "stopped moving by more than
  sampling noise".
- The trend metric was also wrong twice and both were caught by tests, not by reading the output:
  regressing on a rolling mean produced a ±0.006 pp/1000 CI (fake precision — it shrinks with the
  smoothing, not with the evidence), and the slope was computed on 0/1 flags but labelled `pp`.
  Now: independent 200-episode blocks, reported in percentage points. Artifacts:
  `src/lab_ca/plots.py`, `tests/test_plots.py` (13 offline tests, incl. guards for both traps).
- `plots.py --compare runs/a runs/b` overlays runs (win rate with binomial CI + reward, PNG + HTML)
  and writes a per-run metrics table — the instrument for reading an A/B honestly.
- **Verdict on `p3_100k` with the corrected metrics:** convergence still not declared, and for a
  real reason — the win rate was **still climbing at +7.7 ± 4.4 pp per 1000 episodes** over the last
  2000 episodes (49.6 → 53.8 → 57.2 → 63.6 % in 500-episode blocks). So the run never plateaued; the
  old "not converged" hid *why*. Artifact: `runs/p3_100k/metrics.json`.
- Also ruled out as the bottleneck: exploration. ε reached its floor (0.02) at **episode 376 of
  3650** (89.7 % of the run near-greedy) and the whole 2 % → 60 % climb happened there.

### Phase 3, tier 1 — update budget (the lever the evidence pointed at)

The loop is server-bound (5.29 ms per env step) while one gradient update on the `[64,64]` net is
0.56 ms — so SB3's default of 1 update per 4 env steps leaves most of the CPU budget idle
(measured: `spikes/001-update-cost/bench_updates.py`, committed).

- `TrainingConfig` + CLI gained `net_arch`, `train_freq`, `gradient_steps`, `batch_size`,
  `buffer_size`, `learning_rate`, `gamma`, `target_update_interval`, `exploration_fraction`,
  `exploration_final_eps`; `run_config.json` already dumps the dataclass, so any run's knobs are
  checkable afterwards. Artifacts: `src/lab_ca/train.py`, `tests/test_train.py` (CLI→config test).
- **Experiment `p3b_100k`** — same seed, same opponent, same 100 k steps as `p3_100k`, one knob:
  `train_freq=1, gradient_steps=4, batch_size=128` → **16× more updates per env step** (measured
  3.95/step vs 0.25), with `target_update_interval` 10 000 → 1000 so the target net stays as fresh
  *per update* as before. Cost: 189 → 99 fps, so ~17 min for 100 k steps instead of 8.8.
- Judging rule, fixed before the result was seen: the new config wins if its final-200 win rate
  clears the baseline's by more than the bands overlap **and** its late slope is smaller in
  magnitude (i.e. it flattens sooner, not just higher). Seed noise is unknown, so this is a
  single-seed screen; confirming a winner needs 2 seeds.
- **Result — it clears both halves:** final 200 episodes **57.5 % → 67.0 %** (+9.5 pp, ~2 σ of the
  ±3.5 pp window bands) and the late trend drops from **+7.7 ± 4.4** to **+3.5 ± 9.2 pp / 1 000**
  (CI now contains zero). Episodes 3650 → 4018 at the same 100 k steps; steps per episode 27.4 →
  24.9 (closing games out rather than trading). Cost: 189 → 99 fps, so 17 min per 100 k steps.
  Artifacts: `runs/p3b_100k/{curve_check.png,curve_check.html,metrics.json,train.log,notes.md}` and
  the A/B `runs/p3_ab.{png,html,json}`. **Caveat: one seed** — seed variance is still unmeasured, and
  these are training-time win rates (ε = 0.02, moving policy), not the frozen-policy evaluation of
  Phase 4. Full write-up in `runs/p3b_100k/notes.md`.

### Phase 3, tier 1b — seed sweep: the update budget is confirmed, capacity is rejected

Nine runs (3 arms × 3 seeds, 100 k steps each, 100 k steps ~6/14/17.5 min per arm).
Artifacts: `runs/sweep_notes.md` (full tables + caveats), `runs/sweep_aggregate.{png,html,json}`,
`spikes/003-sweep-stats/arm_stats.py`, `scripts/run_seed_sweep.sh`.

| Arm | mean final-200 (3 seeds) | seeds | spread |
|---|---|---|---|
| A defaults | 51.3 % | 40.5, 56.0, 57.5 | 17.0 pp |
| **B 16× updates** | **73.5 %** | 67.0, 76.0, 77.5 | 10.5 pp |
| C B + `[256,256]` | 49.2 % | 27.5, 41.5, 78.5 | **51.0 pp** |

- **B confirmed:** +22.2 pp over A, and the arms **completely separate** (A's best seed 57.5 % < B's
  worst 67.0 %). Welch t = −3.49 (df 3.3, p = 0.034); the exact permutation test can only reach
  **p = 0.10** at 3+3 runs — that *is* the ceiling for this sample size, and complete separation is
  the most extreme outcome available. B also learns earlier (whole-run average 60.6–68.0 % vs A's
  32.8–45.0 %) with shorter episodes (24.8–26.2 vs 27.4–28.7 steps).
- **C rejected:** mean indistinguishable from plain defaults (Welch p = 0.90) and 24 pp below B, with a
  **51 pp spread** — one seed beat every B run, another fell below every A run. Capacity + 16× updates
  is a lottery, and its episodes are the longest (28.5–31.0 steps). Checked rather than assumed: the
  TD loss shows **no divergence** in C (first-half ≈1.5, last-half ≈1.9, max ≈2.6, comparable to B —
  and arm A is the one with the big spikes at max ≈5.1–5.9), so the honest statement is
  "higher-variance, no mean benefit", not "capacity diverges".
- Convergence: one run per arm declared it (A-1 @ 3080, B-2 @ 3567, C-1 @ 3271); the arms differ by
  *level*, not by whether they settle.
- **Decision:** adopt B and stop tuning the architecture. Remaining headroom is the observation (no
  speed order, no revealed opponent bench ⇒ the six switch actions are nearly blind), which changes
  the observation interface and therefore needs its own spec. Extra compute is better spent on 200 k
  steps of B than on a wider net.
- One-seed screens would have misled here: C's seed 0 alone reads "capacity hurts" (41.5 %), seed 1
  alone reads "capacity is the best arm" (78.5 %). The spread is the finding.
- New dependency: `scipy>=1.13` (statistics only — Welch tests in the sweep analysis and CIs in
  Phase 4; nothing in the training path imports it).

### Phase 3, tier 1c — adopting B (and the bug that adoption exposed)

- B is now the **code default** (`TrainingConfig`), so `python -m lab_ca.train` reproduces the
  adopted config with no flags; the SB3 baseline stays reachable via the documented flag set.
- **Bug found by reading the artifact back, not by reading the code:** `main()` builds the config
  from `args.*`, and the argparse defaults were still arm A after the dataclass moved to arm B — so
  a no-flag run trained **arm A** (recorded 0.25 updates/step, 239 fps) while the dataclass, README
  and this plan all said arm B. The launch was killed at ~23 k steps, before producing anything
  reportable; `run_config.json` is what caught it. Fixed by deriving every flag default from
  `TrainingConfig()` (one source of truth) and pinned by two tests: CLI/dataclass parity
  field-by-field, and a plain `--timesteps N --run-name X` run whose *effective* config must be B.
  The sweep is unaffected — every sweep run passed its config explicitly.
- 200 k-step run of B: `runs/p3_200k_2` (the `_2` is because `make_run_dir` never overwrites, and
  the killed arm-A attempt had taken `runs/p3_200k`; that stale directory was deleted). Single
  seed — the across-seed spread of arm B is already characterised at 100 k (10.5 pp).
- **Result (200 k, arm B):** 7 774 episodes, 27 min, 0 unfinished. Final 200 episodes **87.5 %**,
  above every 100 k seed (67.0–77.5 %), i.e. ≈ +14 pp over the 100 k mean — with the n = 1 caveat
  stated in `runs/p3_200k_2/notes.md`. Late trend **+5.2 ± 2.9 pp/1000 (CI excludes 0)**: still
  climbing, convergence **not declared**. Blocks: 37.7 → 67.1 → 62.2 → 70.4 → 72.3 → 76.6 → 79.9 →
  82.3 %. So "convergence is hard by 200 k" is really "200 k is still on the ascending part of the
  curve"; the binding constraint was the training budget, not exploration (ε floor at ~2 % of the
  run) or capacity (arm C: no mean gain, 51 pp spread).
- `config_signature` now includes the step budget: 100 k and 200 k runs of the same knobs are
  different experiments, not repeat seeds, so they must not average into one envelope. Figure:
  `runs/b_100k_vs_200k.{png,html,json}`.
- If convergence *declared* is wanted, the evidence points at more steps (300–500 k) rather than
  more architecture work; the alternative lever is the observation (tier 3), which changes the
  interface and needs its own spec.

## Phase 4 — Evaluation

Checkpoint to evaluate: **arm B** (16× update budget, `[64,64]`, batch 128, target 1 000) is the
adopted training config — see the tier 1b result in Phase 3. Evaluate **all three B seeds**
(`p3b_100k`, `sweep_b_seed1`, `sweep_b_seed2`) if the budget allows: the 10.5 pp across-seed spread
is itself a report number, and a single-seed evaluation would hide it. `sweep_b_seed1` is the
strongest single checkpoint (77.5 % final-200). Do not evaluate arm C — it was rejected for
variance, not for mean.

Two passes, because parallel metrics and watchable battles pull in opposite directions (you can't spectate 8 battles at once):

**Pass 1 — metrics (parallel, unwatched) — done**
- [x] `src/lab_ca/evaluate.py`: load checkpoint, run N battles vs Random and vs Heuristic at `max_concurrent_battles=8`, aggregate win rates — artifact: `runs/<run>/eval.json` (nine files: the three arm-A seeds, the three arm-B seeds at 100k, B at 200k, B at 400k, and a random-policy floor row; every one records `git_commit 5bbca44`, `git_dirty false`, and the provenance of the training run the checkpoint came from)
- [x] Evaluation win rates clearly above the random baseline — artifact: `runs/phase4_eval.md` + `runs/phase4_eval.png/.html` (200 decided battles per bar, Wilson 95% intervals). Frozen greedy policy vs the heuristic: arm A 48.5/51.5/64.0 (mean **54.7 %**), arm B at 100k 88.0/75.0/77.0 (mean **80.0 %**), B at 200k 84.0 %, **B at 400k 87.5 %**. The scripted-random floor scores **1.0 %** vs the heuristic and **49.0 %** vs random — the coin flip it has to be, which is what makes the rest of the column believable.
  The frozen-policy evaluation reproduces the training-time arm ranking a third time: **arm A's best run (64.0 %) is below arm B's worst (75.0 %)** — no overlap at n=3 either way.

**Pass 2 — watchable demo (sequential, recorded locally) — done, with one substitution**
- [x] Demo mode in `evaluate.py` (`--demo`): a handful of battles at `max_concurrent_battles=1` so exactly one battle is live at a time, `save_replays` enabled, recognizable agent username — artifact: `runs/p3_400k/demo/` replay files saved **locally** (no upload to the public replay service). Two requested battles were counted; a third, unrequested battle that the environment starts as it closes is flagged in the payload rather than passed off as part of the demo.
- [ ] Live spectate path verified: with the demo running, the single in-progress battle is watchable at `http://localhost:8000` — **not achievable in this deployment, evidenced rather than assumed**: the checkout is the server package only (`showdown/` has no client directory) and the page served at the server root is Showdown's stub, which rewrites the browser to `<host>.psim.us` (the public client) — verified by loading `http://127.0.0.1:8000/<battle-tag>` and landing on the public client's lobby. The spectate URL is still printed by the demo (`http://localhost:8000/gen9customgame-…`) and would work against a server that ships the client. Substituted artifact: a self-contained transcript per counted battle (below), which is what makes a saved demo game actually watchable here.
- [x] Local replay capture works against the `--no-security` server — artifact: `runs/p3_400k/demo/` replay HTML (full protocol log embedded, written by poke-env from locally tracked messages, no upload). Note on the fallback the plan allowed for: the replay's *rendering* fetches `play.pokemonshowdown.com/js/replay-embed.js`, i.e. the public client, so the demo also writes `runs/p3_400k/demo/battle-<tag>-transcript.html` — the embedded log as a standalone page (33 and 17 move lines for the two counted battles, `|win|lab_ca_dqn`, no network) plus `transcript_view.png`.
- [x] Replay file paths recorded into `eval.json` so the report links a few concrete example games — artifact: `runs/p3_400k/eval.json` gains a `demo_replays` list (the agent's own copy of each counted battle) beside `demo.counted_battles`/`demo.unrequested_battles` and the other side's copy of each file. The demo section carries its own `git_commit`/`git_dirty` (`7a84e92`, clean) rather than inheriting the pass-1 stamp, since the results and the replays are two different runs of the same script.

Replay attribution was fixed while doing this: the first version zipped a sorted file list against the outcome list, which attached a win to a different battle's replay and presented the stray third battle as part of the demo. Replays are now grouped by battle tag and their result is read out of the file itself (`|win|<username>`), which is checked in the payload against the counted outcome.


## Phase 5 — Report assets — done

- [x] `src/lab_ca/plots.py` renders all three graphs + `metrics.json` — artifact: `report_assets/curve_check.png/.html` (the adopted 400k run, three panels: reward, win rate, steps), `report_assets/{reward,success,steps}.png/.html` (the same curves as standalone panels, as the plan suggested), and `report_assets/metrics.json` (14,519 episodes, final-200 **81.0 %**, convergence at episode 13,781). Also in the folder, because the report needs them: `sweep_aggregate.png/.html/.json` (the arm decision, mean per config with the min–max envelope over 3 seeds each: **A 51.3 % / B 73.5 % / C 49.2 %**) and `phase4_eval.png/.html/.md` (the frozen-policy evaluation).
- [x] Graph sanity pass (axes labelled, raw + smoothed, no truncated series) — artifact: `report_assets/SANITY.md` plus the mechanical check it reports, `spikes/005-report-asset-audit/audit_report_assets.py` — **52 checks, all passing**. It recomputes `metrics.json` from `episodes.jsonl` rather than trusting it, compares every drawn series against the run's own episode count (the "no truncation" requirement, made mechanical: all panels draw to 14,519 of 14,519), and measures each drawn CI/envelope off the artists, so a band that was never drawn cannot pass as "the seeds agree". Visual QA by a second model caught real defects in two of the three figures (legends over the data; a legend running into the title; an unlabelled zero line; "win rate" vs "success rate" in one figure); all fixed and re-verified — final verdicts: `curve_check` and `phase4_eval` **report-ready**, aggregate re-QA'd after the fixes.
- [x] Cleanup pass: delete `spikes/000-environment-smoke/`, README documents reproducing a run — artifact: the spike is gone (its version pins and server findings moved into `docs/poke-env-0.16.1-gotchas.md`, and the spec's Phase 0 pointer now points there); `README.md` documents setup, a training run, the plotting commands and the test commands; `.gitignore` keeps `runs/` and `logs/` out of the repo.

