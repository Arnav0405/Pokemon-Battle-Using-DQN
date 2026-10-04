# Spike 000 — Environment smoke test (Phase 0)

## Question

Given a local Pokémon Showdown server started with `--no-security`, when two `poke-env`
`RandomPlayer`s are pointed at `ws://localhost:8000`, then a battle completes over the
websocket bridge and each side reports the expected win/loss — proving server + bridge work
on this Windows machine **before** any RL code exists.

## Setup (what Phase 0 installed)

| Component | Version / pin |
|---|---|
| OS | Windows 11, git-bash (MSYS) shell |
| Node.js | v22.23.3 (Showdown requires >= 22) |
| pokemon-showdown | v0.11.11, commit `9fb3a5b`, cloned to `showdown/` (gitignored) |
| Server config | `config/config.js` from `config-example.js`, `bindaddress = '127.0.0.1'` |
| Python | 3.13.7 (uv-managed) via the `l_lab` uv **workspace** venv at `../.venv` |
| poke-env | 0.16.1 (pinned in `pyproject.toml`) |
| gymnasium / pettingzoo / websockets / numpy | 1.3.0 / 1.27.0 / 16.1.1 / 2.5.3 |

Reproduce:

```bash
scripts/showdown_setup.sh                 # clone + npm install + loopback config (idempotent)
scripts/showdown_start.sh                 # run in an interactive terminal; leave it up
../.venv/Scripts/python.exe spikes/000-environment-smoke/smoke_battle.py --battles 6 --concurrent 3
```

## Results

Each run asserts `finished_reports == 2 * battles` and `wins == losses == battles`.

| battles | max_concurrent_battles | wall time | s / battle |
|---|---|---|---|
| 1 | 1 | 4.56s | 4.56 (includes websocket handshake) |
| 2 | 1 | 2.74s | 1.37 |
| 6 | 3 | 3.20s | 0.53 |
| 24 | 8 | 4.75s | 0.20 |

All four runs PASS. Battle length varied 27–120 turns; formats checked present on the local
server: `gen9randombattle`, `gen9customgame`, `gen9ou`, `gen9nationaldex`.

**Throughput implication (Phase 2/3):** ~44 battles/min single-threaded → ~300 battles/min at
`max_concurrent_battles=8`. A 5k-episode run is roughly 20–30 min of *battle simulation*, not
hours. This confirms the spec's compute note that parallel battles beat GPU for this workload.

## Verdict: VALIDATED

Core question answered yes, with evidence, on this machine.

### What worked

- Clone → `npm install` (354 packages) → `node build` (TypeScript → `dist/`) → server serving
  HTTP 200 on `127.0.0.1:8000`, all with no manual fixes.
- poke-env 0.16.1 installed cleanly on Python 3.13.7 (gymnasium 1.3.0 / pettingzoo 1.27.0
  / websockets 16.1.1 all have cp313 wheels).
- Two `RandomPlayer`s completing real battles, and `max_concurrent_battles` scaling nearly
  linearly up to 8 (0.53s → 0.20s per battle).

### What didn't / gotchas

1. **Server started from a non-interactive parent dies immediately.** Launched with no TTY,
   the launcher exits (code 1) right after spawning its network worker, reporting
   `stdin is not a tty`; the **worker keeps listening on port 8000**, so battles still work
   but the parent is gone. That string appears nowhere in the checkout or in `node_modules`
   (a `readline`/REPL repro with piped stdin does not print it), so the message most likely
   comes from the process-launcher layer rather than Showdown itself — treat the *behavior*
   (parent may exit, worker survives) as the finding, not the exact cause.
   **Consequence:** start the server from an interactive terminal and leave it running;
   do not have `train.py` spawn it headlessly. `winpty` is available in git-bash if a
   scripted launch is needed later (untested).
2. **Orphan workers hold the port.** Because of (1), a dead parent leaves a `node.exe`
   listening on 8000; the next start fails with `EADDRINUSE`. Kill with
   `taskkill /PID <pid> /F` — find the pid via `netstat -ano | grep :8000`.
3. **`bindaddress` in `config/config.js` is ignored in practice.** The loader returns
   `127.0.0.1` correctly, but the spawned worker still binds `0.0.0.0` (its env
   `PSBINDADDR` is computed as `Config.bindaddress || "0.0.0.0"` in a scope where the value
   is missing). With `--no-security` the server is therefore reachable on the LAN, not just
   loopback. Accepted for now; the PC-facing mitigation is to keep the server running only
   while training.
4. **Startup cost is dominated by the build.** `node pokemon-showdown start` runs `node build`
   every time unless `--skip-build` is passed (~30s vs ~2s). `scripts/showdown_start.sh`
   intentionally keeps the build for correctness; add `--skip-build` when restarting often.

### Surprises — poke-env 0.16.1 API differs from the spec's §3/§4 assumptions

The spec was written against an older `SinglesEnv`. Actual API in 0.16.1:

| Spec assumed | Reality in poke-env 0.16.1 |
|---|---|
| `env.embed()` override | `embed_battle(battle)` — abstract; **no default embedding exists**, it must be written |
| `env.calc_reward()` override | `calc_reward(battle)` ✓ (plus a ready `reward_computing_helper(battle, fainted_value=…, hp_value=…, status_value=…, victory_value=…)`) |
| "enable poke-env action masking" | `SinglesEnv.get_action_mask(battle) -> list[int]` (length 26); env observations may be dicts of `{"observation", "action_mask"}` |
| `SinglesEnv` is a single-agent env | It is a **two-agent** `PokeEnv` (`account_configuration1/2`); single-agent training wraps it: `SingleAgentWrapper(env, opponent)` + `Monitor` |
| opponent is a `Player` on the far side | ✓ `SingleAgentWrapper` drives the opponent in-process via its `choose_move`; `SimpleHeuristicsPlayer` and `RandomPlayer` are ready-made baselines |
| action space "discrete over moves + switches" | `Discrete(26)` for gen9: 0–5 switch, 6–9 move, 10–13 mega, 14–17 z-move, 18–21 dynamax, 22–25 terastire |

**Algorithm risk for Phase 2:** SB3 core `DQN` has no action masking, and sb3-contrib ships
`MaskablePPO` but **no `MaskableDQN`**. poke-env's own official example trains with PPO + a
custom masked actor-critic policy. Keeping the spec's pinned DQN therefore needs a custom
masked `DQNPolicy` (mask Q-values with the `action_mask` in `env.observation_space`'s dict —
which masks both the greedy action *and* the TD target's `max`) — tractable, ~50–80 lines,
but it is real work and a deviation from "off-the-shelf SB3 defaults".

### Recommendation for the real build

- Keep the server as an **external prerequisite** (documented, started by hand), not something
  the training script spawns.
- Use `gen9customgame` + explicit packed teams for both sides (format exists, takes supplied
  teams, singles) — matches the spec's fixed-team requirement.
- Write `embed_battle` yourself; the official example's 12-feature vector is a reasonable floor,
  but a slightly wider vector (HP fractions, base power, type multipliers, boosts, status
  flags, faint fractions) is worth the extra features for an MLP.
- Decide the DQN-masking approach before Phase 2 starts (see plan.md).
- Budget battles, not GPU: 8 concurrent battles ≈ 0.2s/battle.
