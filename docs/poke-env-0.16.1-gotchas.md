# poke-env 0.16.1 gotchas (local Showdown, Windows)

Findings from building the Phase-1 wrapper (`src/lab_ca/env.py`). Each one cost a debugging
cycle; each is worked around in our code. Line numbers are poke-env 0.16.1 as installed in
`../.venv/Lib/site-packages/poke_env/`.

## Teardown: `close()` / `reset_battles()` can hang forever

`PokeEnv.close()` (`environment/env.py:548-582`) blocks the calling thread in three ways:

1. **Racy queue drain.** It empties the players' queues with `while not q.empty(): q.get()`
   (`:575-582`), but `_AsyncQueue.get()` (`:54-58`) schedules `asyncio.wait_for(async_get(),
   timeout=None)` and then calls `res.result()` with **no timeout**. The event loop's own
   consumer (`_choose_move` sits in `await self.order_queue.async_get()`, `:112`) can take the
   item between the `empty()` check and the `get()`, leaving that `get()` waiting forever.
   Observed as an indefinite hang inside `close()`.
2. **Silently dropped forfeit.** `_EnvPlayer` builds both queues with `maxsize=1` (`:100-101`),
   and `put()` is `loop.call_soon_threadsafe(self.queue.put_nowait, item)` (`:80-81`). If an
   item is still queued, `put_nowait` raises `QueueFull` *inside the loop callback*: the loop's
   exception handler logs it, the forfeit is discarded, and the battle keeps running.
3. **Unbounded challenge wait.** `self._challenge_task.result()` (`:567`) has no timeout, so a
   battle that never ends on the server hangs the process.

Related: `_battle_finished_callback` (`:153-154`) is a bare
`battle_queue.queue.put_nowait(battle)` with `maxsize=1`; if the queue already holds an item it
raises, and the `notify_all()` that releases the challenge waiter immediately after it
(`player/player.py:379-382`) never runs.

**Our workaround.** `PokemonSinglesEnv.close()` drains both queues for both players
non-blockingly *first* — one loop-side coroutine that never awaits (`_drain_nowait`), so nothing
can interleave. That frees the cap-1 slots and removes the race. It then queues the same orders
the base class does, waits a bounded `CLOSE_TIMEOUT_S`, cancels the challenge if the window
expires, and clears state without any blocking queue read. `reset_battles()` is guarded the same
way; `Player.reset_battles()` raises `EnvironmentError` when a battle never finished
(`player/player.py:675-681`), so `close()` drops that bookkeeping as a last resort.

## Actions must be numpy integers

`SinglesEnv.action_to_order` calls `action.item()`, so a plain Python `int` raises
`AttributeError: 'int' object has no attribute 'item'` from inside poke-env — and because the
crash leaves a half-played battle behind, the failure surfaces later as a teardown hang rather
than a clean traceback. numpy `int64` (what SB3 emits) is fine. Our subclass coerces with
`np.int64(action)` so both work.

## Shape of the API (vs. the original spec)

- `SinglesEnv` is a two-agent `PokeEnv`; single-agent use wraps it: `SingleAgentWrapper(env, opponent)`.
- No default embedding exists: `embed_battle(battle)` is abstract, the vector is ours to write.
- Observation spaces are declared as plain `Box`es; the env itself wraps them into
  `{"observation": <vector>, "action_mask": <0/1 vector>}`, so the legality mask arrives with
  the observation for free.
- `gen9customgame` runs at level 100 and has team preview; in **singles** `_teampreview` always
  returns a random order regardless of `choose_on_teampreview`, so team preview never consumes an
  environment action.
- Battle end is signalled by a condition variable, not by the `battle_queue`, so draining that
  queue during teardown does not block the challenge from completing.

## Server process and toolchain

Phase 0's environment smoke test lived in `spikes/000-environment-smoke/` and was removed in the
Phase 5 cleanup; its measurements and conclusions are here and in the spec's risk table.

Pins the probe ran against (all still installed in `../.venv`):

| Component | Version |
|---|---|
| OS / shell | Windows 11, git-bash (MSYS) |
| Node.js | v22.23.3 (Showdown requires ≥ 22) |
| pokemon-showdown | v0.11.11, commit `9fb3a5b`, cloned to `showdown/` (gitignored) |
| poke-env | 0.16.1 |
| gymnasium / pettingzoo / websockets / numpy | 1.3.0 / 1.27.0 / 16.1.1 / 2.5.3 |

- **Start the server from an interactive terminal.** Launched from a non-interactive parent it
  exits (code 1) right after spawning its worker, reporting `stdin is not a tty`. The **worker keeps
  listening on 8000**, so battles still work and the failure is easy to miss; the parentless worker
  then holds the port (`EADDRINUSE` on the next start). `train.py` and `evaluate.py` never spawn it.
- **Orphans are killed by pid, not by hope:** `netstat -ano | grep :8000`, then
  `taskkill /F /PID <pid>` — with **single** slashes. Path translation is off in this shell, so a
  `//F` written for a POSIX-ish tool reaches `taskkill` literally, the kill fails, and the port
  stays held.
- **`bindaddress` in `config/config.js` is ignored in practice.** The loader returns `127.0.0.1`,
  but the worker binds `0.0.0.0` (`Config.bindaddress || "0.0.0.0"` evaluated where the value is
  missing). With `--no-security` the server is LAN-reachable, so run it only while training.
- **`--skip-build` decides start-up cost:** ~30 s with `node build` versus ~2 s without.
  `scripts/showdown_start.sh` keeps the build by default and takes `--skip-build`.
- **Throughput scales with concurrency, not compute:** 4.56 s for 1 battle single-threaded,
  0.53 s at 3×, **0.20 s at 8×** — so prefer `max_concurrent_battles` over a GPU. The evaluation
  harness uses 4 (short runs of ~40 s keep a server restart cheap).
