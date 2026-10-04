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
