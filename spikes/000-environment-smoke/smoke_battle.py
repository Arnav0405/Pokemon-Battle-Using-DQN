"""Phase 0 spike: prove the local Showdown server + poke-env bridge works on this machine.

Two built-in RandomPlayers battle each other over the local websocket bridge, and we
measure wall time per battle so Phase 2 can size the training budget.
Throwaway code -- delete once Phase 1 owns the real environment wrapper.

Run (server must already be up):
    <venv>/python.exe spikes/000-environment-smoke/smoke_battle.py [--battles N] [--concurrent K]
"""

import argparse
import asyncio
import time

from poke_env.player import RandomPlayer

BATTLE_FORMAT = "gen9randombattle"
BATTLE_TIMEOUT_S = 900


async def run(n_battles: int, concurrent: int) -> int:
    p1 = RandomPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=concurrent)
    p2 = RandomPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=concurrent)

    started = time.perf_counter()
    await asyncio.wait_for(p1.battle_against(p2, n_battles=n_battles), timeout=BATTLE_TIMEOUT_S)
    elapsed = time.perf_counter() - started

    finished = p1.n_finished_battles + p2.n_finished_battles
    wins = p1.n_won_battles + p2.n_won_battles
    losses = p1.n_lost_battles + p2.n_lost_battles
    turns = sorted(b.turn for b in p1.battles.values())
    # each finished battle reports once from each side, so one battle == 2 finished reports
    battles = finished // 2

    print(f"battle_format    : {BATTLE_FORMAT}")
    print(f"battles          : {battles} (requested {n_battles})")
    print(f"max_concurrent   : {concurrent}")
    print(f"finished reports : {finished}")
    print(f"wins / losses    : {wins} / {losses}")
    print(f"turns per battle : {turns}")
    print(f"wall time        : {elapsed:.2f}s")
    print(f"time per battle  : {elapsed / max(battles, 1):.2f}s")

    ok = finished == 2 * n_battles and wins == n_battles and losses == n_battles
    print("SMOKE RESULT     :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=1)
    ap.add_argument("--concurrent", type=int, default=1)
    args = ap.parse_args()
    return asyncio.run(run(args.battles, args.concurrent))


if __name__ == "__main__":
    raise SystemExit(main())
