#!/usr/bin/env bash
# Phase 4, Pass 1 -- frozen-policy evaluation of every checkpoint the report needs.
#
#   bash scripts/run_phase4_eval.sh                     # 200 battles/opponent, 4 parallel envs
#   BATTLES=400 CONCURRENT=8 bash scripts/run_phase4_eval.sh
#   FORCE=1 bash scripts/run_phase4_eval.sh             # redo runs that already have eval.json
#
# Sequential across checkpoints: each evaluation already runs several parallel single-battle envs and
# the one Node server is the shared bottleneck, so stacking more parallelism would only make the
# battles-per-second numbers incomparable and the server less stable.
#
# The server is the fragile part of this pipeline. Two failure modes were observed while building it,
# both of which silently cost a whole evaluation:
#   * the process dies outright -- its log ends in a Node cluster traceback, a worker restarting into
#     an already-bound port being the usual cause -- after which every worker gets
#     WinError 1225 ConnectionRefused;
#   * the process stops answering HTTP while still holding the port, so "just start another one"
#     crashes the new instance with EADDRINUSE.
# So: probe the port before reacting, kill a stale listener before starting a fresh server, and retry
# an evaluation that produced no eval.json. eval.json is written only after every opponent finished,
# which makes its presence a reliable completion marker -- and makes this script resumable.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
VP="${VP:-C:/Users/Arnav Waghdhare/Desktop/Arnav20/Coding/Python/l_lab/.venv/Scripts/python.exe}"
[ -x "$VP" ] || { echo "venv python not found at $VP" >&2; exit 1; }

BATTLES="${BATTLES:-200}"
CONCURRENT="${CONCURRENT:-4}"
ATTEMPTS="${ATTEMPTS:-3}"
SERVER_LOG="logs/showdown_server.log"

server_up() { curl -s -o /dev/null --max-time 8 http://127.0.0.1:8000/; }

# PIDs currently LISTENING on 8000 -- netstat -ano prints them in the last column.
port_pids() {
  netstat -ano 2>/dev/null | grep -E "[:.]8000[[:space:]].*LISTENING" | awk '{print $NF}' | sort -u
}

kill_listeners() {
  local pid
  for pid in $(port_pids); do
    echo "[server] port 8000 still held by pid $pid -- killing it"
    # single slashes: MSYS path conversion is disabled here, so //F reaches taskkill literally and is
    # rejected as an invalid option
    taskkill /F /PID "$pid" >/dev/null 2>&1 || true
  done
}

ensure_server() {
  server_up && return 0
  kill_listeners
  for _ in $(seq 1 20); do
    [ -z "$(port_pids)" ] && break
    sleep 1
  done
  # append, never truncate: a crash traceback in this file is the only evidence of why a run died
  echo "=== server start $(date -u +%H:%M:%S) ===" >> "$SERVER_LOG"
  nohup bash scripts/showdown_start.sh --skip-build >> "$SERVER_LOG" 2>&1 &
  for _ in $(seq 1 45); do
    sleep 2
    server_up && { echo "[server] up"; return 0; }
  done
  echo "[server] would not come up within 90s; aborting" >&2
  return 1
}

echo "battles=$BATTLES per opponent, concurrent=$CONCURRENT envs"
ensure_server || exit 1

# arm A = SB3 defaults (the baseline the sweep rejected); arm B = the adopted config, at 100k
# (3 seeds), 200k and 400k (the run that finally declared convergence). Arm C is not evaluated: it
# was rejected for across-seed variance, so an evaluation of one of its checkpoints would be a
# number with no meaning behind it.
RUNS=(
  p3_100k sweep_a_seed1 sweep_a_seed2
  p3b_100k sweep_b_seed1 sweep_b_seed2
  p3_200k_2 p3_400k
)

for run in "${RUNS[@]}"; do
  checkpoint="runs/${run}/model_final.zip"
  out="runs/${run}/eval.json"
  if [ ! -f "$checkpoint" ]; then
    echo "MISSING $checkpoint -- skipping $run"
    continue
  fi
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== $run === already evaluated (FORCE=1 to redo)"
    continue
  fi
  for attempt in $(seq 1 "$ATTEMPTS"); do
    ensure_server || exit 1
    echo "=== $run === (attempt $attempt/$ATTEMPTS)"
    "$VP" -u -m lab_ca.evaluate --checkpoint "$checkpoint" \
      --battles "$BATTLES" --concurrent "$CONCURRENT"
    [ -f "$out" ] && break
    echo "=== $run === produced no eval.json; restarting the server and retrying"
  done
  [ -f "$out" ] || echo "!!! $run FAILED after $ATTEMPTS attempts"
done

echo "=== random-policy floor (harness sanity: a scripted-random agent should sit near 50% vs"
echo "    random and clearly lose to the heuristic) ==="
if [ -f runs/eval_random_floor.json ] && [ "${FORCE:-0}" != "1" ]; then
  echo "floor already evaluated (FORCE=1 to redo)"
else
  for attempt in $(seq 1 "$ATTEMPTS"); do
    ensure_server || exit 1
    "$VP" -u -m lab_ca.evaluate --agent random --out runs/eval_random_floor.json \
      --battles "$BATTLES" --concurrent "$CONCURRENT"
    [ -f runs/eval_random_floor.json ] && break
  done
fi

echo "=== summary table + figure ==="
"$VP" -u -m lab_ca.evaluate --summarize runs/*/eval.json runs/eval_random_floor.json \
  --out runs/phase4_eval
