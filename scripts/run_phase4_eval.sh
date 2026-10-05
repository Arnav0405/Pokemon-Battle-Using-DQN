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
# THIS SCRIPT DOES NOT TOUCH THE SERVER -- it verifies it and stops. An earlier version tried to be
# self-healing (kill whatever holds port 8000, start a fresh server, retry). That was worse than the
# problem: the server answers in milliseconds when it is up, so a single timed-out probe during a
# busy or restarting moment is not evidence of death, and the reaction destroyed a healthy server
# twice, turning a hiccup into a lost evaluation. Two real failure modes are worth knowing about,
# both of which make every worker fail with WinError 1225 ConnectionRefused:
#   * the process dies outright -- its log ends in a Node cluster traceback (a worker restarting into
#     an already-bound port is the usual cause);
#   * it answers nothing while still holding the port, so "start another one" only gives the new
#     instance EADDRINUSE. Recover by hand: kill the process holding 8000, wait for the port to free,
#     then start one server. Give a fresh cluster a full minute before judging it dead.
# Start the server the documented way, from its own terminal:
#
#   bash scripts/showdown_start.sh --skip-build
#
# eval.json is written only after every opponent finished, so its presence is a reliable completion
# marker: this script skips runs that already have one and can be resumed with the same command.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
VP="${VP:-C:/Users/Arnav Waghdhare/Desktop/Arnav20/Coding/Python/l_lab/.venv/Scripts/python.exe}"
[ -x "$VP" ] || { echo "venv python not found at $VP" >&2; exit 1; }

BATTLES="${BATTLES:-200}"
CONCURRENT="${CONCURRENT:-4}"
ATTEMPTS="${ATTEMPTS:-2}"

# Curl note, and the reason the first version of this script kept "losing" the server:
# `curl -s -o /dev/null` exits 23 (write error) in this environment even when the request returned a
# clean 200 -- it writes the body to /dev/null through a code path MSYS reports as a failure. So
# `curl -o /dev/null <url> && ...` is ALWAYS false here, and every check built on it declared a
# healthy server dead (then killed it, or refused to start). Compare the status code instead.
http_code() { curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://127.0.0.1:8000/ 2>/dev/null; }

require_server() {
  local _attempt
  for _attempt in 1 2 3; do
    [ "$(http_code)" = "200" ] && return 0
    sleep 5
  done
  echo "server at 127.0.0.1:8000 is not answering (http_code=$(http_code))." >&2
  echo "start it in its own terminal:  bash scripts/showdown_start.sh --skip-build" >&2
  return 1
}

echo "battles=$BATTLES per opponent, concurrent=$CONCURRENT envs"
require_server || exit 2

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
    echo "=== $run === (attempt $attempt/$ATTEMPTS)"
    "$VP" -u -m lab_ca.evaluate --checkpoint "$checkpoint" \
      --battles "$BATTLES" --concurrent "$CONCURRENT"
    [ -f "$out" ] && break
    echo "=== $run === produced no eval.json"
  done
  [ -f "$out" ] || echo "!!! $run FAILED after $ATTEMPTS attempts"
done

echo "=== random-policy floor (harness sanity: a scripted-random agent should sit near 50% vs"
echo "    random and clearly lose to the heuristic) ==="
if [ -f runs/eval_random_floor.json ] && [ "${FORCE:-0}" != "1" ]; then
  echo "floor already evaluated (FORCE=1 to redo)"
else
  require_server || exit 2
  "$VP" -u -m lab_ca.evaluate --agent random --out runs/eval_random_floor.json \
    --battles "$BATTLES" --concurrent "$CONCURRENT"
fi

echo "=== summary table + figure ==="
"$VP" -u -m lab_ca.evaluate --summarize runs/*/eval.json runs/eval_random_floor.json \
  --out runs/phase4_eval
