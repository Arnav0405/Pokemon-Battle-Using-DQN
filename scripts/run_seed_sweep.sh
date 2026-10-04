#!/usr/bin/env bash
# Phase 3, tier 1b — seed sweep: does the update budget really win, and does capacity help?
#
#   bash scripts/run_seed_sweep.sh            # all arms, sequential (~1.6 h)
#   STEPS=50000 bash scripts/run_seed_sweep.sh
#
# The Showdown server must already be running (bash scripts/showdown_start.sh) — this script
# never spawns it, same as train.py.
#
# Arms differ only in the training config; seed, opponent and step count are identical:
#   A  SB3 defaults          0.25 updates per env step   (seed 0 already exists as `p3_100k`)
#   B  16x update budget     train_freq=1 gradient_steps=4 batch_size=128 target_update_interval=1000
#                            (seed 0 already exists as `p3b_100k`)
#   C  B + capacity          net_arch=256,256
#
# Sequential on purpose: the Node server and this CPU are the shared resources, so running arms in
# parallel would make their throughput (and thus their learning-per-wall-clock) incomparable.
#
# Ordering note: A1, B1 come first so a 2-seed A-vs-B read exists after ~25 min, then A2, B2 widen
# it to 3 seeds, and the capacity arm runs last.
#
# Caveat recorded here because it shapes how the results must be read: the *battle* RNG lives in the
# Showdown server and is not seeded per battle, so run-to-run spread combines agent-seed noise with
# battle-outcome noise. That is the variance that matters when deciding whether a config wins, but
# it is not a controlled paired comparison.

set -u
cd "$(dirname "$0")/.."
PY="${PY:-../.venv/Scripts/python.exe}"
STEPS="${STEPS:-100000}"
COMMON="--timesteps $STEPS --opponent heuristic --checkpoint-every 10000"
BUDGET="--train-freq 1 --gradient-steps 4 --batch-size 128 --target-update-interval 1000"
CAPACITY="--net-arch 256,256"

run() {
  local name="$1"
  shift
  local started
  started=$(date +%s)
  echo "[sweep] $(date +%H:%M:%S) start $name"
  "$PY" -m lab_ca.train $COMMON --run-name "$name" $* > "/tmp/sweep_${name}.log" 2>&1
  local status=$?
  cp "/tmp/sweep_${name}.log" "runs/${name}/train.log" 2>/dev/null || true
  echo "[sweep] $(date +%H:%M:%S) done  $name status=$status elapsed=$(( $(date +%s) - started ))s"
}

run sweep_a_seed1 --seed 1
run sweep_b_seed1 --seed 1 $BUDGET
run sweep_a_seed2 --seed 2
run sweep_b_seed2 --seed 2 $BUDGET
run sweep_c_seed0 --seed 0 $BUDGET $CAPACITY
run sweep_c_seed1 --seed 1 $BUDGET $CAPACITY
run sweep_c_seed2 --seed 2 $BUDGET $CAPACITY

echo "[sweep] all arms finished $(date +%H:%M:%S)"
