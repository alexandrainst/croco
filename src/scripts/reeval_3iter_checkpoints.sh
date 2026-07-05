#!/usr/bin/env bash
# Re-evaluate all checkpoints that were previously run with only 3 iterations.
# This script evaluates with 10 iterations for tighter confidence intervals.
#
# Models to re-eval (all have checkpoint-100 through checkpoint-625):
#   - croco-munin-apertus-8b-da (max_reward baseline)
#   - croco-munin-apertus-8b-da-generated (generated mode)
#   - croco-munin-apertus-8b-da-gold (gold_chosen mode)
#   - croco-munin-apertus-8b-da-ls (label smoothing α=0.05)
#   - croco-munin-apertus-8b-da-simpo (SimPO loss β=2.0 γ=0.5)
#
# Usage:
#   tmux new-session -d -s reeval3 "bash -lc 'bash ~/croco/src/scripts/reeval_3iter_checkpoints.sh 2>&1 | tee ~/croco/reeval_3iter.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }
run() { log "RUN: $*"; "$@" && log "OK" || log "FAILED ($?): $*"; }

log "===== Re-eval 3-iter checkpoints: sync repo ====="
run git pull --ff-only

GMEM=0.5
ITERATIONS=10

# Model directories to re-evaluate (all have checkpoints 100-625)
MODELS=(
    "croco-munin-apertus-8b-da"
    "croco-munin-apertus-8b-da-generated"
    "croco-munin-apertus-8b-da-gold"
    "croco-munin-apertus-8b-da-ls"
    "croco-munin-apertus-8b-da-simpo"
)

for MODEL in "${MODELS[@]}"; do
    log "===== Evaluating $MODEL (10 iterations, all checkpoints) ====="
    run uv run src/scripts/eval_checkpoints.py \
        -m "models/$MODEL" \
        -l da \
        --num-iterations "$ITERATIONS" \
        --gpu-memory-utilization "$GMEM" \
        --no-include-final
done

log "===== Re-eval complete ====="
