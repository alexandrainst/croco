#!/usr/bin/env bash
# SimPO-full ablation runner: current config uses TRL's sigmoid_norm
# length-normalised DPO (β=2.0) after the custom ref-free SimPO loss was found
# broken. Reuses the already-built max_reward pairs (no candidate generation),
# then trains + evaluates. Self-updates the repo first.
#
# Manual launch:
#   tmux new-session -d -s sfull "bash -lc 'bash ~/croco/src/scripts/simpo_full_queue.sh 2>&1 | tee /tmp/simpo_full_queue.log'"
set -Eeuo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }
run() {
    local status

    log "RUN: $*"
    if "$@"; then
        log "OK"
    else
        status=$?
        log "FAILED ($status): $*"
        return "$status"
    fi
}
GMEM=0.5
DIR=croco-munin-apertus-8b-da-simpo-full
PAIRS=data/pairs_apertus.jsonl
CACHE=data/candidates_cache.jsonl

# TRL optimization: persistent datasets cache (avoids /tmp disappearing mid-run)
export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
export TMPDIR=~/croco/.tmp
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR"

log "===== SimPO-full: sync repo ====="
run git pull --ff-only

log "===== SimPO-full: train (sigmoid_norm, β=2.0; reuse max_reward pairs) ====="
run uv run src/scripts/run_pipeline.py -c config/danish-apertus-simpo-full.yaml \
  --dataset-output "$PAIRS" --candidate-cache "$CACHE" --skip-build

log "===== SimPO-full: final eval (10 iterations) ====="
run uv run euroeval --model "models/$DIR" \
  --language da --num-iterations 10 --gpu-memory-utilization $GMEM --save-results
log "===== SimPO-full: checkpoint evals (10 iterations) ====="
run uv run src/scripts/eval_checkpoints.py -m "models/$DIR" \
  -l da --num-iterations 10 --gpu-memory-utilization $GMEM --no-include-final
log "===== SimPO-full DONE ====="
