#!/usr/bin/env bash
# SimPO-tuned ablation runner: length-normalised loss (sigmoid_norm) with beta
# raised 0.1 -> 2.0. Reuses the already-built max_reward pairs (no candidate
# generation), then trains + evaluates. Self-updates the repo first.
#
# Manual launch:
#   tmux new-session -d -s stuned "bash -lc 'bash ~/croco/src/scripts/simpo_tuned_queue.sh 2>&1 | tee /tmp/simpo_tuned_queue.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }
run() { log "RUN: $*"; "$@" && log "OK" || log "FAILED ($?): $*"; }
GMEM=0.5
DIR=croco-munin-apertus-8b-da-simpo-tuned
PAIRS=data/pairs_apertus.jsonl
CACHE=data/candidates_cache.jsonl

log "===== SimPO-tuned: sync repo ====="
run git pull --ff-only

log "===== SimPO-tuned: train (β=2.0, sigmoid_norm; reuse max_reward pairs) ====="
run uv run src/scripts/run_pipeline.py -c config/danish-apertus-simpo-tuned.yaml \
  --dataset-output "$PAIRS" --candidate-cache "$CACHE" --skip-build

log "===== SimPO-tuned: final eval (10 iterations) ====="
run uv run euroeval --model "models/$DIR" \
  --language da --num-iterations 10 --gpu-memory-utilization $GMEM --save-results
log "===== SimPO-tuned: checkpoint evals (3 iterations) ====="
run uv run src/scripts/eval_checkpoints.py -m "models/$DIR" \
  -l da --num-iterations 3 --gpu-memory-utilization $GMEM --no-include-final
log "===== SimPO-tuned DONE ====="
