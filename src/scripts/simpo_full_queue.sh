#!/usr/bin/env bash
# SimPO-full ablation runner: true reference-free SimPO loss (loss_type: simpo,
# β=2.0, target margin γ=0.5). Reuses the already-built max_reward pairs (no
# candidate generation), then trains + evaluates. Self-updates the repo first.
#
# Manual launch:
#   tmux new-session -d -s sfull "bash -lc 'bash ~/croco/src/scripts/simpo_full_queue.sh 2>&1 | tee /tmp/simpo_full_queue.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }
run() { log "RUN: $*"; "$@" && log "OK" || log "FAILED ($?): $*"; }
GMEM=0.5
DIR=croco-munin-apertus-8b-da-simpo-full
PAIRS=data/pairs_apertus.jsonl
CACHE=data/candidates_cache.jsonl

log "===== SimPO-full: sync repo ====="
run git pull --ff-only

log "===== SimPO-full: train (ref-free simpo, β=2.0, γ=0.5; reuse max_reward pairs) ====="
run uv run src/scripts/run_pipeline.py -c config/danish-apertus-simpo-full.yaml \
  --dataset-output "$PAIRS" --candidate-cache "$CACHE" --skip-build

log "===== SimPO-full: final eval (10 iterations) ====="
run uv run euroeval --model "models/$DIR" \
  --language da --num-iterations 10 --gpu-memory-utilization $GMEM --save-results
log "===== SimPO-full: checkpoint evals (3 iterations) ====="
run uv run src/scripts/eval_checkpoints.py -m "models/$DIR" \
  -l da --num-iterations 3 --gpu-memory-utilization $GMEM --no-include-final
log "===== SimPO-full DONE ====="
