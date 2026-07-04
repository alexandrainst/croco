#!/usr/bin/env bash
# Llama RM ablation runner. Launch AFTER the ls/simpo ablations complete to use the
# winning DPO objective. Self-updates the repo first.
#
# Manual launch:
#   tmux new-session -d -s llamarm "bash -lc 'bash ~/croco/llama_rm_queue.sh 2>&1 | tee /tmp/llama_rm_queue.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }
run() { log "RUN: $*"; "$@" && log "OK" || log "FAILED ($?): $*"; }
GMEM=0.5
DIR=croco-munin-apertus-8b-da-llamarm

# TRL optimization: persistent datasets cache (avoids /tmp disappearing mid-run)
export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
mkdir -p "$HF_DATASETS_CACHE"

log "===== Llama RM: sync repo ====="
run git pull --ff-only

log "===== Llama RM: config check ====="
log "Config uses vanilla DPO (default)"
log "OPTIONAL: Edit config/danish-apertus-llama-rm.yaml to match ls/simpo winner"
log "  - Label smoothing wins: add label_smoothing: 0.05"
log "  - SimPO wins: add loss_type: sigmoid_norm"
log "  - Vanilla DPO wins: leave as is (current default)"

log "===== Llama RM: STEP 1 - Re-score candidate cache ====="
run uv run src/scripts/rescore_candidates.py \
  -i data/candidates_cache.jsonl \
  -o data/candidates_cache_llama_rm.jsonl \
  --reward-model-id Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M

log "===== Llama RM: STEP 2 - Build max_reward pairs + DPO train ====="
run uv run src/scripts/run_pipeline.py \
  -c config/danish-apertus-llama-rm.yaml \
  --dataset-output data/pairs_llama_rm.jsonl \
  --candidate-cache data/candidates_cache_llama_rm.jsonl

log "===== Llama RM: final eval (local adapter, 3 iterations) ====="
run uv run euroeval --model "models/$DIR" \
  --language da --num-iterations 10 --gpu-memory-utilization $GMEM --save-results
log "===== Llama RM: checkpoint evals (3 iterations) ====="
run uv run src/scripts/eval_checkpoints.py -m "models/$DIR" \
  -l da --num-iterations 10 --gpu-memory-utilization $GMEM --no-include-final
log "===== Llama RM DONE ====="
