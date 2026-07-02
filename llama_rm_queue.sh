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

log "===== Llama RM: sync repo ====="
run git pull --ff-only

log "===== Llama RM: update config with ls/simpo winner ====="
# Check which loss won (edit this before running based on ls/simpo results):
#   - If label smoothing wins: add `label_smoothing: 0.05` to dpo block
#   - If SimPO wins: add `loss_type: sigmoid_norm` to dpo block
#   - If vanilla DPO wins: leave as is
log "WARNING: Review config/danish-apertus-llama-rm.yaml dpo block before proceeding"
log "Current config uses vanilla DPO (no modifications)"

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
  --language da --num-iterations 3 --gpu-memory-utilization $GMEM --save-results
log "===== Llama RM: checkpoint evals (3 iterations) ====="
run uv run src/scripts/eval_checkpoints.py -m "models/$DIR" \
  -l da --num-iterations 3 --gpu-memory-utilization $GMEM --no-include-final
log "===== Llama RM DONE ====="
