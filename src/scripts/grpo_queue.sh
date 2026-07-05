#!/usr/bin/env bash
# GRPO online-RL baseline runner. Launch ONLY when the GPU is free (i.e. after the
# live ablations.sh queue has finished). Self-updates the repo first so it never
# touches the working tree while another run is in flight.
#
# Manual launch:
#   tmux new-session -d -s grpo "bash -lc 'bash ~/croco/src/scripts/grpo_queue.sh 2>&1 | tee /tmp/grpo_queue.log'"
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
DIR=croco-munin-apertus-8b-da-grpo

# TRL optimization: persistent datasets cache (avoids /tmp disappearing mid-run)
export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
export TMPDIR=~/croco/.tmp
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR"

log "===== GRPO: sync repo ====="
run git pull --ff-only

log "===== GRPO SMOKE TEST (de-risk memory + reward template) ====="
if uv run src/scripts/train_grpo.py -c config/danish-micro-grpo.yaml; then
  log "GRPO SMOKE OK - proceeding to full run"
  rm -rf models/_smoke_grpo
else
  log "GRPO SMOKE FAILED - aborting (check vLLM-colocate memory / reward template)"
  exit 1
fi

log "===== GRPO FULL: online training ====="
run uv run src/scripts/train_grpo.py -c config/danish-apertus-grpo.yaml

log "===== GRPO: final eval (local adapter, 10 iterations) ====="
run uv run euroeval --model "models/$DIR" \
  --language da --num-iterations 10 --gpu-memory-utilization $GMEM --save-results
log "===== GRPO: checkpoint evals (10 iterations) ====="
run uv run src/scripts/eval_checkpoints.py -m "models/$DIR" \
  -l da --num-iterations 10 --gpu-memory-utilization $GMEM --no-include-final
log "===== GRPO DONE ====="
