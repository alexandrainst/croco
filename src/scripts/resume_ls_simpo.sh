#!/usr/bin/env bash
# Resume the ablation suite from the pre-flight gate onward (ls + simpo), after the
# original ablations.sh aborted (2026-06-30) on a transient /tmp arrow-cache failure
# in TRL's precompute_ref_log_probs path - a knob neither ls nor simpo uses. This
# does NOT repeat the already-completed recovery evals or the generated run.
#
#   tmux new-session -d -s queue "bash -lc 'bash ~/croco/resume_ls_simpo.sh 2>&1 | tee ablations.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }
run() { log "RUN: $*"; "$@" && log "OK" || log "FAILED ($?): $*"; }
PAIRS=data/pairs_apertus.jsonl
CACHE=data/candidates_cache.jsonl
GMEM=0.5

eval_model() {  # $1 = model dir under models/
  local dir="$1"
  log "===== $dir: final eval (10 iterations) ====="
  run uv run euroeval --model "danish-foundation-models/$dir" \
    --language da --num-iterations 10 --gpu-memory-utilization $GMEM --save-results
  log "===== $dir: checkpoint evals (3 iterations) ====="
  run uv run src/scripts/eval_checkpoints.py -m "models/$dir" \
    -l da --num-iterations 3 --gpu-memory-utilization $GMEM --no-include-final
}

run_ablation() {  # $1 = config, $2 = model dir
  local cfg="$1" dir="$2"
  log "===== ABLATION $dir: train + upload (reusing built max_reward pairs) ====="
  run uv run src/scripts/run_pipeline.py -c "$cfg" \
    --dataset-output "$PAIRS" --candidate-cache "$CACHE" --skip-build
  eval_model "$dir"
}

log "===== sync repo ====="
run git pull --ff-only

log "===== PRE-FLIGHT: micro run of the DPO knobs ls/simpo use ====="
if uv run src/scripts/run_pipeline.py -c config/danish-micro-ablation.yaml \
     --dataset-output data/pairs_micro_ablation.jsonl \
     --candidate-cache /tmp/micro_ablation_cache.jsonl; then
  log "PRE-FLIGHT OK - proceeding to ablations"
else
  log "PRE-FLIGHT FAILED - aborting ablations (recipe knobs need fixing)"
  exit 1
fi

run_ablation config/danish-apertus-ls.yaml    croco-munin-apertus-8b-da-ls
run_ablation config/danish-apertus-simpo.yaml croco-munin-apertus-8b-da-simpo
log "===== ls/simpo DONE ====="
