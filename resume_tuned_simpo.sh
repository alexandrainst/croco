#!/usr/bin/env bash
# Resume the SimPO ablation suite with the tuned (beta=2.0) and full (ref-free + gamma)
# variants, validating the custom `simpo` loss via a micro pre-flight on GPU before the
# 8h full run. Launched AFTER resume_ls_simpo.sh completes (watch its log for
# `===== ls/simpo DONE =====`). The tuned run uses the standard sigmoid_norm path and
# always proceeds; the full run is gated on the pre-flight (SOFT-SKIP on failure, never
# a hard abort, since the tuned run does not depend on the custom loss code).
#
#   tmux new-session -d -s tqueue "bash -lc 'bash ~/croco/resume_tuned_simpo.sh 2>&1 | tee tuned_ablations.log'"
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

log "===== PRE-FLIGHT: micro run of the custom simpo loss ====="
if uv run src/scripts/run_pipeline.py -c config/danish-micro-simpo.yaml \
     --dataset-output data/pairs_micro_simpo.jsonl \
     --candidate-cache /tmp/micro_simpo_cache.jsonl; then
  SIMPO_OK=1
  log "PRE-FLIGHT OK - proceeding to full simpo run"
else
  SIMPO_OK=0
  log "PRE-FLIGHT FAILED - skipping full simpo run (custom loss needs fixing); tuned run will still proceed"
fi

run_ablation config/danish-apertus-simpo-tuned.yaml croco-munin-apertus-8b-da-simpo-tuned

if [ "$SIMPO_OK" = "1" ]; then
  run_ablation config/danish-apertus-simpo-full.yaml croco-munin-apertus-8b-da-simpo-full
else
  log "===== SKIPPING full simpo (pre-flight failed) ====="
fi

log "===== tuned simpo DONE ====="
