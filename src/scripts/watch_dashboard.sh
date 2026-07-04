#!/usr/bin/env bash
# Watch dashboard builder — auto-regenerates croco_dashboard.html every 60s.
# Reads data from sparkie over SSH, rebuilds locally.
#
# Usage:
#   tmux new-session -d -s dashboard_watch "bash -lc 'bash ~/croco/src/scripts/watch_dashboard.sh 2>&1 | tee ~/croco/dashboard_watch.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }

# All current models (add new ones as they complete training)
MODELS=(
    "models/croco-munin-apertus-8b-da"
    "models/croco-munin-apertus-8b-da-generated"
    "models/croco-munin-apertus-8b-da-gold"
    "models/croco-munin-apertus-8b-da-ls"
    "models/croco-munin-apertus-8b-da-simpo"
    "models/croco-munin-apertus-8b-da-simpo-tuned"
    "models/croco-munin-apertus-8b-da-simpo-full"
)

MODEL_OPTS=()
for m in "${MODELS[@]}"; do
    MODEL_OPTS+=("-m" "$m")
done

log "===== Dashboard watch started (60s refresh) ====="
log "Models: ${MODELS[*]}"

INTERVAL=60

while true; do
    log "Regenerating dashboard..."
    uv run src/scripts/build_dashboard.py \
        "${MODEL_OPTS[@]}" \
        -r euroeval_benchmark_results.jsonl \
        -o croco_dashboard.html \
        --ssh-host sparkie \
        --remote-root ~/croco \
        --watch "$INTERVAL" 2>&1 | tee -a ~/croco/dashboard_watch.log
done
