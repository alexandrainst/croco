#!/usr/bin/env bash
# Auto-launch Llama RM ablation when GPU becomes free.
# Monitors the 'queue' tmux session and launches llamarm when it finishes.
#
# Usage:
#   tmux new-session -d -s auto_rm "bash -lc 'bash ~/croco/src/scripts/auto_launch_llama_rm.sh 2>&1 | tee ~/croco/auto_rm_launch.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }

log "===== Auto-launch Llama RM monitor started ====="
log "Monitoring 'queue' tmux session for completion..."

wait_for_queue() {
    # Wait until queue session doesn't exist
    while tmux has-session -t queue 2>/dev/null; do
        log "  queue session still running ($(tmux list-sessions | grep queue))"
        sleep 60
    done
    # Double-check GPU is free
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
        log "  GPU still in use by vLLM processes..."
        sleep 30
    done
}

launch_llama_rm() {
    log "===== Queue finished, GPU free — launching Llama RM ====="
    cd ~/croco
    bash src/scripts/llama_rm_queue.sh 2>&1 | tee /tmp/llama_rm_queue.log
    log "===== Llama RM DONE ====="
}

# Main loop
wait_for_queue
launch_llama_rm
