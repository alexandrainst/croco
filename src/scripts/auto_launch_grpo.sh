#!/usr/bin/env bash
# Auto-launch GRPO baseline when Llama RM finishes.
# Monitors the 'llamarm' tmux session and launches grpo when it finishes.
#
# Usage:
#   tmux new-session -d -s auto_grpo "bash -lc 'bash ~/croco/src/scripts/auto_launch_grpo.sh 2>&1 | tee ~/croco/auto_grpo_launch.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }

log "===== Auto-launch GRPO monitor started ====="
log "Monitoring 'llamarm' tmux session for completion..."

wait_for_llamarm() {
    # Wait until llamarm session doesn't exist
    while tmux has-session -t llamarm 2>/dev/null; do
        log "  llamarm session still running"
        sleep 60
    done
    # Double-check GPU is free
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
        log "  GPU still in use by vLLM processes..."
        sleep 30
    done
}

launch_grpo() {
    log "===== Llama RM finished, GPU free — launching GRPO ====="
    cd ~/croco
    bash src/scripts/grpo_queue.sh 2>&1 | tee /tmp/grpo_queue.log
    log "===== GRPO DONE ====="
}

# Main loop
wait_for_llamarm
launch_grpo
