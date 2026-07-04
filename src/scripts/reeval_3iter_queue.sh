#!/usr/bin/env bash
# Queue re-evaluation of 3-iteration checkpoints, waiting for GPU availability.
# Monitors GPU and stuned session, launches re-eval when both are free.
#
# Usage:
#   tmux new-session -d -s reeval3_queue "bash -lc 'bash ~/croco/src/scripts/reeval_3iter_queue.sh 2>&1 | tee ~/croco/reeval_3iter_queue.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }

STABLE_MINUTES=${STABLE_MINUTES:-3}

gpu_busy() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .
}

session_working() {
    tmux has-session -t "$1" 2>/dev/null || return 1
    tmux list-panes -t "$1" -F '#{pane_current_command}' 2>/dev/null \
        | grep -qvE '^(bash|-bash|zsh|-zsh|sh|fish|tmux)$'
}

wait_for_gpu() {
    local stable=0
    log "Waiting for GPU to be free (idle for ${STABLE_MINUTES} min)..."
    while (( stable < STABLE_MINUTES )); do
        if gpu_busy; then
            (( stable > 0 )) && log "  GPU busy, resetting"
            stable=0
        else
            stable=$((stable + 1))
            log "  GPU idle ${stable}/${STABLE_MINUTES} min"
        fi
        sleep 60
    done
}

log "===== Re-eval 3-iter queue monitor started ====="
wait_for_gpu
log "===== GPU free — launching re-eval ====="
bash ~/croco/src/scripts/reeval_3iter_checkpoints.sh 2>&1 | tee ~/croco/reeval_3iter.log
log "===== Re-eval complete ====="
