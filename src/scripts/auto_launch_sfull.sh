#!/usr/bin/env bash
# Auto-launch the SimPO-full ablation once the SimPO-tuned stage has finished and
# the GPU is free.
#
# Usage:
#   tmux new-session -d -s auto_sfull "bash -lc 'bash ~/croco/src/scripts/auto_launch_sfull.sh 2>&1 | tee ~/croco/auto_sfull_launch.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }

WATCH=stuned           # previous-stage tmux session to wait on
STABLE_MINUTES=${STABLE_MINUTES:-3}

gpu_busy() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .
}

# A session is "working" only if it exists AND some pane runs a non-shell command;
# a session that is gone, or only running an interactive shell, is not working.
session_working() {
    tmux has-session -t "$1" 2>/dev/null || return 1
    tmux list-panes -t "$1" -F '#{pane_current_command}' 2>/dev/null \
        | grep -qvE '^(bash|-bash|zsh|-zsh|sh|fish|tmux)$'
}

wait_for_stage() {
    local name=$1 stable=0
    log "Waiting for '$name' to finish (GPU idle + session idle for ${STABLE_MINUTES} min)..."
    while (( stable < STABLE_MINUTES )); do
        if gpu_busy || session_working "$name"; then
            (( stable > 0 )) && log "  still active, resetting"
            stable=0
        else
            stable=$((stable + 1))
            log "  idle ${stable}/${STABLE_MINUTES} min"
        fi
        sleep 60
    done
}

log "===== Auto-launch SimPO-full monitor started ====="
wait_for_stage "$WATCH"
log "===== '$WATCH' finished, GPU free — launching SimPO-full in its own session ====="
tmux new-session -d -s sfull \
    "bash -lc 'bash ~/croco/src/scripts/simpo_full_queue.sh 2>&1 | tee ~/croco/simpo_full_rerun.log'"
log "===== SimPO-full launched (session: sfull) ====="
