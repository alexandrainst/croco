#!/usr/bin/env bash
# Auto-launch the Llama-RM ablation once the previous stage (the ls/simpo queue)
# has finished and the GPU is free.
#
# Why not just `while tmux has-session -t queue`: a finished queue can leave its
# tmux session alive as an idle shell, which stalls a session-existence wait
# forever. Instead we wait until the GPU has been free AND the watched session is
# gone/idle for several consecutive minutes — checked together so a brief
# between-phase gap (GPU momentarily free while the queue is still mid-run) can't
# fire us early. For a fully deterministic run, prefer the sequential runner
# `run_chain.sh` over these chained monitors.
#
# Usage:
#   tmux new-session -d -s auto_rm "bash -lc 'bash ~/croco/src/scripts/auto_launch_llama_rm.sh 2>&1 | tee ~/croco/auto_rm_launch.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }

WATCH=queue          # previous-stage tmux session to wait on
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
            (( stable > 0 )) && log "  still active (gpu_busy=$(gpu_busy && echo y || echo n)), resetting"
            stable=0
        else
            stable=$((stable + 1))
            log "  idle ${stable}/${STABLE_MINUTES} min"
        fi
        sleep 60
    done
}

log "===== Auto-launch Llama RM monitor started ====="
wait_for_stage "$WATCH"
log "===== '$WATCH' finished, GPU free — launching Llama RM in its own session ====="
tmux new-session -d -s llamarm \
    "bash -lc 'bash ~/croco/src/scripts/llama_rm_queue.sh 2>&1 | tee /tmp/llama_rm_queue.log'"
log "===== Llama RM launched (session: llamarm) ====="
