#!/usr/bin/env bash
set -uo pipefail
cd ~/croco
log() { echo "[$(date '+%F %T')] $*"; }

does_session_exist() { tmux has-session -t "$1" 2>/dev/null; }

session_working() {
    tmux has-session -t "$1" 2>/dev/null || return 1
    tmux list-panes -t "$1" -F '#{pane_current_command}' 2>/dev/null | grep -qvE '^(bash|-bash|zsh|-zsh|sh|fish|tmux)$'
}

wait_for_session() {
    local session="$1"
    log "Waiting for $session to finish..."
    while does_session_exist "$session"; do
        if session_working "$session"; then
            log "  $session still running..."
        else
            log "  $session finishing up..."
        fi
        sleep 60
    done
    log "  $session complete."
}

log "===== Full eval queue started ====="
wait_for_session "reeval3"

log "===== Launching SimPO-full ====="
bash ~/croco/src/scripts/simpo_full_queue.sh 2>&1 | tee ~/croco/simpo_full_queued.log
wait_for_session "sfull"

log "===== Launching Llama RM ====="
bash ~/croco/src/scripts/llama_rm_queue.sh 2>&1 | tee ~/croco/llama_rm_queued.log
wait_for_session "llamarm"

log "===== Launching GRPO ====="
bash ~/croco/src/scripts/grpo_queue.sh 2>&1 | tee ~/croco/grpo_queued.log

log "===== All evaluations complete ====="
