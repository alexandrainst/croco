#!/usr/bin/env bash
# Queue re-evaluation of 3-iteration checkpoints.
# Waits for BOTH stuned and sfull training sessions to complete, then runs re-eval.
#
# Usage:
#   tmux new-session -d -s reeval3_queue "bash -lc 'bash ~/croco/src/scripts/reeval_3iter_queue.sh 2>&1 | tee ~/croco/reeval_3iter_queue.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }

# A session is "working" only if it exists AND some pane runs a non-shell command
does_session_exist() {
    tmux has-session -t "$1" 2>/dev/null
}

session_working() {
    tmux has-session -t "$1" 2>/dev/null || return 1
    tmux list-panes -t "$1" -F '#{pane_current_command}' 2>/dev/null \
        | grep -qvE '^(bash|-bash|zsh|-zsh|sh|fish|tmux)$'
}

wait_for_training() {
    log "Waiting for stuned and sfull training sessions to finish..."
    while does_session_exist "stuned" || does_session_exist "sfull"; do
        if session_working "stuned"; then
            log "  stuned still training..."
        elif does_session_exist "stuned"; then
            log "  stuned finishing up..."
        fi
        if session_working "sfull"; then
            log "  sfull still training..."
        elif does_session_exist "sfull"; then
            log "  sfull finishing up..."
        fi
        sleep 60
    done
    log "  Both training sessions complete."
}

log "===== Re-eval 3-iter queue monitor started ====="
wait_for_training
log "===== Training complete — launching re-eval ====="
bash ~/croco/src/scripts/reeval_3iter_checkpoints.sh 2>&1 | tee ~/croco/reeval_3iter.log
log "===== Re-eval complete ====="
