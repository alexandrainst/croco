#!/usr/bin/env bash
set -uo pipefail
cd ~/croco
log() { echo "[$(date '+%F %T')] $*"; }
run() { log "RUN: $*"; "$@" && log "OK" || log "FAILED ($?): $*"; }

wait_for_session() {
    local session="$1"
    log "Waiting for $session to finish..."
    while tmux has-session -t "$session" 2>/dev/null; do
        if tmux capture-pane -t "$session" -p 2>/dev/null | grep -q 'Pane is dead'; then
            log "  $session complete (pane dead)"
            break
        fi
        log "  $session still running..."
        sleep 60
    done
    tmux kill-session -t "$session" 2>/dev/null || true
    log "  $session session cleaned up."
}

wait_for_any_training() {
    log "Checking for active training sessions..."
    local sessions=("stuned_rerun" "stuned" "sfull" "llamarm" "grpo" "queue")
    for sess in "${sessions[@]}"; do
        if tmux has-session -t "$sess" 2>/dev/null; then
            log "  Found: $sess - waiting..."
            wait_for_session "$sess"
        fi
    done
    log "  All training sessions complete."
}

log "===== Full eval queue started ====="

# Wait for any existing training
wait_for_any_training

# Launch and wait for re-eval (needs GPU free)
log "===== Launching 3-iter re-eval (10 iterations with --force) ====="
tmux new-session -d -s reeval3 "bash -lc 'bash ~/croco/src/scripts/reeval_3iter_checkpoints.sh 2>&1 | tee ~/croco/reeval_3iter_queued.log'"
wait_for_session "reeval3"

# Launch and wait for SimPO-full
log "===== Launching SimPO-full ====="
tmux new-session -d -s sfull "bash -lc 'bash ~/croco/src/scripts/simpo_full_queue.sh 2>&1 | tee ~/croco/simpo_full_queued.log'"
wait_for_session "sfull"

# Launch and wait for Llama RM
log "===== Launching Llama RM ====="
tmux new-session -d -s llamarm "bash -lc 'bash ~/croco/src/scripts/llama_rm_queue.sh 2>&1 | tee ~/croco/llama_rm_queued.log'"
wait_for_session "llamarm"

# Launch and wait for GRPO
log "===== Launching GRPO ====="
tmux new-session -d -s grpo "bash -lc 'bash ~/croco/src/scripts/grpo_queue.sh 2>&1 | tee ~/croco/grpo_queued.log'"
wait_for_session "grpo"

log "===== All complete ====="
