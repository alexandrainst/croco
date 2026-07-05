#!/usr/bin/env bash
set -Eeuo pipefail
cd ~/croco
log() { echo "[$(date '+%F %T')] $*"; }
run() {
    local status

    log "RUN: $*"
    if "$@"; then
        log "OK"
    else
        status=$?
        log "FAILED ($status): $*"
        return "$status"
    fi
}

run_reeval_queue() {
    local statuses
    local queue_status
    local tee_status

    if bash ~/croco/src/scripts/reeval_3iter_queue.sh 2>&1 \
        | tee ~/croco/reeval_3iter_queue.log; then
        return 0
    fi

    statuses=("${PIPESTATUS[@]}")
    queue_status="${statuses[0]:-1}"
    tee_status="${statuses[1]:-0}"

    if [ "$queue_status" -ne 0 ]; then
        log "3-iter checkpoint re-eval queue failed (exit $queue_status)."
        return "$queue_status"
    fi

    log "3-iter checkpoint re-eval queue logging failed (tee exit $tee_status)."
    return "$tee_status"
}

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

# Run the single queued 3-iteration checkpoint re-eval path (needs GPU free)
log "===== Running 3-iter checkpoint re-eval queue ====="
run_reeval_queue

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
