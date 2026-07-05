#!/usr/bin/env bash
# Queue checkpoint re-evaluation behind Sparkie training and GPU activity.
#
# Usage:
#   tmux new-session -d -s reeval3_queue \
#     "bash -lc 'bash ~/croco/src/scripts/reeval_3iter_queue.sh \
#     2>&1 | tee ~/croco/reeval_3iter_queue.log'"
set -uo pipefail
cd ~/croco

log() { echo "[$(date "+%F %T")] $*"; }

CHECK_INTERVAL_SECONDS=60
REQUIRED_IDLE_CHECKS=3

# Sessions that either run GPU training or auto-launch the next GPU training stage.
# Keep stuned_rerun here until the current SimPO-tuned rerun has completed.
BLOCKING_SESSIONS=(
    "stuned_rerun"
    "stuned"
    "sfull"
    "llamarm"
    "grpo"
    "queue"
    "auto_sfull"
    "auto_rm"
    "auto_grpo"
)

session_has_live_pane() {
    local session="$1"
    tmux list-panes -t "$session" -F '#{pane_dead}' 2>/dev/null \
        | grep -q '^0$'
}

active_blocking_sessions() {
    local session
    for session in "${BLOCKING_SESSIONS[@]}"; do
        if session_has_live_pane "$session"; then
            echo "$session"
        fi
    done
}

gpu_work_pids() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF {print $1}'
}

join_lines() {
    tr '\n' ' ' | sed 's/[[:space:]]*$//'
}

wait_for_gpu_slot() {
    local idle_checks=0
    local sessions=""
    local pids=""

    log "Waiting for training/auto-launch sessions and GPU work to clear..."
    log "Blocking sessions: ${BLOCKING_SESSIONS[*]}"

    while [ "$idle_checks" -lt "$REQUIRED_IDLE_CHECKS" ]; do
        sessions="$(active_blocking_sessions | join_lines)"
        pids="$(gpu_work_pids | join_lines)"

        if [ -z "$sessions" ] && [ -z "$pids" ]; then
            idle_checks=$((idle_checks + 1))
            log "  GPU slot clear ($idle_checks/$REQUIRED_IDLE_CHECKS)."
        else
            idle_checks=0
            if [ -n "$sessions" ]; then
                log "  Waiting on tmux sessions: $sessions"
            fi
            if [ -n "$pids" ]; then
                log "  Waiting on GPU process PIDs: $pids"
            fi
        fi

        if [ "$idle_checks" -lt "$REQUIRED_IDLE_CHECKS" ]; then
            sleep "$CHECK_INTERVAL_SECONDS"
        fi
    done

    log "GPU slot clear for $REQUIRED_IDLE_CHECKS consecutive checks."
}

log "===== Re-eval 3-iter queue monitor started ====="
wait_for_gpu_slot
log "===== Launching 3-iter checkpoint re-eval ====="
bash ~/croco/src/scripts/reeval_3iter_checkpoints.sh 2>&1 | tee ~/croco/reeval_3iter.log
log "===== Re-eval complete ====="
