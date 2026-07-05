#!/usr/bin/env bash
# Canonical Sparkie queue: old 3-iteration checkpoint re-evals, then SimPO-full,
# then GRPO. Launch this script in one tmux session; it delegates to each stage
# and stops on the first failure.
#
# Usage:
#   tmux new-session -d -s reeval_sfull_grpo \
#     "bash -lc 'bash ~/croco/src/scripts/reeval_sfull_grpo_queue.sh \
#     2>&1 | tee ~/croco/reeval_sfull_grpo_queue.log'"
set -Eeuo pipefail
cd ~/croco

log() { echo "[$(date "+%F %T")] $*"; }

run_stage() {
    local label="$1"
    local script="$2"
    local logfile="$3"
    local -a statuses
    local script_status
    local tee_status

    log "===== Running $label ====="

    set +e
    bash "$script" 2>&1 | tee "$logfile"
    statuses=("${PIPESTATUS[@]}")
    set -e

    script_status="${statuses[0]:-1}"
    tee_status="${statuses[1]:-0}"

    if [ "$script_status" -ne 0 ]; then
        log "===== $label failed (exit $script_status); stopping queue ====="
        return "$script_status"
    fi

    if [ "$tee_status" -ne 0 ]; then
        log "===== $label logging failed (tee exit $tee_status); stopping queue ====="
        return "$tee_status"
    fi

    log "===== $label complete ====="
}

log "===== Canonical Sparkie queue started ====="
log "Order: 3-iter checkpoint re-evals -> SimPO-full -> GRPO"

# The helper waits for existing training/GPU work, including the live
# stuned_rerun session, before it starts the checkpoint re-evaluations.
run_stage \
    "3-iteration checkpoint re-evals" \
    "$HOME/croco/src/scripts/reeval_3iter_queue.sh" \
    "$HOME/croco/reeval_3iter_queue.log"

run_stage \
    "SimPO-full" \
    "$HOME/croco/src/scripts/simpo_full_queue.sh" \
    "$HOME/croco/simpo_full_queued.log"

run_stage \
    "GRPO" \
    "$HOME/croco/src/scripts/grpo_queue.sh" \
    "$HOME/croco/grpo_queued.log"

log "===== Canonical Sparkie queue complete ====="
