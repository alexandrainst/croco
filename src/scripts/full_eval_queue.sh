#!/usr/bin/env bash
set -uo pipefail
cd ~/croco
log() { echo "[$(date '+%F %T')] $*"; }

# Wait for a tmux session's pane to die
wait_for_session() {
    local session="$1"
    log "Waiting for $session to finish..."
    while tmux has-session -t "$session" 2>/dev/null; do
        if ! tmux capture-pane -t "$session" -p 2>/dev/null | grep -q 'Pane is dead'; then
            log "  $session still running..."
        else
            log "  $session complete (pane dead)"
            break
        fi
        sleep 30
    done
    tmux kill-session -t "$session" 2>/dev/null || true
    log "  $session session cleaned up."
}

log "===== Full eval queue started ====="
wait_for_session "reeval3"

# simpo_full_queue.sh runs inline - it blocks until training+eval complete
log "===== Launching SimPO-full (this will take several hours) ====="
bash ~/croco/src/scripts/simpo_full_queue.sh 2>&1 | tee ~/croco/simpo_full_queued.log
log "===== SimPO-full complete ====="

log "===== Launching Llama RM (this will take several hours) ====="
bash ~/croco/src/scripts/llama_rm_queue.sh 2>&1 | tee ~/croco/llama_rm_queued.log
log "===== Llama RM complete ====="

log "===== Launching GRPO (this will take several hours) ====="
bash ~/croco/src/scripts/grpo_queue.sh 2>&1 | tee ~/croco/grpo_queued.log
log "===== GRPO complete ====="

log "===== All evaluations complete ====="
