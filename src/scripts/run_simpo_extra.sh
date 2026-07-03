#!/usr/bin/env bash
# One-off follow-on used when the in-flight 'chain' session was launched with an
# older run_chain.sh that only covered Llama-RM -> GRPO. This waits for that chain
# to finish and the GPU to free, then runs the two SimPO ablations that were added
# later (tuned, then full), so the full queue completes without a restart.
#
# Launch:
#   tmux new-session -d -s chain2 "bash -lc 'bash ~/croco/src/scripts/run_simpo_extra.sh 2>&1 | tee ~/croco/chain2.log'"
set -uo pipefail
cd ~/croco
log() { echo "[$(date "+%F %T")] $*"; }

WATCH=${WATCH:-chain}

log "===== SimPO-extra follow-on: waiting for '$WATCH' to finish ====="
# 'chain' is a self-closing command session, so waiting for it to disappear is a
# reliable "previous stages done" signal; the GPU check guards the hand-off.
while tmux has-session -t "$WATCH" 2>/dev/null; do
    sleep 60
done
log "'$WATCH' gone; confirming GPU is free..."
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    sleep 30
done

log "===== GPU free — running SimPO-tuned then SimPO-full ====="
bash src/scripts/simpo_tuned_queue.sh 2>&1 | tee /tmp/simpo_tuned_queue.log
echo "SIMPO_TUNED_EXIT ${PIPESTATUS[0]} $(date '+%F %T')"
bash src/scripts/simpo_full_queue.sh 2>&1 | tee /tmp/simpo_full_queue.log
echo "SIMPO_FULL_EXIT ${PIPESTATUS[0]} $(date '+%F %T')"
log "===== SimPO-extra DONE ====="
