#!/usr/bin/env bash
# Deterministic sequential runner for all remaining ablations, in order:
#   1. Llama-RM   2. GRPO   3. SimPO-tuned (β=2.0)   4. SimPO-full (ref-free)
#
# Preferred over the auto_launch_*.sh monitors: there is no inter-session polling,
# so a lingering/idle tmux session can't stall the chain. Each stage runs to
# completion before the next starts, so they never contend for the GPU.
#
# Launch detached in its own self-closing session:
#   tmux new-session -d -s chain "bash -lc 'bash ~/croco/src/scripts/run_chain.sh 2>&1 | tee ~/croco/chain.log'"
set -uo pipefail
cd ~/croco
echo "CHAIN_START $(date '+%F %T')"

stage() {  # $1 = queue script, $2 = log basename
    echo "STAGE_START $2 $(date '+%F %T')"
    bash "src/scripts/$1" 2>&1 | tee "/tmp/$2.log"
    echo "STAGE_EXIT $2 ${PIPESTATUS[0]} $(date '+%F %T')"
}

stage llama_rm_queue.sh    llama_rm_queue
stage grpo_queue.sh        grpo_queue
stage simpo_tuned_queue.sh simpo_tuned_queue
stage simpo_full_queue.sh  simpo_full_queue

echo "CHAIN_DONE $(date '+%F %T')"
