#!/usr/bin/env bash
# Deterministic sequential runner for the remaining ablations: Llama-RM, then GRPO.
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

bash src/scripts/llama_rm_queue.sh 2>&1 | tee /tmp/llama_rm_queue.log
echo "LLAMARM_EXIT ${PIPESTATUS[0]} $(date '+%F %T')"

bash src/scripts/grpo_queue.sh 2>&1 | tee /tmp/grpo_queue.log
echo "GRPO_EXIT ${PIPESTATUS[0]} $(date '+%F %T')"

echo "CHAIN_DONE $(date '+%F %T')"
