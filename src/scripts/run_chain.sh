#!/usr/bin/env bash
# Deprecated compatibility wrapper. Use the canonical Sparkie queue instead:
#   3-iteration checkpoint re-evals -> SimPO-full -> GRPO
#
# Launch detached in its own self-closing session:
#   tmux new-session -d -s reeval_sfull_grpo \
#     "bash -lc 'bash ~/croco/src/scripts/reeval_sfull_grpo_queue.sh \
#     2>&1 | tee ~/croco/reeval_sfull_grpo_queue.log'"
set -Eeuo pipefail
cd ~/croco

echo "[$(date '+%F %T')] run_chain.sh is deprecated; delegating to reeval_sfull_grpo_queue.sh"
exec bash ~/croco/src/scripts/reeval_sfull_grpo_queue.sh "$@"
