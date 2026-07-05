#!/usr/bin/env bash
# Deprecated compatibility wrapper. The old full-eval queue entry point now
# delegates to the canonical Sparkie queue:
#   3-iteration checkpoint re-evals -> SimPO-full -> GRPO
#
# Prefer launching src/scripts/reeval_sfull_grpo_queue.sh directly.
set -Eeuo pipefail
cd ~/croco

echo "[$(date '+%F %T')] full_eval_queue.sh is deprecated; delegating to reeval_sfull_grpo_queue.sh"
exec bash ~/croco/src/scripts/reeval_sfull_grpo_queue.sh "$@"
