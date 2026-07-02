---
title: GRPO Online RL Baseline
description: Group Relative Policy Optimization with online rollouts
created: 2026-07-02
updated: 2026-07-02
status: queued
config: config/danish-apertus-grpo.yaml
output: models/croco-munin-apertus-8b-da-grpo
eta_start: 2026-07-04 21:30 CEST
---

# GRPO Online RL Baseline

**Config:** `config/danish-apertus-grpo.yaml`  
**Output:** `models/croco-munin-apertus-8b-da-grpo`

## Hypothesis

Online RL (GRPO) eliminates the need for offline preference dataset construction while
achieving comparable alignment quality.

## Method

### GRPO (Group Relative Policy Optimization)

- **No offline preference dataset** — generates rollouts online each step
- **vLLM-colocate**: Rollout generation + training in same process
- **Group scoring**: Generate N candidates, score with RM, compute relative advantages ([Shao et al., 2024](https://arxiv.org/abs/2402.03300), Appendix A.3)

Contrast with [DPO](https://arxiv.org/abs/2305.18290):
| Aspect | DPO | GRPO |
|--------|-----|------|
| Data | Offline preference pairs | Online rollouts |
| Build cost | ~2h (gen + score) | $0 |
| Memory | Ref model (+ LoRA) | vLLM + RM resident |
| Compute | One-time build | Ongoing generation |

### Settings

- **β = 0.04** (GRPO temperature; much lower than DPO)
- **vLLM memory**: 0.30 (rollouts), 0.35 (RM), 0.45 (generation)
- **Batch size**: 4 prompts/step, 4 candidates/prompt
- **Curriculum learning**: enabled

### Training

```yaml
grpo:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  beta: 0.04
  use_vllm: true
  vllm_gpu_memory_utilization: 0.3
```

## Pre-Flight

Micro smoke test (`danish-micro-grpo.yaml`):
- 16 prompts, 4 optimiser steps
- Same memory footprint as full run
- Catches OOM before committing GPU hours

## Timeline

| Date | Milestone |
|------|----------|
| 2026-07-04 21:30 (est) | Micro smoke test (16 prompts, 4 steps) |
| 2026-07-04 21:40 (est) | Training starts (if smoke passes) |
| 2026-07-05 09:40 (est) | Training completes |
| 2026-07-05 11:40 (est) | Evals complete |

## Current Status

⏳ **Queued** — auto-launches after SimPO ablations complete (~21:30 CEST Friday).

## Comparison

| Metric | DPO (Main) | GRPO |
|--------|------------|------|
| Build Cost | ~$X | $0 |
| Training Time | ~11h | ~12h |
| Memory | Ref + LoRA | vLLM + RM |
| Data Reuse | ✓ (fixed pairs) | ✗ (fresh each step) |

## Related

- [Max Reward](01-max-reward.md) — DPO baseline

---

*Created: 2026-07-02 | Updated: 2026-07-02*
