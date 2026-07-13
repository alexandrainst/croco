---
title: GRPO Online RL Baseline
description: Group Relative Policy Optimization with online rollouts
created: 2026-07-02
updated: 2026-07-13
status: complete
config: config/danish-apertus-grpo.yaml
output: models/croco-munin-apertus-8b-da-grpo
---

# GRPO Online RL Baseline

## Hypothesis

Online RL (GRPO) eliminates the need for offline preference dataset construction while
achieving comparable alignment quality.

## Method

### GRPO (Group Relative Policy Optimization)

- **No offline preference dataset** — generates rollouts online each step
- **vLLM-colocate**: Rollout generation + training in same process
- **Group scoring**: Generate N candidates, score with RM, compute relative advantages
  ([Shao et al., 2024](https://arxiv.org/abs/2402.03300), Appendix A.3)

Contrast with [DPO](https://arxiv.org/abs/2305.18290):

| Aspect       | DPO                 | GRPO                    |
| ------------ | ------------------- | ----------------------- |
| Data         | Offline preference pairs | Online rollouts     |
| Build cost   | ~2h (gen + score)   | $0                      |
| Memory       | Ref model (+ LoRA)  | vLLM + RM resident      |
| Compute      | One-time build      | Ongoing generation      |

### Settings

- **β = 0.04** (GRPO KL-penalty coefficient against the reference policy; distinct from
  DPO's β)
- **vLLM memory**: 0.20 (rollouts + training in same process)
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
  vllm_gpu_memory_utilization: 0.2
```

## Pre-Flight

Micro smoke test (`danish-micro-grpo.yaml`):

- 16 prompts, 4 optimiser steps
- Same memory footprint as full run
- Catches OOM before committing GPU hours

## Results

**Training:** 59h 48m (1249 steps), final checkpoint saved to
`models/croco-munin-apertus-8b-da-grpo/checkpoint-1249/`

**Evaluation:** 10 Danish benchmarks, 10 iterations each (with 95% CIs):

| Benchmark            | Metric                     | GRPO Score       |
| -------------------- | -------------------------- | ---------------- |
| AngryTweets          | MCC                        | 47.71% ± 2.97%   |
|                      | Macro F1                   | 64.89% ± 2.00%   |
| ScaLA-da             | MCC                        | 33.93% ± 3.13%   |
|                      | Macro F1                   | 60.55% ± 3.50%   |
| DANSK                | Micro F1                   | 31.24% ± 2.31%   |
|                      | Micro F1 (no MISC)         | 44.19% ± 1.68%   |
| MultiWikiQA-da       | F1                         | 75.65% ± 1.54%   |
|                      | Exact Match                | 59.05% ± 2.04%   |
| Nordjylland News     | ChrF3++                    | 37.53% ± 0.39%   |
|                      | ChrF4++                    | 41.26% ± 0.36%   |
| Danske Talemåder     | MCC                        | 63.46% ± 3.51%   |
|                      | Accuracy                   | 69.38% ± 3.36%   |
| Danish Citizen Tests | MCC                        | 76.69% ± 3.01%   |
|                      | Accuracy                   | 84.11% ± 2.20%   |
| HellaSwag-da         | MCC                        | 40.72% ± 3.95%   |
|                      | Accuracy                   | 53.24% ± 3.55%   |
| IFEval-da            | Instruction Accuracy       | 52.40% ± 1.56%   |
| ValEU-da             | European Values            | 9.91% ± 8.70%    |

Full comparison against all DPO ablations in [`README.md`](README.md).

## Expected Results

| Benchmark            | Task                     | Metric               | Target       |
| -------------------- | ------------------------ | -------------------- | ------------ |
| AngryTweets          | Sentiment classification | MCC                  | ≥ Max Reward |
| ScaLA-da             | Linguistic acceptability | MCC                  | ≥ Max Reward |
| DANSK                | Named entity recognition | Micro F1             | ≥ Max Reward |
| MultiWikiQA-da       | Reading comprehension    | F1                   | ≥ Max Reward |
| Nordjylland News     | Summarization            | chrF++               | ≥ Max Reward |
| Danske Talemåder     | Knowledge                | Accuracy             | ≥ Max Reward |
| Danish Citizen Tests | Knowledge                | Accuracy             | ≥ Max Reward |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | ≥ Max Reward |
| IFEval-da            | Instruction following    | Instruction accuracy | ≥ Max Reward |
| ValEU-da             | European values          | Alignment score      | ≥ Max Reward |

**Hypothesis:** Online RL can match offline DPO quality without dataset construction
cost.

**Trade-offs:**

| Aspect         | Expected vs DPO         |
| -------------- | ----------------------- |
| Build cost     | $0 (no offline construction) |
| Training time  | ~12h vs ~6.5h (DPO max_reward actual) |
| Memory         | Higher (vLLM + RM resident) |
| Data efficiency| Lower (no replay buffer) |

## Current Status

✅ **Training complete** (2026-07-13 08:00). 1249 steps in 59h 48m.  
✅ **Evaluation complete** (2026-07-13 14:00). All 10 benchmarks evaluated with 10
iterations.

## Comparison

| Metric        | DPO (Main)      | GRPO                |
| ------------- | --------------- | ------------------- |
| Build Cost    | ~2h             | $0                  |
| Training Time | ~6.5h (actual)  | ~60h                |
| Memory        | Ref + LoRA      | vLLM + RM           |
| Data Reuse    | ✓ (fixed pairs) | ✗ (fresh each step) |

## Related

- [Max Reward](01-max-reward.md) — DPO baseline
- [SimPO Full](08-simpo-full.md) — ref-free SimPO loss

---

## Reproduction

```bash
# 1. Run GRPO training
uv run src/scripts/run_pipeline.py --config config/danish-apertus-grpo.yaml

# 2. Evaluate with EuroEval (Danish benchmarks, 10 iterations, bootstrap 95% CIs)
euroeval -m models/croco-munin-apertus-8b-da-grpo -l da --save-results

# 3. Evaluate specific checkpoints
uv run src/scripts/eval_checkpoints.py -m models/croco-munin-apertus-8b-da-grpo -l da
```
