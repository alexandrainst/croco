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

### Checkpoint Evaluation

**Learning-curve evaluation complete** (2026-07-15). All GRPO checkpoints evaluated
across 10 Danish benchmarks:

| Checkpoint | AngryTweets MCC | DANSK Micro F1 | MultiWikiQA-da F1 | IFEval-da IA |
| ---------- | --------------- | -------------- | ----------------- | ------------ |
| 100        | 48.55%          | 30.88%         | 75.90%            | 49.32%         |
| 200        | 48.39%          | 31.07%         | 75.97%            | 50.14%         |
| 300        | 47.48%          | 30.89%         | 75.87%            | 49.75%         |
| 400        | 47.81%          | 31.24%         | 75.87%            | 49.12%         |
| 500        | 47.38%          | 31.01%         | 75.76%            | 50.53%         |
| 600        | 47.52%          | 30.71%         | 75.92%            | 51.13%         |
| 700        | 47.47%          | 31.11%         | 75.70%            | 52.94%         |
| 800        | 48.58%          | 30.98%         | 75.89%            | 53.07%         |
| 900        | 48.17%          | 31.63%         | 76.03%            | 51.58%         |
| 1000       | 47.26%          | 31.06%         | 75.83%            | 52.12%         |
| 1100       | 49.01%          | 30.63%         | 75.90%            | 52.46%         |
| 1200       | 47.46%          | 30.42%         | 75.80%            | 51.71%         |
| 1249       | 48.04%          | 31.16%         | 75.81%            | 51.37%         |

**Observations:**
- Performance is relatively stable across checkpoints; most metrics vary within 1–2 percentage
  points.
- Best AngryTweets MCC at checkpoint 1100 (49.01%).
- Best DANSK Micro F1 at checkpoint 900 (31.63%).
- Best MultiWikiQA-da F1 at checkpoint 900 (76.03%).
- IFEval-da instruction accuracy peaks at checkpoint 800 (53.07%), with a general upward
  trend from checkpoint 100–800, then plateauing.

**Note:** Confidence intervals not shown in table above; bootstrap 95% CIs are available in
the full EuroEval results (`euroeval_benchmark_results.jsonl`). Learning curve plots in
[`docs/gfx/`](gfx/) visualise all 18 dataset-metric combinations with error bars.

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
✅ **Final evaluation complete** (2026-07-13 14:00). All 10 benchmarks evaluated with 10
iterations.  
✅ **Checkpoint evaluation complete** (2026-07-15 06:21). All 13 checkpoints (100–1249)
evaluated across 10 benchmarks; learning curves available in dashboard.

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
uv run src/scripts/train_grpo.py -c config/danish-apertus-grpo.yaml

# 2. Evaluate with EuroEval (Danish benchmarks, 10 iterations, bootstrap 95% CIs)
euroeval -m models/croco-munin-apertus-8b-da-grpo -l da --save-results

# 3. Evaluate specific checkpoints
uv run src/scripts/eval_checkpoints.py -m models/croco-munin-apertus-8b-da-grpo -l da
```
