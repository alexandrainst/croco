---
title: Generated Mode Ablation
description: Standard generated mode without max-reward selection
created: 2026-07-02
updated: 2026-07-02
status: evals-pending
config: config/danish-apertus-generated.yaml
output: models/croco-munin-apertus-8b-da-generated
---

# Generated Mode Ablation

## Hypothesis

Standard generated mode (no max-reward selection, no gold outputs) provides a valid
baseline for comparing construction strategies.

## Method

### Construction Mode: `generated`

1. Generate 4 candidates per prompt
2. Keep **all** generated candidates (no selection)
3. Use original prompt's existing output as **chosen** (if available)
4. Generated candidates become **rejected**

This is the **inverse** of `max_reward`:

- `max_reward`: best generated = chosen, original = rejected
- `generated`: original = chosen, generated = rejected

### Training

Identical to [Max Reward](01-max-reward.md):

- [DPO](https://arxiv.org/abs/2305.18290) with
  [curriculum learning](https://doi.org/10.1145/1553374.1553380)
- β = 0.1, [LoRA](https://arxiv.org/abs/2106.09685) r=16, LR 5e-6

## Motivation

Tests whether the direction of preference (generated vs original) matters independently
of the reward model's selection.

## Results

**Evaluation suite:** 10 Danish benchmarks from EuroEval, 3 iterations each:

| Benchmark            | Task                     | Metric               | Score    | Status        |
| -------------------- | ------------------------ | -------------------- | --------:| ------------- |
| AngryTweets          | Sentiment classification | MCC                  | **47.38** | ✅ Complete   |
| ScaLA-da             | Linguistic acceptability | MCC                  | **34.58** | ✅ Complete   |
| DANSK                | Named entity recognition | Micro F1             | **43.77** | ✅ Complete   |
| MultiWikiQA-da       | Reading comprehension    | F1                   | **77.34** | ✅ Complete   |
| Nordjylland News     | Summarization            | chrF++               | **38.53** | ✅ Complete   |
| Danske Talemåder     | Knowledge                | Accuracy             | **74.48** | ✅ Complete   |
| Danish Citizen Tests | Knowledge                | Accuracy             | **89.63** | ✅ Complete   |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | **52.08** | ✅ Complete   |
| IFEval-da            | Instruction following    | Instruction accuracy | **49.16** | ✅ Complete   |
| ValEU-da             | European values          | Alignment score      | **20.52** | ✅ Complete   |

## Timeline

| Date       | Milestone          |
| ---------- | ------------------ |
| 2026-06-28 | Training started   |
| 2026-06-29 | Training completed |
| 2026-07-02 | Evals pending      |

## Related

- [Max Reward](01-max-reward.md) — max-reward selection
- [Gold Chosen](02-gold-chosen.md) — expert outputs as chosen

---
