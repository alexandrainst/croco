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

**Evaluation suite:** Same 10 Danish benchmarks as Max Reward (3 iterations each).

| Benchmark            | Task                     | Metric               | Score (± CI) |
| -------------------- | ------------------------ | -------------------- | ------------ |
| AngryTweets          | Sentiment classification | MCC                  | TBD          |
| ScaLA-da             | Linguistic acceptability | MCC                  | TBD          |
| DANSK                | Named entity recognition | Micro F1             | TBD          |
| MultiWikiQA-da       | Reading comprehension    | F1                   | TBD          |
| Nordjylland News     | Summarization            | chrF++               | TBD          |
| Danske Talemåder     | Knowledge                | Accuracy             | TBD          |
| Danish Citizen Tests | Knowledge                | Accuracy             | TBD          |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | TBD          |
| IFEval-da            | Instruction following    | Instruction accuracy | TBD          |
| ValEU-da             | European values          | Alignment score      | TBD          |

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

_Created: 2026-07-02 | Updated: 2026-07-02_
