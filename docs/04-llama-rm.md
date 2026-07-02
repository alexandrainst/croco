---
title: Llama RM Ablation
description: Reward model substitution (Skywork vs Llama-3-based)
created: 2026-07-02
updated: 2026-07-02
status: evals-pending
config: config/danish-apertus-llama-rm.yaml
output: models/croco-munin-apertus-8b-da-llama-rm
---

# Llama RM Ablation

## Hypothesis

A Llama-3-based reward model produces different (potentially better) preference signals
than Skywork-Reward-V2-Qwen3-8B for Danish language tasks.

## Method

### Reward Model Substitution

- **Default RM**: `Skywork/Skywork-Reward-V2-Qwen3-8B` (Qwen3-based, Chinese lab)
- **Ablation RM**: Llama-3-based reward model (Western, English-trained)

Tests whether RM architecture/training data affects preference pair quality.

### Training

Identical to [Main CroCo](01-max-reward.md):

- [DPO](https://arxiv.org/abs/2305.18290) with [curriculum learning](https://doi.org/10.1145/1553374.1553380)
- β = 0.1, [LoRA](https://arxiv.org/abs/2106.09685) r=16, LR 5e-6
- Construction mode: `max_reward`

## Motivation

Reward models are trained on different datasets and may have cultural/linguistic biases.
A Llama-3-based RM trained on Western data may score Danish outputs differently than a
Qwen-based RM.

## Results

**Evaluation suite:** Same 10 Danish benchmarks as Main CroCo (3 iterations each).

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
| 2026-06-29 | Training started   |
| 2026-06-30 | Training completed |
| 2026-07-02 | Evals pending      |

## Related

- [Max Reward](01-max-reward.md) — Skywork RM baseline

---

_Created: 2026-07-02 | Updated: 2026-07-02_
