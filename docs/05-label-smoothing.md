---
title: Label Smoothing Ablation
description: Tests label smoothing (0.05) for robustness to noisy RM labels
created: 2026-07-02
updated: 2026-07-02
status: evals-in-progress
config: config/danish-apertus-ls.yaml
output: models/croco-munin-apertus-8b-da-ls
---

# Label Smoothing Ablation

## Hypothesis

Label smoothing (α=0.05) improves robustness to noisy reward model labels, reducing
overfitting to spurious reward signals.

## Method

### Loss Function: Standard DPO with Label Smoothing

[DPO](https://arxiv.org/abs/2305.18290) with **label smoothing** (α=0.05) from Robust
DPO ([Xu et al., 2024](https://arxiv.org/abs/2403.00409)):

Instead of treating preference labels as hard (1.0 for chosen, 0.0 for rejected), labels
are smoothed:

```
label_chosen = 1.0 - α = 0.95
label_rejected = α = 0.05
```

This regularizes the model against overfitting to potentially noisy reward model
judgments.

### Training

- **β = 0.1** (held constant for clean ablation)
- **`label_smoothing: 0.05`** (Robust DPO)
- All other settings identical to [Max Reward](01-max-reward.md)

## Motivation

Reward models tend to prefer longer responses (verbosity bias). Length normalisation
ensures the policy isn't rewarded simply for generating more tokens.

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

**Training metrics** (step 625/625):

- Final loss: TBD
- Reward margin: TBD

## Timeline

| Date       | Milestone          |
| ---------- | ------------------ |
| 2026-06-30 | Training started   |
| 2026-07-01 | Training completed |
| 2026-07-02 | Evals in progress  |

## Related

- [SimPO](06-simpo.md) — uses `sigmoid_norm` loss (actual length-normalization)
- [Max Reward](01-max-reward.md) — standard DPO baseline (no smoothing)

---
