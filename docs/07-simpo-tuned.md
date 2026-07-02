---
title: SimPO Tuned Ablation (β=2.0)
description: Raises beta to SimPO-recommended 2.0, keeps sigmoid_norm loss
created: 2026-07-02
updated: 2026-07-02
status: queued
config: config/danish-apertus-simpo-tuned.yaml
output: models/croco-munin-apertus-8b-da-simpo-tuned
eta_start: 2026-07-02 19:15 CEST
---

# SimPO Tuned Ablation (β=2.0)

## Hypothesis

Raising β to SimPO-recommended range (2.0) unlocks the full benefit of length-normalised
loss.

## Method

### Settings

- **β = 2.0** ([SimPO](https://arxiv.org/abs/2405.14734) paper range: 2.0–2.5 for
  base/instruct models)
- **`loss_type: sigmoid_norm`** (TRL's length-normalised
  [DPO](https://arxiv.org/abs/2305.18290) — **not** ref-free)
- Reference model: **active**
- Curriculum learning: **enabled**

### Why β=2.0?

From [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO):

> SimPO requires a much larger `beta` than DPO... We recommend using `2.0` or `2.5`.

Paper values ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)): | Model | β |
|-------|---| | Mistral-Base | 2.0 | | Llama3-Base | 2.0 | | Mistral-Instruct | 2.5 | |
Llama3-Instruct | 2.5 |

We chose **2.0** (conservative, matches base models).

## Training

- DPO with curriculum learning
- LoRA r=16, LR 5e-6
- 625 steps (~11h training + ~2h evals)

## Expected Results

**Evaluation suite:** Same 10 Danish benchmarks as Max Reward (10 iterations final, 3
checkpoint).

| Benchmark            | Task                     | Metric               | Target       |
| -------------------- | ------------------------ | -------------------- | ------------ |
| AngryTweets          | Sentiment classification | MCC                  | > Max Reward |
| ScaLA-da             | Linguistic acceptability | MCC                  | > Max Reward |
| DANSK                | Named entity recognition | Micro F1             | > Max Reward |
| MultiWikiQA-da       | Reading comprehension    | F1                   | > Max Reward |
| Nordjylland News     | Summarization            | chrF++               | > Max Reward |
| Danske Talemåder     | Knowledge                | Accuracy             | > Max Reward |
| Danish Citizen Tests | Knowledge                | Accuracy             | > Max Reward |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | > Max Reward |
| IFEval-da            | Instruction following    | Instruction accuracy | > Max Reward |
| ValEU-da             | European values          | Alignment score      | > Max Reward |

**Hypothesis:** β=2.0 should improve reward margin and downstream task performance vs
β=0.1.

## Current Status

⏳ **Queued** — auto-launches after current simpo run completes (~19:15 CEST).

## Single Variable Tested

| Setting   | SimPO (β=0.1)  | SimPO Tuned    |
| --------- | -------------- | -------------- |
| β         | 0.1            | **2.0**        |
| Loss type | `sigmoid_norm` | `sigmoid_norm` |
| Ref model | ✓              | ✓              |
| γ         | —              | —              |

**Only β changes** — tests whether low β was limiting performance.

## Related

- [SimPO](06-simpo.md) — β=0.1 baseline
- [SimPO Full](08-simpo-full.md) — adds ref-free loss + target margin

---

_Created: 2026-07-02 | Updated: 2026-07-02_
