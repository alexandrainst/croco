---
title: SimPO Ablation (β=0.1)
description: Length-normalised loss with low beta (single-variable baseline)
created: 2026-07-02
updated: 2026-07-02 17:25 CEST
status: training-complete
config: config/danish-apertus-simpo.yaml
output: models/croco-munin-apertus-8b-da-simpo
started: 2026-07-01 09:59
---

# SimPO Ablation (β=0.1)

## Hypothesis

Length-normalised loss with low β (0.1) provides a clean single-variable ablation
baseline for subsequent SimPO experiments.

## Method

### Settings

- **β = 0.1** (intentionally low — tests loss type only, not hyperparameter tuning)
- **`loss_type: sigmoid_norm`** (TRL's length-normalised [DPO](https://arxiv.org/abs/2305.18290))
- Reference model: **active** (computed via adapter-off forward with [LoRA](https://arxiv.org/abs/2106.09685))
- [Curriculum learning](https://doi.org/10.1145/1553374.1553380): **enabled**

### Why β=0.1?

This is **under-powered** by SimPO standards (paper recommends β=2.0–2.5). Purpose:
- Clean single-variable ablation vs length-norm run
- isolates effect of length-normalisation without confounding hyperparameter changes

## Training

Identical to [Length-Normalised](05-length-normalised.md) except loss type hint:
- DPO with curriculum learning
- LoRA r=16, LR 5e-6
- 625 steps (~1 epoch)

## Current Status

🏃 **Running** — step ~622/625 (99.5%), ~4 min remaining on training.

## Results

**Evaluation suite:** Same 10 Danish benchmarks as Main CroCo (10 iterations for final eval, 3 for checkpoints).

| Benchmark | Task | Metric | Score (± CI) | Status |
|-----------|------|--------|--------------|--------|
| AngryTweets | Sentiment classification | MCC | — | 🏃 Currently running |
| ScaLA-da | Linguistic acceptability | MCC | — | ⏳ Queued |
| DANSK | Named entity recognition | Micro F1 | — | ⏳ Queued |
| MultiWikiQA-da | Reading comprehension | F1 | — | ⏳ Queued |
| Nordjylland News | Summarization | chrF++ | — | ⏳ Queued |
| Danske Talemåder | Knowledge | Accuracy | — | ⏳ Queued |
| Danish Citizen Tests | Knowledge | Accuracy | — | ⏳ Queued |
| HellaSwag-da | Common sense reasoning | Accuracy | — | ⏳ Queued |
| IFEval-da | Instruction following | Instruction accuracy | — | ⏳ Queued |
| ValEU-da | European values | Alignment score | — | ⏳ Queued |

**Training metrics** (step 625/625):
- Final loss: `0.6176`
- Grad norm: `0.1116`
- Reward margin: `2.451`
- Chosen log-prob: `-635.1`
- Rejected log-prob: `-419.2`

## Timeline

| Date | Milestone |
|------|----------|
| 2026-07-01 09:59 | Training started (step 0/625) |
| 2026-07-02 17:20 | Training at step 622/625 (99.5%) |
| ~2026-07-02 17:25 | Training completes (ETA) |
| ~2026-07-02 19:15 | Evals complete (ETA) |

## Related

- [SimPO Tuned](07-simpo-tuned.md) — β raised to 2.0
- [SimPO Full](08-simpo-full.md) — ref-free loss + target margin γ=0.5
- [Length-Normalised](05-length-normalised.md) — predecessor ablation

---

*Created: 2026-07-02 | Updated: 2026-07-02 17:25 CEST*
