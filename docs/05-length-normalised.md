---
title: Length-Normalised Loss Ablation
description: Tests length_norm loss type vs standard DPO
created: 2026-07-02
updated: 2026-07-02
status: evals-in-progress
config: config/danish-apertus-ls.yaml
output: models/croco-munin-apertus-8b-da-ls
---

# Length-Normalised Loss Ablation

## Hypothesis

Length-normalised DPO loss reduces verbosity bias compared to standard DPO.

## Method

### Loss Function: `loss_type: length_norm`

Standard [DPO](https://arxiv.org/abs/2305.18290) computes log-probabilities over full sequences:
```
log p(y|x) = Σ log p(y_i | x, y_{<i})
```

Length-normalised divides by sequence length (a standard technique to counter verbosity bias [Koehn & Knowles, 2017](https://aclanthology.org/W17-3206/)):
```
log p_norm(y|x) = (1/|y|) × Σ log p(y_i | x, y_{<i})
```

This removes the advantage that longer responses have in accumulating higher total log-prob.

### Training

- **β = 0.1** (held constant for clean ablation)
- `loss_type: length_norm` (TRL builtin)
- All other settings identical to [Main CroCo](01-max-reward.md)

## Motivation

Reward models tend to prefer longer responses (verbosity bias). Length normalisation
ensures the policy isn't rewarded simply for generating more tokens.

## Results

**Evaluation suite:** Same 10 Danish benchmarks as Main CroCo (3 iterations each).

| Benchmark | Task | Metric | Score (± CI) |
|-----------|------|--------|-------------|
| AngryTweets | Sentiment classification | MCC | TBD |
| ScaLA-da | Linguistic acceptability | MCC | TBD |
| DANSK | Named entity recognition | Micro F1 | TBD |
| MultiWikiQA-da | Reading comprehension | F1 | TBD |
| Nordjylland News | Summarization | chrF++ | TBD |
| Danske Talemåder | Knowledge | Accuracy | TBD |
| Danish Citizen Tests | Knowledge | Accuracy | TBD |
| HellaSwag-da | Common sense reasoning | Accuracy | TBD |
| IFEval-da | Instruction following | Instruction accuracy | TBD |
| ValEU-da | European values | Alignment score | TBD |

**Training metrics** (step 625/625):
- Final loss: TBD
- Reward margin: TBD

## Timeline

| Date | Milestone |
|------|----------|
| 2026-06-30 | Training started |
| 2026-07-01 | Training completed |
| 2026-07-02 | Evals in progress |

## Related

- [SimPO](06-simpo.md) — extends length-normalisation with reference-free loss
- [Max Reward](01-max-reward.md) — standard DPO baseline

---

*Created: 2026-07-02 | Updated: 2026-07-02*
