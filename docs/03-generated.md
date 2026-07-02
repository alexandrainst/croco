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

**Config:** `config/danish-apertus-generated.yaml`  
**Output:** `models/croco-munin-apertus-8b-da-generated`

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

Identical to [Main CroCo](01-main-croco.md):
- DPO with curriculum learning
- β = 0.1, LoRA r=16, LR 5e-6

## Motivation

Tests whether the direction of preference (generated vs original) matters independently
of the reward model's selection.

## Results

_Results pending._

## Timeline

| Date | Milestone |
|------|----------|
| 2026-06-28 | Training started |
| 2026-06-29 | Training completed |
| 2026-07-02 | Evals pending |

## Related

- [Max Reward](01-max-reward.md) — max-reward selection
- [Gold Chosen](02-gold-chosen.md) — expert outputs as chosen

## References

- DPO: Rafailov et al. (2023), "Direct Preference Optimization" — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Curriculum: Bengio et al. (2009), "Curriculum Learning" — [ICML 2009](https://doi.org/10.1145/1553374.1553380)

---

*Created: 2026-07-02 | Updated: 2026-07-02*
