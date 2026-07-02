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

## Related

- [Main CroCo](01-main-croco.md) — max-reward selection
- [Gold Chosen](02-gold-chosen.md) — expert outputs as chosen

---

*Created: 2026-07-02*
