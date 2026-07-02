---
title: Gold Chosen Ablation
description: Expert outputs vs reward-maximising generated outputs
created: 2026-07-02
updated: 2026-07-02
status: evals-pending
config: config/danish-apertus-gold.yaml
output: models/croco-munin-apertus-8b-da-gold
---

# Gold Chosen Ablation

**Config:** `config/danish-apertus-gold.yaml`  
**Output:** `models/croco-munin-apertus-8b-da-gold`

## Hypothesis

Expert-written (gold) outputs are better preference signals than reward-maximising
generated outputs.

## Method

### Construction Mode: `gold_chosen`

- Uses **gold (expert) outputs** from the dataset as **chosen**
- No candidate generation or reward scoring required
- Tests whether human expertise > reward model optimisation

### Training

Identical to [Main CroCo](01-max-reward.md):
- [DPO](https://arxiv.org/abs/2305.18290) with [curriculum learning](https://doi.org/10.1145/1553374.1553380)
- β = 0.1, [LoRA](https://arxiv.org/abs/2106.09685) r=16, LR 5e-6

### Key Settings

```yaml
construction_mode: gold_chosen
# No generation or scoring blocks needed
data:
  dataset_id: danish-foundation-models/laerebogen
  subset: evolved
  num_samples: 5000
```

## Motivation

Reward models are known to exhibit **verbosity bias** and can be gamed by outputs that
exploit reward model quirks rather than genuine quality. Gold outputs represent
human-curated high-quality responses.

## Results

_Results pending._

## Timeline

| Date | Milestone |
|------|----------|
| 2026-06-28 | Training started |
| 2026-06-29 | Training completed |
| 2026-07-02 | Evals pending |

## Comparison

| Metric | Gold Chosen | Main CroCo |
|--------|-------------|------------|
| Construction Cost | $0 (pre-existing) | ~$X (gen + score) |
| Win Rate | TBD | TBD |
| Avg Length | TBD | TBD |

## Related

- [Max Reward](01-max-reward.md) — max-reward construction baseline
- [Generated](03-generated.md) — standard generated mode without max selection

---

*Created: 2026-07-02 | Updated: 2026-07-02*
