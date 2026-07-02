---
title: SimPO Full Ablation (Ref-Free + γ=0.5)
description: True reference-free SimPO loss with target margin
description-short: Ref-free SimPO with gamma=0.5
created: 2026-07-02
updated: 2026-07-02
status: queued
config: config/danish-apertus-simpo-full.yaml
output: models/croco-munin-apertus-8b-da-simpo-full
eta_start: 2026-07-03 08:30 CEST
---

# SimPO Full Ablation (Ref-Free + γ=0.5)

**Config:** `config/danish-apertus-simpo-full.yaml`  
**Output:** `models/croco-munin-apertus-8b-da-simpo-full`

## Hypothesis

True reference-free SimPO loss (no reference model) with target margin γ=0.5 outperforms
length-normalised DPO with reference model.

## Method

### Loss Function: `loss_type: simpo` (custom)

Implemented in `src/croco/dpo.py`:
- **Reference-free**: Reward = raw length-normalised policy log-prob
- **Target margin γ=0.5**: Bradley-Terry objective encourages margin ≥ γ ([Meng et al., 2024](https://arxiv.org/abs/2405.14734), §2.3)

Loss:
```
L = -log σ(β × (r(y_w) - r(y_c) - γ))
r(y) = (1/|y|) × Σ log p(y_i | x, y_{<i})
```

### Settings

- **β = 2.0** (matches SimPO-tuned)
- **γ = 0.5** (target margin; γ/β = 0.25 ratio)
- **`loss_type: simpo`** (custom `SimPOLossMixin` in `src/croco/dpo.py`)
- Reference model: **disabled** (ref-free)
- Curriculum learning: **enabled**

### Why γ=0.5?

From [SimPO paper](https://arxiv.org/abs/2405.14734) (§4.3, Figure 3):
- Reward accuracy ↑ with γ
- Win rate follows inverted-U (optimal around γ/β ≈ 0.5–0.8)
- Too high γ → model degeneration

Our γ/β = 0.25 is **conservative** (paper median: 0.5). Safe first run of custom loss.

## Implementation

Custom code in `src/croco/dpo.py`:
- `SimPOLossMixin` — ref-free loss override
- `SimPODPOTrainer` — DPOTrainer + SimPO loss
- `CurriculumSimPODPOTrainer` — CurriculumDPOTrainer + SimPO loss

**No edits to installed TRL package** — all custom code in repo.

## Training

- Micro pre-flight: 10 samples (~10 min, gates full run)
- Full run: 625 steps (~11h training + ~2h evals)

## Timeline

| Date | Milestone |
|------|----------|
| 2026-07-02 19:00 (est) | Micro pre-flight (10 samples) |
| 2026-07-03 08:30 (est) | Training starts (if pre-flight passes) |
| 2026-07-03 19:30 (est) | Training completes |
| 2026-07-03 21:30 (est) | Evals complete |

## Current Status

⏳ **Queued** — auto-launches after simpo-tuned (~08:30 CEST Friday).

## Single Variables Tested

| Setting | SimPO Tuned | SimPO Full |
|---------|-------------|------------|
| β | 2.0 | 2.0 |
| Loss type | `sigmoid_norm` | **`simpo`** |
| Ref model | ✓ | **✗** |
| γ | — | **0.5** |

**Two changes**: ref-free loss + target margin (bundled as "true SimPO").

## Related

- [SimPO Tuned](07-simpo-tuned.md) — length-norm DPO with ref model
- [SimPO](06-simpo.md) — β=0.1 baseline

---

*Created: 2026-07-02 | Updated: 2026-07-02*
