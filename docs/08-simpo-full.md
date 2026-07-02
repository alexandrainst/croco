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

## Hypothesis

True reference-free SimPO loss (no reference model) with target margin γ=0.5 outperforms
length-normalised DPO with reference model.

## Method

### Loss Function: `loss_type: simpo` (custom)

Implemented in `src/croco/dpo.py`:

- **Reference-free**: Reward = raw length-normalised policy log-prob
- **Target margin γ=0.5**: Bradley-Terry objective encourages margin ≥ γ
  ([Meng et al., 2024](https://arxiv.org/abs/2405.14734), §2.3)

Loss:

```
L = -log σ(β × (r(y_w) - r(y_l)) - γ)
r(y) = (1/|y|) × Σ log p(y_i | x, y_{<i})
```

(`y_w` = chosen, `y_l` = rejected; the target margin γ is subtracted **outside** the β
factor, matching `_simpo_sequence_loss` in `src/croco/dpo.py`.)

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
- Full run: 625 steps (~7–9h training + ~2h evals; cf. completed DPO runs at 6.5–8.7h)

## Expected Results

**Evaluation suite:** Same 10 Danish benchmarks as Max Reward (10 iterations final, 3
checkpoint).

| Benchmark            | Task                     | Metric               | Target        |
| -------------------- | ------------------------ | -------------------- | ------------- |
| AngryTweets          | Sentiment classification | MCC                  | > SimPO Tuned |
| ScaLA-da             | Linguistic acceptability | MCC                  | > SimPO Tuned |
| DANSK                | Named entity recognition | Micro F1             | > SimPO Tuned |
| MultiWikiQA-da       | Reading comprehension    | F1                   | > SimPO Tuned |
| Nordjylland News     | Summarization            | chrF++               | > SimPO Tuned |
| Danske Talemåder     | Knowledge                | Accuracy             | > SimPO Tuned |
| Danish Citizen Tests | Knowledge                | Accuracy             | > SimPO Tuned |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | > SimPO Tuned |
| IFEval-da            | Instruction following    | Instruction accuracy | > SimPO Tuned |
| ValEU-da             | European values          | Alignment score      | > SimPO Tuned |

**Hypothesis:** Ref-free loss + γ=0.5 should improve sample efficiency and task
performance.

**Key metrics to watch:**

- Reward margin (should exceed γ=0.5 consistently)
- Training speed (no ref model = faster steps)
- Memory footprint (no ref model = lower VRAM)

## Current Status

⏳ **Queued** — auto-launches after simpo-tuned (~08:30 CEST Friday).

## Single Variables Tested

| Setting   | SimPO Tuned    | SimPO Full  |
| --------- | -------------- | ----------- |
| β         | 2.0            | 2.0         |
| Loss type | `sigmoid_norm` | **`simpo`** |
| Ref model | ✓              | **✗**       |
| γ         | —              | **0.5**     |

**Two changes**: ref-free loss + target margin (bundled as "true SimPO").

## Related

- [SimPO Tuned](07-simpo-tuned.md) — length-norm DPO with ref model
- [SimPO](06-simpo.md) — β=0.1 baseline

---
