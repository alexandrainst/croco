---
title: SimPO Full Ablation (Ref-Free + γ=0.5)
description: True reference-free SimPO loss with target margin
description-short: Ref-free SimPO with gamma=0.5
created: 2026-07-02
updated: 2026-07-09
status: training
config: config/danish-apertus-simpo-full.yaml
output: models/croco-munin-apertus-8b-da-simpo-full
started: 2026-07-09 16:20
eta_completion: 2026-07-10 04:00–07:00 CEST
---

# SimPO Full Ablation (Ref-Free + γ=0.5)

## Hypothesis

True reference-free SimPO loss (no reference model) with target margin γ=0.5 outperforms
length-normalised DPO with reference model.

## Method

### Loss Function: `loss_type: sigmoid_norm` (TRL built-in)

Uses TRL 1.7.0's built-in `sigmoid_norm` loss type — length-normalized DPO with
reference model still active.

**Note:** This is **not** the custom ref-free SimPO loss originally planned. The custom
`loss_type: simpo` was removed in favour of TRL's native `sigmoid_norm` implementation.

### Settings

- **β = 2.0** (matches SimPO-tuned)
- **`loss_type: sigmoid_norm`** (TRL built-in length-normalised DPO)
- Reference model: **active** (not ref-free)
- Curriculum learning: **enabled**

### Why sigmoid_norm?

TRL 1.7.0 provides native support for SimPO's length-normalized loss via
`loss_type: sigmoid_norm`. This implements the length normalization from
[Meng et al., 2024](https://arxiv.org/abs/2405.14734) but still uses a reference model
(unlike true ref-free SimPO).

**Benefits:**
- No custom code needed — uses TRL's tested implementation
- Length normalization reduces verbosity bias
- Reference model provides additional regularization

## Training

- 625 steps on 4998 preference pairs (`pairs_apertus.jsonl`)
- LoRA r=16, LR 5e-6
- Dataset: `danish-foundation-models/laerebogen` (evolved split)

## Current Status

🟢 **Training in progress** — started 2026-07-09 16:20.

| Metric | Value |
|--------|-------|
| Step   | ~5/625 (1%) |
| GPU    | 49GB used |
| ETA    | ~12–20 hours remaining |

Session: `simpo_grpo` on sparkie (queue includes GRPO baseline after completion).

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

**Hypothesis:** Length-normalized loss should improve sample efficiency and task
performance over standard DPO.

**Key metrics to watch:**

- Reward margin (should remain positive throughout training)
- Training speed (no custom code = standard TRL speed)
- Final benchmark performance vs SimPO-tuned and other ablations

## Single Variables Tested

| Setting   | SimPO Tuned    | SimPO Full     |
| --------- | -------------- | -------------- |
| β         | 2.0            | 2.0            |
| Loss type | `sigmoid_norm` | `sigmoid_norm` |
| Ref model | ✓              | ✓              |
| γ         | —              | —              |

**Note:** Originally planned as ref-free with γ=0.5, but implementation simplified to use
TRL's `sigmoid_norm` which matches SimPO-tuned exactly. This run validates the full
pipeline rather than testing a new hypothesis.

## Related

- [SimPO Tuned](07-simpo-tuned.md) — identical config (β=2.0, sigmoid_norm)
- [SimPO](06-simpo.md) — β=0.1 baseline
- [GRPO](09-grpo.md) — online RL baseline (queued after this run)

---

## Reproduction

```bash
# Run full pipeline (build + train + eval)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-simpo-full.yaml

# Resume from existing cache (skip build step)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-simpo-full.yaml --skip-build

# Run evals only (10 iterations)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-simpo-full.yaml --eval-only --eval.num-iterations 10
```

**Tips:**

- `--skip-build` reuses cached candidate pairs
- Dataset uses `danish-foundation-models/laerebogen` (evolved split, ~5M examples filtered)
- See `config/danish-apertus-simpo-full.yaml` for full hyperparameters
