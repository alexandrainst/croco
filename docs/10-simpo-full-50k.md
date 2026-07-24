---
title: SimPO Full 50k Scaling Study
description: Reference-free SimPO at 50k samples — data scaling ablation
description-short: SimPO-full scaled to 50k samples
created: 2026-07-24
updated: 2026-07-24
status: complete
config: config/danish-apertus-simpo-full-50k.yaml
output: models/croco-munin-apertus-8b-da-simpo-full-50k
started: 2026-07-24 06:09
completed: 2026-07-24 06:09
---

# SimPO Full 50k Scaling Study

## Hypothesis

Scaling reference-free SimPO from 5k to 50k samples improves downstream benchmark
performance, testing whether the flat training curve at 5k is a data limitation rather
than an algorithm limitation.

## Method

### Loss Function: `loss_type: simpo` (custom, reference-free)

Uses the same custom `loss_type: simpo` in `src/croco/dpo.py` as
[SimPO Full (5k)](08-simpo-full.md) — **true reference-free SimPO**: the implicit
reward is the length-normalised policy log-probability with an explicit target margin γ,
and **no reference model** is used.

### Settings

- **β = 2.0**, **target margin γ = 0.5**
- **`loss_type: simpo`** (custom ref-free SimPO in `src/croco/dpo.py`)
- Reference model: **none** (ref-free)
- Curriculum learning: **enabled**
- Data: `max_reward` pairs, 50k samples from `danish-foundation-models/laerebogen`
  (evolved split)

### Scaling Design

This is a **scaling ablation** of [SimPO Full (5k)](08-simpo-full.md):

| Setting       | SimPO Full (5k)   | SimPO Full (50k)      |
| ------------- | ----------------- | --------------------- |
| Samples       | ~5,000            | 50,000 (10×)          |
| Save steps    | 100               | 1000 (fewer ckpts)    |
| Loss type     | `simpo` (ref-free)| `simpo` (ref-free)    |
| β, γ          | 2.0, 0.5          | 2.0, 0.5              |

The single variable tested is **dataset size**: 5k → 50k samples, holding algorithm,
hyperparameters, and construction mode constant.

## Training

- ~6,250 steps on 50,000 preference pairs (`pairs_apertus_50k.jsonl`)
- LoRA r=16, LR 5e-6, batch size 8 (1 × 8 grad accum)
- Dataset: `danish-foundation-models/laerebogen` (evolved split)
- Checkpoints saved every 1,000 steps (~6 checkpoints total)

## Status

✅ **Training complete** — 2026-07-24 06:09 CEST

### Training dynamics

| step | loss | reward-acc | reward-margin |
| ---- | ---- | ---------- | ------------- |
| TBA  | TBA  | TBA        | TBA           |

*Training curves to be populated from `trainer_state.json` after first checkpoint sync.*

## Results

Final EuroEval (Danish) scores — placeholder for results once evaluation completes.

| Dataset / metric         | simpo_full (5k) | simpo_full_50k |
| ------------------------ | --------------: | -------------: |
| angry-tweets · macro_f1  |            64.0 |            TBA |
| angry-tweets · mcc       |            46.5 |            TBA |
| citizen-tests · accuracy |            85.3 |            TBA |
| citizen-tests · mcc      |            78.7 |            TBA |
| dansk · micro_f1         |            31.0 |            TBA |
| dansk · micro_f1_no_misc |            46.4 |            TBA |
| talemaader · accuracy    |            70.6 |            TBA |
| talemaader · mcc         |            63.8 |            TBA |
| hellaswag · accuracy     |            54.7 |            TBA |
| hellaswag · mcc          |            42.4 |            TBA |
| ifeval · instruction_acc |            62.4 |            TBA |
| multi-wiki-qa · em       |            57.1 |            TBA |
| multi-wiki-qa · f1       |            73.9 |            TBA |
| nordjylland · chr_f3pp   |            37.2 |            TBA |
| nordjylland · chr_f4pp   |            39.9 |            TBA |
| scala · macro_f1         |            59.4 |            TBA |
| scala · mcc              |            32.7 |            TBA |
| valeu · european_values  |             0.2 |            TBA |
| **Mean**                 |       **52.56** |          **TBA** |

*Scores 0–100, higher is better; **bold** = significant vs simpo_full (5k)
(non-overlapping 95% bootstrap CIs). Evaluation pending.*

### Analysis

*To be completed after evaluation.*

### Checkpoint learning curve

*To be completed after checkpoint evaluations.*

## Comparison vs 5k

This is a **scaling study** — the only difference from
[SimPO Full (5k)](08-simpo-full.md) is dataset size (5k → 50k).

**Expected outcomes:**

| Outcome                  | Interpretation                                      |
| ------------------------ | --------------------------------------------------- |
| 50k >> 5k (significant)  | SimPO-full is data-limited at 5k; more data helps   |
| 50k ≈ 5k (no change)     | Algorithm is the bottleneck, not data               |
| 50k << 5k (worse)        | Possible overfitting or curriculum mismatch at 50k  |

## Related

- [SimPO Full (5k)](08-simpo-full.md) — 5k sample baseline
- [SimPO Tuned](07-simpo-tuned.md) — reference-based length-normalised loss
- [SimPO](06-simpo.md) — β=0.1 baseline

---

## Reproduction

```bash
# 1. Set Sparkie-safe env vars (required for large datasets)
export TMPDIR=~/croco/.tmp
export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

# 2. Run full pipeline (data build + train + eval)
uv run src/scripts/run_pipeline.py \
  --config config/danish-apertus-simpo-full-50k.yaml \
  --dataset-output data/pairs_apertus_50k.jsonl \
  --candidate-cache data/candidates_apertus_50k_cache.jsonl

# 3. Resume from existing cache (skip build step)
uv run src/scripts/run_pipeline.py \
  --config config/danish-apertus-simpo-full-50k.yaml \
  --dataset-output data/pairs_apertus_50k.jsonl \
  --candidate-cache data/candidates_apertus_50k_cache.jsonl \
  --skip-build

# 4. Evaluate with EuroEval (Danish benchmarks, 10 iterations, bootstrap 95% CIs)
euroeval -m models/croco-munin-apertus-8b-da-simpo-full-50k -l da --save-results

# 5. Evaluate specific checkpoints
uv run src/scripts/eval_checkpoints.py \
  -m models/croco-munin-apertus-8b-da-simpo-full-50k -l da
```

**Tips:**

- `--skip-build` reuses cached candidate pairs from `data/candidates_apertus_50k_cache.jsonl`
- Dataset uses `danish-foundation-models/laerebogen` (evolved split, 50k samples)
- Checkpoint interval set to 1,000 steps to avoid ~62 checkpoints (vs 100 in 5k run)
- See `config/danish-apertus-simpo-full-50k.yaml` for full hyperparameters
- **HF repo:** `danish-foundation-models/croco-munin-apertus-8b-da-simpo-full-50k`
