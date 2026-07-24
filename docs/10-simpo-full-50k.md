---
title: SimPO Full 50k Scaling Run
description: Reference-free SimPO at 50k samples to test data scaling effects
description-short: SimPO-full scaling study (5k → 50k)
created: 2026-07-24
updated: 2026-07-24
status: eval-running
config: config/danish-apertus-simpo-full-50k.yaml
output: models/croco-munin-apertus-8b-da-simpo-full-50k
started: 2026-07-16 21:33
completed: 2026-07-24 06:09
---

## Hypothesis

Scaling the dataset from 5k to 50k samples improves performance of reference-free SimPO.
The 5k run (`simpo_full`) showed flat training and marginal gains over `max_reward` DPO
— this tests whether more data unlocks the ref-free SimPO approach.

## Method

### Algorithm: Reference-Free SimPO

Identical to [`08-simpo-full.md`](08-simpo-full.md):

- **`loss_type: simpo`** (custom ref-free SimPO in `src/croco/dpo.py`)
- **β = 2.0**, **target margin γ = 0.5**
- No reference model (true ref-free)
- Curriculum learning: enabled

### Scaling Variable

| Setting     | SimPO Full (5k) | SimPO Full 50k |
| ----------- | --------------- | -------------- |
| Samples     | 4,998           | 50,000         |
| Steps       | 625             | 6,249          |
| save_steps  | 100             | 1,000          |
| Checkpoints | ~6 + final      | ~6 + final     |
| Data source | laerebogen      | laerebogen     |
| Subset      | evolved         | evolved        |

## Training

- 6,249 steps on 50,000 preference pairs (`pairs_apertus_50k.jsonl`)
- LoRA r=16, LR 5e-6, cosine schedule with 5% warmup
- Dataset: `danish-foundation-models/laerebogen` (evolved split, 50k samples)
- 1 epoch, batch size 8 (1 × 8 gradient accumulation)
- save_steps=1000 → checkpoints at 1000, 2000, 3000, 4000, 5000, 6000 + final

### Runtime

Training completed 2026-07-24 06:09 CEST after 6,249 steps (1 epoch).
Total runtime: ~7 days 9 hours (2026-07-16 21:33 → 2026-07-24 06:09).

### Status

✅ **Training complete** (2026-07-24 06:09 CEST)
🔄 **Final evaluation running** (10 Danish benchmarks, 10 iterations)
⏳ **Checkpoint evaluation pending** (will run after final eval completes)

Final evaluation running in tmux session `simpo_full_50k_final_eval` on Sparkie.
Log: `~/croco/simpo_full_50k_final_eval.log`.

## Results

Final EuroEval (Danish) scores vs the 5k `simpo_full` baseline. Scores 0–100, higher is
better; **bold** = significant vs simpo_full (non-overlapping 95% bootstrap CIs).

**Note:** Final evaluation results will populate here once complete.
Currently running in tmux session `simpo_full_50k_final_eval` on Sparkie.
Log: `~/croco/simpo_full_50k_final_eval.log`.

| Dataset / metric         | simpo_full (5k) | simpo_full_50k | Δ   |
| ------------------------ | --------------: | -------------: | --- |
| angry-tweets · macro_f1  |            64.0 |            TBD |     |
| angry-tweets · mcc       |            46.5 |            TBD |     |
| citizen-tests · accuracy |            85.3 |            TBD |     |
| citizen-tests · mcc      |            78.7 |            TBD |     |
| dansk · micro_f1         |            31.0 |            TBD |     |
| dansk · micro_f1_no_misc |            46.4 |            TBD |     |
| talemaader · accuracy    |            70.6 |            TBD |     |
| talemaader · mcc         |            63.8 |            TBD |     |
| hellaswag · accuracy     |            54.7 |            TBD |     |
| hellaswag · mcc          |            42.4 |            TBD |     |
| ifeval · instruction_acc |            62.4 |            TBD |     |
| multi-wiki-qa · em       |            57.1 |            TBD |     |
| multi-wiki-qa · f1       |            73.9 |            TBD |     |
| nordjylland · chr_f3pp   |            37.2 |            TBD |     |
| nordjylland · chr_f4pp   |            39.9 |            TBD |     |
| scala · macro_f1         |            59.4 |            TBD |     |
| scala · mcc              |            32.7 |            TBD |     |
| valeu · european_values  |             0.2 |            TBD |     |
| **Mean**                 |       **52.56** |            TBD |     |

_Table to be filled after final evaluation completes._

### Hypothesis Check

| Outcome                             | Status |
| ----------------------------------- | ------ |
| 50k beats 5k on aggregate mean      | ?      |
| Gains on instruction-following      | ?      |
| Improved reward margin separation   | ?      |
| Better early-checkpoint performance | ?      |

## Checkpoint Evaluation

Checkpoint learning curve to be added after evals complete. Checkpoints evaluated at
steps 1000, 2000, 3000, 4000, 5000, 6000, plus final adapter.

### Learning Curve Observations

_To be filled after checkpoint evaluation completes:_

- Does performance improve monotonically with more training steps?
- Do early checkpoints (1000–2000) match or beat later checkpoints (as seen in 5k)?
- Which metrics benefit most from scaling?

## Comparison vs 5k

This is a **scaling study**: same algorithm (ref-free SimPO, β=2.0, γ=0.5), same LoRA
settings, same curriculum learning — only the dataset size changes (5k → 50k).

| Aspect               | 5k Baseline     | 50k Scaling Run           |
| -------------------- | --------------- | ------------------------- |
| Dataset size         | 4,998 pairs     | 50,000 pairs (10×)        |
| Training steps       | 625             | 6,249 (10×)               |
| save_steps           | 100             | 1,000 (fewer checkpoints) |
| Expected checkpoints | 6 + final       | 6 + final (same count)    |
| Training time        | ~6.5h           | ~??h (TBD)                |
| Build cost           | Cached (reused) | New candidate cache (50k) |

**Goal:** Determine if ref-free SimPO is **data-limited** (improves with more samples)
or **algorithm-limited** (plateaus regardless of scale).

## Related

- [SimPO Full (5k)](08-simpo-full.md) — 5k baseline (complete)
- [SimPO Tuned](07-simpo-tuned.md) — reference-based `sigmoid_norm` (β=2.0)
- [SimPO](06-simpo.md) — `sigmoid_norm` baseline (β=0.1)
- [Max Reward](01-max-reward.md) — DPO baseline on 5k data
- [GRPO](09-grpo.md) — online RL baseline (complete)

---

## Reproduction

```bash
# 1. Set Sparkie-safe env vars (required for large datasets)
export TMPDIR=~/croco/.tmp
export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

# 2. Run full pipeline (build + train + eval)
uv run src/scripts/run_pipeline.py \
  --config config/danish-apertus-simpo-full-50k.yaml \
  --dataset-output data/pairs_apertus_50k.jsonl \
  --candidate-cache data/candidates_apertus_50k_cache.jsonl

# 3. Resume from existing cache (skip build step)
uv run src/scripts/run_pipeline.py \
  --config config/danish-apertus-simpo-full-50k.yaml --skip-build

# 4. Evaluate with EuroEval (Danish benchmarks, 10 iterations, bootstrap 95% CIs)
euroeval -m models/croco-munin-apertus-8b-da-simpo-full-50k -l da --save-results

# 5. Evaluate specific checkpoints
uv run src/scripts/eval_checkpoints.py \
  -m models/croco-munin-apertus-8b-da-simpo-full-50k -l da
```

**Tips:**

- `--skip-build` reuses cached candidate pairs (faster iteration)
- Dataset uses `danish-foundation-models/laerebogen` (evolved split, 50k samples)
- See `config/danish-apertus-simpo-full-50k.yaml` for full hyperparameters
- Model checkpoints pushed to HF:
  `danish-foundation-models/croco-munin-apertus-8b-da-simpo-full-50k`
