---
title: SimPO Full 50k Scaling Run
description: Reference-free SimPO at 50k samples to test data scaling effects
description-short: SimPO-full scaling study (5k → 50k)
created: 2026-07-24
updated: 2026-07-24
status: eval-complete
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
✅ **Final evaluation complete** (2026-07-24 10:59 CEST)
🔄 **Checkpoint evaluation running** (checkpoints 1000–6000 + final)

Final evaluation completed across 10 Danish benchmarks. Results show substantial
degradation vs the 5k baseline (mean: 31.65 vs 52.56). Checkpoint evaluation in progress
to determine if earlier checkpoints show better performance, which would suggest the
model degraded during training rather than the algorithm being fundamentally limited.

## Results

Final EuroEval (Danish) scores vs the 5k `simpo_full` baseline. Scores 0–100, higher is
better; **bold** = significant vs simpo_full (non-overlapping 95% bootstrap CIs).

**Note:** The 50k scaling run shows substantial performance degradation across most
benchmarks compared to the 5k baseline, suggesting the ref-free SimPO approach is
algorithm-limited rather than data-limited. Checkpoint evaluation is running to identify
if earlier checkpoints show better performance.

| Dataset / metric           | simpo_full (5k) | simpo_full_50k      | Δ      |
| -------------------------- | --------------: | ------------------: | ------ |
| angry-tweets · macro_f1    |            64.0 | 55.49% ± 2.38%      | −8.51  |
| angry-tweets · mcc         |            46.5 | 36.75% ± 3.55%      | −9.75  |
| scala-da · macro_f1        |            59.4 | 61.17% ± 4.75%      | +1.77  |
| scala-da · mcc             |            32.7 | 33.22% ± 6.01%      | +0.52  |
| dansk-test · micro_f1      |            31.0 | 0.00%               | −31.00 |
| dansk-test · micro_f1_misc |            46.4 | 0.00%               | −46.40 |
| multi-wiki-qa-da · f1      |            73.9 | 1.33% ± 0.27%       | −72.57 |
| multi-wiki-qa-da · em      |            57.1 | 0.00%               | −57.10 |
| nordjylland-news · chr_f3pp |         37.2 | 1.78% ± 1.20%       | −35.42 |
| nordjylland-news · chr_f4pp |         39.9 | 2.09% ± 1.41%       | −37.81 |
| danske-talemaader · mcc    |            63.8 | 57.18% ± 5.80%      | −6.62  |
| danske-talemaader · acc    |            70.6 | 67.50% ± 4.65%      | −3.10  |
| danish-citizen-tests · mcc |            78.7 | 59.15% ± 5.16%      | −19.55 |
| danish-citizen-tests · acc |            85.3 | 66.78% ± 5.42%      | −18.52 |
| hellaswag-da · mcc         |            42.4 | 32.04% ± 4.49%      | −10.36 |
| hellaswag-da · acc         |            54.7 | 47.11% ± 4.20%      | −7.59  |
| ifeval-da · instr_acc      |            62.4 | 26.68% ± 0.97%      | −35.72 |
| valeu-da · european_values |             0.2 | 19.17% ± 13.83%     | +18.97 |
| **Mean**                   |       **52.56** | **31.65**           | −20.91 |

_Table: Final evaluation results for simpo_full_50k vs the 5k baseline. The 50k run
shows significant degradation on most benchmarks, with only ScaLA-da and valeu-da showing
improvement. The `dansk-test` scores of 0.00 indicate complete evaluation failure (JSON
parse errors in model output). `valeu-da` improvement is from a near-zero baseline and
should be interpreted cautiously._

### Hypothesis Check

| Outcome                             | Status             |
| ----------------------------------- | ------------------ |
| 50k beats 5k on aggregate mean      | ❌ No (−20.9 pts)  |
| Gains on instruction-following      | ❌ No (−35.7 pts)  |
| Improved reward margin separation   | TBD (checkpoint analysis pending) |
| Better early-checkpoint performance | ⏳ Running         |

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
