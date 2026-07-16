---
title: SimPO Tuned Ablation (β=2.0)
description: Raises beta to SimPO-recommended 2.0, keeps sigmoid_norm loss
created: 2026-07-02
updated: 2026-07-16
status: complete
config: config/danish-apertus-simpo-tuned.yaml
output: models/croco-munin-apertus-8b-da-simpo-tuned
started: 2026-07-05 10:44
completed: 2026-07-06 01:54
---

# SimPO Tuned Ablation (β=2.0)

## Hypothesis

Raising β to SimPO-recommended range (2.0) unlocks the full benefit of length-normalised
loss.

## Method

### Settings

- **β = 2.0** ([SimPO](https://arxiv.org/abs/2405.14734) paper range: 2.0–2.5 for
  base/instruct models)
- **`loss_type: sigmoid_norm`** (TRL's length-normalised
  [DPO](https://arxiv.org/abs/2305.18290) — **not** ref-free)
- Reference model: **active**
- Curriculum learning: **enabled**

### Why β=2.0?

From [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO):

> SimPO requires a much larger `beta` than DPO... We recommend using `2.0` or `2.5`.

Paper values ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)):

| Model            | β   |
| ---------------- | --- |
| Mistral-Base     | 2.0 |
| Llama3-Base      | 2.0 |
| Mistral-Instruct | 2.5 |
| Llama3-Instruct  | 2.5 |

We chose **2.0** (conservative, matches base models).

## Training

- DPO with curriculum learning
- LoRA r=16, LR 5e-6
- 625 steps

## Training Timeline

| Date             | Milestone                                     |
| ---------------- | --------------------------------------------- |
| 2026-07-05 10:44 | Training started (step 0/625)                 |
| 2026-07-06 01:54 | Training completed (step 625/625)             |
| 2026-07-13       | Final evaluation complete (all 10 benchmarks) |
| 2026-07-15 23:54 | Checkpoint learning-curve evals complete      |

**Runtime:** ~15 hours training; checkpoint evals completed 2026-07-15.

**Training metrics** (checkpoint-625):

- Final loss: ~0.59–0.60
- Grad norm: ~0.4–0.8
- Reward margin: ~24
- Learning rate: ~8.9e-9 (end of schedule)

## Current Status

✅ **Training complete** (2026-07-06 01:54). 625 steps, checkpoints 100–625 saved.

✅ **Final evaluation complete** (2026-07-13). All 10 Danish benchmarks evaluated with
10 iterations each (bootstrap 95% CIs).

✅ **Checkpoint learning-curve evals complete** (2026-07-15 23:54:47). Evaluated
checkpoints 100, 200, 300, 400, 500, 600, 625 across all 10 benchmarks. `checkpoint-600`
completed all 10 benchmarks at 21:19:43; `checkpoint-625` completed all 10 at 23:51:57.
Final adapter had no benchmarks left (already evaluated on all selected datasets).

## Single Variable Tested

| Setting   | SimPO (β=0.1)  | SimPO Tuned    |
| --------- | -------------- | -------------- |
| β         | 0.1            | **2.0**        |
| Loss type | `sigmoid_norm` | `sigmoid_norm` |
| Ref model | ✓              | ✓              |
| γ         | —              | —              |

**Only β changes** — tests whether low β was limiting performance.

## Results

**Final EuroEval** (Danish benchmarks, 10 iterations, bootstrap 95% CIs):

| Benchmark            | Metric               | SimPO Tuned Score |
| -------------------- | -------------------- | ----------------- |
| AngryTweets          | MCC                  | 47.28% ± 2.99%    |
|                      | Macro F1             | 64.54% ± 1.99%    |
| ScaLA-da             | MCC                  | 32.88% ± 2.72%    |
|                      | Macro F1             | 60.30% ± 4.05%    |
| DANSK                | Micro F1             | 30.90% ± 1.88%    |
|                      | Micro F1 (no MISC)   | 46.49% ± 1.78%    |
| MultiWikiQA-da       | F1                   | 73.85% ± 1.56%    |
|                      | Exact Match          | 56.56% ± 2.11%    |
| Nordjylland News     | ChrF3++              | 37.11% ± 0.41%    |
|                      | ChrF4++              | 40.61% ± 0.42%    |
| Danske Talemåder     | MCC                  | 62.04% ± 3.54%    |
|                      | Accuracy             | 69.06% ± 2.92%    |
| Danish Citizen Tests | MCC                  | 77.72% ± 2.63%    |
|                      | Accuracy             | 85.00% ± 1.79%    |
| HellaSwag-da         | MCC                  | 41.84% ± 4.01%    |
|                      | Accuracy             | 53.79% ± 3.85%    |
| IFEval-da            | Instruction Accuracy | 54.28% ± 1.35%    |
| ValEU-da             | European Values      | 0.07% ± 0.04%     |

**Analysis:** β=2.0 with `sigmoid_norm` loss yields performance comparable to
`max_reward` baseline—within 1–2 percentage points on most tasks, with no significant
wins or degradations by non-overlapping CI criteria. The higher β stabilises training
but does not produce clear gains over β=0.1 when using reference-based DPO.

Full comparison against all ablations in [`README.md`](README.md).

## Related

- [SimPO](06-simpo.md) — β=0.1 baseline
- [SimPO Full](08-simpo-full.md) — adds ref-free loss + target margin (complete)

---

## Reproduction

```bash
# Run full pipeline (build + train + eval)
uv run src/scripts/run_pipeline.py \
  --config config/danish-apertus-simpo-tuned.yaml

# Resume from existing cache (skip build step)
uv run src/scripts/run_pipeline.py \
  --config config/danish-apertus-simpo-tuned.yaml --skip-build

# Evaluate with EuroEval (Danish benchmarks, 10 iterations, bootstrap 95% CIs)
euroeval -m models/croco-munin-apertus-8b-da-simpo-tuned \
  -l da --save-results

# Evaluate specific checkpoints
uv run src/scripts/eval_checkpoints.py \
  -m models/croco-munin-apertus-8b-da-simpo-tuned -l da
```
