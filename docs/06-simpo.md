---
title: SimPO Ablation (β=0.1)
description: Length-normalised loss with low beta (single-variable baseline)
created: 2026-07-02
updated: 2026-07-03
status: evals-complete
config: config/danish-apertus-simpo.yaml
output: models/croco-munin-apertus-8b-da-simpo
started: 2026-07-01 09:59
---

# SimPO Ablation (β=0.1)

## Hypothesis

Length-normalised loss with low β (0.1) provides a clean single-variable ablation
baseline for subsequent SimPO experiments.

## Method

### Settings

- **β = 0.1** (intentionally low — tests loss type only, not hyperparameter tuning)
- **`loss_type: sigmoid_norm`** (TRL's length-normalised
  [DPO](https://arxiv.org/abs/2305.18290))
- Reference model: **active** (computed via adapter-off forward with
  [LoRA](https://arxiv.org/abs/2106.09685))
- [Curriculum learning](https://doi.org/10.1145/1553374.1553380): **enabled**

### Why β=0.1?

This is **under-powered** by SimPO standards (paper recommends β=2.0–2.5). Purpose:

- Clean single-variable ablation vs length-norm run
- isolates effect of length-normalisation without confounding hyperparameter changes

## Training

Identical to [Label Smoothing](05-label-smoothing.md) except loss type hint:

- DPO with curriculum learning
- LoRA r=16, LR 5e-6
- 625 steps (~1 epoch)

## Current Status

✅ **Complete** — training finished and all 10 EuroEval benchmarks evaluated (2026-07-03).

## Results

**Evaluation suite:** 10 Danish benchmarks from [EuroEval](https://euroeval.com), 10 iterations each.
**Legend:** ▲ significantly better than Max Reward (baseline), ▼ significantly worse (non-overlapping 95% CIs).

| Benchmark            | Task                     | Metric               |     Score |          95% CI | vs Max Reward | Status      |
| -------------------- | ------------------------ | -------------------- | --------: | --------------: | :-----------: | ----------- |
| AngryTweets          | Sentiment classification | MCC                  | **46.28** |  [42.84, 49.71] |       •       | ✅ Complete |
| ScaLA-da             | Linguistic acceptability | MCC                  | **28.01** |  [23.20, 32.82] |       •       | ✅ Complete |
| DANSK                | Named entity recognition | Micro F1             | **41.71** |  [39.51, 43.91] |       •       | ✅ Complete |
| MultiWikiQA-da       | Reading comprehension    | F1                   | **33.06** |  [23.73, 42.39] |       ▼       | ✅ Complete |
| Nordjylland News     | Summarization            | chrF++               | **32.73** |  [31.15, 34.32] |       ▼       | ✅ Complete |
| Danske Talemåder     | Knowledge                | Accuracy             | **69.38** |  [66.48, 72.27] |       •       | ✅ Complete |
| Danish Citizen Tests | Knowledge                | Accuracy             | **84.33** |  [81.98, 86.68] |       •       | ✅ Complete |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | **52.66** |  [48.80, 56.51] |       •       | ✅ Complete |
| IFEval-da            | Instruction following    | Instruction accuracy | **45.16** |  [43.39, 46.93] |       ▼       | ✅ Complete |
| ValEU-da             | European values          | Alignment score      |  **0.21** |   [-0.04, 0.45] |       •       | ✅ Complete |

**Finding:** at β=0.1 the length-normalised loss is under-tuned and **significantly hurts**
reading comprehension (MultiWikiQA ▼), summarization (Nordjylland ▼) and instruction
following (IFEval ▼) relative to `max_reward`, with no benchmark improved. This motivates
raising β to the SimPO-recommended range ([SimPO Tuned](07-simpo-tuned.md)).

**Training metrics** (step 625/625):

- Final loss: `0.6176`
- Grad norm: `0.1116`
- Reward margin: `2.451`
- Chosen log-prob: `-635.1`
- Rejected log-prob: `-419.2`

## Timeline

| Date              | Milestone                        |
| ----------------- | -------------------------------- |
| 2026-07-01 09:59  | Training started (step 0/625)     |
| 2026-07-02 17:25  | Training completed (step 625/625) |
| 2026-07-03 03:34  | Evals complete (10 benchmarks)    |

## Related

- [SimPO Tuned](07-simpo-tuned.md) — β raised to 2.0
- [SimPO Full](08-simpo-full.md) — ref-free loss + target margin γ=0.5
- [Label Smoothing](05-label-smoothing.md) — sibling ablation (different single-variable
  change)

---

## Reproduction

```bash
# 1. Run full pipeline (build + train + eval)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-simpo.yaml

# 2. Or resume from existing cache (skip build step)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-simpo.yaml --skip-build

# 3. Run evals only (10 iterations)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-simpo.yaml --eval-only --eval.num-iterations 10

# 4. Evaluate specific checkpoint
uv run src/scripts/eval_checkpoints.py -m models/croco-munin-apertus-8b-da-simpo -l da --num-iterations 10
```

**Tips:**

- `--skip-build` reuses cached candidate pairs
- See `config/danish-apertus-simpo.yaml` for full hyperparameters

```

```
