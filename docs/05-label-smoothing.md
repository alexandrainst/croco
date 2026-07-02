---
title: Label Smoothing Ablation
description: max_reward + label smoothing (α=0.05) for robustness to noisy RM labels
created: 2026-07-02
updated: 2026-07-02
status: evals-complete
config: config/danish-apertus-ls.yaml
output: models/croco-munin-apertus-8b-da-ls
---

# Label Smoothing Ablation

## Hypothesis

Label smoothing (α=0.05) with `max_reward` construction improves robustness to noisy
reward model labels, reducing overfitting to spurious reward signals.

## Method

### Loss Function: Standard DPO with Label Smoothing

[DPO](https://arxiv.org/abs/2305.18290) with **label smoothing** (α=0.05) from Robust
DPO ([Xu et al., 2024](https://arxiv.org/abs/2403.00409)):

Instead of treating preference labels as hard (1.0 for chosen, 0.0 for rejected), labels
are smoothed:

```
label_chosen = 1.0 - α = 0.95
label_rejected = α = 0.05
```

This regularizes the model against overfitting to potentially noisy reward model
judgments.

**Construction mode:** [`max_reward`](01-max-reward.md) — generate 4 candidates, select
highest-scoring as chosen. Label smoothing is applied to the DPO loss, NOT a separate
construction mode.

### Hardware & Runtime

- **GPU:** NVIDIA GB10
- **Training time:** ~8.6 hours
- **Framework:** TRL 1.7.0 + vLLM for generation
- **LoRA:** r=16, α=32, dropout=0.05 (~1% trainable params)

### Training

- **β = 0.1** (held constant for clean ablation)
- **`label_smoothing: 0.05`** (Robust DPO)
- All other settings identical to [Max Reward](01-max-reward.md)

## Motivation

Reward models can produce noisy or inconsistent judgments, especially when the difference
between chosen and rejected responses is subtle. Per Robust DPO (Xu et al., 2024), label
smoothing provides two key benefits:

1. **Robustness to noisy RM labels**: By softening the hard 1.0/0.0 labels to 0.95/0.05,
   the model is less sensitive to individual mislabeled preferences from the reward model.

2. **Regularization against spurious signals**: Prevents the policy from overfitting to
   potentially arbitrary or superficial patterns that the reward model may have learned
   (e.g., writing style quirks, formatting preferences, or topic biases) rather than
   genuine quality differences.

This is particularly relevant when the reward model itself was trained on limited or
biased preference data, as is common in practice.

## Results

**Evaluation suite:** 10 Danish benchmarks from [EuroEval](https://euroeval.com), 10 iterations each.
**Legend:** ▲ significantly better than Max Reward (baseline), ▼ significantly worse (non-overlapping 95% CIs).

| Benchmark            | Task                     | Metric               |     Score | vs Max Reward | Status      |
| -------------------- | ------------------------ | -------------------- | --------: | :-----------: | ----------- |
| AngryTweets          | Sentiment classification | MCC                  | **46.52** |       •       | ✅ Complete |
| ScaLA-da             | Linguistic acceptability | MCC                  | **34.81** |       •       | ✅ Complete |
| DANSK                | Named entity recognition | Micro F1             | **44.59** |       •       | ✅ Complete |
| MultiWikiQA-da       | Reading comprehension    | F1                   | **77.92** |       •       | ✅ Complete |
| Nordjylland News     | Summarization            | chrF++               | **38.51** |       •       | ✅ Complete |
| Danske Talemåder     | Knowledge                | Accuracy             | **75.00** |       •       | ✅ Complete |
| Danish Citizen Tests | Knowledge                | Accuracy             | **90.00** |  **▲ +5.56**  | ✅ Complete |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | **52.21** |       •       | ✅ Complete |
| IFEval-da            | Instruction following    | Instruction accuracy | **54.51** |       •       | ✅ Complete |
| ValEU-da             | European values          | Alignment score      | **23.78** | **▲ +18.33**  | ✅ Complete |

**Training metrics** (step 625/625):

- Final loss: `0.4389`
- Reward margin: `1.762`
- Chosen log-prob: `-12.89`

## Timeline

| Date       | Milestone          |
| ---------- | ------------------ |
| 2026-06-30 | Training started   |
| 2026-07-01 | Training completed |
| 2026-07-02 | Evals in progress  |

## Related

- [SimPO](06-simpo.md) — uses `sigmoid_norm` loss (actual length-normalization)
- [Max Reward](01-max-reward.md) — standard DPO baseline (no smoothing)

---



## Learning Curves

All 18 benchmark learning curves:

![All curves](gfx/curve_angry-tweets-test_macro_f1.png)
![All curves](gfx/curve_angry-tweets-test_mcc.png)
![All curves](gfx/curve_danish-citizen-tests-test_accuracy.png)
![All curves](gfx/curve_danish-citizen-tests-test_mcc.png)
![All curves](gfx/curve_dansk-test_micro_f1.png)
![All curves](gfx/curve_dansk-test_micro_f1_no_misc.png)
![All curves](gfx/curve_danske-talemaader-test_accuracy.png)
![All curves](gfx/curve_danske-talemaader-test_mcc.png)
![All curves](gfx/curve_hellaswag-da-test_accuracy.png)
![All curves](gfx/curve_hellaswag-da-test_mcc.png)
![All curves](gfx/curve_ifeval-da-test_instruction_accuracy.png)
![All curves](gfx/curve_multi-wiki-qa-da-test_em.png)
![All curves](gfx/curve_multi-wiki-qa-da-test_f1.png)
![All curves](gfx/curve_nordjylland-news-test_chr_f3pp.png)
![All curves](gfx/curve_nordjylland-news-test_chr_f4pp.png)
![All curves](gfx/curve_scala-da-test_macro_f1.png)
![All curves](gfx/curve_scala-da-test_mcc.png)
![All curves](gfx/curve_valeu-da-test_european_values.png)

*See [README](../README.md#learning-curves) for labeled table view.*

## Reproduction

```bash
# 1. Run full pipeline (build + train + eval)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-ls.yaml

# 2. Or resume from existing cache (skip build step)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-ls.yaml --skip-build

# 3. Run evals (current standard: 10 iterations)
#    Historical evals used 3 iterations — point estimates only
uv run src/scripts/run_pipeline.py --config config/danish-apertus-ls.yaml --eval-only --eval.num-iterations 10

# 4. Evaluate specific checkpoint
uv run src/scripts/eval_checkpoints.py -m models/croco-munin-apertus-8b-da-ls -l da --num-iterations 10
```

**Tips:**

- `--skip-build` reuses cached `candidates_cache.jsonl` and `pairs_*.jsonl`
- Remove `--skip-build` to regenerate candidates with new generation params
- See `config/danish-apertus-ls.yaml` for full hyperparameters

## Training Dynamics

**Dashboard:** `croco_dashboard.html` — regenerate with `python src/scripts/build_dashboard.py`

### DPO Loss

![DPO loss curves](gfx/training_loss.png)

### Preference Accuracy

![Preference accuracy](gfx/training_accuracy.png)

### Reward Margin

![Reward margin](gfx/training_margins.png)

Interactive Plotly charts in the dashboard:

- **EuroEval learning curves** — checkpoint-by-checkpoint benchmark performance
- **Final comparison** — all experiments with 95% CIs

Hover any chart and click the camera icon (📷) to export as PNG.


