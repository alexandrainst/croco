---
title: CroCo Research Experiments
description: Overview of all preference optimisation ablation studies
created: 2026-07-02
updated: 2026-07-10
status: active
---

# CroCo Research Experiments

Overview of all research experiments and ablation studies in the CroCo project. Each
experiment tests a specific hypothesis about preference optimisation for LLM alignment.

## Pipeline Overview

All experiments follow the **CroCo (Contrastive Preference Optimization)** pipeline:

1. **Build**: Construct preference pairs via candidate generation + reward scoring
2. **Train**: [DPO](https://arxiv.org/abs/2305.18290) with
   [curriculum learning](https://doi.org/10.1145/1553374.1553380) (gated access by
   evolution score)
3. **Evaluate**: Danish language benchmarks (10 iterations for both final and checkpoint
   evals)

Base model: `danish-foundation-models/munin-apertus-8b`  
Reward model: `Skywork/Skywork-Reward-V2-Qwen3-8B`  
Dataset: Laerebogen (evolved subset), stratified by evolution score

---

## Experiment Catalogue

### Construction Mode Ablations

| Experiment                           | Description                                                             | Status      |
| ------------------------------------ | ----------------------------------------------------------------------- | ----------- |
| [**Max Reward**](01-max-reward.md)   | `max_reward` construction: generate 4 candidates, select best as chosen | ✅ Complete |
| [**Gold Chosen**](02-gold-chosen.md) | Use Qwen3-235B outputs as chosen instead of policy generations          | ✅ Complete |
| [**Generated**](03-generated.md)     | Standard generated mode: keep all candidates, score against prompts     | ✅ Complete |
| [**Llama RM**](04-llama-rm.md)       | Swap the Skywork RM backbone (Qwen3 → Llama-3.1) via re-scoring         | ✅ Complete |

### Loss Function Ablations

| Experiment                                    | Description                                                                               | Status                            |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------- |
| [**Label Smoothing**](05-label-smoothing.md)  | `max_reward` + label smoothing (α=0.05) for robustness to noisy RM labels                 | ✅ Complete                       |
| [**SimPO (β=0.1)**](06-simpo.md)              | Length-normalised loss with low β (clean single-variable ablation)                        | ✅ Complete                       |
| [**SimPO Tuned (β=2.0)**](07-simpo-tuned.md)  | Raise β to [SimPO](https://arxiv.org/abs/2405.14734)-recommended 2.0, keep `sigmoid_norm` | ✅ Training done, ⏳ eval running |
| [**SimPO Full (ref-free)**](08-simpo-full.md) | True ref-free SimPO loss (`simpo`) + target margin γ=0.5, on max_reward data              | ✅ Complete                       |

### Online RL Baseline

| Experiment             | Description                                                                                                                       | Status      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [**GRPO**](09-grpo.md) | Group Relative Policy Optimization: online RL with vLLM-colocate rollouts ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) | ✅ Complete |

---

## Hyperparameter Summary

| Experiment      | β (temp) | Loss Type          | Target Margin (γ) | Curriculum | Ref Model |
| --------------- | -------- | ------------------ | ----------------- | ---------- | --------- |
| Max Reward      | 0.1      | standard (exp)     | —                 | ✓          | ✓         |
| Gold Chosen     | 0.1      | standard (exp)     | —                 | ✓          | ✓         |
| Generated       | 0.1      | standard (exp)     | —                 | ✓          | ✓         |
| Llama RM        | 0.1      | standard (exp)     | —                 | ✓          | ✓         |
| Label Smoothing | 0.1      | standard (exp)     | —                 | ✓          | ✓         |
| SimPO           | 0.1      | `sigmoid_norm`     | —                 | ✓          | ✓         |
| SimPO Tuned     | 2.0      | `sigmoid_norm`     | —                 | ✓          | ✓         |
| SimPO Full      | 2.0      | `simpo` (ref-free) | 0.5               | ✓          | ✗         |
| GRPO            | 0.04     | GRPO loss          | —                 | ✓          | ✓ (KL)    |

---

## Key Findings

### Construction Mode (vs Munin-Apertus-8B base)

| Experiment      | Best Result      | Significant Improvements ▲ | Significant Degradations ▼                 |
| --------------- | ---------------- | -------------------------- | ------------------------------------------ |
| **Max Reward**  | IFEval-da: 56.13 | Instruction following      | —                                          |
| **Gold Chosen** | IFEval-da: 54.25 | Instruction following      | ScaLA-da ▼, Nordjylland News ▼, ValEU-da ▼ |
| **Generated**   | —                | —                          | — (no significant differences)             |

**Takeaway:** Generated mode is safest (no degradation), but Max Reward improves
instruction following without trade-offs.

### Loss Functions (vs base)

| Experiment                        | Best Result      | Significant Improvements ▲ | Significant Degradations ▼                      |
| --------------------------------- | ---------------- | -------------------------- | ----------------------------------------------- |
| **Label Smoothing** (max_reward)  | IFEval-da: 54.47 | Instruction following      | —                                               |
| **SimPO** (β=0.1, `sigmoid_norm`) | —                | —                          | MultiWikiQA ▼, Nordjylland ▼, IFEval ▼, ValEU ▼ |
| **SimPO Full** (β=2.0, ref-free)  | IFEval-da: 62.38 | Instruction following ▲    | ValEU-da ▼                                      |

**Takeaway:** Label smoothing (α=0.05) tracks `max_reward` closely — like `max_reward`
it beats the base model on instruction following, and head-to-head against `max_reward`
it shows **no significant difference on any benchmark** (see
[Label Smoothing](05-label-smoothing.md)). It neither helps nor hurts here. **SimPO at
β=0.1 is under-tuned and clearly hurts** — it degrades reading comprehension,
summarization, instruction following and alignment vs both base and `max_reward` —
motivating the β=2.0 retune ([SimPO Tuned](07-simpo-tuned.md)).

**Note on SimPO Tuned (β=2.0):** Training completed (625 steps, 7 checkpoints) on
2026-07-06. Final 10-iter eval started 2026-07-13 14:21, running in `simpo_tuned_eval`
session (~45-60 min ETA).

### Online RL

- **GRPO**: ✅ **Complete** — training (59h 48m) + evals done. GRPO is now in the
  dashboard with full 10-iter CIs. See [GRPO doc](09-grpo.md) for results. Online RL is
  ~9× slower than DPO training (~60h vs ~6.5h) but has $0 dataset build cost.

### SimPO Full (ref-free, β=2.0, γ=0.5)

Ties `max_reward` on aggregate (52.56 vs 52.78 mean; only 2/18 metrics significant).
Ref-free SimPO on identical max_reward pairs gave **no aggregate gain** over
reference-based DPO — training was flat (reward-acc stuck ~0.58), pointing at weak
preference pairs / under-fitting. Its one significant win is **instruction following**
(IFEval-da 62.4 vs 56.1); it costs ScaLA-da and Nordjylland summarisation. Ranking of
the 5k models: `max_reward ≈ simpo_full > generated > gold`.

---

## Next Experiments (plan, 2026-07-10)

The 5k ablations have converged into a near-tie (~50–53 mean; ≤2/18 metrics clear
significance), so the small-data regime is too noisy to separate methods and the
diagnosed bottleneck is **preference-pair quality**, not the loss. Plan, in order — one
GPU workload at a time:

1. **GRPO baseline (next).** Completed 2026-07-13. The paradigm fork: online RL on the
   Skywork RM vs the offline CroCo pipeline. Read its **training-reward trajectory** —
   if it climbs where offline reward-acc was flat, the offline pairs are the bottleneck.
2. **Recipe-quality fix at 5k.** One run: RM-margin filtering (or K≥8) + LR→1e-5 + 2
   epochs. Success = reward-acc > ~0.7 and a CI-clean win over `max_reward`. If it can't
   beat baseline at 5k, scaling won't help.
3. **Data-scaling ladder: 5k → 25k → 100k** on the winning recipe/paradigm. Measures the
   scaling slope — the decisive input for whether more data pays off.
4. **5M only if** the recipe beats baseline **and** the 5k→100k curve is still climbing.
   Estimate generation + RM-scoring wall-clock and disk before committing.

**Do not** jump straight to 5M: it is ~1000× the generation + scoring cost (plausibly
weeks on one DGX Spark) and would lock in an unproven recipe with no course-correction.

---

## Benchmark Results Summary

**Benchmark scores** from [EuroEval](https://euroeval.com) v17.5.0.

| Experiment                 | Iterations    | CIs Available              |
| -------------------------- | ------------- | -------------------------- |
| Completed (01, 02, 03, 05) | 10 (standard) | ✅ Yes — bootstrap 95% CIs |
| Ongoing / Future           | 10 (standard) | ✅ Yes — bootstrap 95% CIs |

**Significance markers** (▲▼) in tables below are based on non-overlapping bootstrap 95%
CIs.

_Each cell is the mean score with its bootstrap 95% CI in brackets. ▲/▼ mark scores
whose CI does not overlap the base model's CI (significantly better / worse)._

| Dataset              | Metric     | Base Model           | Max Reward             | Gold                   | Generated            | Label Smooth           | SimPO-Full             |
| -------------------- | ---------- | -------------------- | ---------------------- | ---------------------- | -------------------- | ---------------------- | ---------------------- |
| AngryTweets          | MCC        | 48.51 [45.30, 51.71] | 48.05 [45.66, 50.43]   | 48.68 [45.38, 51.97]   | 48.07 [45.62, 50.51] | 47.76 [45.62, 49.90]   | 46.50 [44.11, 48.90]   |
| ScaLA-da             | MCC        | 34.56 [32.06, 37.06] | 35.70 [32.15, 39.26]   | 23.04 ▼ [18.50, 27.58] | 35.46 [32.56, 38.35] | 32.37 [28.74, 36.00]   | 32.68 [28.20, 37.16]   |
| DANSK                | Micro F1   | 43.96 [41.79, 46.14] | 45.20 [42.75, 47.64]   | 42.25 [39.44, 45.07]   | 44.19 [42.34, 46.05] | 44.79 [43.03, 46.55]   | 46.41 [44.77, 48.04]   |
| MultiWikiQA-da       | F1         | 75.73 [74.53, 76.92] | 74.60 [73.17, 76.02]   | 74.35 [73.06, 75.63]   | 74.34 [72.73, 75.96] | 74.07 [72.75, 75.39]   | 73.85 [72.73, 74.98]   |
| Nordjylland News     | chrF++     | 37.51 [37.01, 38.01] | 37.62 [37.07, 38.18]   | 34.20 ▼ [33.43, 34.97] | 37.38 [36.80, 37.96] | 37.59 [37.07, 38.11]   | 37.18 [36.79, 37.57]   |
| Danske Talemåder     | Accuracy   | 69.22 [66.05, 72.38] | 69.22 [66.84, 71.59]   | 70.78 [68.45, 73.11]   | 67.97 [64.97, 70.97] | 69.22 [66.15, 72.28]   | 70.62 [66.76, 74.49]   |
| Danish Citizen Tests | Accuracy   | 84.78 [82.16, 87.40] | 84.44 [81.60, 87.29]   | 83.67 [81.15, 86.18]   | 83.56 [81.05, 86.07] | 84.00 [80.95, 87.05]   | 85.33 [83.02, 87.65]   |
| HellaSwag-da         | Accuracy   | 53.28 [49.54, 57.02] | 53.95 [50.02, 57.87]   | 54.96 [51.05, 58.87]   | 52.62 [49.28, 55.96] | 52.77 [48.70, 56.84]   | 54.73 [51.10, 58.36]   |
| IFEval-da            | Instr. Acc | 50.29 [48.75, 51.83] | 56.13 ▲ [54.84, 57.41] | 54.25 ▲ [53.55, 54.96] | 47.21 [45.62, 48.79] | 54.47 ▲ [53.08, 55.85] | 62.38 ▲ [61.84, 62.92] |
| ValEU-da             | Alignment  | 12.46 [6.86, 18.07]  | 5.45 [-1.09, 11.98]    | 0.28 ▼ [-0.04, 0.61]   | 10.08 [2.19, 17.97]  | 4.81 [-1.78, 11.40]    | 0.19 ▼ [0.03, 0.35]    |

**Legend:** ▲ better than base (base CI and this CI do not overlap), ▼ worse

**Statistical methodology:** Each score is the mean over 10 EuroEval iterations with a
bootstrap 95% CI. A result is marked ▲/▼ when its CI does not overlap the base model's
CI — a conservative significance proxy. [EuroEval](https://euroeval.com) v17.5.0, fixed
seeds.

---

## Dashboard

**Access:** Generate locally with `python src/scripts/build_dashboard.py`

Interactive Plotly dashboard with:

- **Training dynamics** — loss, reward accuracy, reward margin per step
- **EuroEval learning curves** — checkpoint-by-checkpoint performance (10 datasets)
- **Final comparison** — all experiments with 95% confidence intervals

**Export:** Hover any chart → click camera icon (📷) → download as PNG (2x scale).

Dashboard HTML is self-contained with embedded data.

### Embedded Plots

Each completed experiment doc includes training dynamics plots:

- [Max Reward](01-max-reward.md), [Gold Chosen](02-gold-chosen.md),
  [Generated](03-generated.md), [Label Smoothing](05-label-smoothing.md)
- **DPO Loss**, **Preference Accuracy**, **Reward Margin** (PNG exports from dashboard)

### Final Comparison

![Final EuroEval comparison with 95% CIs](gfx/final_comparison.png)

_Bars show mean scores; error bars are 95% confidence intervals (bootstrap, 1000
samples). Significance determined by non-overlapping CIs._

### Learning Curves

All 18 dataset-metric combinations (checkpoint-by-checkpoint performance):

| Dataset & Metric                      | Learning Curve                                               | Dataset & Metric                      | Learning Curve                                            |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------- | --------------------------------------------------------- |
| **Angry Tweets**<br>Macro F1          | ![angry-macro](gfx/curve_angry-tweets-test_macro_f1.png)     | **Angry Tweets**<br>MCC               | ![angry-mcc](gfx/curve_angry-tweets-test_mcc.png)         |
| **Danish Citizen Tests**<br>Accuracy  | ![dct-acc](gfx/curve_danish-citizen-tests-test_accuracy.png) | **Danish Citizen Tests**<br>MCC       | ![dct-mcc](gfx/curve_danish-citizen-tests-test_mcc.png)   |
| **Dansk (NER)**<br>Micro F1           | ![dansk-f1](gfx/curve_dansk-test_micro_f1.png)               | **Dansk (NER)**<br>Micro F1 (no misc) | ![dansk-f1-nm](gfx/curve_dansk-test_micro_f1_no_misc.png) |
| **Danske Talemåder**<br>Accuracy      | ![dt-acc](gfx/curve_danske-talemaader-test_accuracy.png)     | **Danske Talemåder**<br>MCC           | ![dt-mcc](gfx/curve_danske-talemaader-test_mcc.png)       |
| **Hellaswag-da**<br>Accuracy          | ![hs-acc](gfx/curve_hellaswag-da-test_accuracy.png)          | **Hellaswag-da**<br>MCC               | ![hs-mcc](gfx/curve_hellaswag-da-test_mcc.png)            |
| **IFEval-da**<br>Instruction Accuracy | ![ifeval](gfx/curve_ifeval-da-test_instruction_accuracy.png) | **Multi-Wiki QA-da**<br>Exact Match   | ![mw-em](gfx/curve_multi-wiki-qa-da-test_em.png)          |
| **Multi-Wiki QA-da**<br>F1            | ![mw-f1](gfx/curve_multi-wiki-qa-da-test_f1.png)             | **Nordjylland News**<br>chrF3++       | ![nn-f3](gfx/curve_nordjylland-news-test_chr_f3pp.png)    |
| **Nordjylland News**<br>chrF4++       | ![nn-f4](gfx/curve_nordjylland-news-test_chr_f4pp.png)       | **ScaLA-da**<br>Macro F1              | ![scala-f1](gfx/curve_scala-da-test_macro_f1.png)         |
| **ScaLA-da**<br>MCC                   | ![scala-mcc](gfx/curve_scala-da-test_mcc.png)                | **ValEU-da**<br>European Values       | ![valeu](gfx/curve_valeu-da-test_european_values.png)     |

_Error bars show 95% CIs (bootstrap, 1000 samples); runs dodged horizontally for
visibility._

---

## Configs

All configs in `config/` directory:

| Config                            | Construction Mode | Description                                    |
| --------------------------------- | ----------------- | ---------------------------------------------- |
| `danish-apertus.yaml`             | `max_reward`      | Select best-scoring candidate                  |
| `danish-apertus-gold.yaml`        | `gold_chosen`     | Use Qwen3-235B outputs as chosen               |
| `danish-apertus-generated.yaml`   | `generated`       | Keep all candidates, score all                 |
| `danish-apertus-ls.yaml`          | `max_reward`      | DPO with label smoothing (α=0.05)              |
| `danish-apertus-simpo.yaml`       | `max_reward`      | Length-normalised loss (`sigmoid_norm`, β=0.1) |
| `danish-apertus-simpo-tuned.yaml` | `max_reward`      | Length-normalised loss (`sigmoid_norm`, β=2.0) |
| `danish-apertus-simpo-full.yaml`  | `max_reward`      | Ref-free SimPO (`simpo`, β=2.0, γ=0.5)         |
| `danish-apertus-llama-rm.yaml`    | `max_reward`      | Llama-3.1-based reward model                   |
| `danish-apertus-grpo.yaml`        | — (online RL)     | GRPO online RL baseline                        |

---

## Timeline

| Date             | Milestone                                                        |
| ---------------- | ---------------------------------------------------------------- |
| 2026-06-26       | Micro ablation runs (debug)                                      |
| 2026-06-28       | Initial CroCo runs (main, gold, generated)                       |
| 2026-06-29       | RM ablation started (Llama vs Skywork)                           |
| 2026-06-30       | Loss ablations started (ls, simpo)                               |
| 2026-07-01       | SimPO (β=0.1) complete                                           |
| 2026-07-02       | SimPO ablations queued (tuned, full)                             |
| 2026-07-03       | Llama RM training complete                                       |
| 2026-07-04       | Llama RM evals complete                                          |
| 2026-07-05       | SimPO-tuned training started                                     |
| 2026-07-06       | SimPO-tuned training complete                                    |
| 2026-07-08       | Label smoothing evals complete                                   |
| 2026-07-09 16:20 | SimPO Full training started (→ GRPO queued, → SimPO-tuned evals) |
| 2026-07-10       | SimPO Full complete; GRPO next, then recipe-fix + scaling ladder |
