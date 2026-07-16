---
title: CroCo Research Experiments
description: Overview of all preference optimisation ablation studies
created: 2026-07-02
updated: 2026-07-16
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

| Experiment                           | Description                                          | Status      |
| ------------------------------------ | ---------------------------------------------------- | ----------- |
| [**Max Reward**](01-max-reward.md)   | `max_reward` construction: 4 candidates, select best | ✅ Complete |
| [**Gold Chosen**](02-gold-chosen.md) | Use Qwen3-235B outputs as chosen                     | ✅ Complete |
| [**Generated**](03-generated.md)     | Standard generated mode, all candidates              | ✅ Complete |
| [**Llama RM**](04-llama-rm.md)       | Swap Skywork RM backbone to Llama-3.1                | ✅ Complete |

### Loss Function Ablations

| Experiment                                    | Description                             | Status      |
| --------------------------------------------- | --------------------------------------- | ----------- |
| [**Label Smoothing**](05-label-smoothing.md)  | `max_reward` + label smoothing (α=0.05) | ✅ Complete |
| [**SimPO (β=0.1)**](06-simpo.md)              | Length-normalised loss, low β ablation  | ✅ Complete |
| [**SimPO Tuned (β=2.0)**](07-simpo-tuned.md)  | β=2.0, `sigmoid_norm` loss              | ✅ Complete |
| [**SimPO Full (ref-free)**](08-simpo-full.md) | Ref-free SimPO loss, γ=0.5              | ✅ Complete |
| **SimPO Full 50k**                            | SimPO-full at 50k scale                 | 🕐 Planned  |

### Online RL Baseline

| Experiment             | Description                           | Status      |
| ---------------------- | ------------------------------------- | ----------- |
| [**GRPO**](09-grpo.md) | Online RL with vLLM-colocate rollouts | ✅ Complete |

---

## Hyperparameter Summary

| Experiment      | β (temp) | Loss Type      | Target Margin (γ) | Curriculum | Ref Model |
| --------------- | -------- | -------------- | ----------------- | ---------- | --------- |
| Max Reward      | 0.1      | standard (exp) | —                 | ✓          | ✓         |
| Gold Chosen     | 0.1      | standard (exp) | —                 | ✓          | ✓         |
| Generated       | 0.1      | standard (exp) | —                 | ✓          | ✓         |
| Llama RM        | 0.1      | standard (exp) | —                 | ✓          | ✓         |
| Label Smoothing | 0.1      | standard (exp) | —                 | ✓          | ✓         |
| SimPO           | 0.1      | `sigmoid_norm` | —                 | ✓          | ✓         |
| SimPO Tuned     | 2.0      | `sigmoid_norm` | —                 | ✓          | ✓         |
| SimPO Full      | 2.0      | `simpo`        | 0.5               | ✓          | ✗         |
|                 |          | (ref-free)     |                   |            |           |
| GRPO            | 0.04     | GRPO loss      | —                 | ✓          | ✓ (KL)    |

---

## Key Findings

### Construction Mode (vs Munin-Apertus-8B base)

| Experiment      | Best Result      | Significant Improvements ▲ | Significant Degradations ▼      |
| --------------- | ---------------- | -------------------------- | ------------------------------- |
| **Max Reward**  | IFEval-da: 56.13 | Instruction following      | —                               |
| **Gold Chosen** | IFEval-da: 54.25 | Instruction following      | ScaLA-da ▼, Nordjylland News ▼, |
|                 |                  |                            | ValEU-da ▼                      |
| **Generated**   | —                | —                          | — (no significant differences)  |

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
2026-07-06. Evals complete 2026-07-13.

### Online RL

- **GRPO**: ✅ **Complete** — training (59h 48m) + evals done. GRPO is now in the
  dashboard with full 10-iter CIs. See [GRPO doc](09-grpo.md) for results. Online RL is
  ~9× slower than DPO training (~60h vs ~6.5h) but has $0 dataset build cost.

### SimPO Full (ref-free, β=2.0, γ=0.5)

Ties `max_reward` on aggregate (52.56 vs 52.78 mean; only 2/18 metrics significant).
Ref-free SimPO on identical max_reward pairs gave **no aggregate gain** over
reference-based DPO — training was flat (reward-acc stuck ~0.58), pointing at weak
preference pairs / under-fitting. Its one significant win is **instruction following**
(IFEval-da 62.4 vs 56.1); it costs ScaLA-da and Nordjylland summarisation. Checkpoint
evals completed 2026-07-16 20:45:45 CEST: the curve peaks early on most metrics
(`checkpoint-100`/`200` dominate), with only IFEval-da and Dansk no-misc best at the
final adapter. Ranking of the 5k models: `max_reward ≈ simpo_full > generated > gold`.

---

## Completed Experiments (2026-07-15)

The 5k ablations have converged into a near-tie (~50–53 mean; ≤2/18 metrics clear
significance), so the small-data regime is too noisy to separate methods and the
diagnosed bottleneck is **preference-pair quality**, not the loss.

### Recently Completed

- **SimPO Tuned** (β=2.0, `sigmoid_norm`) — Final eval complete 2026-07-13. Performance
  comparable to `max_reward` baseline, no significant wins/degradations.
- **GRPO baseline** — Training + final eval complete 2026-07-13. Online RL paradigm
  comparison: ~60h training vs ~6.5h for DPO, but $0 dataset build cost.
- **GRPO checkpoint evals** — Learning-curve evaluation complete 2026-07-15 06:21. All
  checkpoints (100–1249) evaluated across 10 Danish benchmarks.
- **SimPO-tuned checkpoint evals** — Completed 2026-07-15 23:54:47. Checkpoints 100,
  200, 300, 400, 500, 600, 625 evaluated. `checkpoint-600` completed all 10 benchmarks
  at 21:19:43; `checkpoint-625` completed all 10 at 23:51:57. Final SimPO-tuned adapter
  had no benchmarks left (already evaluated on all selected datasets).
- **SimPO-full checkpoint evals** — Completed 2026-07-16 20:45:45. Checkpoints 100,
  200, 300, 400, 500, 600, 625 evaluated, plus the already-evaluated final adapter.
  Early checkpoints (`100`/`200`) dominate most metrics; the final adapter only clearly
  wins IFEval-da and Dansk no-misc.

### In Progress

- **SimPO-full 50k** (`config/danish-apertus-simpo-full-50k.yaml`) — Launched on
  Sparkie 2026-07-16 in tmux session `simpo_full_50k`. Tests whether scaling data
  (5k → 50k) unlocks the potential of ref-free SimPO.

### Planned

- **Post-50k analysis** — Evaluate the 50k run after training completes and compare
  against the 5k SimPO-full curve.

### Next Steps

Pending follow-up (no GPU work):

1. **Dashboard update & learning-curve comparison** — Rebuild dashboard, export new
   plots, compare checkpoint-by-checkpoint trajectories across all experiments.

Later research (to be scoped):

1. **Recipe-quality fix at 5k** — RM-margin filtering (or K≥8) + LR→1e-5 + 2 epochs.
2. **Data-scaling ladder: 5k → 25k → 100k** on winning recipe.
3. **5M only if** recipe beats baseline **and** 5k→100k curve still climbing.

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

| Dataset              | Metric     | Base Model    | Max Reward      | Gold            | Generated     | Label Smooth    | SimPO-Full      | SimPO Tuned   |
| -------------------- | ---------- | ------------- | --------------- | --------------- | ------------- | --------------- | --------------- | ------------- |
| AngryTweets          | MCC        | 48.51 [45.30, | 48.05 [45.66,   | 48.68 [45.38,   | 48.07 [45.62, | 47.76 [45.62,   | 46.50 [44.11,   | 47.28 [44.28, |
|                      |            | 51.71]        | 50.43]          | 51.97]          | 50.51]        | 49.90]          | 48.90]          | 50.28]        |
| ScaLA-da             | MCC        | 34.56 [32.06, | 35.70 [32.15,   | 23.04 ▼ [18.50, | 35.46 [32.56, | 32.37 [28.74,   | 32.68 [28.20,   | 32.88 [30.16, |
|                      |            | 37.06]        | 39.26]          | 27.58]          | 38.35]        | 36.00]          | 37.16]          | 35.60]        |
| DANSK                | Micro F1   | 43.96 [41.79, | 45.20 [42.75,   | 42.25 [39.44,   | 44.19 [42.34, | 44.79 [43.03,   | 46.41 [44.77,   | 30.90 [29.02, |
|                      |            | 46.14]        | 47.64]          | 45.07]          | 46.05]        | 46.55]          | 48.04]          | 32.78]        |
| MultiWikiQA-da       | F1         | 75.73 [74.53, | 74.60 [73.17,   | 74.35 [73.06,   | 74.34 [72.73, | 74.07 [72.75,   | 73.85 [72.73,   | 73.85 [72.29, |
|                      |            | 76.92]        | 76.02]          | 75.63]          | 75.96]        | 75.39]          | 74.98]          | 75.41]        |
| Nordjylland News     | chrF++     | 37.51 [37.01, | 37.62 [37.07,   | 34.20 ▼ [33.43, | 37.38 [36.80, | 37.59 [37.07,   | 37.18 [36.79,   | 37.11 [36.70, |
|                      |            | 38.01]        | 38.18]          | 34.97]          | 37.96]        | 38.11]          | 37.57]          | 37.53]        |
| Danske Talemåder     | Accuracy   | 69.22 [66.05, | 69.22 [66.84,   | 70.78 [68.45,   | 67.97 [64.97, | 69.22 [66.15,   | 70.62 [66.76,   | 69.06 [66.15, |
|                      |            | 72.38]        | 71.59]          | 73.11]          | 70.97]        | 72.28]          | 74.49]          | 71.98]        |
| Danish Citizen Tests | Accuracy   | 84.78 [82.16, | 84.44 [81.60,   | 83.67 [81.15,   | 83.56 [81.05, | 84.00 [80.95,   | 85.33 [83.02,   | 85.00 [83.21, |
|                      |            | 87.40]        | 87.29]          | 86.18]          | 86.07]        | 87.05]          | 87.65]          | 86.79]        |
| HellaSwag-da         | Accuracy   | 53.28 [49.54, | 53.95 [50.02,   | 54.96 [51.05,   | 52.62 [49.28, | 52.77 [48.70,   | 54.73 [51.10,   | 53.79 [49.94, |
|                      |            | 57.02]        | 57.87]          | 58.87]          | 55.96]        | 56.84]          | 58.36]          | 57.63]        |
| IFEval-da            | Instr. Acc | 50.29 [48.75, | 56.13 ▲ [54.84, | 54.25 ▲ [53.55, | 47.21 [45.62, | 54.47 ▲ [53.08, | 62.38 ▲ [61.84, | 54.28 [52.93, |
|                      |            | 51.83]        | 57.41]          | 54.96]          | 48.79]        | 55.85]          | 62.92]          | 55.64]        |
| ValEU-da             | Alignment  | 12.46 [6.86,  | 5.45 [-1.09,    | 0.28 ▼ [-0.04,  | 10.08 [2.19,  | 4.81 [-1.78,    | 0.19 ▼ [0.03,   | 0.07 [-0.03,  |
|                      |            | 18.07]        | 11.98]          | 0.61]           | 17.97]        | 11.40]          | 0.35]           | 0.17]         |

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

- **Angry Tweets**
    - Macro F1: ![angry-macro](gfx/curve_angry-tweets-test_macro_f1.png)
    - MCC: ![angry-mcc](gfx/curve_angry-tweets-test_mcc.png)
- **Danish Citizen Tests**
    - Accuracy: ![dct-acc](gfx/curve_danish-citizen-tests-test_accuracy.png)
    - MCC: ![dct-mcc](gfx/curve_danish-citizen-tests-test_mcc.png)
- **Dansk (NER)**
    - Micro F1: ![dansk-f1](gfx/curve_dansk-test_micro_f1.png)
    - Micro F1 (no misc): ![dansk-f1-nm](gfx/curve_dansk-test_micro_f1_no_misc.png)
- **Danske Talemåder**
    - Accuracy: ![dt-acc](gfx/curve_danske-talemaader-test_accuracy.png)
    - MCC: ![dt-mcc](gfx/curve_danske-talemaader-test_mcc.png)
- **Hellaswag-da**
    - Accuracy: ![hs-acc](gfx/curve_hellaswag-da-test_accuracy.png)
    - MCC: ![hs-mcc](gfx/curve_hellaswag-da-test_mcc.png)
- **IFEval-da**
    - Instruction Accuracy: ![ifeval](gfx/curve_ifeval-da-test_instruction_accuracy.png)
- **Multi-Wiki QA-da**
    - Exact Match: ![mw-em](gfx/curve_multi-wiki-qa-da-test_em.png)
    - F1: ![mw-f1](gfx/curve_multi-wiki-qa-da-test_f1.png)
- **Nordjylland News**
    - chrF3++: ![nn-f3](gfx/curve_nordjylland-news-test_chr_f3pp.png)
    - chrF4++: ![nn-f4](gfx/curve_nordjylland-news-test_chr_f4pp.png)
- **ScaLA-da**
    - Macro F1: ![scala-f1](gfx/curve_scala-da-test_macro_f1.png)
    - MCC: ![scala-mcc](gfx/curve_scala-da-test_mcc.png)
- **ValEU-da**
    - European Values: ![valeu](gfx/curve_valeu-da-test_european_values.png)

_Error bars show 95% CIs (bootstrap, 1000 samples); runs dodged horizontally for
visibility._

---

## Configs

All configs in `config/` directory:

| Config                              | Construction Mode | Description                                     |
| ----------------------------------- | ----------------- | ----------------------------------------------- |
| `danish-apertus.yaml`               | `max_reward`      | Select best-scoring candidate                   |
| `danish-apertus-gold.yaml`          | `gold_chosen`     | Use Qwen3-235B outputs as chosen                |
| `danish-apertus-generated.yaml`     | `generated`       | Keep all candidates, score all                  |
| `danish-apertus-ls.yaml`            | `max_reward`      | DPO with label smoothing (α=0.05)               |
| `danish-apertus-simpo.yaml`         | `max_reward`      | Length-normalised loss (`sigmoid_norm`, β=0.1)  |
| `danish-apertus-simpo-tuned.yaml`   | `max_reward`      | Length-normalised loss (`sigmoid_norm`, β=2.0)  |
| `danish-apertus-simpo-full.yaml`    | `max_reward`      | Ref-free SimPO (`simpo`, β=2.0, γ=0.5)          |
| `danish-apertus-simpo-full-50k.yaml`| `max_reward`      | Ref-free SimPO, 50k samples (_planned_)         |
| `danish-apertus-llama-rm.yaml`      | `max_reward`      | Llama-3.1-based reward model                    |
| `danish-apertus-grpo.yaml`          | — (online RL)     | GRPO online RL baseline                         |

---

## Timeline

| Date             | Milestone                                               |
| ---------------- | ------------------------------------------------------- |
| 2026-06-26       | Micro ablation runs (debug)                             |
| 2026-06-28       | Initial CroCo runs (main, gold, generated)              |
| 2026-06-29       | RM ablation started (Llama vs Skywork)                  |
| 2026-06-30       | Loss ablations started (ls, simpo)                      |
| 2026-07-01       | SimPO (β=0.1) complete                                  |
| 2026-07-02       | SimPO ablations queued (tuned, full)                    |
| 2026-07-03       | Llama RM training complete                              |
| 2026-07-04       | Llama RM evals complete                                 |
| 2026-07-05       | SimPO-tuned training started                            |
| 2026-07-06       | SimPO-tuned training complete                           |
| 2026-07-08       | Label smoothing evals complete                          |
| 2026-07-09 16:20 | SimPO Full training started (→ SimPO-tuned evals)       |
| 2026-07-10       | SimPO Full complete; GRPO training started              |
| 2026-07-13       | GRPO + SimPO-tuned final evals complete                 |
| 2026-07-13+      | Checkpoint evals started (one GPU workload at a time)   |
| 2026-07-15 06:21 | GRPO checkpoint evals complete (13 checkpoints)         |
| 2026-07-15 21:19 | SimPO-tuned checkpoint-600 complete (all 10 benchmarks) |
| 2026-07-15 23:51 | SimPO-tuned checkpoint-625 complete (all 10 benchmarks) |
| 2026-07-15 23:54 | SimPO-tuned checkpoint eval queue complete              |
