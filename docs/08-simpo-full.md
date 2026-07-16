---
title: SimPO Full Ablation (Ref-Free + γ=0.5)
description: True reference-free SimPO loss with target margin
description-short: Ref-free SimPO with gamma=0.5
created: 2026-07-02
updated: 2026-07-10
status: complete
config: config/danish-apertus-simpo-full.yaml
output: models/croco-munin-apertus-8b-da-simpo-full
started: 2026-07-09 16:20
completed: 2026-07-10
---

# SimPO Full Ablation (Ref-Free + γ=0.5)

## Hypothesis

True reference-free SimPO loss (no reference model) with target margin γ=0.5 outperforms
length-normalised DPO with reference model.

## Method

### Loss Function: `loss_type: simpo` (custom, reference-free)

Uses the custom `loss_type: simpo` in `src/croco/dpo.py` — **true reference-free
SimPO**: the implicit reward is the length-normalised policy log-probability with an
explicit target margin γ, and **no reference model** is used.

### Settings

- **β = 2.0**, **target margin γ = 0.5**
- **`loss_type: simpo`** (custom ref-free SimPO in `src/croco/dpo.py`)
- Reference model: **none** (ref-free)
- Curriculum learning: **enabled**
- Data: `max_reward` pairs (`data/pairs_apertus.jsonl`), reused via `--skip-build`

### Why ref-free SimPO?

Ref-free SimPO ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)) drops the
reference model entirely and adds a target margin γ. This isolates the loss from the
reference-based `sigmoid_norm` used in SimPO-tuned — the single variable tested here is
**ref-free SimPO (this run) vs reference-based `sigmoid_norm` (SimPO-tuned)**, both on
identical max_reward data.

## Training

- 625 steps on 4998 preference pairs (`pairs_apertus.jsonl`)
- LoRA r=16, LR 5e-6
- Dataset: `danish-foundation-models/laerebogen` (evolved split)

## Status

✅ **Complete** — trained 2026-07-09→10, 625 steps, 7 checkpoints
(`checkpoint-100`…`625`). Final eval done; checkpoint evals (10 iterations each)
completed 2026-07-16 20:45:45 CEST.

### Training dynamics

| step | loss | reward-acc | reward-margin |
| ---- | ---- | ---------- | ------------- |
| 10   | 1.05 | 0.61       | 0.18          |
| 80   | 1.28 | 0.43       | −0.34         |
| 240  | 0.98 | 0.61       | 0.97          |
| 400  | 0.90 | 0.58       | 1.25          |
| 560  | 0.96 | 0.59       | 1.36          |

Loss is essentially **flat** across all 625 steps and reward accuracy never leaves the
0.52–0.66 band (chance = 0.5). The model barely learns to separate chosen from rejected
— a sign of under-fitting and/or weak preference pairs (max_reward best-of-4 gives small
reward gaps).

## Results

Final EuroEval (Danish) scores vs the construction-mode baselines. Scores 0–100, higher
is better; **bold** = significant vs max_reward (non-overlapping 95% bootstrap CIs).

| Dataset / metric         | max_reward | generated |  gold | simpo_full |
| ------------------------ | ---------: | --------: | ----: | ---------: |
| angry-tweets · macro_f1  |       64.9 |      65.0 |  64.9 |       64.0 |
| angry-tweets · mcc       |       48.0 |      48.1 |  48.7 |       46.5 |
| citizen-tests · accuracy |       84.4 |      83.6 |  83.7 |       85.3 |
| citizen-tests · mcc      |       77.6 |      76.1 |  75.8 |       78.7 |
| dansk · micro_f1         |       31.5 |      30.7 |  29.8 |       31.0 |
| dansk · micro_f1_no_misc |       45.2 |      44.2 |  42.3 |       46.4 |
| talemaader · accuracy    |       69.2 |      68.0 |  70.8 |       70.6 |
| talemaader · mcc         |       62.6 |      61.0 |  64.6 |       63.8 |
| hellaswag · accuracy     |       53.9 |      52.6 |  55.0 |       54.7 |
| hellaswag · mcc          |       41.6 |      39.7 |  42.4 |       42.4 |
| ifeval · instruction_acc |       56.1 |      47.2 |  54.3 |   **62.4** |
| multi-wiki-qa · em       |       57.5 |      58.5 |  58.8 |       57.1 |
| multi-wiki-qa · f1       |       74.6 |      74.3 |  74.3 |       73.9 |
| nordjylland · chr_f3pp   |       37.6 |      37.4 |  34.2 |       37.2 |
| nordjylland · chr_f4pp   |       41.2 |      41.1 |  35.1 |   **39.9** |
| scala · macro_f1         |       62.8 |      62.6 |  47.5 |       59.4 |
| scala · mcc              |       35.7 |      35.5 |  23.0 |       32.7 |
| valeu · european_values  |        5.4 |      10.1 |   0.3 |        0.2 |
| **Mean**                 |  **52.78** |     51.98 | 50.30 |  **52.56** |

### Analysis

- **vs max_reward (same data): a wash** — 52.56 vs 52.78, only 2 of 18 metrics
  significant. Swapping reference-based DPO for ref-free SimPO on identical max_reward
  pairs gave no aggregate gain, consistent with the flat training curve.
- **Only significant win: instruction-following** — ifeval-da +6.3 (62.4 vs 56.1), the
  expected SimPO length-normalisation effect (less verbosity / format drift).
- **Cost: grammaticality / length-sensitive tasks** — ScaLA-da −3.4 macro_f1 / −3.0 mcc
  and Nordjylland summarisation chr_f4pp −1.3 (significant).
- **vs generated (+0.6)** and **vs gold_chosen (+2.3)**: simpo_full beats both; gold
  collapses on ScaLA (47.5) and Nordjylland (35) from off-policy distribution shift.
- **Ranking:** max_reward ≈ simpo_full > generated > gold.

Caveats: single-run final eval; only ifeval and nordjylland-chr_f4pp clear the CI bar vs
max_reward. `valeu-da` is unreliable (all models near-zero) — exclude from means.

### Checkpoint learning curve

Checkpoint evals covered checkpoints 100, 200, 300, 400, 500, 600, 625, plus the
final adapter. The curve does **not** show late-stage improvement: early checkpoints are
usually as good as or better than the final model.

| Pattern | Best checkpoint(s) | Notes |
| ------- | ------------------ | ----- |
| Broad aggregate | 100–200 | Best on most QA, social, summarisation, and grammar metrics. |
| Danish Citizen Tests | 300 | Peak accuracy 86.00 and MCC 79.60; final is lower. |
| HellaSwag-da | 400 | Peak accuracy 55.27 and MCC 43.15. |
| IFEval-da | final | Final adapter is best at 62.38 instruction accuracy. |
| Dansk no-misc | final | Final adapter narrowly wins at 46.41. |

By the end of training, the final adapter only clearly improves IFEval-da and Dansk
`micro_f1_no_misc`; most QA, social, summarisation, grammar, and values metrics peak
earlier. `checkpoint-625` is not a new optimum, which reinforces the flat-training
interpretation above and motivates testing whether more data, not more 5k training, is
the limiting factor.

## Single Variables Tested

| Setting   | SimPO Tuned    | SimPO Full         |
| --------- | -------------- | ------------------ |
| β         | 2.0            | 2.0                |
| Loss type | `sigmoid_norm` | `simpo` (ref-free) |
| Ref model | ✓              | ✗                  |
| γ         | —              | 0.5                |

The single variable vs SimPO-tuned is the **loss**: reference-based length-normalised
`sigmoid_norm` → true reference-free SimPO with target margin γ=0.5, on identical
max_reward data.

## Related

- [SimPO Tuned](07-simpo-tuned.md) — identical config (β=2.0, sigmoid_norm)
- [SimPO](06-simpo.md) — β=0.1 baseline
- [GRPO](09-grpo.md) — online RL baseline (complete)

---

## Reproduction

```bash
# Run full pipeline (build + train + eval)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-simpo-full.yaml

# Resume from existing cache (skip build step)
uv run src/scripts/run_pipeline.py --config config/danish-apertus-simpo-full.yaml --skip-build

# 3. Evaluate with EuroEval (Danish benchmarks, 10 iterations, bootstrap 95% CIs)
euroeval -m models/croco-munin-apertus-8b-da-simpo-full -l da --save-results

# 4. Evaluate specific checkpoints
uv run src/scripts/eval_checkpoints.py \
  -m models/croco-munin-apertus-8b-da-simpo-full -l da
```

**Tips:**

- `--skip-build` reuses cached candidate pairs
- Dataset uses `danish-foundation-models/laerebogen` (evolved split, ~5M examples
  filtered)
- See `config/danish-apertus-simpo-full.yaml` for full hyperparameters
