---
title: Max Reward Ablation
description: max_reward construction mode baseline
created: 2026-07-02
updated: 2026-07-02
status: evals-complete
config: config/danish-apertus.yaml
output: models/croco-munin-apertus-8b-da
---

# Max Reward Ablation

## Hypothesis

The `max_reward` construction mode produces high-quality preference pairs by:  
(1) selecting the best available example (gold or generated) as **chosen**, and  
(2) using a statistically low-reward candidate (mean − 2×σ) as **rejected**, ensuring a
clear reward margin.

## Method

### Construction Mode: `max_reward`

1. Generate 4 candidates per prompt using policy model
   ([vLLM](https://github.com/vllm-project/vllm), temp=0.7)
2. Score all candidates with Skywork-Reward-V2-Qwen3-8B
3. **Chosen**: highest-reward candidate (either gold from dataset OR max-reward
   generation)
4. **Rejected**: candidate closest to (mean − 2×σ) of generated candidates

### Hardware & Runtime

- **GPU:** NVIDIA GB10
- **Training time:** ~6.5 hours
- **Framework:** TRL 1.7.0 + vLLM for generation
- **LoRA:** r=16, α=32, dropout=0.05 (~1% trainable params)

- Final loss: `0.5190`
- Reward accuracy: variable (see training dynamics)
- Eval: 10 iterations on full EuroEval suite

### Training

- **[DPO](https://arxiv.org/abs/2305.18290)** with
  [curriculum learning](https://doi.org/10.1145/1553374.1553380) (gated access by
  evolution score)
- **β = 0.1** (standard DPO temperature)
- **[LoRA](https://arxiv.org/abs/2106.09685)**: r=16, α=32, dropout=0.05
- **Learning rate**: 5e-6, cosine schedule, 5% warmup
- **Steps**: 625 (1 epoch over ~5k preference pairs)

### Key Settings

```yaml
construction_mode: max_reward
score_gold_output: true
data:
    dataset_id: danish-foundation-models/laerebogen
    subset: evolved
    num_samples: 5000
dpo:
    curriculum: true
    beta: 0.1
    lora_r: 16
```

## Results

**Evaluation suite:** 10 Danish benchmarks from [EuroEval](https://euroeval.com), 10 iterations each.
**Note:** Significance compared to **Munin-Apertus-8B base model** (pre-CroCo), not to this experiment.

| Benchmark            | Task                     | Metric               |     Score |          95% CI | vs Base Model | Status      |
| -------------------- | ------------------------ | -------------------- | --------: | --------------: | :-----------: | ----------- |
| AngryTweets          | Sentiment classification | MCC                  | **48.05** |  [45.66, 50.43] |       •       | ✅ Complete |
| ScaLA-da             | Linguistic acceptability | MCC                  | **35.70** |  [32.15, 39.26] |       •       | ✅ Complete |
| DANSK                | Named entity recognition | Micro F1             | **45.20** |  [42.75, 47.64] |       •       | ✅ Complete |
| MultiWikiQA-da       | Reading comprehension    | F1                   | **74.60** |  [73.17, 76.02] |       •       | ✅ Complete |
| Nordjylland News     | Summarization            | chrF++               | **37.62** |  [37.07, 38.18] |       •       | ✅ Complete |
| Danske Talemåder     | Knowledge                | Accuracy             | **69.22** |  [66.84, 71.59] |       •       | ✅ Complete |
| Danish Citizen Tests | Knowledge                | Accuracy             | **84.44** |  [81.60, 87.29] |       •       | ✅ Complete |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | **53.95** |  [50.02, 57.87] |       •       | ✅ Complete |
| IFEval-da            | Instruction following    | Instruction accuracy | **56.13** |  [54.84, 57.41] |       ▲       | ✅ Complete |
| ValEU-da             | European values          | Alignment score      |  **5.45** |  [-1.09, 11.98] |       •       | ✅ Complete |



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

## Comparison

| Metric                | Max Reward | Gold Chosen | Generated |
| --------------------- | ---------- | ----------- | --------- |
| Win Rate (Arena-Hard) | TBD        | TBD         | TBD       |
| AlpacaEval 2 LC       | TBD        | TBD         | TBD       |
| Avg Response Length   | TBD        | TBD         | TBD       |

## Related

- [Gold Chosen ablation](02-gold-chosen.md) — replaces max-reward with expert outputs
- [SimPO ablations](06-simpo.md) — tests alternative loss functions
- [Llama RM ablation](04-llama-rm.md) — tests alternative reward model

---

## Reproduction

```bash
# 1. Run full pipeline (build + train + eval)
uv run src/scripts/run_pipeline.py --config config/danish-apertus.yaml

# 2. Or resume from existing cache (skip build step)
uv run src/scripts/run_pipeline.py --config config/danish-apertus.yaml --skip-build

# 3. Run evals (standard: 10 iterations, bootstrap 95% CIs)
uv run src/scripts/run_pipeline.py --config config/danish-apertus.yaml --eval-only --eval.num-iterations 10

# 4. Evaluate specific checkpoint
uv run src/scripts/eval_checkpoints.py -m models/croco-munin-apertus-8b-da -l da --num-iterations 10
```

**Tips:**

- `--skip-build` reuses cached `candidates_cache.jsonl` and `pairs_*.jsonl`
- Remove `--skip-build` to regenerate candidates with new generation params
- See `config/danish-apertus.yaml` for full hyperparameters

## Training Dynamics

**Dashboard:** `croco_dashboard.html` — regenerate with `python src/scripts/build_dashboard.py`

### DPO Loss

![DPO loss curves](gfx/training_loss.png)

### Preference Accuracy

![Preference accuracy](gfx/training_accuracy.png)

### Reward Margin

![Reward margin](gfx/training_margins.png)

Interactive Plotly charts in the dashboard:

- **EuroEval learning curves** — checkpoint-by-checkpointbenchmark performance
- **Final comparison** — all experiments with 95% CIs

Hover any chart and click the camera icon (📷) to export as PNG.


