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
- Eval: 3 iterations on full EuroEval suite

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

**Evaluation suite:** 10 Danish benchmarks from [EuroEval](https://euroeval.com), 3 iterations each.
**Note:** Significance compared to **Munin-Apertus-8B base model** (pre-CroCo), not to this experiment.

| Benchmark            | Task                     | Metric               |     Score | vs Base Model | Status      |
| -------------------- | ------------------------ | -------------------- | --------: | :-----------: | ----------- |
| AngryTweets          | Sentiment classification | MCC                  | **48.68** |       •       | ✅ Complete |
| ScaLA-da             | Linguistic acceptability | MCC                  | **35.70** |       •       | ✅ Complete |
| DANSK                | Named entity recognition | Micro F1             | **45.20** |       •       | ✅ Complete |
| MultiWikiQA-da       | Reading comprehension    | F1                   | **74.60** |       •       | ✅ Complete |
| Nordjylland News     | Summarization            | chrF++               | **37.62** |       •       | ✅ Complete |
| Danske Talemåder     | Knowledge                | Accuracy             | **70.78** |       •       | ✅ Complete |
| Danish Citizen Tests | Knowledge                | Accuracy             | **84.44** |       •       | ✅ Complete |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | **54.96** |       •       | ✅ Complete |
| IFEval-da            | Instruction following    | Instruction accuracy | **56.13** |       ▲       | ✅ Complete |
| ValEU-da             | European values          | Alignment score      |  **5.45** |       •       | ✅ Complete |

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

# 3. Run evals only (3 iterations)
uv run src/scripts/run_pipeline.py --config config/danish-apertus.yaml --eval-only --eval.num-iterations 3

# 4. Evaluate specific checkpoint
uv run src/scripts/eval_checkpoints.py -m models/croco-munin-apertus-8b-da -l da --num-iterations 3
```

**Tips:**

- `--skip-build` reuses cached `candidates_cache.jsonl` and `pairs_*.jsonl`
- Remove `--skip-build` to regenerate candidates with new generation params
- See `config/danish-apertus.yaml` for full hyperparameters

## Training Dynamics

**Dashboard:** `ssh sparkie ~/croco/croco_dashboard.html` (auto-refreshes every 60s)

Interactive Plotly charts:

- **DPO loss curves** — per-step training loss across all experiments
- **Reward margins** — chosen vs rejected separation over training
- **EuroEval learning curves** — checkpoint-by-checkpointbenchmark performance
- **Final comparison** — all experiments with 95% CIs

Hover any chart and click the camera icon (📷) to export as PNG.
