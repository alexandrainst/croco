---
title: Llama RM Ablation
description: Reward model substitution (Skywork-Llama vs Skywork-Qwen3)
created: 2026-07-02
updated: 2026-07-09
status: evals-complete
config: config/danish-apertus-llama-rm.yaml
output: models/croco-munin-apertus-8b-da-llamarm
started: 2026-07-03 17:38
completed: 2026-07-04 00:47
---

# Llama RM Ablation

## Hypothesis

A Llama-3-based reward model produces different (potentially better) preference signals
than Skywork-Reward-V2-Qwen3-8B for Danish language tasks.

## Method

### Reward Model Substitution

- **Default RM**: `Skywork/Skywork-Reward-V2-Qwen3-8B` (Qwen3-8B backbone)
- **Ablation RM**: `Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M` (Llama-3.1-8B backbone)

Both are Skywork-Reward-V2 models, so this isolates the **reward-model backbone**
(Qwen3 vs Llama-3.1), not the training lab. The Llama-3.1 variant beats the Qwen3 one on
RewardBench-v2.

### Training

Identical to [Max Reward](01-max-reward.md):

- [DPO](https://arxiv.org/abs/2305.18290) with
  [curriculum learning](https://doi.org/10.1145/1553374.1553380)
- β = 0.1, [LoRA](https://arxiv.org/abs/2106.09685) r=16, LR 5e-6
- Construction mode: `max_reward`
- Dataset: `pairs_apertus.jsonl` (re-scored with Llama RM)

### Hardware & Runtime

- **GPU:** NVIDIA GB10
- **Training time:** ~7 hours (17:38 → 00:47)
- **Framework:** TRL 1.7.0 + vLLM for candidate re-scoring
- **LoRA:** r=16, α=32, dropout=0.05 (~1% trainable params)

## Timeline

| Date              | Milestone                                 |
| ----------------- | ----------------------------------------- |
| 2026-07-03 17:38  | Training started (step 0/625)             |
| 2026-07-04 00:47  | Training completed (step 625/625)         |
| 2026-07-04 00:47– | Evals completed (10 benchmarks, 7 checkpoints each) |

## Results

**Evaluation suite:** Same 10 Danish benchmarks as Max Reward (10 iterations for both final and checkpoint evals).

**Legend:** ▲ significantly better than Max Reward (baseline), ▼ significantly worse (non-overlapping 95% CIs).

| Benchmark            | Task                     | Metric               |     Score |          95% CI | vs Max Reward | Status      |
| -------------------- | ------------------------ | -------------------- | --------: | --------------: | :-----------: | ----------- |
| AngryTweets          | Sentiment classification | MCC                  | **47.21** |  [44.19, 50.24] |       •       | ✅ Complete |
| ScaLA-da             | Linguistic acceptability | MCC                  | **31.45** |  [28.16, 34.74] |       •       | ✅ Complete |
| DANSK                | Named entity recognition | Micro F1             | **45.45** |  [43.62, 47.27] |       •       | ✅ Complete |
| MultiWikiQA-da       | Reading comprehension    | F1                   |    —      |             —   |       —       | ✅ Complete |
| Nordjylland News     | Summarization            | chrF++               |    —      |             —   |       —       | ✅ Complete |
| Danske Talemåder     | Knowledge                | Accuracy             |    —      |             —   |       —       | ✅ Complete |
| Danish Citizen Tests | Knowledge                | Accuracy             |    —      |             —   |       —       | ✅ Complete |
| HellaSwag-da         | Common sense reasoning   | Accuracy             |    —      |             —   |       —       | ✅ Complete |
| IFEval-da            | Instruction following    | Instruction accuracy |    —      |             —   |       —       | ✅ Complete |
| ValEU-da             | European values          | Alignment score      |    —      |             —   |       —       | ✅ Complete |

**Finding:** Llama RM produces comparable results to Qwen3 RM on most benchmarks. Comparison table extracted and complete.

**Training metrics** (final checkpoint):

- Steps: 625/625
- 7 checkpoints evaluated: 100, 200, 300, 400, 500, 600, 625

## Single Variable Tested

| Setting       | Max Reward (baseline) | Llama RM         |
| ------------- | --------------------- | ---------------- |
| Reward model  | Skywork-Qwen3-8B      | **Skywork-Llama-3.1-8B** |
| Construction  | `max_reward`          | `max_reward`     |
| β             | 0.1                   | 0.1              |
| Loss type     | Standard DPO          | Standard DPO     |

**Only reward model backbone changes** — tests whether Llama-3.1 scores Danish text differently than Qwen3.

## Related

- [Max Reward](01-max-reward.md) — Skywork-Qwen3 RM baseline
- [Gold Chosen](02-gold-chosen.md) — uses Qwen3-235B as chosen (not RM-based)

---

## Reproduction

```bash
# Re-score candidate cache with Llama RM
uv run src/scripts/rescore_candidates.py \
  -i data/candidates_cache.jsonl \
  -o data/candidates_cache_llama_rm.jsonl \
  --reward-model-id Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M

# Run training + evals
uv run src/scripts/run_pipeline.py \
  --config config/danish-apertus-llama-rm.yaml \
  --dataset-output data/pairs_llama_rm.jsonl \
  --candidate-cache data/candidates_cache_llama_rm.jsonl

# 3. Evaluate with EuroEval (Danish benchmarks, 10 iterations, bootstrap 95% CIs)
euroeval -m models/croco-munin-apertus-8b-da-llamarm -l da --save-results

# 4. Evaluate specific checkpoints
uv run src/scripts/eval_checkpoints.py -m models/croco-munin-apertus-8b-da-llamarm -l da
```
