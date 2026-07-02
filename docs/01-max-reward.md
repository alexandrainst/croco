---
title: Max Reward Ablation
description: max_reward construction mode baseline
created: 2026-07-02
updated: 2026-07-02
status: evals-in-progress
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

**Evaluation suite:** 10 Danish benchmarks from EuroEval, 3 iterations each:

| Benchmark            | Task                     | Metric               | Score (± CI) | Status         |
| -------------------- | ------------------------ | -------------------- | ------------ | -------------- |
| AngryTweets          | Sentiment classification | MCC                  | TBD          | 🏃 In progress |
| ScaLA-da             | Linguistic acceptability | MCC                  | TBD          | ⏳ Pending     |
| DANSK                | Named entity recognition | Micro F1             | TBD          | ⏳ Pending     |
| MultiWikiQA-da       | Reading comprehension    | F1                   | TBD          | ⏳ Pending     |
| Nordjylland News     | Summarization            | chrF++               | TBD          | ⏳ Pending     |
| Danske Talemåder     | Knowledge                | Accuracy             | TBD          | ⏳ Pending     |
| Danish Citizen Tests | Knowledge                | Accuracy             | TBD          | ⏳ Pending     |
| HellaSwag-da         | Common sense reasoning   | Accuracy             | TBD          | ⏳ Pending     |
| IFEval-da            | Instruction following    | Instruction accuracy | TBD          | ⏳ Pending     |
| ValEU-da             | European values          | Alignment score      | TBD          | ⏳ Pending     |

**Training metrics** (step 625/625):

- Final loss: TBD
- Reward margin: TBD
- Chosen log-prob: TBD

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

_Created: 2026-07-02 | Updated: 2026-07-02_
