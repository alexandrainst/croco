---
title: Llama RM Ablation
description: Reward model substitution (Skywork vs Llama-3-based)
created: 2026-07-02
updated: 2026-07-02
status: not-started
config: config/danish-apertus-llama-rm.yaml
output: models/croco-munin-apertus-8b-da-llama-rm
---

# Llama RM Ablation

## Hypothesis

A Llama-3-based reward model produces different (potentially better) preference signals
than Skywork-Reward-V2-Qwen3-8B for Danish language tasks.

## Method

### Reward Model Substitution

- **Default RM**: `Skywork/Skywork-Reward-V2-Qwen3-8B` (Qwen3-based, Chinese lab)
- **Ablation RM**: Llama-3-based reward model (Western, English-trained)

Tests whether RM architecture/training data affects preference pair quality.

### Training

Identical to [Max Reward](01-max-reward.md):

- [DPO](https://arxiv.org/abs/2305.18290) with
  [curriculum learning](https://doi.org/10.1145/1553374.1553380)
- β = 0.1, [LoRA](https://arxiv.org/abs/2106.09685) r=16, LR 5e-6
- Construction mode: `max_reward`

### Hardware & Runtime

- **GPU:** NVIDIA GB10 (expected, matching other experiments)
- **Training time:** ~6.5 hours (estimated from Max Reward baseline)
- **Framework:** TRL 1.7.0 + vLLM for candidate re-scoring
- **LoRA:** r=16, α=32, dropout=0.05 (~1% trainable params)

**Training metrics:** ⏳ Pending (experiment not started)

- Expected final loss: ~0.52 (based on Max Reward: 0.5190)
- Expected reward margin: ~1.7-2.0
- Expected eval iterations: 3 on full [EuroEval](https://euroeval.com) suite

## Motivation

Reward models are trained on different datasets and may have cultural/linguistic biases.
A Llama-3-based RM trained on Western data may score Danish outputs differently than a
Qwen-based RM.

## Results

**Status:** ⏳ **Not run yet** — config exists but training has not started.

**Evaluation suite (planned):** Same 10 Danish benchmarks as Max Reward (3 iterations each).

| Benchmark      | Task | Metric | Score | Status     |
| -------------- | ---- | ------ | ----: | ---------- |
| All benchmarks | —    | —      |     — | ⏳ Not run |

## Timeline

**Status:** ⏳ **Queued** — ready to launch after ls/simpo ablations complete.

| Date       | Milestone                                 |
| ---------- | ----------------------------------------- |
| 2026-07-01 | Config created                            |
| 2026-07-02 | Queue script created: `llama_rm_queue.sh` |
| —          | **Awaiting ls/simpo winner**              |
| 2026-07-02 | Evals pending                             |

## Related

- [Max Reward](01-max-reward.md) — Skywork RM baseline

---
