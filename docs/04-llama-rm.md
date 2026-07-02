---
title: Llama RM Ablation
description: Reward model substitution (Skywork vs Llama-3-based)
created: 2026-07-02
updated: 2026-07-02
status: evals-pending
config: config/danish-apertus-llama-rm.yaml
output: models/croco-munin-apertus-8b-da-llama-rm
---

# Llama RM Ablation

**Config:** `config/danish-apertus-llama-rm.yaml`  
**Output:** `models/croco-munin-apertus-8b-da-llama-rm`

## Hypothesis

A Llama-3-based reward model produces different (potentially better) preference signals
than Skywork-Reward-V2-Qwen3-8B for Danish language tasks.

## Method

### Reward Model Substitution

- **Default RM**: `Skywork/Skywork-Reward-V2-Qwen3-8B` (Qwen3-based, Chinese lab)
- **Ablation RM**: Llama-3-based reward model (Western, English-trained)

Tests whether RM architecture/training data affects preference pair quality.

### Training

Identical to [Main CroCo](01-main-croco.md):
- DPO with curriculum learning
- β = 0.1, LoRA r=16, LR 5e-6
- Construction mode: `max_reward`

## Motivation

Reward models are trained on different datasets and may have cultural/linguistic biases.
A Llama-3-based RM trained on Western data may score Danish outputs differently than a
Qwen-based RM.

## Results

_Results pending._

## Timeline

| Date | Milestone |
|------|----------|
| 2026-06-29 | Training started |
| 2026-06-30 | Training completed |
| 2026-07-02 | Evals pending |

## Related

- [Main CroCo](01-main-croco.md) — Skywork RM baseline

## References

- DPO: Rafailov et al. (2023), "Direct Preference Optimization" — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Skywork-Reward-V2-Qwen3-8B: Skywork AI (2024) — [Hugging Face](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)
- Llama 3: Meta AI (2024), "Llama 3 Model Card" — [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

---

*Created: 2026-07-02 | Updated: 2026-07-02*
