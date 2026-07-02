---
title: CroCo Research Experiments
description: Overview of all preference optimisation ablation studies
created: 2026-07-02
updated: 2026-07-02
status: active
---

# CroCo Research Experiments

Overview of all research experiments and ablation studies in the CroCo project. Each
experiment tests a specific hypothesis about preference optimisation for LLM alignment.

## Pipeline Overview

All experiments follow the **CroCo (Contrastive Preference Optimization)** pipeline:

1. **Build**: Construct preference pairs via candidate generation + reward scoring
2. **Train**: DPO with curriculum learning (gated access by evolution score)
3. **Evaluate**: Danish language benchmarks (10 iter final + 3 iter checkpoint evals)

Base model: `danish-foundation-models/munin-apertus-8b`  
Reward model: `Skywork/Skywork-Reward-V2-Qwen3-8B`  
Dataset: Laerebogen (evolved subset), stratified by evolution score

---

## Experiment Catalogue

### Construction Mode Ablations

| Experiment | Description | Status |
|------------|-------------|--------|
| [**Max Reward**](01-max-reward.md) | `max_reward` construction: generate 4 candidates, select best as chosen | ✅ Complete |
| [**Gold Chosen**](02-gold-chosen.md) | Use gold (expert) outputs as chosen instead of max-reward candidates | ✅ Complete |
| [**Generated**](03-generated.md) | Standard generated mode: keep all candidates, score against prompts | ✅ Complete |
| [**Llama RM**](04-llama-rm.md) | Substitute Skywork RM with Llama-3-based reward model | ✅ Complete |

### Loss Function Ablations

| Experiment | Description | Status |
|------------|-------------|--------|
| [**Length-Normalised Loss**](05-length-normalised.md) | Test `loss_type: length_norm` vs standard DPO | ✅ Complete |
| [**SimPO (β=0.1)**](06-simpo.md) | Length-normalised loss with low β (clean single-variable ablation) | 🏃 Running |
| [**SimPO Tuned (β=2.0)**](07-simpo-tuned.md) | Raise β to SimPO-recommended 2.0, keep `sigmoid_norm` | ⏳ Queued |
| [**SimPO Full (ref-free)**](08-simpo-full.md) | True ref-free SimPO loss + target margin γ=0.5 | ⏳ Queued |

### Online RL Baseline

| Experiment | Description | Status |
|------------|-------------|--------|
| [**GRPO**](09-grpo.md) | Group Relative Policy Optimization: online RL with vLLM-colocate rollouts | ⏳ Queued |

---

## Hyperparameter Summary

| Experiment | β (temp) | Loss Type | Target Margin (γ) | Curriculum | Ref Model |
|------------|----------|-----------|-------------------|------------|-----------|
| Max Reward | 0.1 | standard (exp) | — | ✓ | ✓ |
| Gold Chosen | 0.1 | standard (exp) | — | ✓ | ✓ |
| Generated | 0.1 | standard (exp) | — | ✓ | ✓ |
| Llama RM | 0.1 | standard (exp) | — | ✓ | ✓ |
| Length-Norm | 0.1 | `length_norm` | — | ✓ | ✓ |
| SimPO | 0.1 | `sigmoid_norm` | — | ✓ | ✓ |
| SimPO Tuned | 2.0 | `sigmoid_norm` | — | ✓ | ✓ |
| SimPO Full | 2.0 | `simpo` (custom) | 0.5 | ✓ | ✗ |
| GRPO | 0.04 | GRPO loss | — | ✓ | ✗ |

---

## Key Findings (Preliminary)

### Construction Mode

- **`max_reward` vs `gold_chosen`**: Tests whether reward-maximising outputs align
  better with human preferences than expert gold outputs
- **Reward model choice**: Skywork-Reward-V2-Qwen3-8B vs Llama-3-based RM affects
  preference pair quality

### Loss Functions

- **Length normalisation**: Controls for verbosity bias in reward models
- **SimPO**: Reference-free loss removes reference model compute/memory overhead
- **Target margin (γ)**: Encourages larger reward separation between chosen/rejected

### Online RL

- **GRPO**: Eliminates preference dataset construction; learns from online rollouts
  scored by reward model

---

##Configs

All configs in `config/` directory:

```
config/
├── danish-apertus.yaml          # Main run (max_reward)
├── danish-apertus-gold.yaml     # Gold chosen ablation
├── danish-apertus-generated.yaml # Generated mode
├── danish-apertus-llama-rm.yaml # Llama-based RM
├── danish-apertus-ls.yaml       # Length-normalised loss
├── danish-apertus-simpo.yaml    # SimPO β=0.1
├── danish-apertus-simpo-tuned.yaml  # SimPO β=2.0
├── danish-apertus-simpo-full.yaml   # Ref-free SimPO + γ=0.5
├── danish-apertus-grpo.yaml     # GRPO online RL
└── danish-micro-*.yaml          # Smoke tests (10-16 samples)
```

---

## Running Experiments

```bash
# Single experiment
uv run src/scripts/run_pipeline.py -c config/danish-apertus.yaml

# With custom dataset output
uv run src/scripts/run_pipeline.py -c config/danish-apertus.yaml \
  --dataset-output data/pairs_apertus.jsonl --skip-build

# Resume after pre-flight (auto-launches queued runs)
# Monitor runs via tmux sessions: queue, tqueue, grpo
```

---

## Timeline

| Date | Milestone |
|------|-----------|
| 2026-06-28 | Initial CroCo runs (main, gold, generated) |
| 2026-06-29 | RM ablation (Llama vs Skywork) |
| 2026-06-30 | Loss ablations started (ls, simpo) |
| 2026-07-02 | SimPO ablations queued (tuned, full) |
| 2026-07-04 (est) | GRPO baseline completes |

___

## References

Each experiment doc lists its specific references. Key foundational papers:

- **DPO**: Rafailov et al. (2023), "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **SimPO**: Meng et al. (2024), "SimPO: Simple Preference Optimization with a Reference-Free Reward" — [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- **GRPO**: Shao et al. (2024), "DeepSeekMath" — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **LoRA**: Hu et al. (2021), "LoRA: Low-Rank Adaptation" — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Curriculum**: Bengio et al. (2009), "Curriculum Learning" — [ICML 2009](https://doi.org/10.1145/1553374.1553380)

*Last updated: 2026-07-02*
