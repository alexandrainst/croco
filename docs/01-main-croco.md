# Main CroCo Run

**Config:** `config/danish-apertus.yaml`  
**Output:** `models/croco-munin-apertus-8b-da`

## Hypothesis

The standard CroCo pipeline with `max_reward` construction mode produces preference
pairs that lead to better alignment than alternative construction methods.

## Method

### Construction Mode: `max_reward`

1. Generate 4 candidates per prompt using policy model (vLLM, temp=0.7)
2. Score all candidates with Skywork-Reward-V2-Qwen3-8B
3. Select highest-reward candidate as **chosen**
4. Use original prompt's existing output as **rejected** (if available)

### Training

- **DPO** with curriculum learning (gated access by evolution score)
- **β = 0.1** (standard DPO temperature)
- **LoRA**: r=16, α=32, dropout=0.05
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

_Results pending — training completed, evals in progress._

## Timeline

| Date | Milestone |
|------|----------|
| 2026-06-28 | Training started |
| 2026-06-29 | Training completed, evals started |
| 2026-07-02 | Evals in progress |

## Comparison

| Metric | Main CroCo | Gold Chosen | Generated |
|--------|------------|-------------|-----------|
| Win Rate (Arena-Hard) | TBD | TBD | TBD |
| AlpacaEval 2 LC | TBD | TBD | TBD |
| Avg Response Length | TBD | TBD | TBD |

## Related

- [Gold Chosen ablation](02-gold-chosen.md) — replaces max-reward with expert outputs
- [SimPO ablations](06-simpo.md) — tests alternative loss functions
- [Llama RM ablation](04-llama-rm.md) — tests alternative reward model

## References

- **DPO**: Rafailov et al. (2023) — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **LoRA**: Hu et al. (2021) — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Curriculum**: Bengio et al. (2009) — [ICML 2009](https://doi.org/10.1145/1553374.1553380)
- **vLLM**: Kwon et al. (2023) — [GitHub](https://github.com/vllm-project/vllm)

---

*Created: 2026-07-02 | Updated: 2026-07-02*
