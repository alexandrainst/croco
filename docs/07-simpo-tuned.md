# SimPO Tuned Ablation (β=2.0)

**Config:** `config/danish-apertus-simpo-tuned.yaml`  
**Output:** `models/croco-munin-apertus-8b-da-simpo-tuned`

## Hypothesis

Raising β to SimPO-recommended range (2.0) unlocks the full benefit of length-normalised
loss.

## Method

### Settings

- **β = 2.0** (SimPO paper range: 2.0–2.5 for base/instruct models)
- **`loss_type: sigmoid_norm`** (TRL's length-normalised DPO — **not** ref-free)
- Reference model: **active**
- Curriculum learning: **enabled**

### Why β=2.0?

From [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO):
> SimPO requires a much larger `beta` than DPO... We recommend using `2.0` or `2.5`.

Paper values:
| Model | β |
|-------|---|
| Mistral-Base | 2.0 |
| Llama3-Base | 2.0 |
| Mistral-Instruct | 2.5 |
| Llama3-Instruct | 2.5 |

We chose **2.0** (conservative, matches base models).

## Training

- DPO with curriculum learning
- LoRA r=16, LR 5e-6
- 625 steps (~11h training + ~2h evals)

## Timeline

| Date | Milestone |
|------|----------|
| 2026-07-02 19:15 (est) | Training starts (auto-launch) |
| 2026-07-03 06:15 (est) | Training completes |
| 2026-07-03 08:15 (est) | Evals complete |

## Current Status

⏳ **Queued** — auto-launches after current simpo run completes (~19:15 CEST).

## Single Variable Tested

| Setting | SimPO (β=0.1) | SimPO Tuned |
|---------|---------------|-------------|
| β | 0.1 | **2.0** |
| Loss type | `sigmoid_norm` | `sigmoid_norm` |
| Ref model | ✓ | ✓ |
| γ | — | — |

**Only β changes** — tests whether low β was limiting performance.

## Related

- [SimPO](06-simpo.md) — β=0.1 baseline
- [SimPO Full](08-simpo-full.md) — adds ref-free loss + target margin

## References

- SimPO: Meng et al. (2024), "SimPO: Simple Preference Optimization with a Reference-Free Reward" — [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)  
  - Section 4.1: Beta ablation study
  - Official implementation: [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO) (recommends β=2.0–2.5)
- DPO: Rafailov et al. (2023), "Direct Preference Optimization" — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

---

*Created: 2026-07-02 | Updated: 2026-07-02*
