# SimPO Ablation (β=0.1)

**Config:** `config/danish-apertus-simpo.yaml`  
**Output:** `models/croco-munin-apertus-8b-da-simpo`

## Hypothesis

Length-normalised loss with low β (0.1) provides a clean single-variable ablation
baseline for subsequent SimPO experiments.

## Method

### Settings

- **β = 0.1** (intentionally low — tests loss type only, not hyperparameter tuning)
- **`loss_type: sigmoid_norm`** (TRL's length-normalised DPO)
- Reference model: **active** (computed via adapter-off forward with LoRA)
- Curriculum learning: **enabled**

### Why β=0.1?

This is **under-powered** by SimPO standards (paper recommends β=2.0–2.5). Purpose:
- Clean single-variable ablation vs length-norm run
- isolates effect of length-normalisation without confounding hyperparameter changes

## Training

Identical to [Length-Normalised](05-length-normalised.md) except loss type hint:
- DPO with curriculum learning
- LoRA r=16, LR 5e-6
- 625 steps (~1 epoch)

## Current Status

🏃 **Running** — step ~622/625 (99.5%), ~4 min remaining on training.

## Results

_Results pending — evals start after training completes (~2h)._

## Timeline

| Date | Milestone |
|------|----------|
| 2026-07-01 09:59 | Training started (step 0/625) |
| 2026-07-02 17:20 | Training at step 622/625 (99.5%) |
| ~2026-07-02 17:25 | Training completes (ETA) |
| ~2026-07-02 19:15 | Evals complete (ETA) |

## Related

- [SimPO Tuned](07-simpo-tuned.md) — β raised to 2.0
- [SimPO Full](08-simpo-full.md) — ref-free loss + target margin γ=0.5
- [Length-Normalised](05-length-normalised.md) — predecessor ablation

## References

- **SimPO**: Meng et al. (2024) — [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- **DPO**: Rafailov et al. (2023) — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **Length normalization**: Koehn & Knowles (2017) — [ACL Workshop](https://aclanthology.org/W17-3206/)

---

*Created: 2026-07-02 | Updated: 2026-07-02 17:25 CEST*
