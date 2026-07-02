# Length-Normalised Loss Ablation

**Config:** `config/danish-apertus-ls.yaml`  
**Output:** `models/croco-munin-apertus-8b-da-ls`

## Hypothesis

Length-normalised DPO loss reduces verbosity bias compared to standard DPO.

## Method

### Loss Function: `loss_type: length_norm`

Standard DPO computes log-probabilities over full sequences:
```
log p(y|x) = Σ log p(y_i | x, y_{<i})
```

Length-normalised divides by sequence length:
```
log p_norm(y|x) = (1/|y|) × Σ log p(y_i | x, y_{<i})
```

This removes the advantage that longer responses have in accumulating higher total
log-prob.

### Training

- **β = 0.1** (held constant for clean ablation)
- `loss_type: length_norm` (TRL builtin)
- All other settings identical to [Main CroCo](01-main-croco.md)

## Motivation

Reward models tend to prefer longer responses (verbosity bias). Length normalisation
ensures the policy isn't rewarded simply for generating more tokens.

## Results

_Results pending — training completed, evals in progress._

## Related

- [SimPO](06-simpo.md) — extends length-normalisation with reference-free loss
- [Main CroCo](01-main-croco.md) — standard DPO baseline

---

*Created: 2026-07-02*
