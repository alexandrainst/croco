# Croco

Experiments with the **CroCo** (Contrastive Comparison) post-training method — a
DPO-based approach using reward model scoring and curriculum learning for Danish
language alignment on Munin-Apertus-8B.

## Stack

- **Python 3.12** + `uv` package manager
- **TRL 1.7.0** (transformers reinforcement learning)
- **vLLM** for candidate generation and reward scoring
- **Plotly + Kaleido** for dashboard visualization
- **EuroEval v17.5.0** for benchmark evaluation (Danish language)
- **Target GPU**: NVIDIA GB10 (or any CUDA-capable GPU with sufficient VRAM)

## Layout

```
croco/
├── config/               # YAML configs for experiments (ablation studies)
├── docs/                 # Experiment documentation + gfx/ for plots
├── models/               # Trained checkpoints (auto-saved by scripts)
├── src/
│   ├── croco/            # Core library modules (imported, not executed)
│   └── scripts/          # All executable scripts (pipeline, eval, utils)
├── tests/                # Pytest test suite
└── makefile              # Convenience targets (check, test, tree)
```

## Running Experiments

### Quick Start

```bash
# Install dependencies
uv sync

# Run a full experiment (data build + DPO training + evaluation)
uv run src/scripts/run_pipeline.py --config config/danish-apertus.yaml

# Resume from existing cache (skip data generation)
uv run src/scripts/run_pipeline.py --config config/danish-apertus.yaml --skip-build

# Evaluate only (no training)
uv run src/scripts/run_pipeline.py --config config/danish-apertus.yaml --eval-only
```

### Ablation Experiments

| Config                           | Construction Mode | Description                      |
| -------------------------------- | ----------------- | -------------------------------- |
| `danish-apertus.yaml`            | `max_reward`      | Select best-scoring candidate    |
| `danish-apertus-gold.yaml`       | `gold_chosen`     | Use Qwen3-235B outputs as chosen |
| `danish-apertus-generated.yaml`  | `generated`       | Keep all candidates, score all   |
| `danish-apertus-ls.yaml`         | `max_reward`      | DPO with label smoothing (α=0.05)|
| `danish-apertus-simpo.yaml`      | `max_reward`      | SimPO loss (γ=0.5, β=2.0)        |
| `danish-apertus-llama-rm.yaml`   | `max_reward`      | Llama-3-based reward model       |

### Queue/Runner Scripts

All in `src/scripts/`:

| Script                        | Purpose                                 |
| ----------------------------- | --------------------------------------- |
| `grpo_queue.sh`               | GRPO baseline training                  |
| `llama_rm_queue.sh`           | Llama RM ablation (rescore → train)     |
| `resume_ls_simpo.sh`          | Resume label smoothing / SimPO runs     |
| `resume_tuned_simpo.sh`       | SimPO tuned → full ablation chain       |
| `update_docs.sh`              | Export dashboard plots to `docs/gfx/`   |

## Testing

```bash
# Run all tests
make test

# Run type checking, linting, formatting
make check

# Individual test files
uv run pytest tests/test_dpo.py -v
```

## Conventions

### Code Style

- 88-character line width, Google-style docstrings
- Type hints: Python 3.12+ syntax (`list[T]`, `X | None`)
- Imports: `import typing as t`, `import collections.abc as c`
- Use f-strings, not %-formatting
- No `print()` — use `logging` module
- Relative imports in modules, absolute in scripts

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add SimPO loss mixin
fix: correct reward margin calculation
docs: update experiment results
```

## Documentation Workflow

### Updating Experiment Plots

When experiments complete or checkpoints are evaluated:

```bash
# Export all plots (training dynamics + 10 learning curves + final comparison)
uv run src/scripts/update_docs.sh
```

This script:

1. Reads existing `croco_dashboard.html` (regenerate if stale)
2. Exports 14 PNG plots to `docs/gfx/`:
   - Training: `training_loss.png`, `training_accuracy.png`, `training_margins.png`
   - Learning curves: `curve_*.png` (10 benchmarks)
   - Final: `final_comparison.png` (bar chart with 95% CIs)
3. Cleans up outdated plot files (`curve_*-test_*.png` patterns)

After running:

```bash
git add docs/gfx/*.png docs/*.md
git commit -m 'docs: update plots'
```

### Dashboard

Generate locally:

```bash
python src/scripts/build_dashboard.py \
  -m models/croco-munin-apertus-8b-da \
  -m models/croco-munin-apertus-8b-da-gold \
  -r euroeval_benchmark_results.jsonl \
  -o croco_dashboard.html
```

Open `croco_dashboard.html` in a browser. Charts are interactive (hover for
details, camera icon to export PNG).

### Experiment Docs

Each experiment has a markdown file in `docs/` (`01-max-reward.md`, etc.):

- Frontmatter metadata (status, config, output path)
- Hypothesis, Method, Results tables, Reproduction commands
- Training dynamics plots (embedded from `gfx/`)

Update docs when:

- New benchmark results are available
- Training completes (add Runtime section)
- Plots are regenerated (run `update_docs.sh` first)

## Gotchas

- **All scripts in `src/scripts/`** — No `.sh` files in root or separate
  `scripts/` directory. Run via `uv run src/scripts/<script>.sh`.
- **Custom TRL code** — Custom losses (SimPO, label smoothing) are in
  `src/croco/dpo.py`, NOT in `.venv/lib/*/site-packages/trl/`. Never edit
  the installed TRL package.
- **Reward model caching** — Candidate cache signature does NOT include the
  reward model. Swapping RMs requires explicit re-scoring
  (`src/scripts/rescore_candidates.py`).
- **LoRA ref-free training** — TRL sets `ref_model=None` when LoRA is enabled;
  reference log probs computed via adapter-off forward (not a bug).
- **Parallel experiments** — Ensure different model directories to avoid
  checkpoint collisions.
- **EuroEval cache** — Results cached in `.euroeval_cache/`. Delete to
  force re-evaluation.
- **GPU memory** — vLLM needs ~20GB VRAM for 8B models at `max_model_len=4096`.
  Reduce length or use `--tensor-parallel-size` if OOM.
- **Significance markers** — ▲▼ in tables = non-overlapping 95% CIs
  (bootstrap, 1000 samples), not p-values.
- **Dashboard regeneration** — If models/results change, regenerate
  dashboard before running `update_docs.sh`, otherwise plots will be stale.

## Remote Execution (sparkie)

Experiments run on the `sparkie` GPU server:

```bash
# Launch experiment in tmux
ssh sparkie
cd ~/croco
tmux new-session -d -s exp1 "bash -lc 'uv run src/scripts/run_pipeline.py ...'"

# Monitor
tmux attach -t exp1
tail -f ~/croco/*.log
```

Session names:

- `queue` — Construction mode ablations
- `tqueue` — SimPO tuned / full ablations
- `grpo` — GRPO baseline
- `llamarm` — Llama RM experiment

Logs: `~/croco/ablations.log`, `~/croco/overnight.log`, `~/croco/run.log`

## Environment

Sparkie-specific:

- `.env` — Environment variables (HF token, cache paths)
- `.euroeval_cache/` — Benchmark evaluation cache
- `.venv/` — Python virtual environment (managed by uv)
- `models/` — Checkpoint output (symlinked to NFS storage)

Copy `.env.example` to `.env` and fill in Hugging Face token if needed.
