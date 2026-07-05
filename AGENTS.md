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

All executable scripts are in `src/scripts/`. Run via:

```bash
uv run src/scripts/<script>.sh
# or on sparkie:
bash src/scripts/<script>.sh
```

| Script | Purpose | Status |
|--------|---------|--------|
| `grpo_queue.sh` | GRPO baseline (micro → apertus) | ⏳ Queued |
| `llama_rm_queue.sh` | Llama-3.1 RM ablation (rescore cache → train) | ⏳ Waiting for queue |
| `update_docs.sh` | Export all 22 plots from dashboard | ✅ Ready |
| `auto_launch_llama_rm.sh` | Monitor `queue` → auto-launch llama_rm when GPU free | ✅ Ready |
| `auto_launch_grpo.sh` | Monitor `llamarm` → auto-launch grpo when GPU free | ✅ Ready |

**Removed scripts:**

- `resume_ls_simpo.sh` — ls/simpo ablations complete
- `resume_tuned_simpo.sh` — SimPO tuned/full complete

### Auto-Launch Scripts

Chain experiments automatically when GPU becomes free:

```bash
# Launch llama_rm when 'queue' session finishes
tmux new-session -d -s auto_rm "bash -lc 'bash ~/croco/src/scripts/auto_launch_llama_rm.sh 2>&1 | tee ~/croco/auto_rm_launch.log'"

# Launch grpo when 'llamarm' session finishes  
tmux new-session -d -s auto_grpo "bash -lc 'bash ~/croco/src/scripts/auto_launch_grpo.sh 2>&1 | tee ~/croco/auto_grpo_launch.log'"
```

**Workflow:** `queue` (simpo evals) → `llamarm` (Llama RM) → `grpo` (GRPO baseline)

Monitor logs: `tail -f ~/croco/auto_rm_launch.log` or `tail -f ~/croco/auto_grpo_launch.log`

### update_docs.sh Details

Exports 22 plots total:

1. **Training dynamics** (3): `training_loss.png`, `training_accuracy.png`, `training_margins.png`
2. **Learning curves** (18): All dataset-metric combinations:
   - Angry Tweets (macro_f1, mcc)
   - Danish Citizen Tests (accuracy, mcc)
   - Dansk NER (micro_f1, micro_f1_no_misc)
   - Danske Talemåder (accuracy, mcc)
   - Hellaswag-da (accuracy, mcc)
   - IFEval-da (instruction_accuracy)
   - Multi-Wiki QA-da (em, f1)
   - Nordjylland News (chr_f3pp, chr_f4pp)
   - ScaLA-da (macro_f1, mcc)
   - ValEU-da (european_values)
3. **Final comparison** (1): `final_comparison.png` (bar chart with 95% CIs)

Requires: `plotly`, `kaleido` installed, `croco_dashboard.html` exists.

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
- **TMPDIR + HF_DATASETS_CACHE** — Both must be set for training runs:
  ```bash
  export TMPDIR=~/croco/.tmp
  export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
  mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
  ```
  HuggingFace datasets creates temp Arrow files in `/tmp/hf_datasets-*`
  regardless of `HF_DATASETS_CACHE`. If `/tmp` is cleaned mid-run, training
  crashes with `FileNotFoundError`. The queue scripts (`src/scripts/*_queue.sh`)
  already set these; use them.
- **One GPU workload at a time on sparkie** — The DGX Spark has 128GB unified
  memory shared between CPU and GPU. Running training + evaluation simultaneously
  causes GPU OOM → kernel wedge → complete system lockup (no SSH, no ping).
  Recovery requires physical power-cycle + potential NVIDIA driver reload.
  **Always check GPU is idle before launching:**
  ```bash
  ssh sparkie 'nvidia-smi --query-compute-apps=pid --format=csv,noheader'
  ssh sparkie 'tmux ls | grep -v dr_scraper'
  ```
  Wait for training to complete before starting evals (or vice versa).
  Queue scripts handle this automatically; don't manually run concurrent jobs.

## Remote Execution (sparkie)

Experiments run on the `sparkie` GPU server. **Always use queue scripts**
(`src/scripts/*_queue.sh`) — they set `TMPDIR` and `HF_DATASETS_CACHE`,
and log output properly. Don't run `run_pipeline.py` directly.

```bash
# Launch via queue script (recommended)
ssh sparkie
cd ~/croco
tmux new-session -d -s exp1 "bash -lc 'bash src/scripts/<script>_queue.sh 2>&1 | tee ~/croco/run.log'"

# Monitor
tmux attach -t exp1
tail -f ~/croco/run.log
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

## Configuration Optimization Tips

Based on TRL 1.7.0 feature investigation:

### Speed Optimizations

```yaml
dpo:
  precompute_ref_log_probs: true  # Skip ref forward pass per step (~30-50% faster)
  torch_compile: true              # PyTorch compilation (~30-50% faster after warmup)
  dataloader_num_workers: 4        # Parallel data loading
```

**Important:** Before enabling `precompute_ref_log_probs`, set both `TMPDIR`
and `HF_DATASETS_CACHE` to persistent storage:

```bash
export TMPDIR=~/croco/.tmp
export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
```

Without this:
- TRL's precomputed reference log probs cache goes to `/tmp` and can disappear
  mid-run (shared filesystem cleanup policies)
- HuggingFace datasets creates temp Arrow files in `/tmp/hf_datasets-*`
  regardless of `HF_DATASETS_CACHE`

Both failures cause `FileNotFoundError` mid-training. The queue scripts
(`src/scripts/*_queue.sh`) already set these.

### Memory Optimizations

```yaml
dpo:
  gradient_checkpointing: true      # Already default
  activation_offloading: true       # +20-30% GPU memory free, ~15% slower
```

### Quality Improvements

```yaml
dpo:
  loss_type: sigmoid_norm    # SimPO length-normalized loss (reduces verbosity bias)
  label_smoothing: 0.05      # Robust to noisy reward scores
  use_weighting: true        # WPO-style preference weighting
```

**Notes:**

- `sigmoid_norm`: Available in TRL 1.7.0 via [`DPOConfig`](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py). Implements SimPO's length-normalized loss but still uses reference model.
- `torch_compile`: First run slow (compilation), subsequent runs faster.
- `activation_offloading`: Only enable if OOM — trades speed for memory.

### Algorithm Support (TRL 1.7.0)

| Algorithm | Support | Use Case |
|-----------|---------|----------|
| **DPO** | ✅ Full | Solid baseline |
| **SimPO** | ⚠️ Partial (`sigmoid_norm`) | Length normalization |
| **KTO** | ✅ Full | If reward noise is an issue |
| **GRPO** | ✅ Full | If verifiable rewards available |
| **ORPO** | ❌ Not in TRL 1.7.0 | Skip |

