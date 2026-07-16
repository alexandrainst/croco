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

# Evaluate an existing model with EuroEval (Danish benchmarks)
euroeval -m models/croco-munin-apertus-8b-da -l da --save-results

# Evaluate all checkpoints for learning curves
uv run src/scripts/eval_checkpoints.py -m models/croco-munin-apertus-8b-da -l da
```

### Ablation Experiments

| Config                          | Construction  | Description                   |
| ------------------------------- | ------------- | ----------------------------- |
| `danish-apertus.yaml`           | `max_reward`  | Select best-scoring candidate |
| `danish-apertus-gold.yaml`      | `gold_chosen` | Use Qwen3-235B outputs as     |
|                                 |               | chosen                        |
| `danish-apertus-generated.yaml` | `generated`   | Keep all candidates, score    |
|                                 |               | all                           |
| `danish-apertus-ls.yaml`        | `max_reward`  | DPO with label smoothing      |
|                                 |               | (α=0.05)                      |
| `danish-apertus-simpo.yaml`     | `max_reward`  | SimPO loss (γ=0.5, β=2.0)     |
| `danish-apertus-llama-rm.yaml`  | `max_reward`  | Llama-3-based reward model    |

### Running on Sparkie (GPU Server)

Training runs on the `sparkie` GPU server. Set environment variables for persistent
caching before launching:

```bash
# On sparkie - set TMPDIR and HF_DATASETS_CACHE to avoid mid-run crashes
export TMPDIR=~/croco/.tmp
export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

# Launch training (example)
cd ~/croco
tmux new-session -d -s exp1 \
  "bash -lc 'uv run src/scripts/run_pipeline.py \
    --config config/danish-apertus.yaml 2>&1 | tee ~/croco/run.log'"

# Monitor
tmux attach -t exp1
tail -f ~/croco/run.log
```

**Important:** Only run one GPU workload at a time. DGX Spark has 128GB unified memory
shared between CPU and GPU. Concurrent training + evaluation causes GPU OOM → kernel
wedge → system lockup (requires physical power-cycle).

**Check GPU is idle before launching:**

```bash
ssh sparkie 'nvidia-smi --query-compute-apps=pid --format=csv,noheader'
ssh sparkie 'tmux ls | grep -v dr_scraper'
```

**Completed model runs (do NOT re-run):**

- Construction mode: `max_reward`, `gold_chosen`, `generated`
- Loss functions: `label_smoothing`, `simpo` (β=0.1), `simpo_tuned` (β=2.0),
  `simpo_full` (ref-free)
- Reward model: `llamarm`
- Online RL: `grpo` (final eval complete 2026-07-13)

**Checkpoint evals:** GRPO and SimPO-tuned complete as of 2026-07-16. GRPO checkpoint
evals completed 2026-07-15 06:21. SimPO-tuned checkpoint evals (checkpoints 100–625)
completed 2026-07-15 23:54:47. SimPO-full checkpoint evals pending.

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

To export training plots from the dashboard:

1. Generate `croco_dashboard.html` (if stale, regenerate with `build_dashboard.py`)
2. Open in a browser and use the camera icon on each chart to export PNG
3. Save to `docs/gfx/` with the standard names:
    - `training_loss.png`, `training_accuracy.png`, `training_margins.png`
    - `curve_<dataset>_<metric>.png` for learning curves
    - `final_comparison.png` for the summary bar chart

Alternatively use Plotly's built-in export or screenshot tooling.

### Dashboard

Generate locally with auto-discovered model directories:

```bash
uv run src/scripts/build_dashboard.py --configs \
  --ssh-host sparkie --remote-root /home/saattrupdan/croco \
  -r euroeval_benchmark_results.jsonl \
  -o croco_dashboard.html
```

For the live local dashboard, `croco-dash` runs the same script in watch mode
(`--watch 300`) pulling checkpoint/result data from sparkie over SSH.

**HF push integration:** Training runs with `push_to_hub: true` upload checkpoints to
`danish-foundation-models/croco-munin-apertus-8b-da*` HF repos. The dashboard fetches
`trainer_state.json` directly from HF, so no local checkpoint copies are needed.
Training curves appear automatically once the first checkpoint pushes.

Open `croco_dashboard.html` in a browser. Charts are interactive (hover for details,
camera icon to export PNG).

### Experiment Docs

Each experiment has a markdown file in `docs/` (`01-max-reward.md`, etc.):

- Frontmatter metadata (status, config, output path)
- Hypothesis, Method, Results tables, Reproduction commands
- Training dynamics plots (embedded from `gfx/`)

Update docs when:

- New benchmark results are available
- Training completes (add Runtime section)
- Plots are exported from the dashboard

## Gotchas

- **`croco-dash` tmux session** — A local tmux session on the laptop refreshes
  `croco_dashboard.html` every 5 minutes by pulling latest checkpoints from sparkie. If
  the dashboard appears stale, check if `croco-dash` is still running.
- **Dashboard HF integration** — Training runs with `push_to_hub: true` upload
  checkpoints to HF. The dashboard fetches `trainer_state.json` directly from HF repos,
  so training curves auto-populate without local checkpoint copies.
- **Dashboard visibility** — Active training runs do not appear in the dashboard until
  the first `checkpoint-*/trainer_state.json` exists (locally or on HF). SimPO-tuned can
  spend over an hour precomputing reference log probs before step 1; with
  `save_steps: 100`, it stays invisible until checkpoint 100 is written and the next
  `croco-dash` refresh runs.

- **Custom TRL code** — Custom losses (SimPO, label smoothing) are in
  `src/croco/dpo.py`, NOT in `.venv/lib/*/site-packages/trl/`. Never edit the installed
  TRL package.
- **Reward model caching** — Candidate cache signature does NOT include the reward
  model. Swapping RMs requires explicit re-scoring
  (`src/scripts/rescore_candidates.py`).
- **LoRA ref-free training** — TRL sets `ref_model=None` when LoRA is enabled. Reference
  log probs computed via adapter-off forward (not a bug).
- **Parallel experiments** — Ensure different model directories to avoid checkpoint
  collisions.
- **EuroEval cache** — Results cached in `.euroeval_cache/`. Old 3-iteration checkpoint
  re-evals completed 2026-07-09; scripts deleted.
- **Checkpoint eval queue complete** — As of 2026-07-16 11:56 CEST, Sparkie has no GPU
  compute processes. The `ckpt_evals_grpo_simpo_tuned` tmux session exists but its queue
  completed 2026-07-15 23:54:47. Do not use stale `auto_launch_*.sh` monitors — they can
  spawn duplicate workloads.
- **GPU memory** — vLLM needs ~20GB VRAM for 8B models at `max_model_len=4096`. Reduce
  length or use `--tensor-parallel-size` if OOM.
- **Significance markers** — ▲▼ in tables = non-overlapping 95% CIs (bootstrap, 1000
  samples), not p-values.
- **Dashboard regeneration** — If models/results change, regenerate dashboard before
  exporting plots, otherwise they will be stale.
- **TMPDIR + HF_DATASETS_CACHE** — Both must be set for training runs:
    ```bash
    export TMPDIR=~/croco/.tmp
    export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
    mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
    ```
    HuggingFace datasets creates temp Arrow files in `/tmp/hf_datasets-*` regardless of
    `HF_DATASETS_CACHE`. If `/tmp` is cleaned mid-run, training crashes with
    `FileNotFoundError`. Set these manually or via your shell profile.
- **One GPU workload at a time on sparkie** — The DGX Spark has 128GB unified memory
  shared between CPU and GPU. Running training + evaluation simultaneously causes GPU
  OOM → kernel wedge → complete system lockup (no SSH, no ping). Recovery requires
  physical power-cycle + potential NVIDIA driver reload. **Always check GPU is idle
  before launching:**
    ```bash
    ssh sparkie 'nvidia-smi --query-compute-apps=pid --format=csv,noheader'
    ssh sparkie 'tmux ls | grep -v dr_scraper'
    ```
    Wait for training to complete before starting evals (or vice versa).

## Remote Execution (sparkie)

Experiments run on the `sparkie` GPU server. See the **Running on Sparkie** subsection
above for environment setup and launch commands.

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
    precompute_ref_log_probs: true # Skip ref forward pass per step (~30-50% faster)
    torch_compile: true # PyTorch compilation (~30-50% faster after warmup)
    dataloader_num_workers: 4 # Parallel data loading
```

**Important:** Before enabling `precompute_ref_log_probs`, set both `TMPDIR` and
`HF_DATASETS_CACHE` to persistent storage:

```bash
export TMPDIR=~/croco/.tmp
export HF_DATASETS_CACHE=~/croco/.hf_datasets_cache
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
```

Without this:

- TRL's precomputed reference log probs cache goes to `/tmp` and can disappear mid-run
  (shared filesystem cleanup policies)
- HuggingFace datasets creates temp Arrow files in `/tmp/hf_datasets-*` regardless of
  `HF_DATASETS_CACHE`

Both failures cause `FileNotFoundError` mid-training. Set these manually or via your
shell profile.

### Memory Optimizations

```yaml
dpo:
    gradient_checkpointing: true # Already default
    activation_offloading: true # +20-30% GPU memory free, ~15% slower
```

### Quality Improvements

```yaml
dpo:
    loss_type: sigmoid_norm # SimPO length-normalized loss (reduces verbosity bias)
    label_smoothing: 0.05 # Robust to noisy reward scores
    use_weighting: true # WPO-style preference weighting
```

**Notes:**

- `sigmoid_norm`: Available in TRL 1.7.0 via
  [`DPOConfig`](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py).
  Implements SimPO's length-normalized loss but still uses reference model.
- `torch_compile`: First run slow (compilation), subsequent runs faster.
- `activation_offloading`: Only enable if OOM — trades speed for memory.

### Algorithm Support (TRL 1.7.0)

| Algorithm | Support                     | Use Case                        |
| --------- | --------------------------- | ------------------------------- |
| **DPO**   | ✅ Full                     | Solid baseline                  |
| **SimPO** | ⚠️ Partial (`sigmoid_norm`) | Length normalization            |
| **KTO**   | ✅ Full                     | If reward noise is an issue     |
| **GRPO**  | ✅ Full                     | If verifiable rewards available |
| **ORPO**  | ❌ Not in TRL 1.7.0         | Skip                            |
