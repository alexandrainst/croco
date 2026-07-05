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

All executable scripts are in `src/scripts/`. Run on sparkie via:

```bash
bash src/scripts/<script>.sh
```

**Experiment queue (as of 2026-07-05):**

| Session | Script | Purpose | Status |
|---------|--------|---------|--------|
| `stuned_rerun` | `simpo_tuned_queue.sh` | SimPO-tuned rerun | 🔴 Running |
| `sfull` | `simpo_full_queue.sh` | SimPO-full | ⏳ After stuned_rerun |
| `llamarm` | `llama_rm_queue.sh` | Llama-3.1 RM | ⏳ After sfull |
| `grpo` | `grpo_queue.sh` | GRPO baseline | ⏳ After llamarm |
| `reeval3_queue` | `reeval_3iter_queue.sh` | Re-eval monitor | ⏳ After GPU idle |
| — | `update_docs.sh` | Export all plots to `docs/gfx/` | ✅ Ready |

**Completed ablations:**

- `queue` — Construction mode (max_reward, gold_chosen, generated)
- `ls` — Label smoothing (α=0.05)
- `simpo` — Initial SimPO (β=2.0)

**Auto-launch monitors:** Start manually to enable chained execution:

```bash
# Launch auto_sfull when stuned finishes (creates session sfull)
tmux new-session -d -s auto_sfull "bash -lc 'bash ~/croco/src/scripts/auto_launch_sfull.sh 2>&1 | tee ~/croco/auto_sfull_launch.log'"

# Launch llama_rm when sfull finishes (creates session llamarm)
tmux new-session -d -s auto_rm "bash -lc 'bash ~/croco/src/scripts/auto_launch_llama_rm.sh 2>&1 | tee ~/croco/auto_rm_launch.log'"

# Launch grpo when llamarm finishes (creates session grpo)
tmux new-session -d -s auto_grpo "bash -lc 'bash ~/croco/src/scripts/auto_launch_grpo.sh 2>&1 | tee ~/croco/auto_grpo_launch.log'"
```

Monitor logs: `tail -f ~/croco/auto_*_launch.log`

**Checkpoint re-evaluation:** launch `reeval_3iter_queue.sh` as the single
queued path for 3-iteration checkpoint re-evals. It waits behind training
sessions, auto-launch monitors, and active GPU processes before running
`reeval_3iter_checkpoints.sh`. `full_eval_queue.sh` delegates to this same queue
script instead of launching the checkpoint script directly.

```bash
tmux new-session -d -s reeval3_queue \
  "bash -lc 'set -o pipefail; bash ~/croco/src/scripts/reeval_3iter_queue.sh \
  2>&1 | tee ~/croco/reeval_3iter_queue.log'"
```

### Auto-Launch Scripts

Chain experiments automatically when GPU becomes free. Monitors wait for both GPU idle and session inactive for 3 consecutive minutes before launching the next stage.

**Chained workflow:**

`stuned` (SimPO-tuned) → `sfull` (SimPO-full) → `llamarm` (Llama RM) → `grpo` (GRPO baseline)

**Launch monitors:**

```bash
# Launch sfull when stuned finishes
tmux new-session -d -s auto_sfull "bash -lc 'bash ~/croco/src/scripts/auto_launch_sfull.sh 2>&1 | tee ~/croco/auto_sfull_launch.log'"

# Launch llama_rm when sfull finishes (or stuned if sfull skipped)
tmux new-session -d -s auto_rm "bash -lc 'bash ~/croco/src/scripts/auto_launch_llama_rm.sh 2>&1 | tee ~/croco/auto_rm_launch.log'"

# Launch grpo when llamarm finishes  
tmux new-session -d -s auto_grpo "bash -lc 'bash ~/croco/src/scripts/auto_launch_grpo.sh 2>&1 | tee ~/croco/auto_grpo_launch.log'"
```

**Monitor logs:** `tail -f ~/croco/auto_*_launch.log`

**Note:** Auto-launch monitors are NOT running by default. Start after launching the first experiment in the chain.

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

Generate locally from discovered config output directories:

```bash
uv run src/scripts/build_dashboard.py \
  -r euroeval_benchmark_results.jsonl \
  -o croco_dashboard.html
```

For the live local dashboard, `croco-dash` runs the same script in watch mode and
pulls checkpoint/result data from sparkie over SSH.

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

- **`croco-dash` tmux session** — A local tmux session on the laptop refreshes
  `croco_dashboard.html` every 5 minutes by pulling latest checkpoints from
  sparkie. If the dashboard appears stale, check if `croco-dash` is still
  running.
- **Dashboard visibility** — Active training runs do not appear in the dashboard
  until the first `checkpoint-*/trainer_state.json` exists. SimPO-tuned can spend
  over an hour precomputing reference log probs before step 1; with
  `save_steps: 100`, it stays invisible until checkpoint 100 is written and the
  next `croco-dash` refresh runs.
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
- **EuroEval cache** — Results cached in `.euroeval_cache/`. Use
  `src/scripts/reeval_3iter_queue.sh` for prior 3-iteration checkpoint results;
  it calls `src/scripts/reeval_3iter_checkpoints.sh`, the only shell script that
  passes EuroEval `--force`. Normal queue scripts must not use `--force`.
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

- `stuned_rerun` — current SimPO-tuned rerun (β=2.0, sigmoid_norm)
- `stuned` — SimPO-tuned (legacy session name)
- `sfull` — SimPO-full (ref-free, γ=0.5)
- `llamarm` — Llama RM ablation
- `grpo` — GRPO baseline
- `reeval3_queue` — checkpoint re-eval monitor; waits behind GPU work
- `auto_*` — Auto-launch monitors (wait for GPU idle before launching next stage)

Logs: `~/croco/simpo_tuned_rerun.log`, `~/croco/simpo_full_rerun.log`,
`~/croco/reeval_3iter_queue.log`, `~/croco/auto_*_launch.log`

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

