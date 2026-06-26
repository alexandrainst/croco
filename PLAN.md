# CroCo training pipeline — implementation plan

## Context

This repo (`alexandrainst/croco`) is a fresh `uv`/Alexandra-template scaffold for
"experiments with the CroCo post-training method". CroCo (_Cross-Lingual Contrastive
Preference Tuning on Self-Generations_, Zhang/Basirat/Elliott, arXiv 2605.26293;
reference code `github.com/jjzha/CroCo`) has a policy model self-generate K responses
per prompt, scores them with an off-the-shelf reward model, builds **contrastive
preference pairs** via the paper's Eq. (2), and **DPO-trains** the policy on them.
Trained models are evaluated with **EuroEval**.

We adapt CroCo to Danish. **No translation step** — we use the Danish
`danish-foundation-models/laerebogen` dataset (`evolved` config, ~4.98M rows; columns
`instruction: str`, `output: str`, `evolution: int` (Evol-Instruct difficulty),
`hash: str`). The dataset is **gated**: the runner must accept its terms on the Hub and
provide an `HUGGINGFACE_API_KEY`. Heavy steps (generation, scoring, DPO) run on the
user's **DGX Spark** (CUDA); the **Mac** is for development and the test suite. Remote
DGX orchestration is a _later_ step; for now the scripts must run on the DGX over SSH.

This plan is intentionally prescriptive: every module lists its public symbols,
signatures, and the non-obvious logic in full. Follow the python skill conventions
throughout (British English; `import typing as t` / `import collections.abc as c`;
`list[T]`/`X | None`; Google docstrings with a blank line after each arg name;
keyword-only calls; no `print` (use a module-level
`logger = logging.getLogger(__name__)`); relative imports inside `src/croco`, absolute
imports in `src/scripts` and `tests`; max line length 88; high-level functions first,
helpers below; protected helpers prefixed with `_`).

### Decisions (confirmed with user — do not revisit)

- **Inference engine: vLLM only.** The two `vllm_*.py` modules are the only GPU-only
  code and are excluded from Mac test collection (see `conftest.py`). All other modules
  import cleanly on the Mac and are fully tested there.
- **Default policy model: `google/gemma-3-12b-it`** (configurable).
- **Reward model: `Skywork/Skywork-Reward-V2-Qwen3-8B`.** Used in both modes.
- **DPO via `trl`** (`DPOTrainer`/`DPOConfig`). DPO only; no SFT.
- **Data: `danish-foundation-models/laerebogen`, config `evolved`.** No translation.
- **Two construction modes** (`construction_mode` config field):
  - `generated`: prompt = `instruction`; policy generates K candidates; RM scores
      them; chosen/rejected via Eq. (2). Dataset `output` unused.
  - `gold_chosen`: prompt = `instruction`; **chosen = dataset `output`**; policy
      generates K candidates, RM scores them; **rejected** = generation nearest
      `mu - 2*sigma`; the gold `output` is RM-scored too so we can require
      `rejected_score < chosen_score`.
- **Curriculum** (`curriculum` config field, default `true`): pairs are saved sorted by
  ascending `evolution`, and training does not shuffle (sequential sampler).

---

## Step 0 — dependencies & tooling (do first)

Run from repo root:

```bash
uv add pydantic pyyaml numpy datasets huggingface-hub click \
       transformers torch trl peft accelerate euroeval
uv add --optional vllm vllm        # DGX-only extra; NOT installed on the Mac
```

- Do **not** add `flash_attn` (EuroEval hard-exits on import if it is installed).
- Edit `makefile` target `install-dependencies`: change
  `uv sync --all-extras --all-groups --python 3.12` to
  `uv sync --all-groups --python 3.12` (so the `vllm` extra is not pulled on the Mac).
  Add a new target:

    ```make
    install-vllm: ## Install the vLLM extra (run on the DGX/CUDA host only)
     @uv sync --all-groups --extra vllm --python 3.12
    ```

- `.env.example`: append a line `HUGGINGFACE_API_KEY=`. In
  `src/scripts/fix_dot_env_file.py`, add to `DESIRED_ENVIRONMENT_VARIABLES`:
  `HUGGINGFACE_API_KEY="Enter your Hugging Face access token (for gated models/datasets):\n> "`.
- If `uv add` reports a resolution conflict between `euroeval`, `trl`, and
  `transformers`, relax by letting uv pick compatible versions (do not pin); if still
  conflicting, move `euroeval` into its own extra `--optional eval euroeval` and
  document `uv sync --extra eval` for the eval host. Record whatever you did in
  `README.md`.

Verify after coding: `make check` and `make test` both pass on the Mac.

---

## File-by-file specification

All modules live in `src/croco/`. Each begins with a one-line module docstring and
`logger = logging.getLogger(__name__)` where it logs.

### `src/croco/data_models.py`

Pydantic v2 `BaseModel`s (frozen where natural). No heavy imports.

```python
import typing as t
from pydantic import BaseModel

class ScoredCandidate(BaseModel):
    """A single self-generated response together with its reward-model score."""
    response: str
    reward_score: float

class DataExample(BaseModel):
    """One source row from laerebogen used to build a preference pair."""
    instruction: str
    output: str
    evolution: int | None = None
    hash: str | None = None

class PreferencePair(BaseModel):
    """A constructed (chosen, rejected) preference pair for DPO."""
    prompt: str
    chosen: str
    rejected: str
    rejected_score: float
    chosen_score: float | None = None
    evolution: int | None = None
    pool_size: int
    mode: t.Literal["generated", "gold_chosen"]
    hash: str | None = None
```

### `src/croco/preference.py` — CroCo Eq. (2), pure (MOST IMPORTANT)

Full implementation (copy verbatim, adjust docstrings):

```python
"""Construction of contrastive preference pairs (CroCo, Eq. 2)."""

import numpy as np

from .data_models import PreferencePair, ScoredCandidate


def build_pair_generated(
    *, prompt: str, candidates: list[ScoredCandidate], evolution: int | None = None,
    hash: str | None = None,
) -> PreferencePair | None:
    """Build a pair where both chosen and rejected are self-generations.

    Chosen is the highest-reward candidate; rejected is the candidate whose reward
    is nearest to ``mu - 2*sigma`` and strictly below the chosen reward.

    Args:
        prompt:
          The instruction prompt.
        candidates:
          The scored self-generations for this prompt.
        evolution (optional):
          The source difficulty level. Defaults to None.
        hash (optional):
          The source row hash. Defaults to None.

    Returns:
        The preference pair, or None if no valid pair exists (pool < 2 or no
        candidate lies strictly below the chosen reward).
    """
    if len(candidates) < 2:
        return None
    chosen = max(candidates, key=lambda candidate: candidate.reward_score)
    rejected = _select_rejected(
        pool=candidates, upper_bound=chosen.reward_score, exclude=chosen
    )
    if rejected is None:
        return None
    return PreferencePair(
        prompt=prompt, chosen=chosen.response, rejected=rejected.response,
        chosen_score=chosen.reward_score, rejected_score=rejected.reward_score,
        evolution=evolution, pool_size=len(candidates), mode="generated", hash=hash,
    )


def build_pair_gold_chosen(
    *, prompt: str, gold_output: str, candidates: list[ScoredCandidate],
    gold_score: float | None = None, evolution: int | None = None,
    hash: str | None = None,
) -> PreferencePair | None:
    """Build a pair where chosen is the dataset's gold output.

    Rejected is the self-generation nearest to ``mu - 2*sigma`` (statistics taken
    over the generations only). If ``gold_score`` is given, the rejected reward
    must be strictly below it.

    Args:
        prompt:
          The instruction prompt.
        gold_output:
          The dataset's reference completion, used as the chosen response.
        candidates:
          The scored self-generations for this prompt.
        gold_score (optional):
          Reward-model score of ``gold_output``, used as an upper bound on the
          rejected reward. Defaults to None.
        evolution (optional):
          The source difficulty level. Defaults to None.
        hash (optional):
          The source row hash. Defaults to None.

    Returns:
        The preference pair, or None if no valid rejected generation exists.
    """
    if len(candidates) < 1:
        return None
    rejected = _select_rejected(
        pool=candidates, upper_bound=gold_score, exclude=None
    )
    if rejected is None:
        return None
    return PreferencePair(
        prompt=prompt, chosen=gold_output, rejected=rejected.response,
        chosen_score=gold_score, rejected_score=rejected.reward_score,
        evolution=evolution, pool_size=len(candidates), mode="gold_chosen", hash=hash,
    )


def _select_rejected(
    *, pool: list[ScoredCandidate], upper_bound: float | None,
    exclude: ScoredCandidate | None,
) -> ScoredCandidate | None:
    """Pick the candidate nearest ``mu - 2*sigma`` subject to constraints.

    ``mu`` and ``sigma`` are the mean and population standard deviation (ddof=0)
    of the pool's rewards, matching the reference implementation.

    Args:
        pool:
          Candidates over which the reward statistics are computed and from which
          the rejected response is drawn.
        upper_bound (optional):
          If given, only candidates with reward strictly below this are eligible.
        exclude (optional):
          A candidate object to skip (identity comparison), e.g. the chosen one.

    Returns:
        The eligible candidate closest to the target reward, or None.
    """
    scores = np.array([candidate.reward_score for candidate in pool], dtype=float)
    target = float(scores.mean() - 2.0 * scores.std())  # population std (ddof=0)
    for candidate in sorted(pool, key=lambda c: abs(c.reward_score - target)):
        if candidate is exclude:
            continue
        if upper_bound is not None and candidate.reward_score >= upper_bound:
            continue
        return candidate
    return None
```

### `src/croco/engines.py` — backend Protocols (pure)

```python
import typing as t

class GenerationEngine(t.Protocol):
    def generate(
        self, *, prompts: list[str], num_candidates: int
    ) -> list[list[str]]:
        """Return, for each prompt, ``num_candidates`` response strings."""
        ...

class ScoringEngine(t.Protocol):
    def score(self, *, prompts: list[str], responses: list[str]) -> list[float]:
        """Reward-score each parallel ``(prompt, response)`` pair."""
        ...
```

### `src/croco/utils.py` — small helpers (pure)

Public functions:

- `setup_logging(*, level: int = logging.INFO) -> None`
- `set_seed(*, seed: int) -> None` (seeds `random`, `numpy`; torch optional but import
  torch at top — torch imports fine on the Mac).
- `read_jsonl(*, path: Path) -> list[dict[str, t.Any]]`
- `write_jsonl(*, path: Path, rows: c.Iterable[dict[str, t.Any]]) -> None`
- `build_conversation(*, instruction: str, response: str) -> list[dict[str, str]]`
  returns
  `[{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]`.
- `build_user_message(*, instruction: str) -> list[dict[str, str]]` returns
  `[{"role": "user", "content": instruction}]`.

### `src/croco/data.py` — laerebogen loader & curriculum (uses `datasets`)

Public functions (import `datasets` at top — fine on the Mac):

- `load_examples(*, config: "DataConfig") -> list[DataExample]`
    1. `ds = datasets.load_dataset(config.dataset_id, name=config.subset, split=config.split)`
    2. apply `evolution_min`/`evolution_max` filter if set;
    3. drop rows whose `instruction` or `output` is empty;
    4. subsample to `config.num_samples` via `_subsample(...)` (stratified by
       `evolution` when `config.stratify_by_evolution`, else uniform), using
       `config.seed`;
    5. map rows to `DataExample(instruction=..., output=..., evolution=..., hash=...)`.
- `sort_by_evolution(*, pairs: list[PreferencePair]) -> list[PreferencePair]` stable
  sort ascending by `evolution` (treat `None` as `-inf` so they come first).
- `_subsample(*, ds, num_samples, stratify_by_evolution, seed) -> datasets.Dataset`
  stratified: group indices by `evolution`, take `round(num_samples * group_frac)` per
  group with a seeded `random.Random(seed)` shuffle; uniform:
  `ds.shuffle(seed=seed).select(range(n))`. If `num_samples >= len(ds)`, return all
  rows.

### `src/croco/dataset.py` — preference dataset IO + TRL formatting (pure-ish)

- `save_pairs(*, pairs: list[PreferencePair], path: Path) -> None` — writes jsonl via
  `utils.write_jsonl` using `pair.model_dump()`.
- `load_pairs(*, path: Path) -> list[PreferencePair]` — reads jsonl and validates each
  row through `PreferencePair`.
- `to_trl_records(*, pairs: list[PreferencePair]) -> list[dict[str, t.Any]]` — converts
  each pair to TRL conversational DPO format:
  `{"prompt": [{"role":"user","content": pair.prompt}],   "chosen": [{"role":"assistant","content": pair.chosen}],   "rejected": [{"role":"assistant","content": pair.rejected}]}`.

### `src/croco/pipeline.py` — orchestration (DI; pure logic, no vLLM import)

```python
def build_preference_dataset(
    *, config: PipelineConfig, examples: list[DataExample],
    generation_engine: GenerationEngine, scoring_engine: ScoringEngine,
) -> list[PreferencePair]:
```

Logic:

1. `prompts = [ex.instruction for ex in examples]`.
2. `generations = generation_engine.generate(prompts=prompts, num_candidates=config.generation.num_candidates)`
   (a `list[list[str]]`, one inner list per prompt).
3. Flatten generations into parallel `(flat_prompts, flat_responses)` and call
   `scoring_engine.score(...)`; regroup into `list[list[ScoredCandidate]]` per prompt.
4. If `config.construction_mode == "gold_chosen"` and `config.score_gold_output`, also
   score each `(ex.instruction, ex.output)` to obtain `gold_scores` (a second
   `scoring_engine.score` call); else `gold_scores = [None] * len(examples)`.
5. Per example, dispatch on `config.construction_mode`:
    - `generated` -> `build_pair_generated(prompt, candidates, evolution, hash)`;
    - `gold_chosen` ->
      `build_pair_gold_chosen(prompt, gold_output=ex.output, candidates, gold_score=gold_scores[i], evolution, hash)`.
6. Collect non-`None` pairs; log how many of `len(examples)` survived; return them
   (unsorted — the build script sorts for curriculum before saving).

### `src/croco/dpo.py` — TRL DPO training + curriculum (uses transformers/trl/peft)

Imports at top: `torch`, `transformers` (`AutoModelForCausalLM`, `AutoTokenizer`),
`datasets`, `trl` (`DPOTrainer`, `DPOConfig`), `peft` (`LoraConfig`),
`torch.utils.data.SequentialSampler`.

- `class CurriculumDPOTrainer(DPOTrainer)`: override `_get_train_sampler` to return
  `SequentialSampler(self.train_dataset)` (accept and ignore any args for cross-version
  compatibility: `def _get_train_sampler(self, *args, **kwargs)`).
- `build_lora_config(*, config: DPOTrainConfig) -> LoraConfig` with `r=config.lora_r`,
  `lora_alpha=config.lora_alpha`, `lora_dropout=config.lora_dropout`,
  `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`,
  `bias="none"`, `task_type="CAUSAL_LM"`.
- `build_dpo_config(*, config: DPOTrainConfig) -> DPOConfig` mapping every field below.
- `train_dpo(*, config: PipelineConfig, dataset_path: Path) -> Path`: load pairs ->
  `to_trl_records` -> `datasets.Dataset.from_list(...)`; load tokenizer + model
  (`AutoModelForCausalLM.from_pretrained(config.policy.model_id, attn_implementation=config.policy.attn_implementation, dtype="auto", use_cache=not gradient_checkpointing)`);
  pick `CurriculumDPOTrainer` when `config.dpo.curriculum` else `DPOTrainer`;
  `ref_model=None`, `peft_config=build_lora_config(...)`; `trainer.train()`;
  `trainer.save_model(config.dpo.output_dir)`; return the output dir.

### `src/croco/evaluation.py` — EuroEval wrapper (uses `euroeval`)

```python
from euroeval import Benchmarker

def evaluate_model(*, model_id_or_path: str | Path, config: PipelineConfig) -> list:
    """Benchmark a model with EuroEval, restricted to the configured language."""
    benchmarker = Benchmarker(
        language=config.eval.language, progress_bar=True, save_results=True,
        num_iterations=config.eval.num_iterations,
    )
    return list(
        benchmarker.benchmark(model=str(model_id_or_path), task=config.eval.tasks)
    )

def extract_scores(*, results: list) -> dict[str, dict[str, float]]:
    """Map each result's dataset name to its aggregated ``results['total']`` dict."""
    return {r.dataset: dict(r.results["total"]) for r in results}
```

Notes: `Benchmarker` has **no** `model` or `batch_size` ctor arg — model goes to
`benchmark(model=...)`. `task=None` evaluates all datasets for the language; pass a list
of task-name strings to restrict. Language code is `"da"` (not `"danish"`).

### `src/croco/vllm_generation.py` — `VLLMGenerationEngine` (GPU only, excluded on Mac)

Top-level `import vllm` and `from vllm import LLM, SamplingParams`;
`from transformers import AutoTokenizer`. Constructor builds the `LLM` once
(`tensor_parallel_size`, `gpu_memory_utilization=0.95`, `max_model_len`,
`trust_remote_code=True`, `enable_prefix_caching=True`) and loads the tokenizer.
`generate(...)`:

- render each prompt:
  `tokenizer.apply_chat_template(build_user_message(instruction=p), tokenize=False, add_generation_prompt=True)`;
- `params = SamplingParams(n=num_candidates, max_tokens=config.generation.max_tokens, temperature=config.generation.temperature, top_p=config.generation.top_p)`;
- `outputs = self.llm.generate(rendered_prompts, params)`;
- return `[[o.text for o in out.outputs] for out in outputs]`.

### `src/croco/vllm_scoring.py` — `VLLMScoringEngine` (GPU only, excluded on Mac)

Top-level `import vllm`; `from vllm import LLM, PoolingParams`;
`from transformers import AutoTokenizer`. Constructor:
`LLM(model=reward.model_id, runner="pooling", enforce_eager=True, max_model_len=reward.max_model_len, tensor_parallel_size=..., trust_remote_code=True, gpu_memory_utilization=0.95, enable_chunked_prefill=False)`.
`score(...)`:

- build `prompt_token_ids` per pair via
  `tokenizer.apply_chat_template(build_conversation(instruction=p, response=r), tokenize=False, add_generation_prompt=False)`;
- `outputs = self.llm.encode(rendered, pooling_params=PoolingParams(activation=False), pooling_task="classify")`;
- return `[float(o.outputs.data.item()) for o in outputs]` (fallback `-999.0` on
  failure, matching the reference).

---

## `config/danish.yaml` (default config — exact content)

```yaml
construction_mode: generated # or: gold_chosen
score_gold_output: true # only used in gold_chosen mode
language: da

policy:
    model_id: google/gemma-3-12b-it
    attn_implementation: sdpa # set flash_attention_2 on the DGX
    max_model_len: 4096

reward:
    model_id: Skywork/Skywork-Reward-V2-Qwen3-8B
    max_model_len: 32768

generation:
    num_candidates: 64
    max_tokens: 4096
    temperature: 0.7
    top_p: 1.0
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.95

data:
    dataset_id: danish-foundation-models/laerebogen
    subset: evolved
    split: train
    num_samples: 20000
    stratify_by_evolution: true
    evolution_min: null
    evolution_max: null
    max_prompt_tokens: 4096
    seed: 42

dpo:
    output_dir: models/croco-gemma-3-12b-da
    learning_rate: 5.0e-6
    lr_scheduler_type: cosine
    warmup_ratio: 0.05
    weight_decay: 0.01
    beta: 0.1
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    num_train_epochs: 1
    max_length: 4096
    bf16: true
    gradient_checkpointing: true
    curriculum: true
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    seed: 42

eval:
    language: da
    tasks: null # null => all Danish datasets
    num_iterations: 10
```

`config.py` mirrors this exactly with pydantic models (`PolicyModelConfig`,
`RewardModelConfig`, `GenerationConfig`, `DataConfig`, `DPOTrainConfig`, `EvalConfig`,
`PipelineConfig`) and `load_config(*, path: Path) -> PipelineConfig` that does
`PipelineConfig(**yaml.safe_load(path.read_text()))`. `DPOTrainConfig.output_dir` is a
`pathlib.Path`. Validate `construction_mode` as `t.Literal["generated", "gold_chosen"]`.

---

## Scripts (`src/scripts/`, run with `uv run`, click CLIs, absolute imports)

- `build_dataset.py` — `--config PATH` (default `config/danish.yaml`), `--output PATH`
  (default from config). Loads config, `set_seed`, `load_examples`, constructs
  `VLLMGenerationEngine` + `VLLMScoringEngine`, calls
  `pipeline.build_preference_dataset`, `sort_by_evolution` when `config.dpo.curriculum`,
  `save_pairs`.
- `train.py` — `--config PATH`, `--dataset PATH`. Calls `dpo.train_dpo`.
- `evaluate.py` — `--config PATH`, `--model PATH-OR-ID`. Calls
  `evaluation.evaluate_model` then logs `extract_scores`.
- `run_pipeline.py` — `--config PATH`. Runs build -> train -> evaluate in sequence.

The vLLM engines are imported **inside** `build_dataset.py`/`run_pipeline.py` only
(scripts are not collected by pytest), so the Mac test run never imports vLLM.

---

## Tests (`tests/`, absolute imports, pytest)

### `conftest.py` (repo root)

```python
import importlib.util
collect_ignore_glob: list[str] = []
if importlib.util.find_spec("vllm") is None:
    collect_ignore_glob.append("src/croco/vllm_*.py")
```

Plus shared fixtures: `make_scored(scores: list[float]) -> list[ScoredCandidate]`, a
`fake_generation_engine`, and a `fake_scoring_engine` (deterministic: e.g. score =
response length, or a lookup keyed on response text) for the integration test.

### `test_preference.py` (worked numbers — must assert these exactly)

- Scores `[1, 2, 3, 10]`: `mean=4.0`, population `std=sqrt(12.5)≈3.5355`,
  `target≈-3.0711`. `build_pair_generated` -> chosen score `10`, **rejected score `1`**.
- All-equal `[5, 5, 5]`: chosen score `5`; no candidate strictly below `5` -> returns
  `None`.
- Single candidate `[7]`: `build_pair_generated` -> `None` (pool < 2).
- `gold_chosen`, `gold_score=8`, candidates `[1, 2, 3, 10]`: rejected score `1` (target
  as above; `10 >= 8` excluded anyway, nearest eligible is `1`).
- `gold_chosen`, `gold_score=None`, candidates `[1, 2, 3, 10]`: rejected score `1`.
- `gold_chosen`, `gold_score=0.5`, candidates `[1, 2, 3, 10]`: every candidate `>= 0.5`
  -> returns `None`.

### `test_data.py`

Build an in-memory
`datasets.Dataset.from_dict({"instruction": [...], "output": [...], "evolution": [...], "hash": [...]})`,
monkeypatch `datasets.load_dataset` to return it, and assert: uniform + stratified
subsample sizes and determinism under a fixed seed; evolution min/max filtering;
empty-row dropping; `sort_by_evolution` orders ascending with `None` first and is
stable.

### `test_dataset.py`

`save_pairs` then `load_pairs` round-trips identical objects; `to_trl_records` produces
the exact chat-list schema above.

### `test_config.py`

`load_config(path=Path("config/danish.yaml"))` parses; defaults match; an invalid
`construction_mode` raises `pydantic.ValidationError`; `output_dir` is a `Path`.

### `test_data_models.py`

Field validation and `model_dump`/round-trip for `PreferencePair`.

### `test_dpo.py`

`build_lora_config` and `build_dpo_config` produce the exact hyperparameters from the
config (no training run). `to_trl_records` + `CurriculumDPOTrainer._get_train_sampler`
returns a `SequentialSampler` (can instantiate the trainer with a tiny dummy dataset, or
test the sampler method in isolation against a stub with a `train_dataset` attribute).

### `test_evaluation.py`

Monkeypatch `croco.evaluation.Benchmarker` with a fake whose `benchmark(...)` records
kwargs and returns objects exposing `.dataset` and `.results`. Assert `evaluate_model`
constructs `Benchmarker(language="da", ...)`, passes `model=<path>` and
`task=config.eval.tasks` to `benchmark`, and that `extract_scores` maps dataset ->
`results["total"]`.

### `test_pipeline.py` (integration, both modes, no GPU)

Drive `build_preference_dataset` with the fake engines for `generated` and
`gold_chosen`; assert the surviving pair count, that `chosen` equals the gold output in
`gold_chosen` mode, and that rejected selection matches `preference.py` on the fake
scores.

---

## Implementation approach (subagent-driven, per user request)

Keep the high-signal core in the main context (`config.py`, `data_models.py`,
`preference.py`, `pipeline.py`, `conftest.py`, and final wiring). Delegate the rest to
parallel subagents grouped to avoid file conflicts, each given this plan section
verbatim plus the python-skill conventions:

1. data layer — `data.py`, `dataset.py`, `utils.py` + `test_data.py`, `test_dataset.py`.
2. training/eval — `dpo.py`, `evaluation.py` + `test_dpo.py`, `test_evaluation.py`.
3. vLLM concretions — `vllm_generation.py`, `vllm_scoring.py` (no Mac tests). Then wire
   `src/scripts/*`, run `make check` + `make test`, fix fallout.

---

## Verification

- `make check` (ruff format/lint incl. `D`/`ANN`/`PLC0415`/`I` + `ty`) passes on the
  Mac.
- `make test` passes: `test_preference.py` proves Eq. (2) numerically (both modes);
  `test_pipeline.py` builds full preference datasets (both modes) with fake engines;
  `test_data.py` proves curriculum sort + stratified subsample; `test_evaluation.py`
  asserts EuroEval is invoked with `language="da"`.
- Manual DGX smoke test (documented, not run here): tiny config (small policy model,
  e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`, `num_samples: 8`, `num_candidates: 4`) via
  `uv run src/scripts/run_pipeline.py --config config/danish.yaml` produces a jsonl pair
  dataset, a LoRA adapter under `dpo.output_dir`, and
  `euroeval_benchmark_results.jsonl`.
- README updated: pipeline overview, the two modes, the curriculum, gated-dataset access
  (`HUGGINGFACE_API_KEY` + accept terms), `make install-vllm` on the DGX, and a
  placeholder section for the later remote-execution work.
