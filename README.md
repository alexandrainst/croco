<!-- This disables the "First line in file should be a top level heading" rule -->
<!-- markdownlint-disable MD041 -->
<a href="https://github.com/alexandrainst/croco">
<img
 src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/alexandra/alexandra-logo.jpeg"
 width="239"
 height="175"
 align="right"
 alt="Alexandra Institute Logo"
/>
</a>

# Croco

Experiments with the CroCo post-training method.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/croco/tree/main/tests)
[![License](https://img.shields.io/github/license/alexandrainst/croco)](https://github.com/alexandrainst/croco/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/croco)](https://github.com/alexandrainst/croco/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/croco/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Dan Saattrup Smart (<dan.smart@alexandra.dk>)

## Setup

### Installation

1. Run `make install`, which sets up a virtual environment and all Python dependencies
   therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

#### Dependency Notes

- **vLLM** is an optional dependency for GPU-based generation and scoring. It is NOT
  installed by default on macOS (requires CUDA). To install on a DGX/CUDA host, run
  `make install-vllm`.
- **EuroEval** is installed by default. Do NOT add `flash_attn` as it causes EuroEval to
  hard-exit on import.

### Adding and Removing Packages

To install new PyPI packages, run:

```bash
uv add <package-name>
```

To remove them again, run:

```bash
uv remove <package-name>
```

To show all installed packages, run:

```bash
uv pip list
```

## All Built-in Commands

The project includes the following convenience commands:

- `make install`: Install the project and its dependencies in a virtual environment.
- `make install-pre-commit`: Install pre-commit hooks for linting, formatting and type
  checking.
- `make check`: Lint and format the code using `ruff`, and type check using `pyrefly`.
- `make test`: Run tests using `pytest` and update the coverage badge in the readme.
- `make docker`: Build a Docker image and run the Docker container.
- `make tree`: Show the project structure as a tree.

## A Word on Modules and Scripts

In the `src` directory there are two subdirectories, `croco`
and `scripts`. This is a brief explanation of the differences between the two.

### Modules

All Python files in the `croco` directory are _modules_
internal to the project package. Examples here could be a general data loading script,
a definition of a model, or a training function. Think of modules as all the building
blocks of a project.

When a module is importing functions/classes from other modules we use the _relative
import_ notation - here's an example:

```python
from .other_module import some_function
```

### Scripts

Python files in the `scripts` folder are scripts, which are short code snippets that
are _external_ to the project package, and which is meant to actually run the code. As
such, _only_ scripts will be called from the terminal. An analogy here is that the
internal `numpy` code are all modules, but the Python code you write where you import
some `numpy` functions and actually run them, that a script.

When importing module functions/classes when you're in a script, you do it like you
would normally import from any other package:

```python
from croco import some_function
```

Note that this is also how we import functions/classes in tests, since each test Python
file is also a Python script, rather than a module.

## Pipeline Overview

The CroCo pipeline implements the **Contrastive Preference Optimization** post-training
method. It consists of three stages: **build**, **train**, and **evaluate**.

### Pipeline Modes

The pipeline supports two operational modes, configured via `construction_mode` in the
YAML config:

- **`generated`**: Generate candidate responses using a policy model, then score them
  using a reward model to construct preference pairs.
- **`existing`**: Use pre-existing candidate responses from the dataset (no generation
  or scoring required).

### Curriculum Learning

When `dpo.curriculum: true`, the pipeline implements **gated access** curriculum
learning:

1. Examples are sorted by their evolution score (difficulty)
2. Training starts with only the easiest examples
3. As the model improves, harder examples are gradually unlocked
4. This stabilises training and improves convergence on challenging examples

Gating is controlled by the `evolution_threshold` parameter, which increases over
training steps.

### Scripts

All scripts are run with `uv run` from the project root:

```bash
# Build the preference dataset (generation + scoring)
uv run src/scripts/build_dataset.py --config config/danish.yaml

# Train using DPO with curriculum learning
uv run src/scripts/train.py --config config/danish.yaml

# Evaluate the trained model
uv run src/scripts/eval_model.py --config config/danish.yaml --model path/to/model

# Run the full pipeline (build -> train -> evaluate)
uv run src/scripts/run_pipeline.py --config config/danish.yaml
```

### Configuration

The default configuration is `config/danish.yaml`. Key sections:

- **`construction_mode`**: `generated` or `existing`
- **`policy`**: Policy model settings (Gemma-3-12B-IT default)
- **`reward`**: Reward model settings (Skywork-Reward-V2-Qwen3-8B default)
- **`generation`**: vLLM generation parameters (candidates, temperature, etc.)
- **`data`**: Dataset configuration (Laerebogen, stratification, sample count)
- **`dpo`**: Training hyperparameters, curriculum settings, LoRA config
- **`eval`**: Evaluation tasks and iterations

### GPU Requirements

The `generated` mode requires **vLLM** for candidate generation and reward scoring,
which needs CUDA. On macOS (no CUDA), vLLM is excluded and only `existing` mode works.

To install vLLM on a DGX/CUDA host:

```bash
make install-vllm
```

This installs the `vllm` extra with the correct CUDA dependencies for Python 3.12.

## Features

### Docker Setup

A Dockerfile is included in the new repositories, which by default runs
`src/scripts/main.py`. You can build the Docker image and run the Docker container by
running `make docker`.

### Automatic Test Coverage Calculation

Run `make test` to test your code, which also updates the "coverage badge" in the
README, showing you how much of your code base that is currently being tested.

### Continuous Integration

Github CI pipelines are included in the repo, running all the tests in the `tests`
directory, as well as building online documentation, if Github Pages has been enabled
for the repository (can be enabled on Github in the repository settings).

### Code Spaces

Code Spaces is a new feature on Github, that allows you to develop on a project
completely in the cloud, without having to do any local setup at all. This repo comes
included with a configuration file for running code spaces on Github. When hosted on
`alexandrainst/croco` then simply press the `<> Code` button
and add a code space to get started, which will open a VSCode window directly in your
browser.
