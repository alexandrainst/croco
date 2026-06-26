#!/usr/bin/env python3
"""Run the complete CroCo pipeline: build -> train -> evaluate.

Each heavy phase runs in its own subprocess. This is essential on unified-memory
systems (e.g. the DGX Spark / GB10), where GPU memory and system RAM share a single
pool: vLLM holds a large fraction of memory during dataset construction and does not
reliably release it in-process. Running each phase as a separate process guarantees
the OS reclaims all memory before the next phase begins, avoiding whole-system OOM.
"""

import logging
import subprocess
import sys
from pathlib import Path

import click

from croco.config import load_config
from croco.pipeline import upload_to_huggingface

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to pipeline configuration YAML file.",
)
@click.option(
    "--dataset-output",
    type=click.Path(path_type=Path),
    default=Path("data/preference_pairs.jsonl"),
    help="Output path for preference pairs. Default: data/preference_pairs.jsonl.",
)
@click.option(
    "--skip-build",
    is_flag=True,
    help="Skip dataset building step (use existing dataset).",
)
@click.option("--skip-train", is_flag=True, help="Skip training step.")
@click.option("--skip-eval", is_flag=True, help="Skip evaluation step.")
def main(
    *,
    config: Path,
    dataset_output: Path,
    skip_build: bool,
    skip_train: bool,
    skip_eval: bool,
) -> None:
    """Run the complete CroCo post-training pipeline.

    This script orchestrates the full pipeline by invoking the standalone phase
    scripts as separate subprocesses:

    1. Build a preference dataset from the configured data source.
    2. Train a policy model using DPO.
    3. Evaluate the trained model on EuroEval benchmarks.

    Running each phase in its own process ensures vLLM and training memory are fully
    released between phases, which is required on unified-memory systems.

    Each step can be skipped using the respective flag.

    Args:
        config:
          Path to the pipeline configuration YAML file.
        dataset_output:
          Path where preference pairs are written and read.
        skip_build:
          Whether to skip the dataset build step.
        skip_train:
          Whether to skip the training step.
        skip_eval:
          Whether to skip the evaluation step.
    """
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(path=config)

    model_output = cfg.dpo.output_dir

    if not skip_build:
        logger.info("=== Step 1: Building preference dataset ===")
        _run_script(
            script="build_dataset.py",
            args=["--config", str(config), "--output", str(dataset_output)],
        )
        logger.info(f"Dataset saved to {dataset_output}")
    else:
        logger.info("Skipping dataset build (using existing dataset)")

    if not skip_train:
        logger.info("=== Step 2: Training with DPO ===")
        _run_script(
            script="train.py",
            args=["--config", str(config), "--dataset", str(dataset_output)],
        )
        logger.info(f"Model saved to {model_output}")
    else:
        logger.info("Skipping training step")
        if not model_output.exists():
            logger.warning(f"Model output directory {model_output} does not exist")

    if cfg.dpo.hf_repo_id:
        logger.info("=== Uploading to Hugging Face Hub ===")
        upload_to_huggingface(
            dataset_path=dataset_output,
            model_path=model_output,
            repo_id=cfg.dpo.hf_repo_id,
            private=False,
        )

    if not skip_eval and not cfg.eval.skip:
        logger.info("=== Step 3: Evaluating model ===")
        _run_script(
            script="eval_model.py",
            args=["--config", str(config), "--model", str(model_output)],
        )
    else:
        logger.info("Skipping evaluation step")

    logger.info("Pipeline complete")


def _run_script(*, script: str, args: list[str]) -> None:
    """Run a phase script as a separate subprocess.

    Args:
        script:
          Filename of the script in the scripts directory to run.
        args:
          Command-line arguments to pass to the script. The subprocess is run with
          ``check=True``, so a non-zero exit aborts the pipeline.
    """
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
