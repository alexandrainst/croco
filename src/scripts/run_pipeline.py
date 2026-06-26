#!/usr/bin/env python3
"""Run the complete CroCo pipeline: build → train → evaluate."""

import logging
from pathlib import Path

import click

from croco.config import load_config
from croco.data import load_examples
from croco.dpo import train_dpo
from croco.evaluation import evaluate_model, extract_scores
from croco.pipeline import build_preference_dataset
from croco.utils import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    This script orchestrates the full pipeline:
    1. Build a preference dataset from the configured data source
    2. Train a policy model using DPO
    3. Evaluate the trained model on EuroEval benchmarks

    Each step can be skipped using the respective flag if you want to run them
    separately or reuse existing artifacts.

    Raises:
        RuntimeError:
          If model output path is not set when evaluation is requested.
    """
    # Load configuration
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(path=config)

    # Set random seed for reproducibility
    set_seed(seed=cfg.data.seed)

    model_output: Path | None = None

    # Step 1: Build dataset
    if not skip_build:
        logger.info("=== Step 1: Building preference dataset ===")

        # Load examples
        logger.info("Loading examples from dataset")
        examples = load_examples(config=cfg.data)
        logger.info(f"Loaded {len(examples)} examples")

        # Import vLLM engines here to avoid module-level imports
        from croco.vllm_generation import VLLMGenerationEngine  # noqa: PLC0415, I001
        from croco.vllm_scoring import VLLMScoringEngine  # noqa: PLC0415, I001

        # Initialise engines
        logger.info("Initialising vLLM generation engine")
        generation_engine = VLLMGenerationEngine(
            model_id=cfg.policy.model_id,
            config=cfg.generation,
            max_model_len=cfg.policy.max_model_len,
        )

        # Scoring engine uses lower GPU memory since generation engine already loaded
        # Use 40% of remaining GPU memory after generation engine
        scoring_gpu_util = min(0.4, 1.0 - cfg.generation.gpu_memory_utilization)
        logger.info(f"Initialising vLLM scoring engine (GPU util: {scoring_gpu_util:.2f})")
        scoring_engine = VLLMScoringEngine(
            config=cfg.reward,
            gpu_memory_utilization=scoring_gpu_util,
        )

        # Build dataset
        logger.info("Building preference dataset")
        build_preference_dataset(
            generation_engine=generation_engine,
            scoring_engine=scoring_engine,
            num_candidates=cfg.generation.num_candidates,
            construction_mode=cfg.construction_mode,
            score_gold_output=cfg.score_gold_output,
            output_path=dataset_output,
            examples=examples,
        )

        logger.info(f"Dataset saved to {dataset_output}")
    else:
        logger.info("Skipping dataset build (using existing dataset)")

    # Step 2: Train
    if not skip_train:
        logger.info("=== Step 2: Training with DPO ===")
        model_output = train_dpo(config=cfg, dataset_path=dataset_output)
        logger.info(f"Model saved to {model_output}")
    else:
        logger.info("Skipping training step")
        model_output = cfg.dpo.output_dir
        if not model_output.exists():
            logger.warning(f"Model output directory {model_output} does not exist")

    # Step 3: Evaluate
    if not skip_eval:
        if model_output is None:
            logger.error("Cannot evaluate: model_output is None")
            msg = (
                "Model output path not set. Either run training or specify --model-dir."
            )
            raise RuntimeError(msg)

        logger.info("=== Step 3: Evaluating model ===")
        results = evaluate_model(model_id_or_path=model_output, config=cfg)

        # Extract and display scores
        scores = extract_scores(results=results)

        logger.info("Evaluation results:")
        for dataset_name, dataset_scores in scores.items():
            logger.info(f"  {dataset_name}:")
            for metric_name, value in dataset_scores.items():
                logger.info(f"    {metric_name}: {value:.4f}")
    else:
        logger.info("Skipping evaluation step")

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
