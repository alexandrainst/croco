#!/usr/bin/env python3
"""Build a preference dataset using the CroCo pipeline."""

import logging
from pathlib import Path

import click

from croco.config import load_config
from croco.data import load_examples
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
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("data/preference_pairs.jsonl"),
    help="Output path for preference pairs. Default: data/preference_pairs.jsonl.",
)
def main(*, config: Path, output: Path) -> None:
    """Build a preference dataset from the configured data source.

    This script loads examples from the Laerebogen dataset, generates candidate
    responses using vLLM, scores them with a reward model, and constructs preference
    pairs sorted by evolution level.
    """
    # Load configuration
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(path=config)

    # Set random seed for reproducibility
    set_seed(seed=cfg.data.seed)

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

    logger.info("Initialising vLLM scoring engine")
    scoring_engine = VLLMScoringEngine(
        config=cfg.reward, gpu_memory_utilization=cfg.reward.gpu_memory_utilization
    )

    # Build dataset
    logger.info("Building preference dataset")
    pairs = build_preference_dataset(
        generation_engine=generation_engine,
        scoring_engine=scoring_engine,
        num_candidates=cfg.generation.num_candidates,
        construction_mode=cfg.construction_mode,
        score_gold_output=cfg.score_gold_output,
        output_path=output,
        examples=examples,
        batch_size=cfg.generation.batch_size,
    )

    logger.info(f"Successfully built dataset with {len(pairs)} pairs")
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
