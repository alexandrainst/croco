#!/usr/bin/env python3
"""Evaluate a trained model using EuroEval."""

import logging
from pathlib import Path

import click

from croco.config import load_config
from croco.evaluation import evaluate_model, extract_scores

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
    "--model",
    "-m",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the trained model directory.",
)
def main(*, config: Path, model_path: Path) -> None:
    """Evaluate a trained model using the EuroEval benchmark suite.

    This script benchmarks a trained model on tasks configured for the target
    language and reports aggregated scores for each dataset.
    """
    # Load configuration
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(path=config)

    # Evaluate
    logger.info(f"Evaluating model at {model_path}")
    results = evaluate_model(model_id_or_path=model_path, config=cfg)

    # Extract and display scores
    scores = extract_scores(results=results)

    logger.info("Evaluation results:")
    for dataset_name, dataset_scores in scores.items():
        logger.info(f"  {dataset_name}:")
        for metric_name, value in dataset_scores.items():
            logger.info(f"    {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
