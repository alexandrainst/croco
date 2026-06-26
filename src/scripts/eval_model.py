#!/usr/bin/env python3
"""Evaluate a trained or base model using EuroEval."""

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
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to the trained model directory. "
        "If not provided, evaluates the base policy model from config."
    ),
)
def main(*, config: Path, model_path: Path | None) -> None:
    """Evaluate a model using the EuroEval benchmark suite.

    This script benchmarks a model on tasks configured for the target
    language and reports aggregated scores for each dataset. If no model
    path is provided, evaluates the base policy model from the config.
    """
    # Load configuration
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(path=config)

    # Determine model to evaluate
    model_id_or_path: str | Path = (
        model_path if model_path is not None else cfg.policy.model_id
    )
    logger.info(f"Evaluating model: {model_id_or_path}")

    # Evaluate
    results = evaluate_model(model_id_or_path=model_id_or_path, config=cfg)

    # Extract and display scores
    scores = extract_scores(results=results)

    logger.info("Evaluation results:")
    for dataset_name, dataset_scores in scores.items():
        logger.info(f"  {dataset_name}:")
        for metric_name, value in dataset_scores.items():
            logger.info(f"    {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
