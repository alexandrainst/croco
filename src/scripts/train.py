#!/usr/bin/env python3
"""Train a policy model using DPO on a preference dataset."""

import logging
from pathlib import Path

import click

from croco.config import load_config
from croco.dpo import train_dpo

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
    "--dataset",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/preference_pairs.jsonl"),
    help="Path to preference pairs (JSONL). Defaults to data/preference_pairs.jsonl.",
)
def main(*, config: Path, dataset: Path) -> None:
    """Train a policy model using Direct Preference Optimisation (DPO).

    This script loads a preference dataset and trains the policy model using either
    standard DPO or CurriculumDPO, depending on the configuration. The trained model
    is saved to the output directory specified in the config.
    """
    # Load configuration
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(path=config)

    # Train
    logger.info(f"Training on dataset {dataset}")
    output_path = train_dpo(config=cfg, dataset_path=dataset)

    logger.info(f"Training complete. Model saved to {output_path}")


if __name__ == "__main__":
    main()
