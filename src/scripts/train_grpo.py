#!/usr/bin/env python3
"""Train a policy model online with GRPO against the reward model.

This is the no-preference-construction baseline: unlike ``train.py`` (offline
CroCo DPO), GRPO needs no pre-built dataset - it generates and scores completions
during training. Evaluate the resulting adapter with the EuroEval CLI
(``euroeval -m <model> -l da``) or ``eval_checkpoints.py`` exactly like the DPO runs.
"""

import logging
from pathlib import Path

import click

from croco.config import load_config
from croco.grpo import train_grpo

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
    help="Path to pipeline configuration YAML file (must contain a 'grpo' block).",
)
def main(*, config: Path) -> None:
    """Train a policy model with Group Relative Policy Optimisation (GRPO).

    Loads the configured prompts and trains the policy online against the reward
    model. The trained adapter is saved to the output directory in the config.
    """
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(path=config)

    output_path = train_grpo(config=cfg)

    logger.info(f"Training complete. Model saved to {output_path}")


if __name__ == "__main__":
    main()
