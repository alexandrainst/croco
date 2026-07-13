#!/usr/bin/env python3
"""Build a preference dataset using the CroCo pipeline."""

import logging
from pathlib import Path

import click
from transformers import AutoTokenizer

from croco.config import load_config
from croco.data import filter_by_prompt_length, load_examples
from croco.pipeline import build_preference_dataset
from croco.utils import build_user_message, set_seed

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
@click.option(
    "--candidate-cache",
    type=click.Path(path_type=Path),
    default=Path("data/candidates_cache.jsonl"),
    help=(
        "Path to the raw-candidate cache. Reused across runs with a matching "
        "generation config so modes can share one generation pass."
    ),
)
def main(*, config: Path, output: Path, candidate_cache: Path) -> None:
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

    # Drop prompts that would not leave room for generation in the context window.
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy.model_id)

    def count_prompt_tokens(instruction: str) -> int:
        rendered = tokenizer.apply_chat_template(  # ty: ignore[unresolved-attribute]
            build_user_message(instruction=instruction),
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer(str(rendered))["input_ids"])  # ty: ignore[call-non-callable]

    examples = filter_by_prompt_length(
        examples=examples,
        count_tokens=count_prompt_tokens,
        max_prompt_tokens=cfg.data.max_prompt_tokens,
    )
    logger.info(f"{len(examples)} examples within prompt-length budget")

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

    # Fingerprint the generation config so cached candidates are only reused when
    # they were produced with the same settings.
    gen = cfg.generation
    generation_signature = "|".join(
        [
            cfg.policy.model_id,
            f"max_model_len={cfg.policy.max_model_len}",
            f"K={gen.num_candidates}",
            f"max_tokens={gen.max_tokens}",
            f"temperature={gen.temperature}",
            f"top_p={gen.top_p}",
        ]
    )

    # Build dataset
    logger.info("Building preference dataset")
    pairs = build_preference_dataset(
        generation_engine=generation_engine,
        scoring_engine=scoring_engine,
        num_candidates=cfg.generation.num_candidates,
        construction_mode=cfg.construction_mode,
        output_path=output,
        examples=examples,
        batch_size=cfg.generation.batch_size,
        candidate_cache_path=candidate_cache,
        generation_signature=generation_signature,
    )

    logger.info(f"Successfully built dataset with {len(pairs)} pairs")
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
