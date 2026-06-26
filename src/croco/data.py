"""Laerebogen data loader with stratified subsampling."""

import collections.abc as c
import logging
import random
import typing as t

import datasets

from .config import DataConfig
from .data_models import DataExample, PreferencePair

logger = logging.getLogger(__name__)


def load_examples(*, config: DataConfig) -> list[DataExample]:
    """Load examples from the Laerebogen dataset.

    Args:
        config:
          Data configuration specifying dataset ID, subset, split, filters, and
          subsampling strategy.

    Returns:
        List of DataExample objects after filtering and subsampling.
    """
    ds = datasets.load_dataset(
        config.dataset_id, name=config.subset, split=config.split
    )

    logger.info(
        "Loaded %d examples from %s (%s, %s)",
        len(ds),
        config.dataset_id,
        config.subset,
        config.split,
    )

    if config.evolution_min is not None:
        ds = ds.filter(lambda row: row["evolution"] >= config.evolution_min)
        logger.info(
            "Filtered to evolution >= %d, %d examples remaining",
            config.evolution_min,
            len(ds),
        )

    if config.evolution_max is not None:
        ds = ds.filter(lambda row: row["evolution"] <= config.evolution_max)
        logger.info(
            "Filtered to evolution <= %d, %d examples remaining",
            config.evolution_max,
            len(ds),
        )

    ds = ds.filter(
        lambda row: row["instruction"].strip() != "" and row["output"].strip() != ""
    )
    logger.info("Filtered out empty instruction/output, %d examples remaining", len(ds))

    ds = _subsample(
        ds=ds,
        num_samples=config.num_samples,
        stratify_by_evolution=config.stratify_by_evolution,
        seed=config.seed,
    )

    examples = [
        DataExample(
            instruction=row["instruction"],
            output=row["output"],
            evolution=row.get("evolution"),
            hash=row.get("hash"),
        )
        for row in ds
    ]

    logger.info("Subsampled to %d examples", len(examples))

    return examples


def filter_by_prompt_length(
    *,
    examples: list[DataExample],
    count_tokens: c.Callable[[str], int],
    max_prompt_tokens: int,
) -> list[DataExample]:
    """Drop examples whose prompt exceeds the token budget.

    Over-long prompts leave no room for generation within the model's context
    window and would otherwise be rejected by vLLM at generation time.

    Args:
        examples:
          The examples to filter.
        count_tokens:
          Callable returning the token count of an instruction's rendered prompt.
        max_prompt_tokens:
          Maximum allowed prompt length in tokens.

    Returns:
        The examples whose prompt fits within the budget.
    """
    kept = [
        example
        for example in examples
        if count_tokens(example.instruction) <= max_prompt_tokens
    ]
    n_dropped = len(examples) - len(kept)
    if n_dropped:
        logger.info(
            "Dropped %d/%d examples exceeding max_prompt_tokens=%d",
            n_dropped,
            len(examples),
            max_prompt_tokens,
        )
    return kept


def sort_by_evolution(*, pairs: list[PreferencePair]) -> list[PreferencePair]:
    """Sort preference pairs by evolution level (ascending).

    Pairs with None evolution are treated as -infinity and come first.
    The sort is stable.

    Args:
        pairs:
          List of preference pairs to sort.

    Returns:
        Sorted list of preference pairs.
    """

    def sort_key(pair: PreferencePair) -> tuple[int, int]:
        evolution = pair.evolution
        if evolution is None:
            return (0, 0)
        return (1, evolution)

    return sorted(pairs, key=sort_key)


def _subsample(
    *, ds: datasets.Dataset, num_samples: int, stratify_by_evolution: bool, seed: int
) -> datasets.Dataset:
    """Subsample a dataset uniformly or with stratification.

    Args:
        ds:
          Dataset to subsample.
        num_samples:
          Target number of samples.
        stratify_by_evolution:
          Whether to stratify by the evolution field.
        seed:
          Random seed for reproducibility.

    Returns:
        Subsampled dataset.
    """
    if num_samples >= len(ds):
        return ds

    if stratify_by_evolution:
        return _stratified_subsample(ds=ds, num_samples=num_samples, seed=seed)

    return _uniform_subsample(ds=ds, num_samples=num_samples, seed=seed)


def _uniform_subsample(
    *, ds: datasets.Dataset, num_samples: int, seed: int
) -> datasets.Dataset:
    """Subsample uniformly at random.

    Args:
        ds:
          Dataset to subsample.
        num_samples:
          Target number of samples.
        seed:
          Random seed for reproducibility.

    Returns:
        Subsampled dataset.
    """
    shuffled = ds.shuffle(seed=seed)
    return shuffled.select(range(num_samples))


def _stratified_subsample(
    *, ds: datasets.Dataset, num_samples: int, seed: int
) -> datasets.Dataset:
    """Subsample with stratification by evolution level.

    Args:
        ds:
          Dataset to subsample (must have an 'evolution' column).
        num_samples:
          Target number of samples.
        seed:
          Random seed for reproducibility.

    Returns:
        Subsampled dataset.
    """
    rng = random.Random(seed)

    # Read the evolution column directly: per-row iteration over the full dataset
    # materialises every row as a dict and is prohibitively slow on millions of rows.
    evolutions = ds["evolution"]
    grouped_indices: dict[t.Any, list[int]] = {}
    for idx, evolution in enumerate(evolutions):
        if evolution not in grouped_indices:
            grouped_indices[evolution] = []
        grouped_indices[evolution].append(idx)

    total_rows = len(ds)
    selected_indices: list[int] = []

    for evolution, indices in grouped_indices.items():
        group_frac = len(indices) / total_rows
        group_samples = round(num_samples * group_frac)

        rng.shuffle(indices)
        selected_indices.extend(indices[:group_samples])

    if len(selected_indices) > num_samples:
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:num_samples]

    logger.info(
        "Stratified subsample: %d samples from %d groups",
        len(selected_indices),
        len(grouped_indices),
    )

    return ds.select(selected_indices)
