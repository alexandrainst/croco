"""Preference dataset save/load and TRL record conversion."""

import typing as t
from pathlib import Path

from .data_models import ExampleCandidates, PreferencePair
from .utils import append_jsonl, read_jsonl, write_jsonl


def save_pairs(*, pairs: list[PreferencePair], path: Path) -> None:
    """Save preference pairs to a JSONL file, overwriting any existing content.

    Args:
        pairs:
          List of preference pairs to save.
        path:
          Output file path.
    """
    rows = [pair.model_dump() for pair in pairs]
    write_jsonl(path=path, rows=rows)


def append_pairs(*, pairs: list[PreferencePair], path: Path) -> None:
    """Append preference pairs to a JSONL file, creating it if needed.

    Used for incremental checkpointing during dataset construction so that a
    long-running build remains usable even if interrupted.

    Args:
        pairs:
          List of preference pairs to append.
        path:
          Output file path.
    """
    append_jsonl(path=path, rows=(pair.model_dump() for pair in pairs))


def load_pairs(*, path: Path) -> list[PreferencePair]:
    """Load preference pairs from a JSONL file.

    Args:
        path:
          Input file path.

    Returns:
        List of validated preference pairs.
    """
    rows = read_jsonl(path=path)
    return [PreferencePair.model_validate(row) for row in rows]


def load_candidate_cache(*, path: Path) -> dict[str, ExampleCandidates]:
    """Load cached self-generations keyed by example hash.

    Records without a hash cannot be keyed and are skipped.

    Args:
        path:
          Path to the candidate cache JSONL file.

    Returns:
        Mapping of example hash to its cached candidates.
    """
    cache: dict[str, ExampleCandidates] = {}
    for row in read_jsonl(path=path):
        record = ExampleCandidates.model_validate(row)
        if record.hash is not None:
            cache[record.hash] = record
    return cache


def append_candidates(*, records: list[ExampleCandidates], path: Path) -> None:
    """Append candidate records to the cache file, creating it if needed.

    Args:
        records:
          Candidate records to append.
        path:
          Path to the candidate cache JSONL file.
    """
    append_jsonl(path=path, rows=(record.model_dump() for record in records))


def to_trl_records(*, pairs: list[PreferencePair]) -> list[dict[str, t.Any]]:
    """Convert preference pairs to TRL conversational DPO format.

    Args:
        pairs:
          List of preference pairs.

    Returns:
        List of records in TRL DPO format with prompt, chosen, and rejected
        as conversation lists.
    """
    records = []
    for pair in pairs:
        record = {
            "prompt": [{"role": "user", "content": pair.prompt}],
            "chosen": [{"role": "assistant", "content": pair.chosen}],
            "rejected": [{"role": "assistant", "content": pair.rejected}],
        }
        records.append(record)
    return records
