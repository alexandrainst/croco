"""Preference dataset save/load and TRL record conversion."""

import typing as t
from pathlib import Path

from .data_models import PreferencePair
from .utils import read_jsonl, write_jsonl


def save_pairs(*, pairs: list[PreferencePair], path: Path) -> None:
    """Save preference pairs to a JSONL file.

    Args:
        pairs:
          List of preference pairs to save.
        path:
          Output file path.
    """
    rows = [pair.model_dump() for pair in pairs]
    write_jsonl(path=path, rows=rows)


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
