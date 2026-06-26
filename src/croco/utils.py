"""Utility functions for logging, seeding, JSONL IO, and chat formatting."""

import collections.abc as c
import json
import logging
import random
import typing as t
from pathlib import Path

import numpy as np
import torch


def setup_logging(*, level: int = logging.INFO) -> None:
    """Configure logging for the CroCo package.

    Args:
        level:
          Logging level. Defaults to INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(*, seed: int) -> None:
    """Set random seeds for reproducibility.

    Seeds Python's random module, NumPy, and optionally PyTorch if available.

    Args:
        seed:
          The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(*, path: Path) -> list[dict[str, t.Any]]:
    """Read a JSONL file and return a list of dictionaries.

    Args:
        path:
          Path to the JSONL file.

    Returns:
        List of dictionaries, one per line in the file.
    """
    rows: list[dict[str, t.Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(*, path: Path, rows: c.Iterable[dict[str, t.Any]]) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        path:
          Path to the output JSONL file.
        rows:
          Iterable of dictionaries to write, one per line.
    """
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_conversation(*, instruction: str, response: str) -> list[dict[str, str]]:
    """Build a chat conversation with user instruction and assistant response.

    Args:
        instruction:
          The user's instruction or prompt.
        response:
          The assistant's response.

    Returns:
        A list of message dicts in chat format:
        [{"role": "user", "content": instruction},
         {"role": "assistant", "content": response}].
    """
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]


def build_user_message(*, instruction: str) -> list[dict[str, str]]:
    """Build a chat message with only a user instruction.

    Args:
        instruction:
          The user's instruction or prompt.

    Returns:
        A list with a single message dict:
        [{"role": "user", "content": instruction}].
    """
    return [{"role": "user", "content": instruction}]
