#!/usr/bin/env python3
"""Compare EuroEval results across models from the benchmark results file."""

import json
import logging
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--results",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=Path("euroeval_benchmark_results.jsonl"),
    help="Path to the EuroEval results JSONL file.",
)
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    required=True,
    help="Model id to include, in column order. Repeat for each model.",
)
def main(*, results: Path, models: tuple[str, ...]) -> None:
    """Print a per-dataset comparison table of EuroEval scores across models.

    Args:
        results:
          Path to the EuroEval results JSONL file.
        models:
          Model ids to compare, in the order they should appear as columns.
    """
    scores = _load_scores(path=results)

    keys = sorted(
        {key for model in models for key in scores.get(model, {})},
        key=lambda key: (key[0], key[1]),
    )
    if not keys:
        logger.warning("No matching results found for the requested models.")
        return

    headers = [_short(model) for model in models]
    rows = [_format_row(scores=scores, models=models, key=key) for key in keys]
    _log_table(
        label_header="dataset / metric", headers=headers, label_rows=keys, rows=rows
    )


def _load_scores(*, path: Path) -> dict[str, dict[tuple[str, str], tuple[float, bool]]]:
    """Load per-(dataset, metric) scores per model from a results file.

    Later records override earlier ones, so re-runs use the most recent score.

    Args:
        path:
          Path to the EuroEval results JSONL file.

    Returns:
        Mapping of model id to {(dataset, metric): (score, lower_is_better)}.
    """
    scores: dict[str, dict[tuple[str, str], tuple[float, bool]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        model_id = record["model_info"]["id"]
        model_scores = scores.setdefault(model_id, {})
        for result in record["evaluation_results"]:
            dataset = result["source_data"]["dataset_name"]
            metric = result["evaluation_name"]
            score = result["score_details"]["score"]
            lower_is_better = result["metric_config"]["lower_is_better"]
            model_scores[dataset, metric] = (score, lower_is_better)
    return scores


def _format_row(
    *,
    scores: dict[str, dict[tuple[str, str], tuple[float, bool]]],
    models: tuple[str, ...],
    key: tuple[str, str],
) -> list[str]:
    """Format one table row of scores for a (dataset, metric) across models.

    Args:
        scores:
          Loaded per-model scores.
        models:
          Model ids in column order.
        key:
          The (dataset, metric) pair for this row.

    Returns:
        The formatted cell values, one per model.
    """
    return [
        f"{scores[model][key][0]:.2f}" if key in scores.get(model, {}) else "-"
        for model in models
    ]


def _log_table(
    *,
    label_header: str,
    headers: list[str],
    label_rows: list[tuple[str, str]],
    rows: list[list[str]],
) -> None:
    """Log an aligned text table.

    Args:
        label_header:
          Header for the leftmost (row-label) column.
        headers:
          Column headers for the model columns.
        label_rows:
          The (dataset, metric) label for each row.
        rows:
          The formatted score cells for each row.
    """
    labels = [f"{dataset} / {metric}" for dataset, metric in label_rows]
    label_width = max([len(label_header), *(len(label) for label in labels)])
    col_widths = [
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]

    header_line = (
        label_header.ljust(label_width)
        + "  "
        + "  ".join(
            header.rjust(width)
            for header, width in zip(headers, col_widths, strict=True)
        )
    )
    logger.info(header_line)
    logger.info("-" * len(header_line))
    for label, row in zip(labels, rows, strict=True):
        line = (
            label.ljust(label_width)
            + "  "
            + "  ".join(
                cell.rjust(width) for cell, width in zip(row, col_widths, strict=True)
            )
        )
        logger.info(line)


def _short(model_id: str) -> str:
    """Return the trailing path component of a model id for a compact header.

    Args:
        model_id:
          The full model id.

    Returns:
        The last path component of the model id.
    """
    return model_id.rsplit("/", maxsplit=1)[-1]


if __name__ == "__main__":
    main()
