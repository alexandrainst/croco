#!/usr/bin/env python3
"""Evaluate every saved DPO checkpoint with EuroEval to trace a learning curve.

Each ``checkpoint-N`` directory (and the final adapter) is benchmarked as a
separate EuroEval run, so the results file ends up with one entry per checkpoint.
Because training runs for a single epoch, a checkpoint at step N has effectively
seen N x effective-batch samples, so the curve over checkpoints approximates the
curve over training-set size.
"""

import logging
import subprocess
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-dir",
    "-m",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="DPO output directory containing checkpoint-* subdirectories.",
)
@click.option(
    "--language",
    "-l",
    default="da",
    help="EuroEval language code to benchmark. Defaults to da.",
)
@click.option(
    "--task",
    "tasks",
    multiple=True,
    help="Restrict to these tasks (repeatable). Omit for the full language suite.",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.5,
    help="vLLM GPU memory utilisation for EuroEval. Defaults to 0.5.",
)
@click.option(
    "--include-final/--no-include-final",
    default=True,
    help="Also evaluate the final adapter in the model directory. Defaults to True.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Forward --force to EuroEval to recompute existing results.",
)
def main(
    *,
    model_dir: Path,
    language: str,
    tasks: tuple[str, ...],
    gpu_memory_utilization: float,
    include_final: bool,
    force: bool,
) -> None:
    """Benchmark every checkpoint in a DPO output directory with EuroEval.

    Args:
        model_dir:
          DPO output directory containing checkpoint-* subdirectories.
        language:
          EuroEval language code to benchmark.
        tasks:
          Tasks to restrict to, or empty for the full language suite.
        gpu_memory_utilization:
          vLLM GPU memory utilisation for EuroEval.
        include_final:
          Whether to also evaluate the final adapter in the model directory.
        force:
          Whether to forward --force to EuroEval.
    """
    targets = _checkpoint_dirs(model_dir=model_dir, include_final=include_final)
    if not targets:
        logger.warning("No checkpoints found in %s", model_dir)
        return

    logger.info(
        "Evaluating %d checkpoints: %s", len(targets), [t.name for t in targets]
    )
    for target in targets:
        logger.info("=== Evaluating %s ===", target)
        _run_euroeval(
            model_path=target,
            language=language,
            tasks=tasks,
            gpu_memory_utilization=gpu_memory_utilization,
            force=force,
        )

    logger.info(
        "Done. Compare with: uv run src/scripts/compare_evals.py %s",
        " ".join(f"-m {target}" for target in targets),
    )


def _checkpoint_dirs(*, model_dir: Path, include_final: bool) -> list[Path]:
    """Return checkpoint directories sorted by training step.

    Args:
        model_dir:
          DPO output directory.
        include_final:
          Whether to append the final adapter directory.

    Returns:
        Checkpoint directories in ascending step order, optionally with the final
        adapter directory last.
    """
    checkpoints = sorted(
        (path for path in model_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    targets = [path for path in checkpoints if (path / "adapter_config.json").exists()]
    if include_final and (model_dir / "adapter_config.json").exists():
        targets.append(model_dir)
    return targets


def _run_euroeval(
    *,
    model_path: Path,
    language: str,
    tasks: tuple[str, ...],
    gpu_memory_utilization: float,
    force: bool,
) -> None:
    """Run EuroEval on a single checkpoint as a subprocess.

    Args:
        model_path:
          Path to the checkpoint adapter directory.
        language:
          EuroEval language code.
        tasks:
          Tasks to restrict to, or empty for the full language suite.
        gpu_memory_utilization:
          vLLM GPU memory utilisation.
        force:
          Whether to recompute existing EuroEval results.
    """
    # EuroEval ships only a console-script entry point (no ``__main__``), so it
    # cannot be run via ``python -m euroeval``; invoke the binary next to the
    # active interpreter instead.
    cmd = [
        str(Path(sys.executable).with_name("euroeval")),
        "--model",
        str(model_path),
        "--language",
        language,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--save-results",
    ]
    for task in tasks:
        cmd += ["--task", task]
    if force:
        cmd.append("--force")
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
