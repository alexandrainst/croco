"""EuroEval subprocess helper for running EuroEval as an external command."""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_euroeval_subprocess(
    *,
    model_path: Path,
    language: str,
    tasks: list[str] | tuple[str, ...],
    gpu_memory_utilization: float,
    force: bool = False,
) -> None:
    """Run EuroEval CLI as a subprocess.

    EuroEval ships only a console-script entry point (no ``__main__``), so it
    cannot be run via ``python -m euroeval``; invoke the binary next to the
    active interpreter instead.

    Args:
        model_path:
            Path to the model/checkpoint directory to evaluate.
        language:
            EuroEval language code.
        tasks:
            Tasks to evaluate, or empty/None for full language suite.
        gpu_memory_utilization:
            vLLM GPU memory utilisation.
        force:
            Whether to recompute existing EuroEval results.
    """
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
    if tasks:
        for task in tasks:
            cmd += ["--task", task]
    if force:
        cmd.append("--force")
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
