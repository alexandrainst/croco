"""EuroEval evaluation wrapper for the CroCo pipeline."""

from pathlib import Path

from euroeval import Benchmarker

from .config import PipelineConfig


def evaluate_model(*, model_id_or_path: str | Path, config: PipelineConfig) -> list:
    """Benchmark a model with EuroEval, restricted to the configured language.

    Args:
        model_id_or_path:
            Path or Hugging Face ID of the model to evaluate.
        config:
            Pipeline configuration containing evaluation settings.

    Returns:
        List of BenchmarkResult objects from EuroEval.
    """
    benchmarker = Benchmarker(
        language=config.eval.language,
        progress_bar=True,
        save_results=True,
        gpu_memory_utilization=config.eval.gpu_memory_utilization,
        num_iterations=10,
    )
    results = benchmarker.benchmark(model=str(model_id_or_path), task=config.eval.tasks)
    return list(results)


def extract_scores(*, results: list) -> dict[str, dict[str, float]]:
    """Extract aggregated scores from EuroEval benchmark results.

    Maps each result's dataset name to its aggregated results['total'] dict.

    Args:
        results:
            List of BenchmarkResult objects from EuroEval.

    Returns:
        Dictionary mapping dataset names to their total score dictionaries.
    """
    return {r.dataset: dict(r.results["total"]) for r in results}
