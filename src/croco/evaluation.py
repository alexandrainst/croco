"""EuroEval evaluation wrapper for the CroCo pipeline."""

from pathlib import Path

from .config import PipelineConfig


def _get_benchmarker() -> type:
    """Lazily import Benchmarker to avoid hard dependency on euroeval.

    Returns:
        The Benchmarker class from euroeval.
    """
    from euroeval import Benchmarker  # ty: ignore[unresolved-import]

    return Benchmarker


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
    Benchmarker = _get_benchmarker()
    benchmarker = Benchmarker(
        language=config.eval.language,
        progress_bar=True,
        save_results=True,
        num_iterations=config.eval.num_iterations,
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
