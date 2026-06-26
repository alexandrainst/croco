"""Pytest configuration and shared fixtures for CroCo tests."""

from __future__ import annotations

import collections.abc as c

import pytest

from croco.data_models import ScoredCandidate
from croco.engines import GenerationEngine, ScoringEngine

# Skip vLLM modules when vLLM is not available (GPU-only)
import importlib.util

collect_ignore_glob: list[str] = []
if importlib.util.find_spec("vllm") is None:
    collect_ignore_glob.append("src/croco/vllm_*.py")

pytest_plugins = []


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "vllm: mark test to run only when vLLM is available (GPU required)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip vLLM tests when vLLM is not available."""
    vllm_skip = pytest.mark.skip(reason="vLLM not available (GPU required)")
    for item in items:
        if "vllm" in item.keywords:
            try:
                import vllm  # noqa: F401, ty: ignore[unresolved-import]
            except ImportError:
                item.add_marker(vllm_skip)


@pytest.fixture
def make_scored() -> c.Callable[[str, float], ScoredCandidate]:
    """Factory fixture for creating ScoredCandidate instances.

    Returns:
        A callable that takes a response string and reward score, returning
        a ScoredCandidate instance.
    """

    def _make(response: str, reward_score: float) -> ScoredCandidate:
        return ScoredCandidate(response=response, reward_score=reward_score)

    return _make


@pytest.fixture
def fake_generation_engine() -> GenerationEngine:
    """Fixture providing a fake generation engine for testing.

    Returns:
        A GenerationEngine implementation that returns deterministic fake
        responses based on the input prompts.
    """

    class FakeGenerationEngine:
        """Fake generation engine for testing."""

        def generate(
            self, *, prompts: list[str], num_candidates: int
        ) -> list[list[str]]:
            """Generate fake responses for testing.

            Returns:
                Nested list of fake response strings, one list per prompt.
            """
            results: list[list[str]] = []
            for i, prompt in enumerate(prompts):
                candidates = [
                    f"Response {i}-{j} to: {prompt}" for j in range(num_candidates)
                ]
                results.append(candidates)
            return results

    return FakeGenerationEngine()


@pytest.fixture
def fake_scoring_engine() -> ScoringEngine:
    """Fixture providing a fake scoring engine for testing.

    Returns:
        A ScoringEngine implementation that returns deterministic fake scores
        based on the length of the response.
    """

    class FakeScoringEngine:
        """Fake scoring engine for testing."""

        def score(self, *, prompts: list[str], responses: list[str]) -> list[float]:
            """Score responses based on their length for testing.

            Returns:
                List of float scores proportional to response length.
            """
            scores: list[float] = []
            for prompt, response in zip(prompts, responses):
                # Simple heuristic: longer responses get higher scores
                score = len(response) / 100.0
                scores.append(score)
            return scores

    return FakeScoringEngine()
