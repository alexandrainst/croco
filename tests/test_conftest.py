"""Tests for conftest fixtures."""

import collections.abc as c

from croco.data_models import ScoredCandidate
from croco.engines import GenerationEngine, ScoringEngine


def test_make_scored_fixture(
    make_scored: c.Callable[[str, float], ScoredCandidate],
) -> None:
    """Test the make_scored fixture creates ScoredCandidate instances."""
    candidate = make_scored("Test response", 0.75)
    assert isinstance(candidate, ScoredCandidate)
    assert candidate.response == "Test response"
    assert candidate.reward_score == 0.75


def test_fake_generation_engine_fixture(
    fake_generation_engine: GenerationEngine,
) -> None:
    """Test the fake_generation_engine fixture."""
    prompts = ["Prompt 1", "Prompt 2"]
    results = fake_generation_engine.generate(prompts=prompts, num_candidates=2)

    assert len(results) == 2
    assert len(results[0]) == 2
    assert len(results[1]) == 2
    assert "Prompt 1" in results[0][0]


def test_fake_scoring_engine_fixture(fake_scoring_engine: ScoringEngine) -> None:
    """Test the fake_scoring_engine fixture."""
    prompts = ["Prompt 1", "Prompt 2"]
    responses = ["Short", "This is a longer response"]
    scores = fake_scoring_engine.score(prompts=prompts, responses=responses)

    assert len(scores) == 2
    # Longer response should have higher score
    assert scores[1] > scores[0]
