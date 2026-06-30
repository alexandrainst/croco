"""Tests for the candidate-cache re-scoring script."""

import importlib.util
from pathlib import Path

import pytest

from croco.data_models import ExampleCandidates, ScoredCandidate

_SPEC = importlib.util.spec_from_file_location(
    "rescore_candidates",
    Path(__file__).resolve().parent.parent
    / "src"
    / "scripts"
    / "rescore_candidates.py",
)
assert _SPEC is not None and _SPEC.loader is not None
rescore_candidates = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rescore_candidates)


class _MappingScoringEngine:
    """Scores each response by a fixed lookup, to verify score alignment."""

    def __init__(self, *, table: dict[str, float]) -> None:
        self.table = table

    def score(self, *, prompts: list[str], responses: list[str]) -> list[float]:
        """Return the looked-up score for each response.

        Returns:
            List of scores, one per response.
        """
        return [self.table[response] for response in responses]


class TestRescoreBatch:
    """The flat score list must be realigned to per-record candidates and gold."""

    def test_realigns_scores_and_preserves_generations(self) -> None:
        """Candidate and gold scores update; everything else is preserved."""
        batch = [
            ExampleCandidates(
                prompt="p1",
                gold_output="g1",
                candidates=[
                    ScoredCandidate(response="a", reward_score=0.0),
                    ScoredCandidate(response="b", reward_score=0.0),
                ],
                gold_score=0.0,
                evolution=3,
                hash="h1",
                signature="sig",
            ),
            ExampleCandidates(
                prompt="p2",
                gold_output="g2",
                candidates=[ScoredCandidate(response="c", reward_score=0.0)],
                gold_score=None,
                evolution=7,
                hash="h2",
                signature="sig",
            ),
        ]
        engine = _MappingScoringEngine(table={"a": 1.0, "b": 2.0, "g1": 3.0, "c": 4.0})

        result = rescore_candidates._rescore_batch(batch=batch, scoring_engine=engine)

        assert [c.reward_score for c in result[0].candidates] == [1.0, 2.0]
        assert result[0].gold_score == 3.0
        assert [c.reward_score for c in result[1].candidates] == [4.0]
        # A record without a gold score must not consume a slot in the flat list.
        assert result[1].gold_score is None
        # Generations and metadata are carried over untouched.
        assert [c.response for c in result[0].candidates] == ["a", "b"]
        assert [c.response for c in result[1].candidates] == ["c"]
        assert (result[0].hash, result[0].signature, result[0].evolution) == (
            "h1",
            "sig",
            3,
        )
        assert result[1].prompt == "p2"


class TestCheckDistinctPaths:
    """Re-scoring a cache onto itself must be rejected."""

    def test_same_path_raises(self) -> None:
        """Identical input and output paths raise a usage error."""
        import click

        with pytest.raises(click.UsageError):
            rescore_candidates._check_distinct_paths(
                input_cache=Path("data/x.jsonl"), output_cache=Path("data/x.jsonl")
            )

    def test_distinct_paths_ok(self) -> None:
        """Different paths pass validation."""
        rescore_candidates._check_distinct_paths(
            input_cache=Path("data/x.jsonl"), output_cache=Path("data/y.jsonl")
        )
