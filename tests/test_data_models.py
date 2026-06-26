"""Tests for CroCo data models."""

import pytest

from croco.data_models import DataExample, PreferencePair, ScoredCandidate


class TestScoredCandidate:
    """Tests for ScoredCandidate model."""

    def test_create_scored_candidate(self) -> None:
        """Test creating a ScoredCandidate instance."""
        candidate = ScoredCandidate(
            response="This is a test response", reward_score=0.75
        )
        assert candidate.response == "This is a test response"
        assert candidate.reward_score == 0.75

    def test_scored_candidate_with_zero_score(self) -> None:
        """Test ScoredCandidate with zero score."""
        candidate = ScoredCandidate(response="Zero score response", reward_score=0.0)
        assert candidate.response == "Zero score response"
        assert candidate.reward_score == 0.0

    def test_scored_candidate_with_negative_score(self) -> None:
        """Test ScoredCandidate with negative score."""
        candidate = ScoredCandidate(
            response="Negative score response", reward_score=-0.5
        )
        assert candidate.response == "Negative score response"
        assert candidate.reward_score == -0.5

    def test_scored_candidate_with_high_score(self) -> None:
        """Test ScoredCandidate with high score."""
        candidate = ScoredCandidate(response="High score response", reward_score=0.99)
        assert candidate.response == "High score response"
        assert candidate.reward_score == 0.99

    def test_model_dump(self) -> None:
        """Test serialising ScoredCandidate to dictionary."""
        candidate = ScoredCandidate(response="Test response", reward_score=0.65)
        dump = candidate.model_dump()
        assert dump == {"response": "Test response", "reward_score": 0.65}

    def test_model_validate(self) -> None:
        """Test deserialising ScoredCandidate from dictionary."""
        data = {"response": "Test response", "reward_score": 0.65}
        candidate = ScoredCandidate.model_validate(data)
        assert candidate.response == "Test response"
        assert candidate.reward_score == 0.65


class TestDataExample:
    """Tests for DataExample model."""

    def test_create_minimal_data_example(self) -> None:
        """Test creating a minimal DataExample instance."""
        example = DataExample(
            instruction="What is the capital of Denmark?", output="Copenhagen"
        )
        assert example.instruction == "What is the capital of Denmark?"
        assert example.output == "Copenhagen"
        assert example.evolution is None
        assert example.hash is None

    def test_create_full_data_example(self) -> None:
        """Test creating a DataExample with all fields."""
        example = DataExample(
            instruction="Translate to Danish",
            output="Hej, hvordan har du det?",
            evolution=3,
            hash="abc123",
        )
        assert example.instruction == "Translate to Danish"
        assert example.output == "Hej, hvordan har du det?"
        assert example.evolution == 3
        assert example.hash == "abc123"

    def test_data_example_with_evolution_zero(self) -> None:
        """Test DataExample with evolution level 0."""
        example = DataExample(
            instruction="Simple instruction", output="Simple output", evolution=0
        )
        assert example.evolution == 0

    def test_model_dump(self) -> None:
        """Test serialising DataExample to dictionary."""
        example = DataExample(
            instruction="Test instruction",
            output="Test output",
            evolution=2,
            hash="xyz789",
        )
        dump = example.model_dump()
        assert dump == {
            "instruction": "Test instruction",
            "output": "Test output",
            "evolution": 2,
            "hash": "xyz789",
        }

    def test_model_validate(self) -> None:
        """Test deserialising DataExample from dictionary."""
        data = {
            "instruction": "Test instruction",
            "output": "Test output",
            "evolution": 2,
            "hash": "xyz789",
        }
        example = DataExample.model_validate(data)
        assert example.instruction == "Test instruction"
        assert example.output == "Test output"
        assert example.evolution == 2
        assert example.hash == "xyz789"


class TestPreferencePair:
    """Tests for PreferencePair model."""

    def test_create_preference_pair_generated_mode(self) -> None:
        """Test creating a PreferencePair in generated mode."""
        pair = PreferencePair(
            prompt="Write a poem",
            chosen="Roses are red",
            rejected="Roses are blue",
            rejected_score=0.4,
            chosen_score=0.8,
            evolution=2,
            pool_size=5,
            mode="generated",
            hash="pair123",
        )
        assert pair.prompt == "Write a poem"
        assert pair.chosen == "Roses are red"
        assert pair.rejected == "Roses are blue"
        assert pair.rejected_score == 0.4
        assert pair.chosen_score == 0.8
        assert pair.evolution == 2
        assert pair.pool_size == 5
        assert pair.mode == "generated"
        assert pair.hash == "pair123"

    def test_create_preference_pair_gold_chosen_mode(self) -> None:
        """Test creating a PreferencePair in gold_chosen mode."""
        pair = PreferencePair(
            prompt="Translate this",
            chosen="Gold translation",
            rejected="Generated translation",
            rejected_score=0.3,
            chosen_score=0.9,
            pool_size=4,
            mode="gold_chosen",
        )
        assert pair.prompt == "Translate this"
        assert pair.chosen == "Gold translation"
        assert pair.rejected == "Generated translation"
        assert pair.rejected_score == 0.3
        assert pair.chosen_score == 0.9
        assert pair.mode == "gold_chosen"
        assert pair.evolution is None

    def test_preference_pair_without_chosen_score(self) -> None:
        """Test PreferencePair without chosen score."""
        pair = PreferencePair(
            prompt="Test prompt",
            chosen="Chosen response",
            rejected="Rejected response",
            rejected_score=0.5,
            pool_size=3,
            mode="generated",
        )
        assert pair.chosen_score is None
        assert pair.rejected_score == 0.5

    def test_preference_pair_validation_mode_literal(self) -> None:
        """Test that mode must be a valid literal."""
        with pytest.raises(ValueError):
            PreferencePair(
                prompt="Test",
                chosen="Chosen",
                rejected="Rejected",
                rejected_score=0.5,
                pool_size=2,
                mode="invalid_mode",  # type: ignore[arg-type]
            )

    def test_model_dump(self) -> None:
        """Test serialising PreferencePair to dictionary."""
        pair = PreferencePair(
            prompt="Test prompt",
            chosen="Chosen response",
            rejected="Rejected response",
            rejected_score=0.4,
            chosen_score=0.8,
            evolution=1,
            pool_size=4,
            mode="generated",
            hash="test123",
        )
        dump = pair.model_dump()
        assert dump == {
            "prompt": "Test prompt",
            "chosen": "Chosen response",
            "rejected": "Rejected response",
            "rejected_score": 0.4,
            "chosen_score": 0.8,
            "evolution": 1,
            "pool_size": 4,
            "mode": "generated",
            "hash": "test123",
        }

    def test_model_validate(self) -> None:
        """Test deserialising PreferencePair from dictionary."""
        data = {
            "prompt": "Test prompt",
            "chosen": "Chosen response",
            "rejected": "Rejected response",
            "rejected_score": 0.4,
            "chosen_score": 0.8,
            "evolution": 1,
            "pool_size": 4,
            "mode": "generated",
            "hash": "test123",
        }
        pair = PreferencePair.model_validate(data)
        assert pair.prompt == "Test prompt"
        assert pair.chosen == "Chosen response"
        assert pair.rejected == "Rejected response"
        assert pair.rejected_score == 0.4
        assert pair.chosen_score == 0.8
        assert pair.evolution == 1
        assert pair.pool_size == 4
        assert pair.mode == "generated"
        assert pair.hash == "test123"
