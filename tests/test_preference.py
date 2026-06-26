"""Tests for CroCo preference pair construction.

Tests include worked numerical examples to verify the pair construction
logic matches the reference implementation (CroCo, Eq. 2).
"""

import numpy as np
import pytest

from croco.data_models import PreferencePair, ScoredCandidate
from croco.preference import (
    _select_rejected,
    build_pair_generated,
    build_pair_gold_chosen,
)


class TestSelectRejected:
    """Tests for _select_rejected helper function."""

    def test_select_rejected_basic(self) -> None:
        """Test selecting rejected candidate closest to mu - 2*sigma."""
        candidates = [
            ScoredCandidate(response="resp_a", reward_score=0.9),
            ScoredCandidate(response="resp_b", reward_score=0.7),
            ScoredCandidate(response="resp_c", reward_score=0.5),
            ScoredCandidate(response="resp_d", reward_score=0.3),
            ScoredCandidate(response="resp_e", reward_score=0.1),
        ]

        # Mean = (0.9 + 0.7 + 0.5 + 0.3 + 0.1) / 5 = 2.5 / 5 = 0.5
        # Population std = sqrt(((0.4)^2 + (0.2)^2 + 0^2 + (-0.2)^2 + (-0.4)^2) / 5)
        #                = sqrt((0.16 + 0.04 + 0 + 0.04 + 0.16) / 5)
        #                = sqrt(0.4 / 5) = sqrt(0.08) ≈ 0.283
        # Target = 0.5 - 2 * 0.283 = 0.5 - 0.566 = -0.066
        # Closest to -0.066 is 0.1 (resp_e)

        result = _select_rejected(pool=candidates, upper_bound=None, exclude=None)
        assert result is not None
        assert result.response == "resp_e"
        assert result.reward_score == 0.1

    def test_select_rejected_with_upper_bound(self) -> None:
        """Test selecting rejected candidate with upper bound constraint."""
        candidates = [
            ScoredCandidate(response="resp_a", reward_score=0.9),
            ScoredCandidate(response="resp_b", reward_score=0.7),
            ScoredCandidate(response="resp_c", reward_score=0.5),
            ScoredCandidate(response="resp_d", reward_score=0.3),
            ScoredCandidate(response="resp_e", reward_score=0.1),
        ]

        # Target is -0.066 (as calculated above)
        # With upper_bound=0.6, candidates with score >= 0.6 are excluded
        # Eligible: resp_c (0.5), resp_d (0.3), resp_e (0.1)
        # Closest to -0.066 is still resp_e (0.1)

        result = _select_rejected(pool=candidates, upper_bound=0.6, exclude=None)
        assert result is not None
        assert result.response == "resp_e"
        assert result.reward_score == 0.1

    def test_select_rejected_with_exclude(self) -> None:
        """Test selecting rejected candidate with exclusion."""
        candidates = [
            ScoredCandidate(response="resp_a", reward_score=0.9),
            ScoredCandidate(response="resp_b", reward_score=0.7),
            ScoredCandidate(response="resp_c", reward_score=0.5),
        ]

        # Mean = (0.9 + 0.7 + 0.5) / 3 = 2.1 / 3 = 0.7
        # Population std = sqrt(((0.2)^2 + 0^2 + (-0.2)^2) / 3)
        #                = sqrt((0.04 + 0 + 0.04) / 3)
        #                = sqrt(0.08 / 3) ≈ sqrt(0.0267) ≈ 0.163
        # Target = 0.7 - 2 * 0.163 = 0.7 - 0.326 = 0.374
        # Closest to 0.374 is resp_c (0.5)
        # If we exclude resp_c, closest is resp_b (0.7)

        # Exclude the candidate closest to target
        to_exclude = candidates[2]  # resp_c with score 0.5
        result = _select_rejected(pool=candidates, upper_bound=None, exclude=to_exclude)
        assert result is not None
        assert result.response == "resp_b"
        assert result.reward_score == 0.7

    def test_select_rejected_all_excluded(self) -> None:
        """Test when all candidates are excluded."""
        candidates = [
            ScoredCandidate(response="resp_a", reward_score=0.9),
        ]

        # All candidates are excluded, should return None
        result = _select_rejected(
            pool=candidates, upper_bound=None, exclude=candidates[0]
        )
        assert result is None

    def test_select_rejected_all_above_upper_bound(self) -> None:
        """Test when all candidates are above upper bound."""
        candidates = [
            ScoredCandidate(response="resp_a", reward_score=0.9),
            ScoredCandidate(response="resp_b", reward_score=0.8),
            ScoredCandidate(response="resp_c", reward_score=0.7),
        ]

        # All candidates have score >= 0.7, upper_bound is 0.7
        result = _select_rejected(pool=candidates, upper_bound=0.7, exclude=None)
        assert result is None


class TestBuildPairGenerated:
    """Tests for build_pair_generated function with worked numbers."""

    def test_build_pair_generated_basic(self) -> None:
        """Test building a pair in generated mode with worked numbers."""
        candidates = [
            ScoredCandidate(response="best", reward_score=0.95),
            ScoredCandidate(response="good", reward_score=0.75),
            ScoredCandidate(response="mediocre", reward_score=0.55),
            ScoredCandidate(response="poor", reward_score=0.35),
            ScoredCandidate(response="worst", reward_score=0.15),
        ]

        # Mean = (0.95 + 0.75 + 0.55 + 0.35 + 0.15) / 5 = 2.75 / 5 = 0.55
        # Population std = sqrt(((0.4)^2 + (0.2)^2 + 0^2 + (-0.2)^2 + (-0.4)^2) / 5)
        #                = sqrt((0.16 + 0.04 + 0 + 0.04 + 0.16) / 5)
        #                = sqrt(0.4 / 5) = sqrt(0.08) ≈ 0.283
        # Target = 0.55 - 2 * 0.283 = 0.55 - 0.566 = -0.016
        # Closest to -0.016 is "worst" (0.15)
        # Chosen is "best" (0.95, highest score)
        # Rejected is "worst" (0.15, closest to target)

        result = build_pair_generated(
            prompt="Test instruction",
            candidates=candidates,
            evolution=2,
            hash="test_hash",
        )

        assert result is not None
        assert result.prompt == "Test instruction"
        assert result.chosen == "best"
        assert result.chosen_score == 0.95
        assert result.rejected == "worst"
        assert result.rejected_score == 0.15
        assert result.evolution == 2
        assert result.pool_size == 5
        assert result.mode == "generated"
        assert result.hash == "test_hash"

    def test_build_pair_generated_insufficient_candidates(self) -> None:
        """Test building pair with fewer than 2 candidates."""
        candidates = [
            ScoredCandidate(response="only_one", reward_score=0.8),
        ]

        result = build_pair_generated(
            prompt="Test instruction",
            candidates=candidates,
        )
        assert result is None

    def test_build_pair_generated_empty_candidates(self) -> None:
        """Test building pair with no candidates."""
        result = build_pair_generated(
            prompt="Test instruction",
            candidates=[],
        )
        assert result is None

    def test_build_pair_generated_no_valid_rejected(self) -> None:
        """Test when no valid rejected candidate exists."""
        candidates = [
            ScoredCandidate(response="same1", reward_score=0.8),
            ScoredCandidate(response="same2", reward_score=0.8),
        ]

        # Mean = 0.8, std = 0, target = 0.8
        # Both candidates have score 0.8, chosen is one of them
        # No candidate strictly below chosen, so rejected selection fails

        result = build_pair_generated(
            prompt="Test instruction",
            candidates=candidates,
        )
        assert result is None

    def test_build_pair_generated_three_candidates(self) -> None:
        """Test with exactly 3 candidates."""
        candidates = [
            ScoredCandidate(response="high", reward_score=0.9),
            ScoredCandidate(response="mid", reward_score=0.5),
            ScoredCandidate(response="low", reward_score=0.1),
        ]

        # Mean = (0.9 + 0.5 + 0.1) / 3 = 1.5 / 3 = 0.5
        # Population std = sqrt(((0.4)^2 + 0^2 + (-0.4)^2) / 3)
        #                = sqrt((0.16 + 0 + 0.16) / 3)
        #                = sqrt(0.32 / 3) ≈ sqrt(0.1067) ≈ 0.327
        # Target = 0.5 - 2 * 0.327 = 0.5 - 0.654 = -0.154
        # Closest to -0.154 is "low" (0.1)
        # Chosen is "high" (0.9)
        # Rejected is "low" (0.1)

        result = build_pair_generated(
            prompt="Test prompt",
            candidates=candidates,
        )

        assert result is not None
        assert result.chosen == "high"
        assert result.chosen_score == 0.9
        assert result.rejected == "low"
        assert result.rejected_score == 0.1
        assert result.pool_size == 3
        assert result.mode == "generated"


class TestBuildPairGoldChosen:
    """Tests for build_pair_gold_chosen function with worked numbers."""

    def test_build_pair_gold_chosen_basic(self) -> None:
        """Test building a pair in gold_chosen mode with worked numbers."""
        gold_output = "Gold standard translation"
        gold_score = 0.92

        candidates = [
            ScoredCandidate(response="gen_1", reward_score=0.85),
            ScoredCandidate(response="gen_2", reward_score=0.65),
            ScoredCandidate(response="gen_3", reward_score=0.45),
            ScoredCandidate(response="gen_4", reward_score=0.25),
        ]

        # Mean = (0.85 + 0.65 + 0.45 + 0.25) / 4 = 2.2 / 4 = 0.55
        # Population std = sqrt(((0.3)^2 + (0.1)^2 + (-0.1)^2 + (-0.3)^2) / 4)
        #                = sqrt((0.09 + 0.01 + 0.01 + 0.09) / 4)
        #                = sqrt(0.2 / 4) = sqrt(0.05) ≈ 0.224
        # Target = 0.55 - 2 * 0.224 = 0.55 - 0.448 = 0.102
        # Closest to 0.102 is "gen_4" (0.25)
        # Chosen is gold_output (0.92)
        # Rejected is "gen_4" (0.25, closest to target)
        # Must be strictly below gold_score (0.92), which 0.25 is

        result = build_pair_gold_chosen(
            prompt="Translate this sentence",
            gold_output=gold_output,
            candidates=candidates,
            gold_score=gold_score,
            evolution=3,
            hash="gold_test_hash",
        )

        assert result is not None
        assert result.prompt == "Translate this sentence"
        assert result.chosen == "Gold standard translation"
        assert result.chosen_score == 0.92
        assert result.rejected == "gen_4"
        assert result.rejected_score == 0.25
        assert result.evolution == 3
        assert result.pool_size == 4
        assert result.mode == "gold_chosen"
        assert result.hash == "gold_test_hash"

    def test_build_pair_gold_chosen_without_gold_score(self) -> None:
        """Test building pair without gold score (no upper bound)."""
        gold_output = "Gold output"

        candidates = [
            ScoredCandidate(response="gen_a", reward_score=0.7),
            ScoredCandidate(response="gen_b", reward_score=0.5),
            ScoredCandidate(response="gen_c", reward_score=0.3),
        ]

        # Mean = (0.7 + 0.5 + 0.3) / 3 = 1.5 / 3 = 0.5
        # Population std = sqrt(((0.2)^2 + 0^2 + (-0.2)^2) / 3)
        #                = sqrt((0.04 + 0 + 0.04) / 3)
        #                = sqrt(0.08 / 3) ≈ 0.163
        # Target = 0.5 - 2 * 0.163 = 0.5 - 0.326 = 0.174
        # Closest to 0.174 is "gen_c" (0.3)

        result = build_pair_gold_chosen(
            prompt="Test prompt",
            gold_output=gold_output,
            candidates=candidates,
        )

        assert result is not None
        assert result.chosen == "Gold output"
        assert result.chosen_score is None
        assert result.rejected == "gen_c"
        assert result.rejected_score == 0.3
        assert result.pool_size == 3
        assert result.mode == "gold_chosen"

    def test_build_pair_gold_chosen_empty_candidates(self) -> None:
        """Test building pair with no candidates."""
        result = build_pair_gold_chosen(
            prompt="Test prompt",
            gold_output="Gold output",
            candidates=[],
        )
        assert result is None

    def test_build_pair_gold_chosen_rejected_above_gold(self) -> None:
        """Test when best rejected candidate is above gold score."""
        gold_output = "Gold output"
        gold_score = 0.5

        candidates = [
            ScoredCandidate(response="gen_high", reward_score=0.9),
            ScoredCandidate(response="gen_mid", reward_score=0.8),
            ScoredCandidate(response="gen_low", reward_score=0.7),
        ]

        # Mean = (0.9 + 0.8 + 0.7) / 3 = 2.4 / 3 = 0.8
        # Population std = sqrt(((0.1)^2 + 0^2 + (-0.1)^2) / 3)
        #                = sqrt((0.01 + 0 + 0.01) / 3)
        #                = sqrt(0.02 / 3) ≈ 0.082
        # Target = 0.8 - 2 * 0.082 = 0.8 - 0.164 = 0.636
        # Closest to 0.636 is "gen_low" (0.7)
        # But 0.7 >= 0.5 (gold_score), so it's excluded
        # All candidates are above gold_score, so rejected selection fails

        result = build_pair_gold_chosen(
            prompt="Test prompt",
            gold_output=gold_output,
            candidates=candidates,
            gold_score=gold_score,
        )
        assert result is None

    def test_build_pair_gold_chosen_partial_exclusion(self) -> None:
        """Test when some candidates are above gold score but not all."""
        gold_output = "Gold output"
        gold_score = 0.6

        candidates = [
            ScoredCandidate(response="gen_above1", reward_score=0.8),
            ScoredCandidate(response="gen_above2", reward_score=0.7),
            ScoredCandidate(response="gen_below", reward_score=0.4),
        ]

        # Mean = (0.8 + 0.7 + 0.4) / 3 = 1.9 / 3 ≈ 0.633
        # Population std = sqrt(((0.167)^2 + (0.067)^2 + (-0.233)^2) / 3)
        #                ≈ sqrt((0.028 + 0.004 + 0.054) / 3)
        #                ≈ sqrt(0.086 / 3) ≈ sqrt(0.029) ≈ 0.170
        # Target = 0.633 - 2 * 0.170 = 0.633 - 0.34 = 0.293
        # Closest to 0.293 is "gen_below" (0.4)
        # 0.4 < 0.6 (gold_score), so it's eligible

        result = build_pair_gold_chosen(
            prompt="Test prompt",
            gold_output=gold_output,
            candidates=candidates,
            gold_score=gold_score,
        )

        assert result is not None
        assert result.chosen == "Gold output"
        assert result.chosen_score == 0.6
        assert result.rejected == "gen_below"
        assert result.rejected_score == 0.4
