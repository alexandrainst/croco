"""Integration tests for the pipeline with fake engines."""

from pathlib import Path

from croco.config import (
    DataConfig,
    DPOTrainConfig,
    EvalConfig,
    GenerationConfig,
    PipelineConfig,
    PolicyModelConfig,
    RewardModelConfig,
)
from croco.data import sort_by_evolution
from croco.data_models import DataExample, PreferencePair, ScoredCandidate
from croco.dataset import load_pairs
from croco.engines import GenerationEngine, ScoringEngine
from croco.pipeline import build_preference_dataset
from croco.preference import build_pair_generated, build_pair_gold_chosen


class DictGenerationEngine(GenerationEngine):
    """Generation engine returning predefined responses keyed by prompt."""

    def __init__(self, responses_by_prompt: dict[str, list[str]]) -> None:
        """Initialise with a prompt-to-responses mapping.

        Args:
            responses_by_prompt:
                Mapping from each prompt to its list of candidate responses.
        """
        self._responses_by_prompt = responses_by_prompt

    def generate(self, *, prompts: list[str], num_candidates: int) -> list[list[str]]:
        """Return the predefined responses for each prompt.

        Args:
            prompts:
                The prompts in the batch.
            num_candidates:
                Number of candidates (ignored).

        Returns:
            The predefined responses per prompt.
        """
        return [self._responses_by_prompt[prompt] for prompt in prompts]


class DictScoringEngine(ScoringEngine):
    """Scoring engine returning predefined scores keyed by response text."""

    def __init__(self, scores_by_response: dict[str, float]) -> None:
        """Initialise with a response-to-score mapping.

        Args:
            scores_by_response:
                Mapping from each response (generation or gold) to its score.
        """
        self._scores_by_response = scores_by_response

    def score(self, *, prompts: list[str], responses: list[str]) -> list[float]:
        """Return the predefined score for each response.

        Args:
            prompts:
                The prompts (ignored).
            responses:
                The responses to score.

        Returns:
            The predefined score per response.
        """
        return [self._scores_by_response[response] for response in responses]


class TestBuildPreferenceDataset:
    """Integration tests calling the real build_preference_dataset."""

    def test_generated_mode_batches_and_checkpoints(self, tmp_path: Path) -> None:
        """Should process all examples across batches and write them to disk."""
        examples = [
            DataExample(
                instruction=f"Prompt {i}", output=f"Gold {i}", evolution=i, hash=f"h{i}"
            )
            for i in range(3)
        ]
        gen_engine = DictGenerationEngine(
            responses_by_prompt={
                f"Prompt {i}": [f"P{i}-A", f"P{i}-B", f"P{i}-C"] for i in range(3)
            }
        )
        score_engine = DictScoringEngine(
            scores_by_response={
                response: score
                for i in range(3)
                for response, score in zip(
                    [f"P{i}-A", f"P{i}-B", f"P{i}-C"], [9.0, 5.0, 1.0], strict=True
                )
            }
        )
        output_path = tmp_path / "pairs.jsonl"

        pairs = build_preference_dataset(
            generation_engine=gen_engine,
            scoring_engine=score_engine,
            num_candidates=3,
            construction_mode="generated",
            score_gold_output=False,
            output_path=output_path,
            examples=examples,
            batch_size=2,
        )

        assert len(pairs) == 3
        assert all(pair.mode == "generated" for pair in pairs)
        assert output_path.exists()
        assert len(load_pairs(path=output_path)) == 3

    def test_max_reward_mode_picks_best_of_gold_and_generations(
        self, tmp_path: Path
    ) -> None:
        """Should choose gold or generation by reward across the combined pool."""
        examples = [
            DataExample(
                instruction="P0", output="Gold0", evolution=1, hash="a"
            ),  # gold best
            DataExample(
                instruction="P1", output="Gold1", evolution=2, hash="b"
            ),  # gen best
        ]
        gen_engine = DictGenerationEngine(
            responses_by_prompt={"P0": ["g0a", "g0b"], "P1": ["g1a", "g1b"]}
        )
        score_engine = DictScoringEngine(
            scores_by_response={
                "g0a": 0.4,
                "g0b": 0.2,
                "Gold0": 0.9,
                "g1a": 0.95,
                "g1b": 0.1,
                "Gold1": 0.5,
            }
        )
        output_path = tmp_path / "pairs.jsonl"

        pairs = build_preference_dataset(
            generation_engine=gen_engine,
            scoring_engine=score_engine,
            num_candidates=2,
            construction_mode="max_reward",
            score_gold_output=True,
            output_path=output_path,
            examples=examples,
        )

        by_hash = {pair.hash: pair for pair in pairs}
        assert by_hash["a"].chosen == "Gold0"
        assert by_hash["b"].chosen == "g1a"
        assert all(pair.mode == "max_reward" for pair in pairs)

    def test_resume_skips_already_built_examples(self, tmp_path: Path) -> None:
        """A second run with resume should skip examples already on disk."""
        examples = [
            DataExample(instruction="P0", output="G0", evolution=1, hash="a"),
            DataExample(instruction="P1", output="G1", evolution=2, hash="b"),
        ]
        gen_engine = DictGenerationEngine(
            responses_by_prompt={"P0": ["g0a", "g0b"], "P1": ["g1a", "g1b"]}
        )
        score_engine = DictScoringEngine(
            scores_by_response={"g0a": 0.9, "g0b": 0.1, "g1a": 0.8, "g1b": 0.2}
        )
        output_path = tmp_path / "pairs.jsonl"

        first = build_preference_dataset(
            generation_engine=gen_engine,
            scoring_engine=score_engine,
            num_candidates=2,
            construction_mode="generated",
            score_gold_output=False,
            output_path=output_path,
            examples=examples[:1],
        )
        assert len(first) == 1

        second = build_preference_dataset(
            generation_engine=gen_engine,
            scoring_engine=score_engine,
            num_candidates=2,
            construction_mode="generated",
            score_gold_output=False,
            output_path=output_path,
            examples=examples,
            resume=True,
        )

        assert len(second) == 2
        assert {pair.hash for pair in second} == {"a", "b"}


class FakeGenerationEngine(GenerationEngine):
    """Fake generation engine for testing."""

    def __init__(self, responses_per_prompt: list[list[str]]) -> None:
        """Initialise with predefined responses.

        Args:
            responses_per_prompt:
                List of response lists, one per prompt.
        """
        self._responses = responses_per_prompt

    def generate(self, *, prompts: list[str], num_candidates: int) -> list[list[str]]:
        """Return predefined responses.

        Args:
            prompts:
                List of prompts (ignored, uses predefined responses).
            num_candidates:
                Number of candidates (ignored, uses predefined responses).

        Returns:
            The predefined responses per prompt.
        """
        return self._responses


class FakeScoringEngine(ScoringEngine):
    """Fake scoring engine for testing."""

    def __init__(self, scores: list[float]) -> None:
        """Initialise with predefined scores.

        Args:
            scores:
                List of scores, one per (prompt, response) pair.
        """
        self._scores = scores

    def score(self, *, prompts: list[str], responses: list[str]) -> list[float]:
        """Return predefined scores.

        Args:
            prompts:
                List of prompts (ignored).
            responses:
                List of responses (ignored).

        Returns:
            The predefined scores.
        """
        return self._scores


def _make_config(
    construction_mode: str = "generated", score_gold_output: bool = True
) -> PipelineConfig:
    """Create a minimal pipeline config for tests.

    Args:
        construction_mode:
            Either "generated" or "gold_chosen".
        score_gold_output:
            Whether to score gold output in gold_chosen mode.

    Returns:
        A minimal PipelineConfig for testing.
    """
    return PipelineConfig(
        construction_mode=construction_mode,
        score_gold_output=score_gold_output,
        language="da",
        policy=PolicyModelConfig(
            model_id="test/policy", attn_implementation="sdpa", max_model_len=4096
        ),
        reward=RewardModelConfig(model_id="test/reward", max_model_len=8192),
        generation=GenerationConfig(
            num_candidates=4,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        ),
        data=DataConfig(
            dataset_id="test/dataset",
            subset="test",
            split="train",
            num_samples=100,
            stratify_by_evolution=False,
            evolution_min=None,
            evolution_max=None,
            max_prompt_tokens=2048,
            seed=42,
        ),
        dpo=DPOTrainConfig(
            output_dir="/tmp/dpo_test",
            learning_rate=5e-6,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            max_length=1024,
            bf16=False,
            gradient_checkpointing=False,
            curriculum=False,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            seed=42,
        ),
        eval=EvalConfig(language="da", tasks=None, num_iterations=1),
    )


class TestBuildPairGenerated:
    """Tests for build_pair_generated with various scenarios."""

    def test_build_pair_generated_basic(self) -> None:
        """Should build a pair with highest score as chosen."""
        candidates = [
            ScoredCandidate(response="Response A", reward_score=5.0),
            ScoredCandidate(response="Response B", reward_score=8.0),
            ScoredCandidate(response="Response C", reward_score=3.0),
        ]

        pair = build_pair_generated(
            prompt="Test instruction", candidates=candidates, evolution=2, hash="abc123"
        )

        assert pair is not None
        assert pair.prompt == "Test instruction"
        assert pair.chosen == "Response B"
        assert pair.chosen_score == 8.0
        assert pair.mode == "generated"
        assert pair.evolution == 2
        assert pair.hash == "abc123"
        assert pair.pool_size == 3

    def test_build_pair_generated_insufficient_candidates(self) -> None:
        """Should return None with fewer than 2 candidates."""
        candidates = [ScoredCandidate(response="Only response", reward_score=7.0)]

        pair = build_pair_generated(prompt="Test", candidates=candidates)

        assert pair is None

    def test_build_pair_generated_selects_target_rejected(self) -> None:
        """Should select rejected nearest to mu - 2*sigma."""
        scores = [10.0, 9.0, 8.0, 7.0, 1.0]
        candidates = [
            ScoredCandidate(response=f"Response {i}", reward_score=score)
            for i, score in enumerate(scores)
        ]

        pair = build_pair_generated(prompt="Test", candidates=candidates)

        assert pair is not None
        assert pair.chosen == "Response 0"
        assert pair.chosen_score == 10.0
        assert pair.rejected_score < pair.chosen_score


class TestBuildPairGoldChosen:
    """Tests for build_pair_gold_chosen with various scenarios."""

    def test_build_pair_gold_chosen_basic(self) -> None:
        """Should build a pair with gold as chosen."""
        candidates = [
            ScoredCandidate(response="Generated A", reward_score=5.0),
            ScoredCandidate(response="Generated B", reward_score=7.0),
        ]

        pair = build_pair_gold_chosen(
            prompt="Test instruction",
            gold_output="Gold standard output",
            candidates=candidates,
            gold_score=9.0,
            evolution=1,
            hash="def456",
        )

        assert pair is not None
        assert pair.prompt == "Test instruction"
        assert pair.chosen == "Gold standard output"
        assert pair.chosen_score == 9.0
        assert pair.mode == "gold_chosen"
        assert pair.evolution == 1
        assert pair.hash == "def456"
        assert pair.pool_size == 2
        assert pair.rejected_score < pair.chosen_score

    def test_build_pair_gold_chosen_requires_candidates(self) -> None:
        """Should return None with no candidates."""
        candidates = []

        pair = build_pair_gold_chosen(
            prompt="Test", gold_output="Gold", candidates=candidates
        )

        assert pair is None

    def test_build_pair_gold_chosen_respects_upper_bound(self) -> None:
        """Should respect gold_score as upper bound for rejected."""
        candidates = [
            ScoredCandidate(response="High score", reward_score=9.5),
            ScoredCandidate(response="Medium score", reward_score=7.0),
            ScoredCandidate(response="Low score", reward_score=3.0),
        ]

        pair = build_pair_gold_chosen(
            prompt="Test", gold_output="Gold", candidates=candidates, gold_score=8.0
        )

        assert pair is not None
        assert pair.chosen_score == 8.0
        assert pair.rejected_score < 8.0


def _build_pairs(
    *,
    examples: list[DataExample],
    config: PipelineConfig,
    gen_engine: GenerationEngine,
    score_engine: ScoringEngine,
) -> tuple[list[PreferencePair], list[float | None]]:
    """Build preference pairs from examples.

    Args:
        examples:
            List of data examples.
        config:
            Pipeline configuration.
        gen_engine:
            Generation engine.
        score_engine:
            Scoring engine.

    Returns:
        Tuple of (pairs, gold_scores).
    """
    prompts = [ex.instruction for ex in examples]
    generations = gen_engine.generate(
        prompts=prompts, num_candidates=config.generation.num_candidates
    )

    flat_prompts: list[str] = []
    flat_responses: list[str] = []

    for i, gens in enumerate(generations):
        for gen in gens:
            flat_prompts.append(prompts[i])
            flat_responses.append(gen)

    flat_scores = score_engine.score(prompts=flat_prompts, responses=flat_responses)

    scored_per_prompt: list[list[ScoredCandidate]] = []
    current_idx = 0
    for i, gens in enumerate(generations):
        scored = [
            ScoredCandidate(
                response=flat_responses[current_idx + j],
                reward_score=flat_scores[current_idx + j],
            )
            for j in range(len(gens))
        ]
        scored_per_prompt.append(scored)
        current_idx += len(gens)

    pairs: list[PreferencePair | None] = []
    for i, ex in enumerate(examples):
        pair = build_pair_generated(
            prompt=ex.instruction,
            candidates=scored_per_prompt[i],
            evolution=ex.evolution,
            hash=ex.hash,
        )
        pairs.append(pair)

    return [p for p in pairs if p is not None], [None] * len(examples)


class TestIntegrationGeneratedMode:
    """Integration tests for generated construction mode."""

    def test_generated_mode_with_multiple_examples(self) -> None:
        """Should process multiple examples in generated mode."""
        examples = [
            DataExample(
                instruction="Prompt 1", output="Output 1", evolution=1, hash="h1"
            ),
            DataExample(
                instruction="Prompt 2", output="Output 2", evolution=2, hash="h2"
            ),
        ]

        all_generations = [
            ["Gen1-A", "Gen1-B", "Gen1-C"],
            ["Gen2-A", "Gen2-B", "Gen2-C"],
        ]

        gen_engine = FakeGenerationEngine(responses_per_prompt=all_generations)

        fake_scores = [7.0, 8.0, 6.0, 9.0, 5.0, 10.0]
        score_engine = FakeScoringEngine(scores=fake_scores)

        config = _make_config(construction_mode="generated")

        pairs, gold_scores = _build_pairs(
            examples=examples,
            config=config,
            gen_engine=gen_engine,
            score_engine=score_engine,
        )

        assert len(pairs) == 2
        assert all(pair.mode == "generated" for pair in pairs)


def _build_pairs_gold(
    *,
    examples: list[DataExample],
    config: PipelineConfig,
    gen_engine: GenerationEngine,
    score_engine: ScoringEngine,
) -> tuple[list[PreferencePair], list[float]]:
    """Build preference pairs in gold_chosen mode.

    Args:
        examples:
            List of data examples.
        config:
            Pipeline configuration.
        gen_engine:
            Generation engine.
        score_engine:
            Scoring engine.

    Returns:
        Tuple of (pairs, gold_scores).
    """
    prompts = [ex.instruction for ex in examples]
    generations = gen_engine.generate(
        prompts=prompts, num_candidates=config.generation.num_candidates
    )

    flat_prompts: list[str] = []
    flat_responses: list[str] = []

    for i, gens in enumerate(generations):
        for gen in gens:
            flat_prompts.append(prompts[i])
            flat_responses.append(gen)

    flat_scores = score_engine.score(prompts=flat_prompts, responses=flat_responses)

    gold_prompts = [ex.instruction for ex in examples]
    gold_responses = [ex.output for ex in examples]

    gold_scores = score_engine.score(prompts=gold_prompts, responses=gold_responses)

    scored_per_prompt: list[list[ScoredCandidate]] = []
    current_idx = 0
    for i, gens in enumerate(generations):
        scored = [
            ScoredCandidate(
                response=flat_responses[current_idx + j],
                reward_score=flat_scores[current_idx + j],
            )
            for j in range(len(gens))
        ]
        scored_per_prompt.append(scored)
        current_idx += len(gens)

    pairs: list[PreferencePair | None] = []
    for i, ex in enumerate(examples):
        pair = build_pair_gold_chosen(
            prompt=ex.instruction,
            gold_output=ex.output,
            candidates=scored_per_prompt[i],
            gold_score=gold_scores[i],
            evolution=ex.evolution,
            hash=ex.hash,
        )
        pairs.append(pair)

    return [p for p in pairs if p is not None], gold_scores


class TestIntegrationGoldChosenMode:
    """Integration tests for gold_chosen construction mode."""

    def test_gold_chosen_mode_with_scoring(self) -> None:
        """Should process examples in gold_chosen mode with gold scoring."""
        examples = [
            DataExample(
                instruction="Prompt 1", output="Gold output 1", evolution=1, hash="h1"
            ),
            DataExample(
                instruction="Prompt 2", output="Gold output 2", evolution=2, hash="h2"
            ),
        ]

        all_generations = [["Gen1-A", "Gen1-B"], ["Gen2-A", "Gen2-B"]]

        gen_engine = FakeGenerationEngine(responses_per_prompt=all_generations)

        generation_scores = [5.0, 7.0, 8.0, 6.0]
        gold_scores_list = [9.0, 10.0]
        # FakeScoringEngine returns all scores on each call, so we can't use it
        # for two calls. Instead, we'll use a stateful scorer
        score_call_count = [0]

        class StatefulFakeScoringEngine(ScoringEngine):
            def score(self, *, prompts: list[str], responses: list[str]) -> list[float]:
                if score_call_count[0] == 0:
                    score_call_count[0] += 1
                    return generation_scores
                else:
                    return gold_scores_list

        score_engine = StatefulFakeScoringEngine()

        config = _make_config(construction_mode="gold_chosen", score_gold_output=True)

        pairs, gold_scores = _build_pairs_gold(
            examples=examples,
            config=config,
            gen_engine=gen_engine,
            score_engine=score_engine,
        )

        assert len(pairs) == 2
        assert all(pair.mode == "gold_chosen" for pair in pairs)
        assert all(pair.chosen_score is not None for pair in pairs)


class TestSortByEvolution:
    """Tests for the sort_by_evolution function."""

    def test_sort_by_evolution_ascending(self) -> None:
        """Should sort pairs by evolution in ascending order."""
        pairs = [
            PreferencePair(
                prompt="p3",
                chosen="c3",
                rejected="r3",
                chosen_score=1.0,
                rejected_score=0.5,
                pool_size=2,
                mode="generated",
                evolution=3,
            ),
            PreferencePair(
                prompt="p1",
                chosen="c1",
                rejected="r1",
                chosen_score=1.0,
                rejected_score=0.5,
                pool_size=2,
                mode="generated",
                evolution=1,
            ),
            PreferencePair(
                prompt="p2",
                chosen="c2",
                rejected="r2",
                chosen_score=1.0,
                rejected_score=0.5,
                pool_size=2,
                mode="generated",
                evolution=2,
            ),
        ]

        sorted_pairs = sort_by_evolution(pairs=pairs)

        assert [p.evolution for p in sorted_pairs] == [1, 2, 3]

    def test_sort_by_evolution_with_none_values(self) -> None:
        """Should place None evolution values first."""
        pairs = [
            PreferencePair(
                prompt="p2",
                chosen="c2",
                rejected="r2",
                chosen_score=1.0,
                rejected_score=0.5,
                pool_size=2,
                mode="generated",
                evolution=2,
            ),
            PreferencePair(
                prompt="p1",
                chosen="c1",
                rejected="r1",
                chosen_score=1.0,
                rejected_score=0.5,
                pool_size=2,
                mode="generated",
                evolution=None,
            ),
            PreferencePair(
                prompt="p3",
                chosen="c3",
                rejected="r3",
                chosen_score=1.0,
                rejected_score=0.5,
                pool_size=2,
                mode="generated",
                evolution=3,
            ),
        ]

        sorted_pairs = sort_by_evolution(pairs=pairs)

        assert [p.evolution for p in sorted_pairs] == [None, 2, 3]
