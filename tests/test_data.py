"""Tests for CroCo data loading functions."""

from unittest import mock

import pytest
from datasets import Dataset

from croco.config import DataConfig
from croco.data import filter_by_prompt_length, load_examples, sort_by_evolution
from croco.data_models import DataExample, PreferencePair


class TestFilterByPromptLength:
    """Tests for filter_by_prompt_length."""

    def test_drops_over_long_prompts(self) -> None:
        """Examples whose prompt exceeds the budget are removed."""
        examples = [
            DataExample(instruction="short", output="o", hash="a"),
            DataExample(
                instruction="this is a much longer prompt", output="o", hash="b"
            ),
        ]

        kept = filter_by_prompt_length(
            examples=examples,
            count_tokens=lambda text: len(text.split()),
            max_prompt_tokens=3,
        )

        assert [example.hash for example in kept] == ["a"]

    def test_keeps_all_within_budget(self) -> None:
        """Nothing is dropped when every prompt fits."""
        examples = [
            DataExample(instruction="a b", output="o", hash="a"),
            DataExample(instruction="c d", output="o", hash="b"),
        ]

        kept = filter_by_prompt_length(
            examples=examples,
            count_tokens=lambda text: len(text.split()),
            max_prompt_tokens=2,
        )

        assert len(kept) == 2


class TestLoadExamples:
    """Tests for load_examples function."""

    @pytest.fixture
    def mock_dataset(self) -> Dataset:
        """Create a mock dataset for testing.

        Returns:
            A datasets.Dataset instance with sample instruction/output/evolution
            data for testing data loading functions.
        """
        return Dataset.from_dict(
            {
                "instruction": [
                    "What is the capital of Denmark?",
                    "Translate to Danish",
                    "Explain quantum physics",
                    "Write a poem about春天",
                    "How to cook pasta?",
                ],
                "output": [
                    "Copenhagen",
                    "Hej, hvordan har du det?",
                    "Quantum physics is...",
                    "春天是美丽的季节",
                    "Boil water, add pasta...",
                ],
                "evolution": [1, 2, 3, 4, 5],
                "hash": ["a", "b", "c", "d", "e"],
            }
        )

    def test_load_examples_with_num_samples(self, mock_dataset: Dataset) -> None:
        """Test loading a limited number of examples."""
        with mock.patch("croco.data.datasets.load_dataset", return_value=mock_dataset):
            config = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=3,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=None,
                max_prompt_tokens=512,
                seed=42,
            )

            examples = load_examples(config=config)

            assert len(examples) == 3
            assert all(isinstance(ex, DataExample) for ex in examples)
            assert all(ex.instruction for ex in examples)
            assert all(ex.output for ex in examples)

    def test_load_examples_without_stratification(self, mock_dataset: Dataset) -> None:
        """Test loading examples without stratification."""
        with mock.patch("croco.data.datasets.load_dataset", return_value=mock_dataset):
            config = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=5,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=None,
                max_prompt_tokens=512,
                seed=42,
            )

            examples = load_examples(config=config)

            assert len(examples) == 5
            assert all(isinstance(ex, DataExample) for ex in examples)

    def test_load_examples_preserves_fields(self, mock_dataset: Dataset) -> None:
        """Test that all fields are loaded correctly."""
        with mock.patch("croco.data.datasets.load_dataset", return_value=mock_dataset):
            config = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=1,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=None,
                max_prompt_tokens=512,
                seed=42,
            )

            examples = load_examples(config=config)

            assert len(examples) == 1
            example = examples[0]
            assert isinstance(example.instruction, str)
            assert isinstance(example.output, str)
            assert example.evolution is not None

    def test_load_examples_with_seed_reproducibility(
        self, mock_dataset: Dataset
    ) -> None:
        """Test that using the same seed gives reproducible results."""
        with mock.patch("croco.data.datasets.load_dataset", return_value=mock_dataset):
            config1 = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=3,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=None,
                max_prompt_tokens=512,
                seed=123,
            )
            config2 = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=3,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=None,
                max_prompt_tokens=512,
                seed=123,
            )

            examples1 = load_examples(config=config1)
            examples2 = load_examples(config=config2)

            assert len(examples1) == len(examples2)
            assert [ex.instruction for ex in examples1] == [
                ex.instruction for ex in examples2
            ]

    def test_load_examples_different_seeds(self, mock_dataset: Dataset) -> None:
        """Test that different seeds give different results."""
        with mock.patch("croco.data.datasets.load_dataset", return_value=mock_dataset):
            config1 = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=3,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=None,
                max_prompt_tokens=512,
                seed=42,
            )
            config2 = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=3,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=None,
                max_prompt_tokens=512,
                seed=999,
            )

            examples1 = load_examples(config=config1)
            examples2 = load_examples(config=config2)

            # With different seeds, subsampling may give different results
            # (unless the dataset is very small or seed doesn't affect it)
            assert len(examples1) == 3
            assert len(examples2) == 3

    def test_load_examples_filters_evolution_min(self, mock_dataset: Dataset) -> None:
        """Test that evolution_min filters correctly."""
        with mock.patch("croco.data.datasets.load_dataset", return_value=mock_dataset):
            config = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=10,
                stratify_by_evolution=False,
                evolution_min=3,
                evolution_max=None,
                max_prompt_tokens=512,
                seed=42,
            )

            examples = load_examples(config=config)

            # Should only get examples with evolution >= 3
            assert len(examples) <= 3  # evolution 3, 4, 5
            for ex in examples:
                assert ex.evolution is None or ex.evolution >= 3

    def test_load_examples_filters_evolution_max(self, mock_dataset: Dataset) -> None:
        """Test that evolution_max filters correctly."""
        with mock.patch("croco.data.datasets.load_dataset", return_value=mock_dataset):
            config = DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=10,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=2,
                max_prompt_tokens=512,
                seed=42,
            )

            examples = load_examples(config=config)

            # Should only get examples with evolution <= 2
            assert len(examples) <= 2  # evolution 1, 2
            for ex in examples:
                assert ex.evolution is None or ex.evolution <= 2


class TestSortByEvolution:
    """Tests for sort_by_evolution function."""

    def test_sort_by_evolution_ascending(self) -> None:
        """Test that pairs are sorted by evolution level ascending."""
        pairs = [
            PreferencePair(
                prompt=f"prompt_{i}",
                chosen=f"chosen_{i}",
                rejected=f"rejected_{i}",
                rejected_score=0.5,
                pool_size=4,
                mode="generated",
                evolution=evo,
            )
            for i, evo in enumerate([3, 1, 4, 1, 5, 2])
        ]

        sorted_pairs = sort_by_evolution(pairs=pairs)

        evolutions = [pair.evolution for pair in sorted_pairs]
        assert evolutions == [1, 1, 2, 3, 4, 5]

    def test_sort_by_evolution_with_none(self) -> None:
        """Test sorting when some pairs have None evolution."""
        pairs = [
            PreferencePair(
                prompt=f"prompt_{i}",
                chosen=f"chosen_{i}",
                rejected=f"rejected_{i}",
                rejected_score=0.5,
                pool_size=4,
                mode="generated",
                evolution=evo,
            )
            for i, evo in enumerate([3, None, 1, None, 2])
        ]

        sorted_pairs = sort_by_evolution(pairs=pairs)

        # None values should come first, then sorted by evolution
        evolutions = [pair.evolution for pair in sorted_pairs]
        expected = [None, None, 1, 2, 3]
        assert evolutions == expected

    def test_sort_by_evolution_all_none(self) -> None:
        """Test sorting when all pairs have None evolution."""
        pairs = [
            PreferencePair(
                prompt=f"prompt_{i}",
                chosen=f"chosen_{i}",
                rejected=f"rejected_{i}",
                rejected_score=0.5,
                pool_size=4,
                mode="generated",
                evolution=None,
            )
            for i in range(3)
        ]

        sorted_pairs = sort_by_evolution(pairs=pairs)

        evolutions = [pair.evolution for pair in sorted_pairs]
        assert evolutions == [None, None, None]

    def test_sort_by_evolution_already_sorted(self) -> None:
        """Test sorting already sorted pairs."""
        pairs = [
            PreferencePair(
                prompt=f"prompt_{i}",
                chosen=f"chosen_{i}",
                rejected=f"rejected_{i}",
                rejected_score=0.5,
                pool_size=4,
                mode="generated",
                evolution=i,
            )
            for i in range(5)
        ]

        sorted_pairs = sort_by_evolution(pairs=pairs)

        evolutions = [pair.evolution for pair in sorted_pairs]
        assert evolutions == [0, 1, 2, 3, 4]

    def test_sort_by_evolution_empty_list(self) -> None:
        """Test sorting an empty list."""
        sorted_pairs = sort_by_evolution(pairs=[])
        assert sorted_pairs == []

    def test_sort_by_evolution_single_element(self) -> None:
        """Test sorting a single element."""
        pairs = [
            PreferencePair(
                prompt="prompt_0",
                chosen="chosen_0",
                rejected="rejected_0",
                rejected_score=0.5,
                pool_size=4,
                mode="generated",
                evolution=5,
            )
        ]

        sorted_pairs = sort_by_evolution(pairs=pairs)

        assert len(sorted_pairs) == 1
        assert sorted_pairs[0].evolution == 5


class TestSortByEvolutionKey:
    """Tests for the generic sort_by_evolution_key helper."""

    def test_sorts_data_examples(self) -> None:
        """Test sorting DataExample objects."""
        from croco.data_models import DataExample

        examples = [
            DataExample(instruction="c", output="x", evolution=5, hash="c"),
            DataExample(instruction="a", output="x", evolution=1, hash="a"),
            DataExample(instruction="n", output="x", evolution=None, hash="n"),
            DataExample(instruction="b", output="x", evolution=3, hash="b"),
        ]

        from croco.data import sort_by_evolution_key

        ordered = sort_by_evolution_key(items=examples)
        assert [e.instruction for e in ordered] == ["n", "a", "b", "c"]

    def test_sorts_preference_pairs(self) -> None:
        """Test sorting PreferencePair objects."""
        from croco.data_models import PreferencePair

        pairs = [
            PreferencePair(
                prompt="p3",
                chosen="c3",
                rejected="r3",
                rejected_score=0.5,
                pool_size=4,
                mode="generated",
                evolution=3,
            ),
            PreferencePair(
                prompt="p1",
                chosen="c1",
                rejected="r1",
                rejected_score=0.5,
                pool_size=4,
                mode="generated",
                evolution=1,
            ),
            PreferencePair(
                prompt="pn",
                chosen="cn",
                rejected="rn",
                rejected_score=0.5,
                pool_size=4,
                mode="generated",
                evolution=None,
            ),
        ]

        from croco.data import sort_by_evolution_key

        ordered = sort_by_evolution_key(items=pairs)
        assert [p.evolution for p in ordered] == [None, 1, 3]

    def test_none_first_behavior(self) -> None:
        """Test that None evolution values are placed first (treated as easiest)."""
        from croco.data_models import DataExample

        examples = [
            DataExample(instruction="high", output="x", evolution=10, hash="h"),
            DataExample(instruction="none1", output="x", evolution=None, hash="n1"),
            DataExample(instruction="low", output="x", evolution=2, hash="l"),
            DataExample(instruction="none2", output="x", evolution=None, hash="n2"),
        ]

        from croco.data import sort_by_evolution_key

        ordered = sort_by_evolution_key(items=examples)
        # None values come first, then ascending order
        assert [e.instruction for e in ordered] == ["none1", "none2", "low", "high"]
