"""Unit tests for the evaluation module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from croco.config import (
    DataConfig,
    DPOTrainConfig,
    EvalConfig,
    GenerationConfig,
    PipelineConfig,
    PolicyModelConfig,
    RewardModelConfig,
)
from croco.evaluation import evaluate_model, extract_scores


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    @patch("croco.evaluation.Benchmarker")
    def test_evaluate_model_calls_benchmarker_correctly(
        self, mock_benchmarker_class: MagicMock
    ) -> None:
        """Should instantiate Benchmarker with correct args and call benchmark."""
        mock_benchmarker = MagicMock()
        mock_benchmarker.benchmark.return_value = []
        mock_benchmarker_class.return_value = mock_benchmarker

        eval_config = EvalConfig(
            language="da", tasks=["mmlu", "xquad"], num_iterations=5
        )
        config = PipelineConfig(
            construction_mode="generated",
            score_gold_output=True,
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
            eval=eval_config,
        )

        results = evaluate_model(
            model_id_or_path=Path("/tmp/test_model"), config=config
        )

        mock_benchmarker_class.assert_called_once_with(
            language="da",
            progress_bar=True,
            save_results=True,
            num_iterations=5,
            gpu_memory_utilization=0.5,
        )
        mock_benchmarker.benchmark.assert_called_once_with(
            model="/tmp/test_model", task=["mmlu", "xquad"]
        )
        assert results == []

    @patch("croco.evaluation.Benchmarker")
    def test_evaluate_model_with_string_path(
        self, mock_benchmarker_class: MagicMock
    ) -> None:
        """The function should accept string paths as well."""
        mock_benchmarker = MagicMock()
        mock_benchmarker.benchmark.return_value = []
        mock_benchmarker_class.return_value = mock_benchmarker

        eval_config = EvalConfig(language="en", tasks=None, num_iterations=10)
        config = PipelineConfig(
            construction_mode="generated",
            score_gold_output=False,
            language="en",
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
            eval=eval_config,
        )

        evaluate_model(model_id_or_path="/models/my_model", config=config)

        mock_benchmarker.benchmark.assert_called_once_with(
            model="/models/my_model", task=None
        )

    @patch("croco.evaluation.Benchmarker")
    def test_evaluate_model_returns_list_of_results(
        self, mock_benchmarker_class: MagicMock
    ) -> None:
        """The function should convert benchmark results to a list."""
        mock_benchmarker = MagicMock()
        mock_result1 = MagicMock()
        mock_result2 = MagicMock()
        mock_benchmarker.benchmark.return_value = iter([mock_result1, mock_result2])
        mock_benchmarker_class.return_value = mock_benchmarker

        eval_config = EvalConfig(language="da", tasks=None, num_iterations=1)
        config = PipelineConfig(
            construction_mode="generated",
            score_gold_output=False,
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
            eval=eval_config,
        )

        results = evaluate_model(model_id_or_path=Path("/tmp/model"), config=config)

        assert len(results) == 2
        assert results[0] is mock_result1
        assert results[1] is mock_result2


class TestExtractScores:
    """Tests for the extract_scores function."""

    def test_extract_scores_empty_results(self) -> None:
        """Should return empty dict for empty results."""
        results = extract_scores(results=[])
        assert results == {}

    def test_extract_scores_single_result(self) -> None:
        """Should extract scores from a single result."""
        mock_result = MagicMock()
        mock_result.dataset = "mmlu"
        mock_result.results = {"total": {"accuracy": 0.75, "f1": 0.80}}

        results = extract_scores(results=[mock_result])

        assert results == {"mmlu": {"accuracy": 0.75, "f1": 0.80}}

    def test_extract_scores_multiple_results(self) -> None:
        """Should extract scores from multiple results."""
        mock_result1 = MagicMock()
        mock_result1.dataset = "mmlu"
        mock_result1.results = {"total": {"accuracy": 0.75, "f1": 0.80}}

        mock_result2 = MagicMock()
        mock_result2.dataset = "xquad"
        mock_result2.results = {"total": {"accuracy": 0.65, "f1": 0.70}}

        mock_result3 = MagicMock()
        mock_result3.dataset = "boolq"
        mock_result3.results = {"total": {"accuracy": 0.85}}

        results = extract_scores(results=[mock_result1, mock_result2, mock_result3])

        assert results == {
            "mmlu": {"accuracy": 0.75, "f1": 0.80},
            "xquad": {"accuracy": 0.65, "f1": 0.70},
            "boolq": {"accuracy": 0.85},
        }

    def test_extract_scores_preserves_all_total_fields(self) -> None:
        """Should preserve all fields in the total results dict."""
        mock_result = MagicMock()
        mock_result.dataset = "custom_dataset"
        mock_result.results = {
            "total": {
                "accuracy": 0.90,
                "f1": 0.88,
                "precision": 0.87,
                "recall": 0.89,
                "custom_metric": 0.95,
            }
        }

        results = extract_scores(results=[mock_result])

        assert results == {
            "custom_dataset": {
                "accuracy": 0.90,
                "f1": 0.88,
                "precision": 0.87,
                "recall": 0.89,
                "custom_metric": 0.95,
            }
        }
