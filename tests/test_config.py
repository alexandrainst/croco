"""Tests for CroCo configuration models."""

from pathlib import Path

import pytest
import yaml

from croco.config import (
    DataConfig,
    DPOTrainConfig,
    EvalConfig,
    GenerationConfig,
    PipelineConfig,
    PolicyModelConfig,
    RewardModelConfig,
    load_config,
)


class TestPolicyModelConfig:
    """Tests for PolicyModelConfig."""

    def test_create_policy_model_config(self) -> None:
        """Test creating a PolicyModelConfig instance."""
        config = PolicyModelConfig(
            model_id="test/policy", attn_implementation="sdpa", max_model_len=1024
        )
        assert config.model_id == "test/policy"
        assert config.attn_implementation == "sdpa"
        assert config.max_model_len == 1024

    def test_policy_model_config_requires_all_fields(self) -> None:
        """Test that all fields are required."""
        with pytest.raises(Exception):  # noqa: B017
            PolicyModelConfig(model_id="test/policy")  # type: ignore[call-arg]


class TestRewardModelConfig:
    """Tests for RewardModelConfig."""

    def test_create_reward_model_config(self) -> None:
        """Test creating a RewardModelConfig instance."""
        config = RewardModelConfig(model_id="test/reward", max_model_len=512)
        assert config.model_id == "test/reward"
        assert config.max_model_len == 512

    def test_reward_model_config_requires_all_fields(self) -> None:
        """Test that all fields are required."""
        with pytest.raises(Exception):  # noqa: B017
            RewardModelConfig(model_id="test/reward")  # type: ignore[call-arg]


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_create_generation_config(self) -> None:
        """Test creating a GenerationConfig instance."""
        config = GenerationConfig(
            num_candidates=4,
            max_tokens=256,
            temperature=0.7,
            top_p=0.95,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        assert config.num_candidates == 4
        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.9

    def test_generation_config_requires_all_fields(self) -> None:
        """Test that all fields are required."""
        with pytest.raises(Exception):  # noqa: B017
            GenerationConfig(num_candidates=4)  # type: ignore[call-arg]


class TestDataConfig:
    """Tests for DataConfig."""

    def test_create_data_config(self) -> None:
        """Test creating a DataConfig instance."""
        config = DataConfig(
            dataset_id="laerebogen",
            subset="danish",
            split="train",
            num_samples=1000,
            stratify_by_evolution=True,
            evolution_min=0,
            evolution_max=5,
            max_prompt_tokens=512,
            seed=42,
        )
        assert config.dataset_id == "laerebogen"
        assert config.subset == "danish"
        assert config.split == "train"
        assert config.num_samples == 1000
        assert config.stratify_by_evolution is True
        assert config.evolution_min == 0
        assert config.evolution_max == 5
        assert config.max_prompt_tokens == 512
        assert config.seed == 42

    def test_data_config_with_none_evolution_bounds(self) -> None:
        """Test DataConfig with None evolution bounds."""
        config = DataConfig(
            dataset_id="laerebogen",
            subset="danish",
            split="train",
            num_samples=1000,
            stratify_by_evolution=False,
            evolution_min=None,
            evolution_max=None,
            max_prompt_tokens=512,
            seed=42,
        )
        assert config.evolution_min is None
        assert config.evolution_max is None
        assert config.stratify_by_evolution is False

    def test_data_config_requires_all_fields(self) -> None:
        """Test that all fields are required."""
        with pytest.raises(Exception):  # noqa: B017
            DataConfig(dataset_id="laerebogen")  # type: ignore[call-arg]


class TestDPOTrainConfig:
    """Tests for DPOTrainConfig."""

    def test_create_dpo_train_config(self) -> None:
        """Test creating a DPOTrainConfig instance."""
        config = DPOTrainConfig(
            output_dir=Path("output"),
            learning_rate=5e-7,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            max_length=1024,
            bf16=True,
            gradient_checkpointing=True,
            curriculum=False,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            seed=42,
        )
        assert config.output_dir == Path("output")
        assert config.learning_rate == 5e-7
        assert config.lr_scheduler_type == "cosine"
        assert config.warmup_ratio == 0.1
        assert config.weight_decay == 0.01
        assert config.beta == 0.1
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 4
        assert config.num_train_epochs == 3
        assert config.max_length == 1024
        assert config.bf16 is True
        assert config.gradient_checkpointing is True
        assert config.curriculum is False
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.seed == 42

    def test_dpo_train_config_requires_all_fields(self) -> None:
        """Test that all fields are required."""
        with pytest.raises(Exception):  # noqa: B017
            DPOTrainConfig(output_dir=Path("output"))  # type: ignore[call-arg]


class TestEvalConfig:
    """Tests for EvalConfig."""

    def test_create_eval_config(self) -> None:
        """Test creating an EvalConfig instance."""
        config = EvalConfig(
            language="en", tasks=["truthfulqa", "mmlu"], num_iterations=3
        )
        assert config.language == "en"
        assert config.tasks == ["truthfulqa", "mmlu"]
        assert config.num_iterations == 3

    def test_eval_config_with_none_tasks(self) -> None:
        """Test EvalConfig with None tasks."""
        config = EvalConfig(language="da", tasks=None, num_iterations=5)
        assert config.language == "da"
        assert config.tasks is None
        assert config.num_iterations == 5

    def test_eval_config_requires_all_fields(self) -> None:
        """Test that all fields are required."""
        with pytest.raises(Exception):  # noqa: B017
            EvalConfig(language="en")  # type: ignore[call-arg]


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_create_minimal_pipeline_config(self) -> None:
        """Test creating a minimal pipeline configuration."""
        config = PipelineConfig(
            construction_mode="generated",
            score_gold_output=False,
            language="en",
            policy=PolicyModelConfig(
                model_id="test/policy", attn_implementation="sdpa", max_model_len=1024
            ),
            reward=RewardModelConfig(model_id="test/reward", max_model_len=512),
            generation=GenerationConfig(
                num_candidates=4,
                max_tokens=256,
                temperature=0.7,
                top_p=0.95,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
            ),
            data=DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=1000,
                stratify_by_evolution=True,
                evolution_min=0,
                evolution_max=5,
                max_prompt_tokens=512,
                seed=42,
            ),
            dpo=DPOTrainConfig(
                output_dir=Path("output"),
                learning_rate=5e-7,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                weight_decay=0.01,
                beta=0.1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=3,
                max_length=1024,
                bf16=True,
                gradient_checkpointing=True,
                curriculum=False,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                seed=42,
            ),
            eval=EvalConfig(language="en", tasks=None, num_iterations=3),
        )
        assert config.construction_mode == "generated"
        assert config.score_gold_output is False
        assert config.language == "en"
        assert config.policy.model_id == "test/policy"
        assert config.reward.model_id == "test/reward"
        assert config.generation.num_candidates == 4
        assert config.data.dataset_id == "laerebogen"
        assert config.dpo.beta == 0.1
        assert config.eval.language == "en"

    def test_create_gold_chosen_pipeline_config(self) -> None:
        """Test creating a gold_chosen mode pipeline configuration."""
        config = PipelineConfig(
            construction_mode="gold_chosen",
            score_gold_output=True,
            language="da",
            policy=PolicyModelConfig(
                model_id="danish/policy",
                attn_implementation="eager",
                max_model_len=2048,
            ),
            reward=RewardModelConfig(model_id="danish/reward", max_model_len=1024),
            generation=GenerationConfig(
                num_candidates=8,
                max_tokens=512,
                temperature=0.8,
                top_p=0.9,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.85,
            ),
            data=DataConfig(
                dataset_id="laerebogen",
                subset="danish",
                split="train",
                num_samples=2000,
                stratify_by_evolution=True,
                evolution_min=1,
                evolution_max=3,
                max_prompt_tokens=256,
                seed=123,
            ),
            dpo=DPOTrainConfig(
                output_dir=Path("dpo-output"),
                learning_rate=1e-6,
                lr_scheduler_type="linear",
                warmup_ratio=0.05,
                weight_decay=0.001,
                beta=0.5,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                num_train_epochs=5,
                max_length=2048,
                bf16=False,
                gradient_checkpointing=False,
                curriculum=True,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                seed=123,
            ),
            eval=EvalConfig(
                language="da", tasks=["mmlu", "hellaswag"], num_iterations=5
            ),
        )
        assert config.construction_mode == "gold_chosen"
        assert config.score_gold_output is True
        assert config.language == "da"
        assert config.policy.attn_implementation == "eager"
        assert config.reward.max_model_len == 1024
        assert config.generation.tensor_parallel_size == 2
        assert config.data.stratify_by_evolution is True
        assert config.dpo.curriculum is True
        assert config.eval.tasks == ["mmlu", "hellaswag"]


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading configuration from a YAML file."""
        config_dict = {
            "construction_mode": "generated",
            "score_gold_output": False,
            "language": "en",
            "policy": {
                "model_id": "test/policy",
                "attn_implementation": "sdpa",
                "max_model_len": 1024,
            },
            "reward": {"model_id": "test/reward", "max_model_len": 512},
            "generation": {
                "num_candidates": 4,
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.95,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            },
            "data": {
                "dataset_id": "laerebogen",
                "subset": "danish",
                "split": "train",
                "num_samples": 1000,
                "stratify_by_evolution": True,
                "evolution_min": 0,
                "evolution_max": 5,
                "max_prompt_tokens": 512,
                "seed": 42,
            },
            "dpo": {
                "output_dir": "output",
                "learning_rate": 5e-7,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "beta": 0.1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "num_train_epochs": 3,
                "max_length": 1024,
                "bf16": True,
                "gradient_checkpointing": True,
                "curriculum": False,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "seed": 42,
            },
            "eval": {"language": "en", "tasks": None, "num_iterations": 3},
        }

        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_dict, f)

        config = load_config(path=config_path)

        assert config.construction_mode == "generated"
        assert config.language == "en"
        assert config.policy.model_id == "test/policy"
        assert config.reward.model_id == "test/reward"
        assert config.generation.num_candidates == 4
        assert config.data.num_samples == 1000
        assert config.dpo.beta == 0.1
        assert config.eval.num_iterations == 3

    def test_load_with_all_fields(self, tmp_path: Path) -> None:
        """Test loading configuration with all fields set."""
        config_dict = {
            "construction_mode": "gold_chosen",
            "score_gold_output": True,
            "language": "da",
            "policy": {
                "model_id": "danish/policy",
                "attn_implementation": "eager",
                "max_model_len": 2048,
            },
            "reward": {"model_id": "danish/reward", "max_model_len": 1024},
            "generation": {
                "num_candidates": 8,
                "max_tokens": 512,
                "temperature": 0.8,
                "top_p": 0.9,
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.85,
            },
            "data": {
                "dataset_id": "laerebogen",
                "subset": "danish",
                "split": "train",
                "num_samples": 2000,
                "stratify_by_evolution": True,
                "evolution_min": 1,
                "evolution_max": 3,
                "max_prompt_tokens": 256,
                "seed": 123,
            },
            "dpo": {
                "output_dir": "dpo-output",
                "learning_rate": 1e-6,
                "lr_scheduler_type": "linear",
                "warmup_ratio": 0.05,
                "weight_decay": 0.001,
                "beta": 0.5,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "num_train_epochs": 5,
                "max_length": 2048,
                "bf16": False,
                "gradient_checkpointing": False,
                "curriculum": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "seed": 123,
            },
            "eval": {
                "language": "da",
                "tasks": ["mmlu", "hellaswag"],
                "num_iterations": 5,
            },
        }

        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_dict, f)

        config = load_config(path=config_path)

        assert config.construction_mode == "gold_chosen"
        assert config.score_gold_output is True
        assert config.language == "da"
        assert config.policy.attn_implementation == "eager"
        assert config.generation.tensor_parallel_size == 2
        assert config.data.stratify_by_evolution is True
        assert config.dpo.curriculum is True
        assert config.eval.tasks == ["mmlu", "hellaswag"]


class TestLengthBudgetValidator:
    """Tests for the prompt + generation length budget validator."""

    def _build(
        self, *, max_model_len: int, max_prompt_tokens: int, max_tokens: int
    ) -> PipelineConfig:
        """Build a pipeline config with the given length settings.

        Args:
            max_model_len:
              Policy context window.
            max_prompt_tokens:
              Maximum allowed prompt length.
            max_tokens:
              Generation length budget.

        Returns:
            The constructed pipeline configuration.
        """
        return PipelineConfig(
            construction_mode="generated",
            score_gold_output=False,
            language="da",
            policy=PolicyModelConfig(
                model_id="test/policy",
                attn_implementation="sdpa",
                max_model_len=max_model_len,
            ),
            reward=RewardModelConfig(model_id="test/reward", max_model_len=8192),
            generation=GenerationConfig(
                num_candidates=4,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
            ),
            data=DataConfig(
                dataset_id="test/dataset",
                subset="test",
                split="train",
                num_samples=10,
                stratify_by_evolution=False,
                evolution_min=None,
                evolution_max=None,
                max_prompt_tokens=max_prompt_tokens,
                seed=42,
            ),
            dpo=DPOTrainConfig(
                output_dir=Path("output"),
                learning_rate=5e-6,
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
                weight_decay=0.01,
                beta=0.1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                num_train_epochs=1,
                max_length=4096,
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

    def test_within_budget_is_valid(self) -> None:
        """A prompt + generation budget within max_model_len is accepted."""
        config = self._build(
            max_model_len=4096, max_prompt_tokens=3072, max_tokens=1024
        )
        assert config.policy.max_model_len == 4096

    def test_over_budget_raises(self) -> None:
        """Exceeding max_model_len raises a ValueError."""
        with pytest.raises(ValueError, match="exceeds policy"):
            self._build(max_model_len=2048, max_prompt_tokens=2048, max_tokens=1024)
