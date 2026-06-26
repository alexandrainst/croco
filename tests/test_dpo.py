"""Unit tests for the DPO training module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from torch.utils.data import SequentialSampler

from croco.config import DPOTrainConfig
from croco.dpo import CurriculumDPOTrainer, build_dpo_config, build_lora_config


class TestBuildLoraConfig:
    """Tests for the LoRA configuration builder."""

    def test_lora_config_values(self) -> None:
        """LoRA config should map all fields from DPOTrainConfig."""
        dpo_config = DPOTrainConfig(
            output_dir=Path("/tmp/test"),
            learning_rate=5e-6,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            max_length=4096,
            bf16=True,
            gradient_checkpointing=True,
            curriculum=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            seed=42,
        )

        lora_config = build_lora_config(config=dpo_config)

        assert lora_config.r == 16
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.05
        assert set(lora_config.target_modules) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        assert lora_config.bias == "none"
        assert lora_config.task_type == "CAUSAL_LM"

    def test_lora_config_different_values(self) -> None:
        """LoRA config should work with different parameter values."""
        dpo_config = DPOTrainConfig(
            output_dir=Path("/tmp/test2"),
            learning_rate=1e-5,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            weight_decay=0.0,
            beta=0.5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            max_length=2048,
            bf16=False,
            gradient_checkpointing=False,
            curriculum=False,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            seed=123,
        )

        lora_config = build_lora_config(config=dpo_config)

        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
        assert lora_config.lora_dropout == 0.1


class TestBuildDpoConfig:
    """Tests for the DPO configuration builder."""

    def test_dpo_config_values(self) -> None:
        """DPO config should map all fields from DPOTrainConfig."""
        dpo_config = DPOTrainConfig(
            output_dir=Path("/tmp/dpo_output"),
            learning_rate=5e-6,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            max_length=4096,
            bf16=True,
            gradient_checkpointing=True,
            curriculum=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            seed=42,
        )

        result = build_dpo_config(config=dpo_config)

        assert result.output_dir == "/tmp/dpo_output"
        assert result.learning_rate == 5e-6
        assert result.lr_scheduler_type == "cosine"
        assert result.warmup_ratio == 0.05
        assert result.weight_decay == 0.01
        assert result.beta == 0.1
        assert result.per_device_train_batch_size == 1
        assert result.gradient_accumulation_steps == 8
        assert result.num_train_epochs == 1
        assert result.max_length == 4096
        assert result.bf16 is True
        assert result.gradient_checkpointing is True
        assert result.seed == 42

    def test_dpo_config_different_values(self) -> None:
        """DPO config should work with different parameter values."""
        dpo_config = DPOTrainConfig(
            output_dir=Path("/tmp/dpo_other"),
            learning_rate=1e-4,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            weight_decay=0.0,
            beta=0.5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=5,
            max_length=2048,
            bf16=False,
            gradient_checkpointing=False,
            curriculum=False,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            seed=999,
        )

        result = build_dpo_config(config=dpo_config)

        assert result.output_dir == "/tmp/dpo_other"
        assert result.learning_rate == 1e-4
        assert result.lr_scheduler_type == "linear"
        assert result.warmup_ratio == 0.1
        assert result.beta == 0.5
        assert result.per_device_train_batch_size == 4
        assert result.gradient_accumulation_steps == 2
        assert result.num_train_epochs == 5
        assert result.bf16 is False
        assert result.gradient_checkpointing is False
        assert result.seed == 999


class TestCurriculumDPOTrainer:
    """Tests for the CurriculumDPOTrainer class."""

    def test_curriculum_dpo_trainer_exists(self) -> None:
        """The CurriculumDPOTrainer class should exist."""
        assert CurriculumDPOTrainer is not None

    def test_curriculum_dpo_trainer_has_sampler_method(self) -> None:
        """The CurriculumDPOTrainer should have _get_train_sampler method."""
        assert hasattr(CurriculumDPOTrainer, "_get_train_sampler")

    def test_curriculum_dpo_trainer_sampler_returns_sequential(self) -> None:
        """The _get_train_sampler method should return a SequentialSampler."""
        # Create a mock dataset with necessary attributes
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        # Create trainer instance with mock data
        with patch.object(CurriculumDPOTrainer, "__init__", return_value=None):
            trainer = CurriculumDPOTrainer()
            trainer.train_dataset = mock_dataset

            # Get the sampler
            sampler = trainer._get_train_sampler()

            # Verify it's a SequentialSampler
            assert isinstance(sampler, SequentialSampler)
