"""Unit tests for the DPO training module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

from croco.config import DPOTrainConfig
from croco.dpo import (
    CurriculumDPOTrainer,
    CurriculumSimPODPOTrainer,
    SimPODPOTrainer,
    _simpo_sequence_loss,
    build_dpo_config,
    build_lora_config,
)


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


class TestSimPOSequenceLoss:
    """Tests for the _simpo_sequence_loss helper function."""

    def test_simpo_loss_formula(self) -> None:
        """SimPO loss should match the expected formula."""
        # Simple case: chosen better than rejected
        chosen_logps = torch.tensor([-1.0, -2.0])
        rejected_logps = torch.tensor([-3.0, -4.0])
        chosen_lengths = torch.tensor([10.0, 10.0])
        rejected_lengths = torch.tensor([10.0, 10.0])
        beta = 2.0
        target_margin = 0.5

        loss, chosen_rewards, rejected_rewards, reward_accuracies = (
            _simpo_sequence_loss(
                chosen_logps=chosen_logps,
                rejected_logps=rejected_logps,
                chosen_lengths=chosen_lengths,
                rejected_lengths=rejected_lengths,
                beta=beta,
                target_margin=target_margin,
            )
        )

        # Verify length normalisation
        chosen_avg = chosen_logps / chosen_lengths  # [-0.1, -0.2]
        rejected_avg = rejected_logps / rejected_lengths  # [-0.3, -0.4]
        delta = chosen_avg - rejected_avg  # [0.2, 0.2]

        # Expected rewards
        expected_chosen_rewards = beta * chosen_avg
        expected_rejected_rewards = beta * rejected_avg

        assert torch.allclose(chosen_rewards, expected_chosen_rewards)
        assert torch.allclose(rejected_rewards, expected_rejected_rewards)

        # Expected loss: -logsigmoid(beta * delta - target_margin)
        expected_loss = -F.logsigmoid(beta * delta - target_margin)
        assert torch.allclose(loss, expected_loss)

        # Chosen should be better (higher reward) in both cases
        assert torch.all(reward_accuracies == 1.0)

    def test_target_margin_raises_loss(self) -> None:
        """Target margin should raise loss when chosen-rejected gap is small."""
        chosen_logps = torch.tensor([-1.0])
        rejected_logps = torch.tensor([-1.0])  # Same as chosen
        chosen_lengths = torch.tensor([10.0])
        rejected_lengths = torch.tensor([10.0])
        beta = 2.0

        # No margin: delta=0, loss = -logsigmoid(0) = log(2) ≈ 0.693
        loss_no_margin, _, _, _ = _simpo_sequence_loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=beta,
            target_margin=0.0,
        )

        # With margin: loss should be higher
        loss_with_margin, _, _, _ = _simpo_sequence_loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=beta,
            target_margin=0.5,
        )

        assert loss_with_margin > loss_no_margin

    def test_reward_accuracies_flag_chosen_greater(self) -> None:
        """Reward accuracies should flag cases where chosen reward > rejected reward."""
        chosen_logps = torch.tensor(
            [-1.0, -5.0]
        )  # First: chosen better; Second: rejected better
        rejected_logps = torch.tensor([-2.0, -4.0])
        chosen_lengths = torch.tensor([10.0, 10.0])
        rejected_lengths = torch.tensor([10.0, 10.0])

        _, chosen_rewards, rejected_rewards, reward_accuracies = _simpo_sequence_loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
            beta=2.0,
            target_margin=0.0,
        )

        # First sample: chosen (-0.1) > rejected (-0.2) → accuracy = 1
        # Second sample: chosen (-0.5) < rejected (-0.4) → accuracy = 0
        assert reward_accuracies[0] == 1.0
        assert reward_accuracies[1] == 0.0


class TestSimPOTrainers:
    """Tests for SimPODPOTrainer and CurriculumSimPODPOTrainer."""

    def test_simpo_dpo_trainer_exists(self) -> None:
        """The SimPODPOTrainer class should exist."""
        assert SimPODPOTrainer is not None

    def test_curriculum_simpo_dpo_trainer_exists(self) -> None:
        """The CurriculumSimPODPOTrainer class should exist."""
        assert CurriculumSimPODPOTrainer is not None

    def test_curriculum_simpo_trainer_has_sampler_method(self) -> None:
        """CurriculumSimPODPOTrainer should have _get_train_sampler method."""
        assert hasattr(CurriculumSimPODPOTrainer, "_get_train_sampler")

    def test_curriculum_simpo_trainer_sampler_returns_sequential(self) -> None:
        """CurriculumSimPODPOTrainer sampler returns SequentialSampler."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)

        with patch.object(CurriculumSimPODPOTrainer, "__init__", return_value=None):
            trainer = CurriculumSimPODPOTrainer()
            trainer.train_dataset = mock_dataset

            sampler = trainer._get_train_sampler()

            assert isinstance(sampler, SequentialSampler)

    def test_simpo_mixin_in_mro(self) -> None:
        """SimPOLossMixin should be in the MRO of SimPO trainers."""
        from croco.dpo import SimPOLossMixin

        assert SimPOLossMixin in SimPODPOTrainer.__mro__
        assert SimPOLossMixin in CurriculumSimPODPOTrainer.__mro__
