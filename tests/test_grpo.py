"""Unit tests for the GRPO (online RL) training module."""

from pathlib import Path

import pytest

from croco.config import GRPOTrainConfig, load_config
from croco.data_models import DataExample
from croco.grpo import _build_grpo_lora_config, _sort_by_evolution, build_grpo_config

_GRPO_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "danish-apertus-grpo.yaml"
)
_DPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "danish-apertus.yaml"


def _grpo_train_config() -> GRPOTrainConfig:
    """Return a GRPOTrainConfig with representative values.

    Returns:
        A populated GRPOTrainConfig.
    """
    return GRPOTrainConfig(
        output_dir=Path("/tmp/grpo"),
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        beta=0.04,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        curriculum=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        seed=42,
    )


class TestConfigSchema:
    """The grpo/dpo training blocks are mutually optional."""

    def test_grpo_config_loads_with_grpo_block(self) -> None:
        """The GRPO config sets the grpo block and leaves dpo unset."""
        cfg = load_config(path=_GRPO_CONFIG)
        assert cfg.grpo is not None
        assert cfg.dpo is None

    def test_dpo_config_leaves_grpo_unset(self) -> None:
        """The DPO config sets the dpo block and leaves grpo unset."""
        cfg = load_config(path=_DPO_CONFIG)
        assert cfg.dpo is not None
        assert cfg.grpo is None


class TestBuildGrpoConfig:
    """Tests for the GRPO configuration builder."""

    def test_sampling_pulled_from_generation_block(self) -> None:
        """Group size and sampling mirror the shared generation block."""
        cfg = load_config(path=_GRPO_CONFIG)
        result = build_grpo_config(config=cfg)

        assert result.num_generations == cfg.generation.num_candidates
        assert result.temperature == cfg.generation.temperature
        assert result.top_p == cfg.generation.top_p
        assert result.max_completion_length == cfg.generation.max_tokens
        assert result.vllm_max_model_length == cfg.policy.max_model_len

    def test_curriculum_disables_shuffle(self) -> None:
        """Curriculum ordering turns off GRPO's dataset shuffle."""
        cfg = load_config(path=_GRPO_CONFIG)
        result = build_grpo_config(config=cfg)
        assert result.shuffle_dataset is (not cfg.grpo.curriculum)

    def test_save_steps_enables_checkpointing(self) -> None:
        """A positive save_steps enables step-based checkpointing."""
        cfg = load_config(path=_GRPO_CONFIG)
        result = build_grpo_config(config=cfg)
        if cfg.grpo.save_steps > 0:
            assert result.save_strategy == "steps"
            assert result.save_steps == cfg.grpo.save_steps
            assert result.save_total_limit is None

    def test_requires_grpo_block(self) -> None:
        """Building from a config without a grpo block raises."""
        cfg = load_config(path=_DPO_CONFIG)
        with pytest.raises(ValueError, match="grpo"):
            build_grpo_config(config=cfg)


class TestBuildGrpoLoraConfig:
    """Tests for the GRPO LoRA configuration builder."""

    def test_lora_config_values(self) -> None:
        """LoRA config maps all fields from GRPOTrainConfig."""
        lora_config = _build_grpo_lora_config(config=_grpo_train_config())

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


class TestSortByEvolution:
    """Curriculum ordering places easy examples first and unknowns last."""

    def test_ascending_with_none_last(self) -> None:
        """Examples sort by ascending evolution, with None pushed to the end."""
        examples = [
            DataExample(instruction="c", output="x", evolution=5, hash="c"),
            DataExample(instruction="a", output="x", evolution=1, hash="a"),
            DataExample(instruction="n", output="x", evolution=None, hash="n"),
            DataExample(instruction="b", output="x", evolution=3, hash="b"),
        ]
        ordered = _sort_by_evolution(examples=examples)
        assert [e.instruction for e in ordered] == ["a", "b", "c", "n"]
