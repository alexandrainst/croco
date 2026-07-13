"""LoRA configuration builder for DPO and GRPO training."""

import typing as t

from peft import LoraConfig


class _HasLoraConfig(t.Protocol):
    """Protocol for training configs with LoRA parameters."""

    lora_r: int
    lora_alpha: int
    lora_dropout: float


def build_lora_config(*, config: _HasLoraConfig) -> LoraConfig:
    """Build a LoRA configuration from training config with LoRA parameters.

    Shared helper for DPO and GRPO training configs. Both configs share the
    same LoRA hyperparameters (r, alpha, dropout) and target modules.

    Args:
        config:
            Training configuration with lora_r, lora_alpha, lora_dropout fields.

    Returns:
        LoraConfig targeting transformer attention and MLP projection layers.
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
