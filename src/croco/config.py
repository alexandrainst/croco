"""Configuration models for the CroCo training pipeline."""

from __future__ import annotations

import pathlib
import typing as t

import yaml
from pydantic import BaseModel, model_validator


def load_config(*, path: pathlib.Path) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file.

    Args:
        path:
            Path to the YAML configuration file.

    Returns:
        The parsed pipeline configuration.
    """
    content = yaml.safe_load(path.read_text())
    return PipelineConfig(**content)


class PolicyModelConfig(BaseModel):
    """Configuration for the policy model."""

    model_id: str
    attn_implementation: str
    max_model_len: int


class RewardModelConfig(BaseModel):
    """Configuration for the reward model."""

    model_id: str
    max_model_len: int
    gpu_memory_utilization: float = 0.35


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    num_candidates: int
    max_tokens: int
    temperature: float
    top_p: float
    tensor_parallel_size: int
    gpu_memory_utilization: float
    batch_size: int = 64
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    # Skip the vision tower on multimodal policy models (no-op on text-only models).
    language_model_only: bool = False


class DataConfig(BaseModel):
    """Configuration for dataset loading."""

    dataset_id: str
    subset: str
    split: str
    num_samples: int
    stratify_by_evolution: bool
    evolution_min: int | None
    evolution_max: int | None
    max_prompt_tokens: int
    seed: int


class DPOTrainConfig(BaseModel):
    """Configuration for DPO training."""

    output_dir: pathlib.Path
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    beta: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    max_length: int
    bf16: bool
    gradient_checkpointing: bool
    curriculum: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    hf_repo_id: str | None = None  # HuggingFace repo ID for uploading model and dataset


class EvalConfig(BaseModel):
    """Configuration for evaluation."""

    language: str
    tasks: list[str] | None
    num_iterations: int
    skip: bool = False  # Skip evaluation (e.g., for LoRA models)
    gpu_memory_utilization: float = 0.5


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    construction_mode: t.Literal["generated", "gold_chosen", "max_reward"]
    score_gold_output: bool
    language: str
    policy: PolicyModelConfig
    reward: RewardModelConfig
    generation: GenerationConfig
    data: DataConfig
    dpo: DPOTrainConfig
    eval: EvalConfig

    @model_validator(mode="after")
    def _check_length_budget(self) -> PipelineConfig:
        """Ensure prompts plus generations fit within the policy context window.

        Returns:
            The validated configuration.

        Raises:
            ValueError:
                If ``max_prompt_tokens + generation.max_tokens`` exceeds the
                policy's ``max_model_len``, which would let vLLM reject prompts.
        """
        budget = self.data.max_prompt_tokens + self.generation.max_tokens
        if budget > self.policy.max_model_len:
            msg = (
                f"max_prompt_tokens ({self.data.max_prompt_tokens}) + max_tokens "
                f"({self.generation.max_tokens}) = {budget} exceeds policy "
                f"max_model_len ({self.policy.max_model_len})"
            )
            raise ValueError(msg)
        return self
