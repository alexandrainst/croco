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
    # Save an intermediate checkpoint every N optimiser steps (0 disables). Each
    # checkpoint is an adapter that can be evaluated to trace the learning curve.
    save_steps: int = 0
    # DPO loss variant (e.g. "sigmoid" for vanilla DPO, "sigmoid_norm" for the
    # SimPO-style length-normalised loss). Passed through to TRL's DPOConfig.
    loss_type: str = "sigmoid"
    # Conservative-DPO label smoothing; >0 assumes that fraction of preference
    # labels are flipped, adding robustness to noisy reward-model judgements.
    label_smoothing: float = 0.0
    # Cache the (frozen) reference log-probabilities once before training instead
    # of recomputing them each step. A speed optimisation, not a recipe change.
    precompute_ref_log_probs: bool = False
    # SimPO target margin γ; only used when loss_type == 'simpo'. Meng et al. 2024
    # (arXiv 2405.14734) recommend γ ≈ 0.5–1.0. Default 0.0 disables the margin.
    target_margin: float = 0.0


class GRPOTrainConfig(BaseModel):
    """Configuration for GRPO (online RL) training.

    The online-RL baseline that skips offline preference construction entirely:
    the policy generates completions during training and the reward model scores
    them on the fly. Rollout sampling (group size, temperature, top_p, completion
    length) and the prompt set are deliberately taken from the shared
    ``generation``/``data`` blocks, so this run draws the SAME self-generation
    distribution as the CroCo pipeline - isolating online-vs-offline preference
    optimisation as the only difference from the max_reward DPO config.
    """

    output_dir: pathlib.Path
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    # KL-penalty coefficient against the (frozen) reference policy.
    beta: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    bf16: bool
    gradient_checkpointing: bool
    # Order prompts easy-to-hard by evolution and disable dataset shuffling, so the
    # online run sees the same curriculum as the DPO runs (GRPO's repeat-sampler is
    # preserved; only its shuffle is turned off).
    curriculum: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    # Use a colocated vLLM engine for fast rollouts. The fraction below is what
    # vLLM may claim of the (unified) GPU memory; keep it modest so the training
    # copy of the policy and the reward model still fit. Tune on the box.
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.3
    hf_repo_id: str | None = None
    hf_token: str | None = None
    save_steps: int = 0
    # Maximum number of checkpoints to keep (None = keep all).
    save_total_limit: int | None = None


class EvalConfig(BaseModel):
    """Configuration for evaluation."""

    language: str
    tasks: list[str] | None
    skip: bool = False  # Skip evaluation (e.g., for LoRA models)
    gpu_memory_utilization: float = 0.5


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    construction_mode: t.Literal["generated", "gold_chosen", "max_reward"]
    language: str
    policy: PolicyModelConfig
    reward: RewardModelConfig
    generation: GenerationConfig
    data: DataConfig
    # Exactly one training block is used per run: ``dpo`` for the offline CroCo
    # pipeline, ``grpo`` for the online-RL baseline. Both default to None so a
    # config need only specify the one it uses.
    dpo: DPOTrainConfig | None = None
    grpo: GRPOTrainConfig | None = None
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
