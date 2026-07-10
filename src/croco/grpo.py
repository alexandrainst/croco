"""TRL GRPO (online RL) training - the no-preference-construction baseline.

This is the ablation that tests whether CroCo's offline contrastive-DPO pipeline
actually beats simply optimising the policy against the reward model online. The
policy generates a group of completions per prompt during training, the reward
model scores them, and GRPO updates the policy with the group-relative advantage.

To keep the comparison tight, the rollout sampling (group size, temperature,
top_p, completion length) and the prompt set are taken from the same
``generation``/``data`` blocks the CroCo pipeline uses, so the only difference
from the winning max_reward DPO run is online-group-relative vs offline-contrastive
preference optimisation.
"""

from __future__ import annotations

import logging
import typing as t
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .config import GRPOTrainConfig, PipelineConfig
from .data import filter_by_prompt_length, load_examples
from .data_models import DataExample
from .utils import build_user_message

logger = logging.getLogger(__name__)


def train_grpo(*, config: PipelineConfig) -> Path:
    """Train a policy model online with GRPO against the reward model.

    Loads the same prompts the CroCo pipeline uses, then lets GRPO generate and
    score completions on the fly (no preference dataset is built). The trained
    adapter is saved to the configured output directory.

    Args:
        config:
            The full pipeline configuration. Its ``grpo`` block must be set.

    Returns:
        Path to the directory where the trained adapter was saved.

    Raises:
        ValueError:
            If the configuration has no ``grpo`` block.
    """
    if config.grpo is None:
        msg = "train_grpo requires a 'grpo' block in the configuration."
        raise ValueError(msg)
    grpo_config = config.grpo

    tokenizer = AutoTokenizer.from_pretrained(config.policy.model_id)

    dataset = _build_prompt_dataset(config=config, tokenizer=tokenizer)
    logger.info("Training on %d prompts", len(dataset))

    peft_config = _build_grpo_lora_config(config=grpo_config)
    grpo_cfg = build_grpo_config(config=config)

    logger.info("Initialising GRPOTrainer (reward model %s)", config.reward.model_id)
    trainer = GRPOTrainer(
        model=config.policy.model_id,
        reward_funcs=config.reward.model_id,
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting GRPO training")
    trainer.train()

    output_dir = Path(grpo_config.output_dir)
    logger.info("Saving model to %s", output_dir)
    trainer.save_model(str(output_dir))

    return output_dir


def build_grpo_config(*, config: PipelineConfig) -> GRPOConfig:
    """Build a TRL GRPOConfig from the pipeline configuration.

    Sampling (group size, temperature, top_p, completion length) is pulled from
    the shared ``generation`` block so the online rollouts match the CroCo
    self-generation distribution.

    Args:
        config:
            The full pipeline configuration with a ``grpo`` block.

    Returns:
        A GRPOConfig with all training and rollout hyperparameters.

    Raises:
        ValueError:
            If the configuration has no ``grpo`` block.
    """
    if config.grpo is None:
        msg = "build_grpo_config requires a 'grpo' block in the configuration."
        raise ValueError(msg)
    grpo = config.grpo

    save_kwargs: dict[str, t.Any] = {}
    if grpo.save_steps > 0:
        # Keep every checkpoint so the learning curve can be evaluated, matching
        # the DPO runs (unless save_total_limit is set).
        save_kwargs = {
            "save_strategy": "steps",
            "save_steps": grpo.save_steps,
            "save_total_limit": grpo.save_total_limit,
        }

    # Push to Hub settings (optional)
    hub_kwargs: dict[str, t.Any] = {}
    if grpo.hf_repo_id is not None:
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": grpo.hf_repo_id,
            "hub_strategy": "checkpoint",  # Push at each save_steps interval
        }
        if grpo.hf_token is not None:
            hub_kwargs["hub_token"] = grpo.hf_token

    return GRPOConfig(
        output_dir=str(grpo.output_dir),
        learning_rate=grpo.learning_rate,
        lr_scheduler_type=grpo.lr_scheduler_type,
        warmup_ratio=grpo.warmup_ratio,
        weight_decay=grpo.weight_decay,
        beta=grpo.beta,
        num_generations=config.generation.num_candidates,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        max_completion_length=config.generation.max_tokens,
        per_device_train_batch_size=grpo.per_device_train_batch_size,
        gradient_accumulation_steps=grpo.gradient_accumulation_steps,
        num_train_epochs=grpo.num_train_epochs,
        bf16=grpo.bf16,
        gradient_checkpointing=grpo.gradient_checkpointing,
        seed=grpo.seed,
        use_vllm=grpo.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=grpo.vllm_gpu_memory_utilization,
        vllm_max_model_length=config.policy.max_model_len,
        # Curriculum = ascending-evolution order with shuffling off; GRPO's
        # repeat-sampler (which groups generations) is otherwise untouched.
        shuffle_dataset=not grpo.curriculum,
        **save_kwargs,
        **hub_kwargs,
    )


def _build_prompt_dataset(*, config: PipelineConfig, tokenizer: object) -> Dataset:
    """Load and format the prompt-only dataset for online rollouts.

    Args:
        config:
            The full pipeline configuration.
        tokenizer:
            Policy tokenizer, used to drop prompts that leave no room for a
            completion within the policy context window.

    Returns:
        A dataset with a conversational ``prompt`` column, ordered by ascending
        evolution when the curriculum is enabled.
    """

    def count_prompt_tokens(instruction: str) -> int:
        rendered = tokenizer.apply_chat_template(  # ty: ignore[unresolved-attribute]
            build_user_message(instruction=instruction),
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer(str(rendered))["input_ids"])  # ty: ignore[call-non-callable]

    examples = load_examples(config=config.data)
    examples = filter_by_prompt_length(
        examples=examples,
        count_tokens=count_prompt_tokens,
        max_prompt_tokens=config.data.max_prompt_tokens,
    )

    if config.grpo is not None and config.grpo.curriculum:
        examples = _sort_by_evolution(examples=examples)

    records = [
        {"prompt": build_user_message(instruction=example.instruction)}
        for example in examples
    ]
    return Dataset.from_list(records)


def _sort_by_evolution(*, examples: list[DataExample]) -> list[DataExample]:
    """Order examples easy-to-hard by evolution, with unknowns last.

    Args:
        examples:
            Examples to order.

    Returns:
        The examples sorted by ascending evolution.
    """
    return sorted(
        examples,
        key=lambda example: (example.evolution is None, example.evolution or 0),
    )


def _build_grpo_lora_config(*, config: GRPOTrainConfig) -> LoraConfig:
    """Build a LoRA configuration for GRPO training.

    Args:
        config:
            The GRPO training configuration.

    Returns:
        LoraConfig targeting the transformer attention and MLP projections.
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
