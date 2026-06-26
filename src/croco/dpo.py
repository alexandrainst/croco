"""TRL DPO training with curriculum learning support."""

from __future__ import annotations

import logging
import typing as t
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from torch.utils.data import Sampler, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .config import DPOTrainConfig, PipelineConfig
from .dataset import load_pairs, to_trl_records

logger = logging.getLogger(__name__)


class CurriculumDPOTrainer(DPOTrainer):
    """DPOTrainer with curriculum learning support.

    Overrides ``_get_train_sampler`` to return a SequentialSampler, which
    respects the curriculum ordering of the dataset.
    """

    def _get_train_sampler(self, *args, **kwargs) -> Sampler | None:
        """Return a sequential sampler for curriculum learning.

        Returns:
            SequentialSampler that iterates through the dataset in order.
        """
        # Type ignore: datasets.Dataset is Sized in practice
        return SequentialSampler(t.cast(t.Any, self.train_dataset))


def build_lora_config(*, config: DPOTrainConfig) -> LoraConfig:
    """Build a LoRA configuration from DPO training config.

    Args:
        config:
          The DPO training configuration.

    Returns:
        LoraConfig with target modules for transformer attention layers.
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


def build_dpo_config(*, config: DPOTrainConfig) -> DPOConfig:
    """Build a TRL DPOConfig from DPO training config.

    Args:
        config:
          The DPO training configuration.

    Returns:
        DPOConfig with all training hyperparameters.
    """
    return DPOConfig(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        beta=config.beta,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_length=config.max_length,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
    )


def train_dpo(*, config: PipelineConfig, dataset_path: Path) -> Path:
    """Train a policy model with DPO.

    Loads preference pairs, converts to TRL format, and trains with either
    CurriculumDPOTrainer (if curriculum learning is enabled) or standard
    DPOTrainer.

    Args:
        config:
          The full pipeline configuration.
        dataset_path:
          Path to the JSONL file containing preference pairs.

    Returns:
        Path to the directory where the trained model was saved.
    """
    dpo_config = config.dpo

    # Load and convert dataset
    logger.info(f"Loading preference pairs from {dataset_path}")
    pairs = load_pairs(path=dataset_path)
    records = to_trl_records(pairs=pairs)
    dataset = Dataset.from_list(records)
    logger.info(f"Loaded {len(dataset)} preference pairs")

    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {config.policy.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.policy.model_id)

    model = AutoModelForCausalLM.from_pretrained(
        config.policy.model_id,
        attn_implementation=config.policy.attn_implementation,
        dtype="auto",
        use_cache=not dpo_config.gradient_checkpointing,
    )

    # Build LoRA and DPO configs
    peft_config = build_lora_config(config=dpo_config)
    dpo_cfg = build_dpo_config(config=dpo_config)

    # Select trainer based on curriculum setting
    if dpo_config.curriculum:
        logger.info("Using CurriculumDPOTrainer for curriculum learning")
        trainer_class: type[DPOTrainer] = CurriculumDPOTrainer
    else:
        logger.info("Using standard DPOTrainer")
        trainer_class = DPOTrainer

    # Initialise trainer
    trainer = trainer_class(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting DPO training")
    trainer.train()

    # Save model
    output_dir = Path(dpo_config.output_dir)
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))

    return output_dir
