"""TRL DPO training with curriculum learning support."""

from __future__ import annotations

import logging
import typing as t
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig
from torch.utils.data import Sampler, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .config import DPOTrainConfig, PipelineConfig
from .data import sort_by_evolution
from .dataset import load_pairs, to_trl_records

logger = logging.getLogger(__name__)


def _simpo_sequence_loss(
    *,
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float,
    target_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SimPO sequence-level loss with length normalisation and target margin.

    Args:
        chosen_logps:
            Log-probabilities of chosen completions (sum over tokens).
        rejected_logps:
            Log-probabilities of rejected completions (sum over tokens).
        chosen_lengths:
            Completion lengths for chosen responses.
        rejected_lengths:
            Completion lengths for rejected responses.
        beta:
            SimPO temperature parameter.
        target_margin:
            Target margin γ; SimPO pull chosen above rejected by at least this amount.

    Returns:
        Tuple of per_sequence_loss, chosen_rewards, rejected_rewards, and
        reward_accuracies.
    """
    chosen_avg = chosen_logps / chosen_lengths.clamp(min=1.0)
    rejected_avg = rejected_logps / rejected_lengths.clamp(min=1.0)
    delta = chosen_avg - rejected_avg
    per_sequence_loss = -F.logsigmoid(beta * delta - target_margin)
    chosen_rewards = beta * chosen_avg
    rejected_rewards = beta * rejected_avg
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    return per_sequence_loss, chosen_rewards, rejected_rewards, reward_accuracies


class SimPOLossMixin:
    """Mixin adding ref-free SimPO loss with target margin to DPOTrainer.

    Overrides ``compute_loss`` to intercept ``loss_type == 'simpo'`` and compute
    the SimPO loss (length-normalised policy log-probs, no reference subtraction,
    with target margin γ). Falls through to super() for other loss types.
    """

    # Declared for type checking; actually provided by DPOTrainer base class.
    beta: float
    loss_types: list[str]
    target_margin: float = 0.0

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute SimPO loss if ``loss_type == 'simpo'``, else defer to parent.

        Returns:
            Loss tensor, optionally with outputs dict if return_outputs is True.
        """
        if "simpo" not in self.loss_types:
            return super().compute_loss(  # ty: ignore
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        return self._compute_simpo_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _compute_simpo_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        """Compute ref-free SimPO loss with length normalisation and target margin.

        Returns:
            Loss tensor, optionally with model outputs dict if return_outputs is True.
        """
        from trl.trainer.utils import selective_log_softmax

        # Forward pass through policy (no reference model)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = inputs["labels"][..., 1:]
        completion_mask = inputs["completion_mask"][..., 1:]

        # Compute per-token log-probs for policy
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[completion_mask == 0] = 0.0

        # Sum over completion length (ld_alpha=None path only)
        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)
        chosen_lengths = completion_mask.sum(dim=1).chunk(2, dim=0)[0]
        rejected_lengths = completion_mask.sum(dim=1).chunk(2, dim=0)[1]

        # Compute SimPO loss
        loss, chosen_rewards, rejected_rewards, reward_accuracies = (
            _simpo_sequence_loss(
                chosen_logps=chosen_logps,
                rejected_logps=rejected_logps,
                chosen_lengths=chosen_lengths,
                rejected_lengths=rejected_lengths,
                beta=self.beta,  # type: ignore[attr-defined]
                target_margin=getattr(self, "target_margin", 0.0),
            )
        )

        # Log metrics (mirror TRL's metric logging)
        logs = {
            "rewards/chosen": self.accelerator.gather(chosen_rewards).mean().item(),  # type: ignore
            "rewards/rejected": self.accelerator.gather(rejected_rewards).mean().item(),  # type: ignore
            "rewards/accuracies": self.accelerator.gather(reward_accuracies)  # type: ignore
            .mean()
            .item(),
            "rewards/margins": self.accelerator.gather(  # type: ignore
                chosen_rewards - rejected_rewards
            )
            .mean()
            .item(),
            "logps/chosen": self.accelerator.gather(chosen_logps).mean().item(),  # type: ignore
            "logps/rejected": self.accelerator.gather(rejected_logps).mean().item(),  # type: ignore
        }
        # Log metrics using TRL's logging mechanism
        for key, value in logs.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=False)  # type: ignore
        self.log("loss", loss.mean().item(), on_step=True, on_epoch=True, prog_bar=True)  # type: ignore

        if return_outputs:
            return loss.mean(), outputs
        return loss.mean()


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


class SimPODPOTrainer(SimPOLossMixin, DPOTrainer):
    """DPOTrainer with ref-free SimPO loss support."""

    pass


class CurriculumSimPODPOTrainer(SimPOLossMixin, CurriculumDPOTrainer):
    """CurriculumDPOTrainer with ref-free SimPO loss support."""

    pass


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
    save_kwargs: dict[str, t.Any] = {}
    if config.save_steps > 0:
        # Keep every checkpoint (save_total_limit=None) so the full learning curve
        # can be evaluated afterwards.
        save_kwargs = {
            "save_strategy": "steps",
            "save_steps": config.save_steps,
            "save_total_limit": None,
        }

    return DPOConfig(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        beta=config.beta,
        loss_type=[config.loss_type],
        label_smoothing=config.label_smoothing,
        precompute_ref_log_probs=config.precompute_ref_log_probs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_length=config.max_length,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        **save_kwargs,
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

    Raises:
        ValueError:
            If the configuration has no ``dpo`` block.
    """
    if config.dpo is None:
        msg = "train_dpo requires a 'dpo' block in the configuration."
        raise ValueError(msg)
    dpo_config = config.dpo

    # Load and convert dataset
    logger.info(f"Loading preference pairs from {dataset_path}")
    pairs = load_pairs(path=dataset_path)
    if dpo_config.curriculum:
        # The checkpointed dataset is in processing order; impose the easy-to-hard
        # curriculum here so the SequentialSampler trains on ascending difficulty.
        pairs = sort_by_evolution(pairs=pairs)
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

    # Select trainer based on curriculum and loss_type
    is_simpo = dpo_config.loss_type == "simpo"
    if is_simpo and dpo_config.curriculum:
        logger.info("Using CurriculumSimPODPOTrainer for SimPO with curriculum")
        trainer_class: type[DPOTrainer] = CurriculumSimPODPOTrainer
    elif is_simpo:
        logger.info("Using SimPODPOTrainer for SimPO")
        trainer_class = SimPODPOTrainer
    elif dpo_config.curriculum:
        logger.info("Using CurriculumDPOTrainer for curriculum learning")
        trainer_class = CurriculumDPOTrainer
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

    # Set SimPO target margin if using SimPO loss
    if is_simpo:
        trainer.target_margin = dpo_config.target_margin  # type: ignore

    # Train
    logger.info("Starting DPO training")
    trainer.train()

    # Save model
    output_dir = Path(dpo_config.output_dir)
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))

    return output_dir
