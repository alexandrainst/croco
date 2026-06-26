"""vLLM-based scoring engine for the CroCo pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from transformers import AutoTokenizer
from vllm import LLM, PoolingParams

from .utils import build_conversation

if TYPE_CHECKING:
    from .config import RewardModelConfig

logger = logging.getLogger(__name__)


class VLLMScoringEngine:
    """Scoring engine backed by vLLM's pooling mode for reward scoring.

    This engine initialises a vLLM LLM instance in pooling mode and uses
    it to score (prompt, response) pairs using a reward model.
    """

    def __init__(
        self,
        *,
        config: RewardModelConfig,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.15,  # Lower default for unified memory systems
        trust_remote_code: bool = False,
    ) -> None:
        """Initialise the vLLM scoring engine.

        Args:
            config:
                Reward model configuration with model ID and max length.
            tensor_parallel_size:
                Number of GPUs to use for tensor parallelism.
            gpu_memory_utilization:
                Fraction of GPU memory to use for model weights and KV cache.
            trust_remote_code:
                Whether to trust remote code from the model repository.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, trust_remote_code=trust_remote_code
        )

        self.llm = LLM(
            model=config.model_id,
            runner="pooling",
            enforce_eager=True,
            max_model_len=config.max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_chunked_prefill=False,
            load_format="fastsafetensors",
        )

    def score(self, *, prompts: list[str], responses: list[str]) -> list[float]:
        """Score each (prompt, response) pair using the reward model.

        Args:
            prompts:
                List of instruction prompts.
            responses:
                List of responses, parallel to prompts.

        Returns:
            List of reward scores, one for each (prompt, response) pair.
            On failure, returns -999.0 for the failed item.
        """
        rendered: list[str] = [
            str(
                self.tokenizer.apply_chat_template(  # ty: ignore[unresolved-attribute]
                    build_conversation(instruction=p, response=r),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            for p, r in zip(prompts, responses, strict=True)
        ]

        # use_activation=False yields raw reward logits (no sigmoid/softmax).
        outputs = self.llm.encode(
            rendered,
            pooling_params=PoolingParams(use_activation=False),
            pooling_task="classify",
        )

        scores: list[float] = []
        for o in outputs:
            try:
                scores.append(float(o.outputs.data.item()))
            except Exception:
                logger.warning("Failed to extract score, using fallback -999.0")
                scores.append(-999.0)

        return scores
