"""vLLM-based generation engine for the CroCo pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import vllm  # noqa: F401, ty: ignore[unresolved-import] (vllm is GPU-only)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams  # ty: ignore[unresolved-import]

from .utils import build_user_message

if TYPE_CHECKING:
    from .config import GenerationConfig

logger = logging.getLogger(__name__)


class VLLMGenerationEngine:
    """Generation engine backed by vLLM for GPU-accelerated text generation.

    This engine initialises a vLLM LLM instance and tokenizer, then uses
    them to generate multiple candidate responses for each input prompt.
    """

    def __init__(
        self,
        *,
        model_id: str,
        config: GenerationConfig,
        max_model_len: int,
        trust_remote_code: bool = False,
    ) -> None:
        """Initialise the vLLM generation engine.

        Args:
            model_id:
                HuggingFace model identifier for the policy model.
            config:
                Generation configuration with hyperparameters.
            max_model_len:
                Maximum sequence length for the model.
            trust_remote_code:
                Whether to trust remote code from the model repository.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=True,
        )

        self._config = config

    def generate(self, *, prompts: list[str], num_candidates: int) -> list[list[str]]:
        """Generate candidate responses for each instruction prompt.

        Args:
            prompts:
                List of instruction prompts to generate responses for.
            num_candidates:
                Number of candidate responses to generate per prompt.

        Returns:
            A list of lists, where each inner list contains the generated
            responses for the corresponding prompt.
        """
        rendered_prompts = [
            self.tokenizer.apply_chat_template(  # ty: ignore[unresolved-attribute]
                build_user_message(instruction=p),
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        params = SamplingParams(
            n=num_candidates,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
        )

        outputs = self.llm.generate(rendered_prompts, params)

        return [[o.text for o in out.outputs] for out in outputs]
