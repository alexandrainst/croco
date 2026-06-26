"""Backend Protocols for generation and scoring engines."""

import typing as t


class GenerationEngine(t.Protocol):
    """Protocol for text generation backends."""

    def generate(self, *, prompts: list[str], num_candidates: int) -> list[list[str]]:
        """Return, for each prompt, ``num_candidates`` response strings.

        Args:
            prompts:
              List of instruction prompts to generate responses for.
            num_candidates:
              Number of candidate responses to generate per prompt.

        Returns:
            A list of lists, where each inner list contains the generated
            responses for the corresponding prompt.
        """
        ...


class ScoringEngine(t.Protocol):
    """Protocol for reward-model scoring backends."""

    def score(self, *, prompts: list[str], responses: list[str]) -> list[float]:
        """Reward-score each parallel ``(prompt, response)`` pair.

        Args:
            prompts:
              List of instruction prompts.
            responses:
              List of responses, parallel to ``prompts``.

        Returns:
            List of reward scores, one for each (prompt, response) pair.
        """
        ...
