"""Pipeline orchestration for building preference datasets."""

import logging
from pathlib import Path

from .data_models import DataExample, PreferencePair, ScoredCandidate
from .dataset import append_pairs, load_pairs
from .engines import GenerationEngine, ScoringEngine
from .preference import (
    build_pair_generated,
    build_pair_gold_chosen,
    build_pair_max_reward,
)

logger = logging.getLogger(__name__)


def build_preference_dataset(
    *,
    generation_engine: GenerationEngine,
    scoring_engine: ScoringEngine,
    num_candidates: int,
    construction_mode: str,
    score_gold_output: bool,
    output_path: Path,
    examples: list[DataExample] | None = None,
    batch_size: int = 64,
    resume: bool = True,
) -> list[PreferencePair]:
    """Build a preference dataset.

    This function runs the full CroCo pipeline in prompt batches:
    1. Generate multiple candidate responses for a batch of instructions
    2. Score all candidates (and the gold outputs, where needed) in bulk
    3. Construct preference pairs (chosen vs rejected) per example
    4. Append the batch's pairs to disk before moving on

    Batching lets vLLM saturate the GPU with its own request queue, and the
    per-batch append provides incremental checkpointing: an interrupted build
    leaves a usable dataset and can be resumed. Pairs are written in processing
    order; curriculum ordering is applied at training time.

    Args:
        generation_engine:
            Engine for generating candidate responses.
        scoring_engine:
            Engine for scoring (prompt, response) pairs with a reward model.
        num_candidates:
            Number of candidate responses to generate per instruction.
        construction_mode:
            One of "generated", "gold_chosen", or "max_reward".
        score_gold_output:
            Whether to score the gold output in "gold_chosen" mode (always scored
            in "max_reward" mode, which requires it).
        output_path:
            Path to save the resulting preference dataset (JSONL).
        examples (optional):
            List of data examples to process. If None, returns an empty dataset.
            Defaults to None.
        batch_size (optional):
            Number of prompts handed to the engines per batch. Defaults to 64.
        resume (optional):
            If True and the output file exists, skip examples whose hash already
            has a pair on disk and append to it. Defaults to True.

    Returns:
        List of all constructed preference pairs (including any pre-existing ones
        when resuming), in processing order.
    """
    if examples is None:
        logger.warning("No examples provided, returning empty dataset")
        return []

    done_hashes: set[str] = set()
    n_existing = 0
    if resume and output_path.exists():
        existing = load_pairs(path=output_path)
        done_hashes = {pair.hash for pair in existing if pair.hash is not None}
        n_existing = len(existing)
        logger.info(
            "Resuming from %s: %d pairs already present", output_path, n_existing
        )

    todo = [
        example
        for example in examples
        if example.hash is None or example.hash not in done_hashes
    ]
    logger.info(
        "Building dataset: mode=%s, K=%d, batch_size=%d, %d examples to process",
        construction_mode,
        num_candidates,
        batch_size,
        len(todo),
    )

    n_built = 0
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        candidates_per_example = _generate_and_score_batch(
            generation_engine=generation_engine,
            scoring_engine=scoring_engine,
            prompts=[example.instruction for example in batch],
            num_candidates=num_candidates,
        )
        gold_scores = _maybe_score_gold(
            construction_mode=construction_mode,
            score_gold_output=score_gold_output,
            scoring_engine=scoring_engine,
            batch=batch,
        )

        batch_pairs: list[PreferencePair] = []
        for example, candidates, gold_score in zip(
            batch, candidates_per_example, gold_scores, strict=True
        ):
            if len(candidates) < 1:
                continue
            pair = _construct_pair(
                construction_mode=construction_mode,
                prompt=example.instruction,
                gold_output=example.output,
                gold_score=gold_score,
                candidates=candidates,
                evolution=example.evolution,
                hash=example.hash,
            )
            if pair is not None:
                batch_pairs.append(pair)

        append_pairs(pairs=batch_pairs, path=output_path)
        n_built += len(batch_pairs)
        logger.info(
            "Processed %d/%d examples, %d pairs built this run",
            min(batch_start + batch_size, len(todo)),
            len(todo),
            n_built,
        )

    logger.info(
        "Constructed %d new preference pairs (%d total in %s)",
        n_built,
        n_existing + n_built,
        output_path,
    )

    return load_pairs(path=output_path)


def _generate_and_score_batch(
    *,
    generation_engine: GenerationEngine,
    scoring_engine: ScoringEngine,
    prompts: list[str],
    num_candidates: int,
) -> list[list[ScoredCandidate]]:
    """Generate and score candidates for a batch of prompts.

    All prompts are generated in one call and all (prompt, candidate) pairs are
    scored in one call, so vLLM can batch the work internally.

    Args:
        generation_engine:
            Engine for generating candidate responses.
        scoring_engine:
            Engine for scoring (prompt, response) pairs.
        prompts:
            The instruction prompts in the batch.
        num_candidates:
            Number of candidates to generate per prompt.

    Returns:
        For each prompt, its list of scored candidates (parallel to ``prompts``).
    """
    generated = generation_engine.generate(
        prompts=prompts, num_candidates=num_candidates
    )

    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    for prompt, responses in zip(prompts, generated, strict=True):
        for response in responses:
            flat_prompts.append(prompt)
            flat_responses.append(response)

    flat_scores = (
        scoring_engine.score(prompts=flat_prompts, responses=flat_responses)
        if flat_responses
        else []
    )

    candidates_per_example: list[list[ScoredCandidate]] = []
    cursor = 0
    for responses in generated:
        chunk_scores = flat_scores[cursor : cursor + len(responses)]
        cursor += len(responses)
        candidates_per_example.append(
            [
                ScoredCandidate(response=response, reward_score=score)
                for response, score in zip(responses, chunk_scores, strict=True)
            ]
        )

    return candidates_per_example


def _maybe_score_gold(
    *,
    construction_mode: str,
    score_gold_output: bool,
    scoring_engine: ScoringEngine,
    batch: list[DataExample],
) -> list[float | None]:
    """Score the gold outputs for a batch where the mode requires it.

    Args:
        construction_mode:
            One of "generated", "gold_chosen", or "max_reward".
        score_gold_output:
            Whether to score the gold output in "gold_chosen" mode.
        scoring_engine:
            Engine for scoring (prompt, response) pairs.
        batch:
            The examples in the batch.

    Returns:
        A gold score per example (parallel to ``batch``), or None where the gold
        output is not scored.
    """
    needs_gold = construction_mode == "max_reward" or (
        construction_mode == "gold_chosen" and score_gold_output
    )
    if not needs_gold:
        return [None] * len(batch)

    scores = scoring_engine.score(
        prompts=[example.instruction for example in batch],
        responses=[example.output for example in batch],
    )
    return list(scores)


def _construct_pair(
    *,
    construction_mode: str,
    prompt: str,
    gold_output: str,
    gold_score: float | None,
    candidates: list[ScoredCandidate],
    evolution: int | None,
    hash: str | None,
) -> PreferencePair | None:
    """Construct a preference pair based on the configured mode.

    Args:
        construction_mode:
            One of "generated", "gold_chosen", or "max_reward".
        prompt:
            The instruction prompt.
        gold_output:
            The dataset's gold output (used in the gold-based modes).
        gold_score:
            Reward-model score of the gold output, or None if not scored.
        candidates:
            Scored candidate responses.
        evolution:
            Source difficulty level.
        hash:
            Source row hash.

    Returns:
        The constructed preference pair, or None if construction failed.
    """
    if construction_mode == "generated":
        return build_pair_generated(
            prompt=prompt, candidates=candidates, evolution=evolution, hash=hash
        )
    elif construction_mode == "gold_chosen":
        return build_pair_gold_chosen(
            prompt=prompt,
            gold_output=gold_output,
            candidates=candidates,
            gold_score=gold_score,
            evolution=evolution,
            hash=hash,
        )
    elif construction_mode == "max_reward":
        if gold_score is None:
            logger.error("max_reward mode requires a gold score; skipping example")
            return None
        return build_pair_max_reward(
            prompt=prompt,
            gold_output=gold_output,
            gold_score=gold_score,
            candidates=candidates,
            evolution=evolution,
            hash=hash,
        )
    else:
        logger.error("Unknown construction mode: %s", construction_mode)
        return None


def upload_to_huggingface(
    *,
    dataset_path: Path,
    model_path: Path,
    repo_id: str,
    config_repo_id: str | None = None,
    private: bool = False,
) -> None:
    """Upload dataset and model to Hugging Face Hub.

    Args:
        dataset_path:
            Path to the preference dataset JSONL file.
        model_path:
            Path to the trained model directory.
        repo_id:
            HuggingFace repo ID for the model (e.g., "user/repo-name").
        config_repo_id:
            Optional separate repo ID for the dataset. If None, uses same repo.
        private:
            Whether to make repos private. Defaults to False.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        logger.warning("huggingface_hub not installed, skipping upload: %s", e)
        return

    api = HfApi()
    dataset_repo = config_repo_id or repo_id

    # Create repos if they don't exist
    try:
        api.create_repo(
            repo_id=dataset_repo, repo_type="dataset", private=private, exist_ok=True
        )
    except Exception as e:
        logger.warning("Failed to create dataset repo: %s", e)

    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as e:
        logger.warning("Failed to create model repo: %s", e)

    # Upload dataset
    logger.info(f"Uploading dataset to {dataset_repo}")
    try:
        api.upload_file(
            path_or_fileobj=str(dataset_path),
            path_in_repo="preference_pairs.jsonl",
            repo_id=dataset_repo,
            repo_type="dataset",
        )
        logger.info(
            f"Dataset uploaded to https://huggingface.co/datasets/{dataset_repo}"
        )
    except Exception as e:
        logger.error("Failed to upload dataset: %s", e)

    # Upload model
    logger.info(f"Uploading model to {repo_id}")
    try:
        api.upload_folder(folder_path=str(model_path), repo_id=repo_id)
        logger.info(f"Model uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error("Failed to upload model: %s", e)
