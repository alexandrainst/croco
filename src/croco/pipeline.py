"""Pipeline orchestration for building preference datasets."""

import logging
from pathlib import Path

from .data import sort_by_evolution
from .data_models import DataExample, PreferencePair, ScoredCandidate
from .dataset import save_pairs
from .engines import GenerationEngine, ScoringEngine
from .preference import build_pair_generated, build_pair_gold_chosen

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
) -> list[PreferencePair]:
    """Build a preference dataset.

    Orchestrates generation, scoring, pair construction, sorting, and saving.

    This function runs the full CroCo pipeline:
    1. Generate multiple candidate responses per instruction
    2. Score all candidates using a reward model
    3. Construct preference pairs (chosen vs rejected)
    4. Sort pairs by difficulty (evolution level)
    5. Save to disk

    Args:
        generation_engine:
            Engine for generating candidate responses.
        scoring_engine:
            Engine for scoring (prompt, response) pairs with a reward model.
        num_candidates:
            Number of candidate responses to generate per instruction.
        construction_mode:
            Either "generated" (both chosen/rejected are generated) or
            "gold_chosen" (chosen is the dataset's gold output).
        score_gold_output:
            Whether to score the gold output when using "gold_chosen" mode.
        output_path:
            Path to save the resulting preference dataset (JSONL).
        examples (optional):
            List of data examples to process. If None, loads from configured source.
            Defaults to None.

    Returns:
        List of constructed preference pairs, sorted by evolution level.
    """
    if examples is None:
        logger.warning("No examples provided, returning empty dataset")
        pairs: list[PreferencePair] = []
        save_pairs(pairs=pairs, path=output_path)
        return pairs

    logger.info("Starting preference dataset construction")
    logger.info(
        "Mode: %s, Candidates per prompt: %d", construction_mode, num_candidates
    )

    all_pairs: list[PreferencePair] = []

    for idx, example in enumerate(examples):
        logger.debug(
            "Processing example %d/%d: %s", idx + 1, len(examples), example.hash
        )

        prompt = example.instruction

        candidates = _generate_and_score_candidates(
            generation_engine=generation_engine,
            scoring_engine=scoring_engine,
            prompt=prompt,
            num_candidates=num_candidates,
        )

        if len(candidates) < 1:
            logger.warning("No candidates generated for example %d, skipping", idx + 1)
            continue

        pair = _construct_pair(
            construction_mode=construction_mode,
            score_gold_output=score_gold_output,
            prompt=prompt,
            gold_output=example.output,
            candidates=candidates,
            scoring_engine=scoring_engine,
            evolution=example.evolution,
            hash=example.hash,
        )

        if pair is None:
            logger.warning("Failed to construct pair for example %d, skipping", idx + 1)
            continue

        all_pairs.append(pair)

    sorted_pairs = sort_by_evolution(pairs=all_pairs)

    logger.info("Constructed %d preference pairs", len(sorted_pairs))

    save_pairs(pairs=sorted_pairs, path=output_path)
    logger.info("Saved preference dataset to %s", output_path)

    return sorted_pairs


def _generate_and_score_candidates(
    *,
    generation_engine: GenerationEngine,
    scoring_engine: ScoringEngine,
    prompt: str,
    num_candidates: int,
) -> list[ScoredCandidate]:
    """Generate and score candidate responses for a single prompt.

    Args:
        generation_engine:
            Engine for generating candidate responses.
        scoring_engine:
            Engine for scoring (prompt, response) pairs.
        prompt:
            The instruction prompt.
        num_candidates:
            Number of candidates to generate.

    Returns:
        List of scored candidates.
    """
    generated = generation_engine.generate(
        prompts=[prompt], num_candidates=num_candidates
    )

    candidates_text = generated[0] if generated else []

    if not candidates_text:
        return []

    scores = scoring_engine.score(
        prompts=[prompt] * len(candidates_text), responses=candidates_text
    )

    scored_candidates = [
        ScoredCandidate(response=response, reward_score=score)
        for response, score in zip(candidates_text, scores, strict=True)
    ]

    logger.debug(
        "Generated %d candidates, scores range: [%.2f, %.2f]",
        len(scored_candidates),
        min(c.reward_score for c in scored_candidates),
        max(c.reward_score for c in scored_candidates),
    )

    return scored_candidates


def _construct_pair(
    *,
    construction_mode: str,
    score_gold_output: bool,
    prompt: str,
    gold_output: str,
    candidates: list[ScoredCandidate],
    scoring_engine: ScoringEngine,
    evolution: int | None,
    hash: str | None,
) -> PreferencePair | None:
    """Construct a preference pair based on the configured mode.

    Args:
        construction_mode:
            Either "generated" or "gold_chosen".
        score_gold_output:
            Whether to score the gold output in "gold_chosen" mode.
        prompt:
            The instruction prompt.
        gold_output:
            The dataset's gold output (used in "gold_chosen" mode).
        candidates:
            Scored candidate responses.
        scoring_engine:
            Engine for scoring the gold output if needed.
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
        gold_score: float | None = None
        if score_gold_output:
            scored = scoring_engine.score(prompts=[prompt], responses=[gold_output])
            gold_score = scored[0] if scored else None
            logger.debug("Gold output score: %.2f", gold_score)

        return build_pair_gold_chosen(
            prompt=prompt,
            gold_output=gold_output,
            candidates=candidates,
            gold_score=gold_score,
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
