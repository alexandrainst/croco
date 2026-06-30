#!/usr/bin/env python3
"""Re-score a candidate cache with a different reward model.

The candidate cache stores raw policy self-generations together with their
reward scores. The generation-config fingerprint (``signature``) deliberately
does *not* include the reward model, so two reward models would otherwise share
one cache file and silently reuse each other's scores. To ablate the reward
model we therefore produce a *separate* cache whose generations are identical
but whose scores come from the new reward model.

Holding the generations fixed and only swapping the reward model is both cheaper
than regenerating (no autoregressive pass, just reward forward passes) and a
cleaner ablation: it isolates the reward model's effect on pair construction from
generation variance. The output cache keeps the original ``signature`` so the
downstream ``build_dataset``/``run_pipeline`` step treats it as a full cache hit
(no GPU generation) and constructs pairs straight from the new scores.
"""

import logging
from pathlib import Path

import click

from croco.config import RewardModelConfig
from croco.data_models import ExampleCandidates, ScoredCandidate
from croco.dataset import append_candidates
from croco.engines import ScoringEngine
from croco.utils import read_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-cache",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/candidates_cache.jsonl"),
    help="Existing candidate cache to re-score. Default: data/candidates_cache.jsonl.",
)
@click.option(
    "--output-cache",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Destination cache for the re-scored records (separate from the input).",
)
@click.option(
    "--reward-model-id",
    required=True,
    help="Reward model to score with, e.g. Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M.",
)
@click.option(
    "--reward-max-model-len",
    type=int,
    default=32768,
    help="Reward model max sequence length. Default: 32768.",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.85,
    help="Fraction of GPU memory for the reward model. Default: 0.85.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Number of cache records scored per encode call. Default: 64.",
)
def main(
    *,
    input_cache: Path,
    output_cache: Path,
    reward_model_id: str,
    reward_max_model_len: int,
    gpu_memory_utilization: float,
    batch_size: int,
) -> None:
    """Re-score every candidate (and gold) in a cache with a new reward model.

    Records already present in the output cache (by hash) are skipped, so the job
    is resumable after an interruption.
    """
    _check_distinct_paths(input_cache=input_cache, output_cache=output_cache)

    logger.info("Loading candidate cache from %s", input_cache)
    records = [
        ExampleCandidates.model_validate(row) for row in read_jsonl(path=input_cache)
    ]
    logger.info("Loaded %d cached records", len(records))

    done = _already_rescored_hashes(path=output_cache)
    if done:
        logger.info("Resuming: %d records already re-scored, skipping them", len(done))
    pending = [r for r in records if r.hash is None or r.hash not in done]
    logger.info("%d records to re-score", len(pending))
    if not pending:
        logger.info("Nothing to do; output cache is complete")
        return

    # Import vLLM here so the script is importable on a GPU-less machine.
    from croco.vllm_scoring import VLLMScoringEngine  # noqa: PLC0415

    logger.info("Initialising reward model %s", reward_model_id)
    scoring_engine = VLLMScoringEngine(
        config=RewardModelConfig(
            model_id=reward_model_id,
            max_model_len=reward_max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        ),
        gpu_memory_utilization=gpu_memory_utilization,
    )

    n_done = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        rescored = _rescore_batch(batch=batch, scoring_engine=scoring_engine)
        append_candidates(records=rescored, path=output_cache)
        n_done += len(rescored)
        logger.info("Re-scored %d/%d records", n_done, len(pending))

    logger.info("Done; re-scored cache written to %s", output_cache)


def _check_distinct_paths(*, input_cache: Path, output_cache: Path) -> None:
    """Reject re-scoring a cache onto itself, which would corrupt the input.

    Args:
        input_cache:
            Source cache path.
        output_cache:
            Destination cache path.

    Raises:
        UsageError:
            If the two paths resolve to the same file.
    """
    if output_cache.resolve() == input_cache.resolve():
        msg = "Output cache must differ from the input cache."
        raise click.UsageError(msg)


def _rescore_batch(
    *, batch: list[ExampleCandidates], scoring_engine: ScoringEngine
) -> list[ExampleCandidates]:
    """Re-score one batch of cache records, preserving generations and signature.

    All candidate responses across the batch (plus each record's gold output, when
    present) are scored in a single call so vLLM can batch them efficiently.

    Args:
        batch:
            Cache records whose candidates and gold outputs should be re-scored.
        scoring_engine:
            Reward-model scoring engine.

    Returns:
        New records identical to the inputs except for the refreshed scores.
    """
    prompts: list[str] = []
    responses: list[str] = []
    # Layout per record: its candidate responses, then its gold output if scored.
    for record in batch:
        for candidate in record.candidates:
            prompts.append(record.prompt)
            responses.append(candidate.response)
        if record.gold_score is not None:
            prompts.append(record.prompt)
            responses.append(record.gold_output)

    scores = scoring_engine.score(prompts=prompts, responses=responses)

    rescored: list[ExampleCandidates] = []
    cursor = 0
    for record in batch:
        candidate_scores = scores[cursor : cursor + len(record.candidates)]
        cursor += len(record.candidates)
        gold_score = record.gold_score
        if gold_score is not None:
            gold_score = scores[cursor]
            cursor += 1
        rescored.append(
            ExampleCandidates(
                prompt=record.prompt,
                gold_output=record.gold_output,
                candidates=[
                    ScoredCandidate(response=candidate.response, reward_score=score)
                    for candidate, score in zip(
                        record.candidates, candidate_scores, strict=True
                    )
                ],
                gold_score=gold_score,
                evolution=record.evolution,
                hash=record.hash,
                signature=record.signature,
            )
        )
    return rescored


def _already_rescored_hashes(*, path: Path) -> set[str]:
    """Return the hashes already present in an output cache, for resumability.

    Args:
        path:
            Output cache path; may not yet exist.

    Returns:
        Set of example hashes already written, empty if the file is absent.
    """
    if not path.exists():
        return set()
    hashes: set[str] = set()
    for row in read_jsonl(path=path):
        record_hash = row.get("hash")
        if record_hash is not None:
            hashes.add(record_hash)
    return hashes


if __name__ == "__main__":
    main()
