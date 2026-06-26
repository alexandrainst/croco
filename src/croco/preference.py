"""Construction of contrastive preference pairs (CroCo, Eq. 2)."""

import logging

import numpy as np

from .data_models import PreferencePair, ScoredCandidate

logger = logging.getLogger(__name__)


def build_pair_generated(
    *,
    prompt: str,
    candidates: list[ScoredCandidate],
    evolution: int | None = None,
    hash: str | None = None,
) -> PreferencePair | None:
    """Build a pair where both chosen and rejected are self-generations.

    Chosen is the highest-reward candidate; rejected is the candidate whose
    reward is nearest to ``mu - 2*sigma`` and strictly below the chosen reward.

    Args:
        prompt:
          The instruction prompt.
        candidates:
          The scored self-generations for this prompt.
        evolution (optional):
          The source difficulty level. Defaults to None.
        hash (optional):
          The source row hash. Defaults to None.

    Returns:
        The preference pair, or None if no valid pair exists (pool < 2 or no
        candidate lies strictly below the chosen reward).
    """
    if len(candidates) < 2:
        return None
    chosen = max(candidates, key=lambda candidate: candidate.reward_score)
    rejected = _select_rejected(
        pool=candidates, upper_bound=chosen.reward_score, exclude=chosen
    )
    if rejected is None:
        return None
    return PreferencePair(
        prompt=prompt,
        chosen=chosen.response,
        rejected=rejected.response,
        chosen_score=chosen.reward_score,
        rejected_score=rejected.reward_score,
        evolution=evolution,
        pool_size=len(candidates),
        mode="generated",
        hash=hash,
    )


def build_pair_gold_chosen(
    *,
    prompt: str,
    gold_output: str,
    candidates: list[ScoredCandidate],
    gold_score: float | None = None,
    evolution: int | None = None,
    hash: str | None = None,
) -> PreferencePair | None:
    """Build a pair where chosen is the dataset's gold output.

    Rejected is the self-generation nearest to ``mu - 2*sigma`` (statistics
    taken over the generations only). If ``gold_score`` is given, the rejected
    reward must be strictly below it.

    Args:
        prompt:
          The instruction prompt.
        gold_output:
          The dataset's reference completion, used as the chosen response.
        candidates:
          The scored self-generations for this prompt.
        gold_score (optional):
          Reward-model score of ``gold_output``, used as an upper bound on the
          rejected reward. Defaults to None.
        evolution (optional):
          The source difficulty level. Defaults to None.
        hash (optional):
          The source row hash. Defaults to None.

    Returns:
        The preference pair, or None if no valid rejected generation exists.
    """
    if len(candidates) < 1:
        return None
    rejected = _select_rejected(pool=candidates, upper_bound=gold_score, exclude=None)
    if rejected is None:
        # No generation scores strictly below the gold output, so there is no valid
        # contrastive pair: skip rather than invert the preference (which would train
        # the policy towards the RM-preferred generation over the gold chosen).
        return None
    return PreferencePair(
        prompt=prompt,
        chosen=gold_output,
        rejected=rejected.response,
        chosen_score=gold_score,
        rejected_score=rejected.reward_score,
        evolution=evolution,
        pool_size=len(candidates),
        mode="gold_chosen",
        hash=hash,
    )


def build_pair_max_reward(
    *,
    prompt: str,
    gold_output: str,
    gold_score: float,
    candidates: list[ScoredCandidate],
    evolution: int | None = None,
    hash: str | None = None,
) -> PreferencePair | None:
    """Build a pair where chosen is the highest-reward of gold and generations.

    The gold output is treated as one more candidate: the chosen response is the
    argmax-reward over ``{gold} + generations``, and the rejected is the candidate
    nearest to ``mu - 2*sigma`` (statistics over the combined pool) that lies
    strictly below the chosen reward. Unlike ``gold_chosen``, this never skips an
    example just because the policy beat gold; it simply prefers whichever scored
    highest.

    Args:
        prompt:
          The instruction prompt.
        gold_output:
          The dataset's reference completion, scored alongside the generations.
        gold_score:
          Reward-model score of ``gold_output``.
        candidates:
          The scored self-generations for this prompt.
        evolution (optional):
          The source difficulty level. Defaults to None.
        hash (optional):
          The source row hash. Defaults to None.

    Returns:
        The preference pair, or None if the combined pool has fewer than two
        candidates or no candidate lies strictly below the chosen reward.
    """
    pool = [*candidates, ScoredCandidate(response=gold_output, reward_score=gold_score)]
    if len(pool) < 2:
        return None
    chosen = max(pool, key=lambda candidate: candidate.reward_score)
    rejected = _select_rejected(
        pool=pool, upper_bound=chosen.reward_score, exclude=chosen
    )
    if rejected is None:
        return None
    return PreferencePair(
        prompt=prompt,
        chosen=chosen.response,
        rejected=rejected.response,
        chosen_score=chosen.reward_score,
        rejected_score=rejected.reward_score,
        evolution=evolution,
        pool_size=len(pool),
        mode="max_reward",
        hash=hash,
    )


def _select_rejected(
    *,
    pool: list[ScoredCandidate],
    upper_bound: float | None,
    exclude: ScoredCandidate | None,
) -> ScoredCandidate | None:
    """Pick the candidate nearest ``mu - 2*sigma`` subject to constraints.

    ``mu`` and ``sigma`` are the mean and population standard deviation (ddof=0)
    of the pool's rewards, matching the reference implementation.

    Args:
        pool:
          Candidates over which the reward statistics are computed and from
          which the rejected response is drawn.
        upper_bound (optional):
          If given, only candidates with reward strictly below this are
          eligible.
        exclude (optional):
          A candidate object to skip (identity comparison), e.g. the chosen one.

    Returns:
        The eligible candidate closest to the target reward, or None.
    """
    scores = np.array([candidate.reward_score for candidate in pool], dtype=float)
    target = float(scores.mean() - 2.0 * scores.std())  # population std (ddof=0)
    for candidate in sorted(pool, key=lambda c: abs(c.reward_score - target)):
        if candidate is exclude:
            continue
        if upper_bound is not None and candidate.reward_score >= upper_bound:
            continue
        return candidate
    return None
