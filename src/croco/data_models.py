"""Pydantic v2 data models for CroCo preference pairs."""

import typing as t

from pydantic import BaseModel


class ScoredCandidate(BaseModel):
    """A single self-generated response together with its reward-model score."""

    response: str
    reward_score: float


class DataExample(BaseModel):
    """One source row from laerebogen used to build a preference pair."""

    instruction: str
    output: str
    evolution: int | None = None
    hash: str | None = None


class PreferencePair(BaseModel):
    """A constructed (chosen, rejected) preference pair for DPO."""

    prompt: str
    chosen: str
    rejected: str
    rejected_score: float
    chosen_score: float | None = None
    evolution: int | None = None
    pool_size: int
    mode: t.Literal["generated", "gold_chosen"]
    hash: str | None = None
