from datetime import datetime

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.corpus import NAME_PATTERN

PromptLanguage = Literal["auto", "zh", "en"]


class TestsetCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    corpus_id: int
    reuse_kg: bool = True
    llm_config_id: int
    embedding_config_id: int
    llm_concurrency: int = Field(default=16, ge=1, le=64)
    llm_max_tokens: int = Field(default=16384, ge=4096, le=32768)
    n_single: int = Field(default=10, ge=0, le=1000)
    n_multi_specific: int = Field(default=10, ge=0, le=1000)
    n_multi_abstract: int = Field(default=10, ge=0, le=1000)
    prompt_language: PromptLanguage = "auto"
    amplify: float = Field(default=1.3, ge=1.0, le=3.0)  # seed-sampling amplification
    # Generation headroom over the requested count. The gate correctly rejects
    # 25-40% of candidates (cross-company mashups, cross-year comparisons,
    # premises the material does not support), so the first batch has to clear
    # that bar before a single QA pair reaches the testset: at 1.2 — 12
    # candidates for 10 — runs routinely delivered 28-29 items. Measured on the
    # same settings: 1.4 delivered 30 in 4 of 7 runs, 1.6 in 2 of 2, and 1.6 was
    # FASTER (120-130 s against 121-205 s): a first batch big enough does not
    # trigger the fallback, and every fallback round rebuilds part of the
    # knowledge graph.
    gen_amplify: float = Field(default=1.6, ge=1.0, le=2.0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v or not NAME_PATTERN.match(v):
            raise ValueError(
                "Name may only contain letters, digits, Chinese characters, spaces, '-' and '_'"
            )
        return v

    @field_validator("n_multi_abstract")
    @classmethod
    def at_least_one_question(cls, v: int, info) -> int:
        total = (info.data.get("n_single") or 0) + (info.data.get("n_multi_specific") or 0) + v
        if total <= 0:
            raise ValueError("At least one question is required")
        return v


class TestsetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    corpus_id: int
    corpus_name: str
    reuse_kg: bool
    llm_name: str
    embedding_name: str
    llm_concurrency: int
    llm_max_tokens: int
    n_single: int
    n_multi_specific: int
    n_multi_abstract: int
    status: str
    progress: float
    stage: str
    stage_done: int
    stage_total: int
    actual_single: int
    actual_multi_specific: int
    actual_multi_abstract: int
    prompt_language: str
    amplify: float
    gen_amplify: float
    run_seed: int | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None
    # True when any QA item was manually edited (derived, not an ORM column).
    edited: bool = False
    # Total QA items (derived). Compared against sum(actual_*) to decide
    # whether the per-type breakdown is meaningful (imports may lack types).
    item_count: int = 0


class TestsetPage(BaseModel):
    total: int
    items: list[TestsetOut]


class TestsetLogEntry(BaseModel):
    time: str
    key: str
    params: dict


class TestsetLogOut(BaseModel):
    entries: list[TestsetLogEntry]
    token_usage: dict
    error: str | None


class TestsetItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    seq: int
    user_input: str
    reference: str
    reference_contexts: list[str]
    synthesizer_name: str
    persona_name: str
    edited: bool
    has_original: bool


class TestsetItemPage(BaseModel):
    total: int
    items: list[TestsetItemOut]


class TestsetItemUpdateIn(BaseModel):
    user_input: str = Field(min_length=1)
    reference: str = ""
