from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.schemas.corpus import NAME_PATTERN


class EvalRunCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    testset_id: int
    rag_system_id: int
    llm_config_id: int
    embedding_config_id: int
    concurrency: int = Field(default=4, ge=1, le=32)
    judge_concurrency: int = Field(default=16, ge=1, le=64)
    use_judge_cache: bool = True
    metrics: list[str] = Field(min_length=1)
    # The endpoint/credential for THIS run (adapters persist only templates)
    base_url: str = Field(min_length=1)
    api_key: str | None = None
    timeout: int = Field(default=120, ge=5, le=600)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        # Same rule as corpora and testsets, and it matters more here: the name
        # becomes a directory under data/evaluation that delete rmtrees, so
        # "../../logs" would remove the logs and "../../data" the whole data
        # directory. The frontend checks this client-side, which is not a guard.
        v = v.strip()
        if not v or not NAME_PATTERN.match(v):
            raise ValueError(
                "Name may only contain letters, digits, Chinese characters, spaces, '-' and '_'"
            )
        return v


class EvalRunOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    testset_id: int
    testset_name: str
    rag_system_id: int
    rag_system_name: str
    run_base_url: str
    timeout: int
    llm_name: str
    embedding_name: str
    concurrency: int
    judge_concurrency: int = 16
    use_judge_cache: bool = True
    metrics: str  # JSON array
    has_contexts: bool
    status: str
    progress: float
    stage: str
    stage_done: int
    stage_total: int
    total_items: int
    failed_items: int
    summary: str  # JSON object {metric: avg}
    # JSON object {"items": n, "curve": {"1": 0.33, …, "k": 0.67}}: share of
    # items whose first k retrieved chunks include a useful one, over the n items
    # that have verdicts. Derived on read from the items' context_verdicts, so it
    # is empty — {"items": 0, "curve": {}} — for runs that never computed
    # context_precision or predate the column.
    hit_at_k: str = '{"items": 0, "curve": {}}'
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class EvalRunPage(BaseModel):
    total: int
    items: list[EvalRunOut]


class EvalLogOut(BaseModel):
    entries: list[dict]
    token_usage: dict
    error: str | None


class EvalItemOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    seq: int
    user_input: str
    reference: str
    answer: str | None
    contexts: str  # JSON array
    scores: str  # JSON object
    # JSON array of 0/1, one per retrieved chunk, in retrieval order — the judge's
    # verdict on each. Empty when context_precision did not run, or when the item
    # was recorded before the column existed.
    context_verdicts: str = "[]"
    error: str | None


class EvalItemPage(BaseModel):
    total: int
    items: list[EvalItemOut]
