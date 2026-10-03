from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ModelConfig(Base):
    __tablename__ = "model_configs"

    # Every later record refers to a model by name, not by id — a testset keeps
    # llm_name, an evaluation keeps the judge's — so two configs sharing a name
    # leaves those records unable to say which one they used. The API checks
    # for this; the database is what holds the rule.
    __table_args__ = (Index("uq_model_configs_name", "name", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False)  # llm | embedding
    api_format: Mapped[str] = mapped_column(String(16), nullable=False)  # openai | anthropic
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    api_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    # Whether this endpoint thinks before answering, and what to send to stop it.
    # Probed once when the config is saved: the field that turns thinking off has
    # a different name on every platform, and one it does not recognise is
    # ignored silently — the model simply keeps thinking and the only symptom is
    # a run several times slower than it should be. `thinking_state` is what the
    # UI shows ("already_off" and "disabled" both mean the column reads 是);
    # `thinking_param` is the JSON body fragment the runtime has to send, empty
    # when there is nothing to send.
    thinking_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    thinking_param: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Corpus(Base):
    __tablename__ = "corpora"

    STATUS_CONVERTING = "converting"
    STATUS_COMPLETED = "completed"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default=STATUS_CONVERTING)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    processed_files: Mapped[int] = mapped_column(Integer, default=0)
    success_files: Mapped[int] = mapped_column(Integer, default=0)
    failed_files: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of file paths
    # Timestamped events, the shape Testset / RagSystemConfig / EvalRun all use.
    # This was the only one of the four with none, which is exactly why its
    # "log" was a summary paragraph while the others read "[time] event".
    log_entries: Mapped[str] = mapped_column(Text, default="[]")
    convert_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RagSystemConfig(Base):
    """A target RAG system adapter (generic HTTP config).

    Lifecycle: configuring (agent or simple test running) -> completed/failed.
    A failed adapter can be edited; saving re-runs the simple test.
    """

    __tablename__ = "rag_system_configs"

    STATUS_CONFIGURING = "configuring"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    platform: Mapped[str] = mapped_column(String(16), default="custom")  # informational
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    api_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    headers: Mapped[str] = mapped_column(Text, default="{}")  # JSON object
    body_template: Mapped[str] = mapped_column(Text, default="")
    answer_path: Mapped[str] = mapped_column(String(256), default="")
    contexts_path: Mapped[str | None] = mapped_column(String(256), nullable=True)
    timeout: Mapped[int] = mapped_column(Integer, default=120)
    # agent provenance (smart fill only)
    llm_config_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    platform_hint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    lang: Mapped[str] = mapped_column(String(8), default="")  # UI locale; probe question follows it
    # lifecycle
    status: Mapped[str] = mapped_column(String(16), default=STATUS_COMPLETED)
    progress: Mapped[float] = mapped_column(Float, default=100.0)
    stage: Mapped[str] = mapped_column(String(32), default="done")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_entries: Mapped[str] = mapped_column(Text, default="[]")  # JSON: [{time, key, params}]
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)


class Testset(Base):
    __tablename__ = "testsets"
    # A corpus may only have one testset generating from it: the knowledge
    # graph, seed list and persona cache under data/corpus/<name>/ are shared
    # files, and the pipeline only serialises per testset, so two runs would
    # interleave writes to them. The API checks for this, but the check and the
    # write are separate statements — two requests can pass the check together.
    # The database is what actually holds the rule.
    __table_args__ = (
        Index("uq_testsets_generating_per_corpus", "corpus_id", unique=True,
              sqlite_where=text("status = 'generating'")),
    )

    STATUS_GENERATING = "generating"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    corpus_id: Mapped[int] = mapped_column(Integer, nullable=False)
    corpus_name: Mapped[str] = mapped_column(String(64), nullable=False)
    reuse_kg: Mapped[bool] = mapped_column(Integer, default=1)
    llm_config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_name: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_name: Mapped[str] = mapped_column(String(128), nullable=False)
    llm_concurrency: Mapped[int] = mapped_column(Integer, default=16)
    llm_max_tokens: Mapped[int] = mapped_column(Integer, default=16384)
    n_single: Mapped[int] = mapped_column(Integer, default=10)
    n_multi_specific: Mapped[int] = mapped_column(Integer, default=10)
    n_multi_abstract: Mapped[int] = mapped_column(Integer, default=10)
    prompt_language: Mapped[str] = mapped_column(String(8), default="auto")  # auto|zh|en
    amplify: Mapped[float] = mapped_column(Float, default=1.3)  # seed-sampling amplification
    gen_amplify: Mapped[float] = mapped_column(Float, default=1.6)  # generation headroom
    run_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)  # reproducibility seed
    status: Mapped[str] = mapped_column(String(16), default=STATUS_GENERATING)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(32), default="init")  # stage code for i18n
    stage_done: Mapped[int] = mapped_column(Integer, default=0)
    stage_total: Mapped[int] = mapped_column(Integer, default=0)
    actual_single: Mapped[int] = mapped_column(Integer, default=0)
    actual_multi_specific: Mapped[int] = mapped_column(Integer, default=0)
    actual_multi_abstract: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_entries: Mapped[str] = mapped_column(Text, default="[]")  # JSON: [{time, key, params}]
    token_usage: Mapped[str] = mapped_column(Text, default="{}")  # JSON: {prompt, completion, calls}
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)


class TestsetItem(Base):
    __tablename__ = "testset_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    testset_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    reference: Mapped[str] = mapped_column(Text, default="")
    reference_contexts: Mapped[str] = mapped_column(Text, default="[]")  # JSON array
    synthesizer_name: Mapped[str] = mapped_column(String(64), default="")
    persona_name: Mapped[str] = mapped_column(String(128), default="")
    edited: Mapped[bool] = mapped_column(Integer, default=0)
    original_user_input: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EvalRun(Base):
    """One evaluation run: a testset executed against a target RAG system."""

    __tablename__ = "eval_runs"

    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    testset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    testset_name: Mapped[str] = mapped_column(String(64), nullable=False)
    rag_system_id: Mapped[int] = mapped_column(Integer, nullable=False)
    rag_system_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # Effective endpoint + credential used by this run (snapshot; default from
    # the adapter, overridable per run — e.g. another assistant's chat_id URL)
    run_base_url: Mapped[str] = mapped_column(String(512), default="")
    run_api_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    timeout: Mapped[int] = mapped_column(Integer, default=120)  # per-run target timeout
    llm_config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_name: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_name: Mapped[str] = mapped_column(String(128), nullable=False)
    concurrency: Mapped[int] = mapped_column(Integer, default=4)  # target system
    judge_concurrency: Mapped[int] = mapped_column(Integer, default=16)  # judge LLM/embedding
    use_judge_cache: Mapped[bool] = mapped_column(Integer, default=1)  # shared judge cache
    metrics: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of metric keys
    has_contexts: Mapped[bool] = mapped_column(Integer, default=0)  # RAG system returns contexts
    status: Mapped[str] = mapped_column(String(16), default=STATUS_RUNNING)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(32), default="init")
    stage_done: Mapped[int] = mapped_column(Integer, default=0)
    stage_total: Mapped[int] = mapped_column(Integer, default=0)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[str] = mapped_column(Text, default="{}")  # JSON: {metric: avg score}
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_entries: Mapped[str] = mapped_column(Text, default="[]")  # JSON: [{time, key, params}]
    token_usage: Mapped[str] = mapped_column(Text, default="{}")  # judge tokens
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)


class EvalRunItem(Base):
    __tablename__ = "eval_run_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    reference: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)  # None = query failed
    contexts: Mapped[str] = mapped_column(Text, default="[]")  # JSON array (from RAG system)
    scores: Mapped[str] = mapped_column(Text, default="{}")  # JSON: {metric: float}
    # JSON array of 0/1, one per retrieved chunk, in retrieval order: the judge's
    # verdict for each. Kept because every rank-based reading — Hit@k, the k
    # curve, where the hit landed — is a view of this one sequence, and the
    # context_precision metric computes it and throws it away.
    context_verdicts: Mapped[str] = mapped_column(Text, default="[]")
    seconds: Mapped[float | None] = mapped_column(Float, nullable=True)  # target call latency
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
