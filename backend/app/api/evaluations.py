import asyncio
import json
import shutil

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.model_configs import get_db
from app.db.models import EvalRun, EvalRunItem, ModelConfig, RagSystemConfig, Testset
from app.schemas.evaluation import (
    EvalItemOut,
    EvalItemPage,
    EvalLogOut,
    EvalRunCreateIn,
    EvalRunOut,
    EvalRunPage,
)
from app.services import eval_run
from app.services.rag_client import normalize_url
from app.services.eval_run import CONTEXT_METRICS, METRIC_KEYS

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


def _get_or_404(run_id: int, db: Session) -> EvalRun:
    run = db.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return run


def _hit_at_k_from(seqs: list[list[int]]) -> dict[str, float]:
    """Share of items whose first k chunks include a useful one, for every k.

    Monotone in k by construction. An item that returned fewer than k chunks is
    read at all of them (`s[:k]` simply runs out), which is the rule the number
    is published under — "the first k chunks it gave, or everything it gave".
    """
    if not seqs:
        return {}
    counts = {k: sum(1 for s in seqs if any(s[:k]))
              for k in range(1, max(len(s) for s in seqs) + 1)}
    return {str(k): round(n / len(seqs), 4) for k, n in counts.items()}


def hit_at_k(db: Session, run_id: int) -> str:
    """The k curve for one run, as JSON: {"items": n, "curve": {"1": …}}.

    `items` is published because the curve is a mean over the items that have
    verdicts, and a run can have only some of them — rescoring one item writes
    its verdicts without touching the rest. Without the count, a curve over one
    item reads exactly like a curve over thirty.

    Derived on read from the items' stored verdicts rather than kept as its own
    summary: the verdicts are the raw material and every k is a view of them, so
    the rule above has one definition instead of two.
    """
    blobs = db.execute(
        select(EvalRunItem.context_verdicts).where(EvalRunItem.run_id == run_id)
    ).scalars().all()
    seqs = []
    for blob in blobs:
        try:
            seq = json.loads(blob) if blob else []
        except ValueError:
            seq = []
        if seq:
            seqs.append(seq)
    return json.dumps({"items": len(seqs), "curve": _hit_at_k_from(seqs)})


def _with_hit_at_k(db: Session, run: EvalRun) -> EvalRun:
    # Not a column: computed per request so it cannot drift from the verdicts.
    run.hit_at_k = hit_at_k(db, run.id)
    return run


@router.get("", response_model=EvalRunPage)
def list_runs(
    name: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> EvalRunPage:
    stmt = select(EvalRun).order_by(EvalRun.id.desc())
    if name:
        stmt = stmt.where(EvalRun.name.contains(name))
    total = len(db.execute(stmt).scalars().all())
    items = db.execute(stmt.offset((page - 1) * page_size).limit(page_size)).scalars().all()
    return EvalRunPage(
        total=total,
        items=[EvalRunOut.model_validate(_with_hit_at_k(db, i)) for i in items],
    )


@router.get("/metrics")
def list_metrics() -> dict:
    """What a run can be created with.

    `requires_contexts` is part of the answer, not a detail to discover: creating
    a run with a context metric against an adapter that has no contexts path is
    rejected, and a caller should be able to see that coming rather than read it
    out of a 400.

    Declared above `/{run_id}` — a literal path has to be matched before the
    parameterised one, or the parameterised one wins and the literal 422s.
    """
    return {
        "metrics": [
            {"key": key, "requires_contexts": key in CONTEXT_METRICS}
            for key in METRIC_KEYS
        ]
    }


@router.get("/check-name")
def check_name(name: str, db: Session = Depends(get_db)) -> dict:
    existing = db.execute(
        select(EvalRun).where(func.lower(EvalRun.name) == name.strip().lower())
    ).scalar_one_or_none()
    return {"available": existing is None}


@router.post("", response_model=EvalRunOut, status_code=201)
async def create_run(payload: EvalRunCreateIn, db: Session = Depends(get_db)) -> EvalRun:
    if not check_name(payload.name, db)["available"]:
        raise HTTPException(status_code=409, detail="Evaluation name already exists")
    ts = db.get(Testset, payload.testset_id)
    if ts is None or ts.status != Testset.STATUS_COMPLETED:
        raise HTTPException(status_code=400, detail="Testset not found or not completed")
    rag = db.get(RagSystemConfig, payload.rag_system_id)
    if rag is None:
        raise HTTPException(status_code=400, detail="RAG system not found")
    if rag.status != RagSystemConfig.STATUS_COMPLETED:
        raise HTTPException(status_code=400, detail="RAG system adapter is not ready")
    llm = db.get(ModelConfig, payload.llm_config_id)
    emb = db.get(ModelConfig, payload.embedding_config_id)
    if llm is None or llm.type != "llm" or emb is None or emb.type != "embedding":
        raise HTTPException(status_code=400, detail="Judge model configs invalid")
    # Context metrics require the RAG system to expose retrieved contexts.
    has_contexts = bool(rag.contexts_path)
    if not has_contexts and any(m in CONTEXT_METRICS for m in payload.metrics):
        raise HTTPException(
            status_code=400,
            detail="Context metrics require a contexts extraction path on the RAG system",
        )
    run = EvalRun(
        name=payload.name.strip(),
        testset_id=ts.id,
        testset_name=ts.name,
        rag_system_id=rag.id,
        rag_system_name=rag.name,
        # Snapshot the endpoint/credential for this run (adapters persist
        # only templates/paths, so both come from the request)
        run_base_url=normalize_url(payload.base_url),
        run_api_key=payload.api_key,
        timeout=payload.timeout,
        llm_config_id=llm.id,
        llm_name=llm.name,
        embedding_config_id=emb.id,
        embedding_name=emb.name,
        concurrency=payload.concurrency,
        judge_concurrency=payload.judge_concurrency,
        use_judge_cache=payload.use_judge_cache,
        metrics=json.dumps(payload.metrics),
        has_contexts=has_contexts,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    asyncio.create_task(eval_run.run_evaluation(run.id))
    return run


@router.get("/{run_id}", response_model=EvalRunOut)
def get_run(run_id: int, db: Session = Depends(get_db)) -> EvalRun:
    return _with_hit_at_k(db, _get_or_404(run_id, db))


@router.get("/{run_id}/log", response_model=EvalLogOut)
def get_run_log(run_id: int, db: Session = Depends(get_db)) -> EvalLogOut:
    run = _get_or_404(run_id, db)
    return EvalLogOut(
        entries=json.loads(run.log_entries),
        token_usage=json.loads(run.token_usage),
        error=run.error,
    )


@router.get("/{run_id}/items", response_model=EvalItemPage)
def list_run_items(
    run_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> EvalItemPage:
    _get_or_404(run_id, db)
    stmt = (
        select(EvalRunItem).where(EvalRunItem.run_id == run_id).order_by(EvalRunItem.seq)
    )
    total = len(db.execute(stmt).scalars().all())
    items = db.execute(stmt.offset((page - 1) * page_size).limit(page_size)).scalars().all()
    return EvalItemPage(total=total, items=[EvalItemOut.model_validate(i) for i in items])


@router.post("/{run_id}/items/{seq}/recompute", response_model=EvalItemOut)
async def recompute_item(
    run_id: int, seq: int, db: Session = Depends(get_db)
) -> EvalRunItem:
    """Re-score one item of a finished run, for a judge call that came back
    unusable and left the item with no score.

    Rescoring reads around the judge cache, so it can actually replace the bad
    answer instead of replaying it. The run's means are refreshed with it.
    """
    run = _get_or_404(run_id, db)
    if run.status != EvalRun.STATUS_COMPLETED:
        raise HTTPException(
            status_code=409,
            detail="Only a completed evaluation can have one item rescored",
        )
    try:
        await eval_run.recompute_item(run_id, seq)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # The service commits in its own session; drop this one's cached row before
    # reading the result back.
    db.expire_all()
    return db.execute(
        select(EvalRunItem).where(EvalRunItem.run_id == run_id, EvalRunItem.seq == seq)
    ).scalar_one()


@router.post("/{run_id}/resume", response_model=EvalRunOut)
async def resume_run(run_id: int, db: Session = Depends(get_db)) -> EvalRun:
    run = _get_or_404(run_id, db)
    if run.status != EvalRun.STATUS_FAILED:
        raise HTTPException(status_code=409, detail="Only a failed evaluation can be resumed")
    run.status = EvalRun.STATUS_RUNNING
    run.error = None
    db.commit()
    db.refresh(run)
    eval_run._log(run.id, "resumed")
    asyncio.create_task(eval_run.run_evaluation(run.id))
    return run


@router.delete("/{run_id}", status_code=204)
def delete_run(run_id: int, db: Session = Depends(get_db)) -> None:
    run = _get_or_404(run_id, db)
    if run.status == EvalRun.STATUS_RUNNING:
        raise HTTPException(status_code=409, detail="Cannot delete an evaluation while running")
    # This run's own artifacts (responses.jsonl + any legacy per-run cache).
    # The shared judge cache under data/evaluation/_judge_cache is NOT touched:
    # it is keyed by judge model and shared by every run.
    shutil.rmtree(eval_run.eval_path_for(run.name), ignore_errors=True)
    db.query(EvalRunItem).filter(EvalRunItem.run_id == run.id).delete()
    db.delete(run)
    db.commit()
