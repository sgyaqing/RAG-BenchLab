import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.model_configs import get_db
from app.db.models import Corpus, ModelConfig, Testset, TestsetItem
from app.schemas.corpus import NAME_PATTERN
from app.schemas.testset import (
    TestsetCreateIn,
    TestsetItemOut,
    TestsetItemPage,
    TestsetItemUpdateIn,
    TestsetLogOut,
    TestsetOut,
    TestsetPage,
)
from app.services import testset_gen

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/testsets", tags=["testsets"])


def _get_or_404(testset_id: int, db: Session) -> Testset:
    ts = db.get(Testset, testset_id)
    if ts is None:
        raise HTTPException(status_code=404, detail="Testset not found")
    return ts


def _item_out(item: TestsetItem) -> TestsetItemOut:
    return TestsetItemOut(
        id=item.id,
        seq=item.seq,
        user_input=item.user_input,
        reference=item.reference,
        reference_contexts=json.loads(item.reference_contexts),
        synthesizer_name=item.synthesizer_name,
        persona_name=item.persona_name,
        edited=bool(item.edited),
        has_original=item.original_user_input is not None,
    )


@router.get("", response_model=TestsetPage)
def list_testsets(
    name: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> TestsetPage:
    stmt = select(Testset).order_by(Testset.id.desc())
    if name:
        stmt = stmt.where(Testset.name.contains(name))
    total = len(db.execute(stmt).scalars().all())
    items = db.execute(stmt.offset((page - 1) * page_size).limit(page_size)).scalars().all()
    # One query for the whole page: which testsets have any edited QA item.
    edited_ids: set[int] = set()
    if items:
        edited_ids = set(
            db.execute(
                select(TestsetItem.testset_id)
                .where(
                    TestsetItem.testset_id.in_([i.id for i in items]),
                    TestsetItem.edited.is_(True),
                )
                .distinct()
            )
            .scalars()
            .all()
        )
    out = []
    counts: dict[int, int] = {}
    if items:
        counts = dict(
            db.execute(
                select(TestsetItem.testset_id, func.count())
                .where(TestsetItem.testset_id.in_([i.id for i in items]))
                .group_by(TestsetItem.testset_id)
            ).all()
        )
    for i in items:
        row = TestsetOut.model_validate(i)
        row.edited = i.id in edited_ids
        row.item_count = counts.get(i.id, 0)
        out.append(row)
    return TestsetPage(total=total, items=out)


@router.get("/check-name")
def check_name(name: str, db: Session = Depends(get_db)) -> dict:
    existing = db.execute(
        select(Testset).where(func.lower(Testset.name) == name.strip().lower())
    ).scalar_one_or_none()
    return {"available": existing is None}


def _other_generating_on_corpus(corpus_id: int, db: Session,
                                exclude_id: int | None = None) -> Testset | None:
    """The testset already generating from this corpus, if any.

    The knowledge graph, seed list and persona cache are shared files under
    data/corpus/<name>/, and the pipeline only serialises per testset — two
    runs would interleave writes to them and could leave a graph that does not
    match the seed list beside it. Every entry point that starts a run has to
    ask this; the create path checking on its own left resume able to start a
    second run on a corpus that was already busy.
    """
    stmt = select(Testset).where(
        Testset.corpus_id == corpus_id,
        Testset.status == Testset.STATUS_GENERATING,
    )
    if exclude_id is not None:
        stmt = stmt.where(Testset.id != exclude_id)
    return db.execute(stmt).scalars().first()


def _refuse_if_corpus_busy(corpus_id: int, db: Session,
                           exclude_id: int | None = None) -> None:
    running = _other_generating_on_corpus(corpus_id, db, exclude_id)
    if running is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Testset '{running.name}' is already generating from this corpus",
        )


def _commit_or_refuse(db: Session, corpus_id: int) -> None:
    """Commit, turning the one-generating-per-corpus index into the same 409.

    The check above and the write below are separate statements, so two requests
    can pass the check together; uq_testsets_generating_per_corpus is what
    actually holds the rule. Naming the testset that won needs a query — the
    constraint only knows the corpus — so this re-reads rather than reporting
    the constraint.
    """
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        _refuse_if_corpus_busy(corpus_id, db)
        raise HTTPException(
            status_code=409,
            detail="Another testset is already generating from this corpus",
        )


@router.post("", response_model=TestsetOut, status_code=201)
async def create_testset(payload: TestsetCreateIn, db: Session = Depends(get_db)) -> Testset:
    if not check_name(payload.name, db)["available"]:
        raise HTTPException(status_code=409, detail="Testset name already exists")
    corpus = db.get(Corpus, payload.corpus_id)
    if corpus is None or corpus.status != Corpus.STATUS_COMPLETED:
        raise HTTPException(status_code=400, detail="Corpus not found or not ready")
    if corpus.success_files == 0:
        raise HTTPException(status_code=400, detail="Corpus has no converted files")
    _refuse_if_corpus_busy(corpus.id, db)
    llm = db.get(ModelConfig, payload.llm_config_id)
    emb = db.get(ModelConfig, payload.embedding_config_id)
    if llm is None or llm.type != "llm":
        raise HTTPException(status_code=400, detail="LLM configuration not found")
    if emb is None or emb.type != "embedding":
        raise HTTPException(status_code=400, detail="Embedding configuration not found")

    ts = Testset(
        name=payload.name,
        corpus_id=corpus.id,
        corpus_name=corpus.name,
        reuse_kg=payload.reuse_kg,
        llm_config_id=llm.id,
        llm_name=llm.name,
        embedding_config_id=emb.id,
        embedding_name=emb.name,
        llm_concurrency=payload.llm_concurrency,
        llm_max_tokens=payload.llm_max_tokens,
        n_single=payload.n_single,
        n_multi_specific=payload.n_multi_specific,
        n_multi_abstract=payload.n_multi_abstract,
        prompt_language=payload.prompt_language,
        amplify=payload.amplify,
        gen_amplify=payload.gen_amplify,
    )
    db.add(ts)
    _commit_or_refuse(db, corpus.id)
    db.refresh(ts)
    asyncio.create_task(testset_gen.run_generation(ts.id))
    return ts


def _synth_type(name: str) -> str | None:
    """Map a ragas synthesizer name to a display type; None if unrecognized."""
    if "multi_hop_abstract" in name:
        return "multi_abstract"
    if "multi_hop" in name:
        return "multi_specific"
    if "single_hop" in name:
        return "single"
    return None


def _parse_import_jsonl(text: str) -> list[dict]:
    """Parse and validate an imported JSONL testset. Any bad line rejects the
    whole file with a 400 naming the first offending line."""
    items: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"Line {lineno}: not valid JSON")
        if not isinstance(obj, dict):
            raise HTTPException(status_code=400, detail=f"Line {lineno}: expected a JSON object")
        for field in ("user_input", "reference"):
            if not isinstance(obj.get(field), str):
                raise HTTPException(
                    status_code=400, detail=f"Line {lineno}: {field} must be a string"
                )
        user_input = testset_gen.normalize_qa_text(obj["user_input"])
        reference = testset_gen.normalize_qa_text(obj["reference"], multiline=True)
        if not user_input:
            raise HTTPException(status_code=400, detail=f"Line {lineno}: user_input is required")
        if not reference:
            raise HTTPException(status_code=400, detail=f"Line {lineno}: reference is required")
        contexts = obj.get("reference_contexts") or []
        if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
            raise HTTPException(
                status_code=400,
                detail=f"Line {lineno}: reference_contexts must be an array of strings",
            )
        # Contexts are corpus excerpts — kept verbatim (no normalization).
        contexts = [c for c in contexts if c.strip()]
        synth = obj.get("synthesizer_name") or ""
        if not isinstance(synth, str):
            raise HTTPException(
                status_code=400, detail=f"Line {lineno}: synthesizer_name must be a string"
            )
        items.append(
            {
                "user_input": user_input,
                "reference": reference,
                "reference_contexts": contexts,
                "synthesizer_name": synth,
            }
        )
    return items


@router.post("/import", response_model=TestsetOut, status_code=201)
def import_testset(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> TestsetOut:
    """Import a testset from a JSONL file. No corpus/KG is attached: corpus_id
    is the 0 sentinel and the frontend shows an "imported from file" label."""
    name = name.strip()
    if not name or len(name) > 64 or not NAME_PATTERN.match(name):
        raise HTTPException(status_code=422, detail="Invalid testset name")
    if not check_name(name, db)["available"]:
        raise HTTPException(status_code=409, detail="Testset name already exists")
    raw = file.file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8")
    items = _parse_import_jsonl(text)
    if not items:
        raise HTTPException(status_code=400, detail="File contains no QA pairs")

    now = datetime.now(timezone.utc)
    types = [_synth_type(i["synthesizer_name"]) for i in items]
    # Per-type counts only when every row's type is recognizable; otherwise
    # the frontend shows just the total (mixed/untyped imports).
    typed = all(types)
    ts = Testset(
        name=name,
        corpus_id=0,
        corpus_name="",
        reuse_kg=False,
        llm_config_id=0,
        llm_name="",
        embedding_config_id=0,
        embedding_name="",
        n_single=0,
        n_multi_specific=0,
        n_multi_abstract=0,
        status=Testset.STATUS_COMPLETED,
        progress=100,
        stage="done",
        completed_at=now,
        actual_single=sum(1 for t in types if t == "single") if typed else 0,
        actual_multi_specific=sum(1 for t in types if t == "multi_specific") if typed else 0,
        actual_multi_abstract=sum(1 for t in types if t == "multi_abstract") if typed else 0,
        log_entries=json.dumps(
            [
                {
                    "time": now.isoformat(),
                    "key": "imported",
                    "params": {"filename": file.filename or "", "count": len(items)},
                }
            ],
            ensure_ascii=False,
        ),
    )
    db.add(ts)
    db.flush()
    for seq, i in enumerate(items, start=1):
        db.add(
            TestsetItem(
                testset_id=ts.id,
                seq=seq,
                user_input=i["user_input"],
                reference=i["reference"],
                reference_contexts=json.dumps(i["reference_contexts"], ensure_ascii=False),
                synthesizer_name=i["synthesizer_name"],
            )
        )
    db.commit()
    db.refresh(ts)
    out = TestsetOut.model_validate(ts)
    out.item_count = len(items)
    logger.info("Imported testset %s: %d QA pairs from %s", ts.name, len(items), file.filename)
    return out


@router.get("/{testset_id}", response_model=TestsetOut)
def get_testset(testset_id: int, db: Session = Depends(get_db)) -> TestsetOut:
    ts = _get_or_404(testset_id, db)
    out = TestsetOut.model_validate(ts)
    out.edited = (
        db.execute(
            select(TestsetItem.id)
            .where(TestsetItem.testset_id == ts.id, TestsetItem.edited.is_(True))
            .limit(1)
        ).scalar_one_or_none()
        is not None
    )
    out.item_count = (
        db.execute(
            select(func.count()).select_from(TestsetItem).where(TestsetItem.testset_id == ts.id)
        ).scalar_one()
    )
    return out


@router.get("/{testset_id}/log", response_model=TestsetLogOut)
def get_testset_log(testset_id: int, db: Session = Depends(get_db)) -> TestsetLogOut:
    ts = _get_or_404(testset_id, db)
    return TestsetLogOut(
        entries=json.loads(ts.log_entries),
        token_usage=json.loads(ts.token_usage),
        error=ts.error,
    )


@router.post("/{testset_id}/resume", response_model=TestsetOut)
async def resume_testset(testset_id: int, db: Session = Depends(get_db)) -> Testset:
    """Re-run a failed testset. LLM disk cache + saved KG make already-finished
    work free; seeds are deterministic, so the pipeline resumes cheaply."""
    ts = _get_or_404(testset_id, db)
    if ts.status != Testset.STATUS_FAILED:
        raise HTTPException(status_code=409, detail="Only failed testsets can be resumed")
    # A previous run may still be unwinding (its status was committed as failed
    # a moment before it leaves the in-flight set). Starting another one would
    # return straight back out of run_generation and leave the row generating
    # with nothing running, so refuse instead.
    if testset_gen.is_running(testset_id):
        raise HTTPException(status_code=409,
                            detail="This testset is still finishing its previous run")
    # Resuming starts a run, so it needs the same corpus check creating does:
    # without it a failed testset could be resumed onto a corpus that is
    # already generating, and the two would interleave writes to the shared
    # graph, seed list and persona cache.
    _refuse_if_corpus_busy(ts.corpus_id, db, exclude_id=ts.id)
    ts.status = Testset.STATUS_GENERATING
    ts.error = None
    ts.progress = 0
    ts.stage = "init"
    ts.stage_done = 0
    ts.stage_total = 0
    ts.completed_at = None
    entries = json.loads(ts.log_entries)
    entries.append(
        {"time": datetime.now(timezone.utc).isoformat(), "key": "resumed", "params": {}}
    )
    ts.log_entries = json.dumps(entries, ensure_ascii=False)
    _commit_or_refuse(db, ts.corpus_id)
    db.refresh(ts)
    asyncio.create_task(testset_gen.run_generation(ts.id))
    return ts


@router.get("/{testset_id}/export")
def export_testset(testset_id: int, db: Session = Depends(get_db)) -> Response:
    """Download a completed testset as JSONL (one QA pair per line)."""
    ts = _get_or_404(testset_id, db)
    if ts.status != Testset.STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail="Only completed testsets can be exported")
    items = (
        db.execute(
            select(TestsetItem)
            .where(TestsetItem.testset_id == ts.id)
            .order_by(TestsetItem.seq)
        )
        .scalars()
        .all()
    )
    lines = [
        json.dumps(
            {
                "user_input": i.user_input,
                "reference": i.reference,
                "reference_contexts": json.loads(i.reference_contexts),
                "synthesizer_name": i.synthesizer_name,
            },
            ensure_ascii=False,
        )
        for i in items
    ]
    body = "\n".join(lines) + ("\n" if lines else "")
    filename = quote(f"{ts.name}.jsonl")
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.delete("/{testset_id}", status_code=204)
def delete_testset(testset_id: int, db: Session = Depends(get_db)) -> None:
    ts = _get_or_404(testset_id, db)
    if ts.status == Testset.STATUS_GENERATING:
        raise HTTPException(status_code=409, detail="Cannot delete a testset while generating")
    shutil.rmtree(testset_gen.testset_dir_for(ts.name), ignore_errors=True)
    db.query(TestsetItem).filter(TestsetItem.testset_id == ts.id).delete()
    db.delete(ts)
    db.commit()


# --------------------------------------------------------------------------
# QA items (edit page)
# --------------------------------------------------------------------------


@router.get("/{testset_id}/items", response_model=TestsetItemPage)
def list_items(
    testset_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> TestsetItemPage:
    _get_or_404(testset_id, db)
    stmt = (
        select(TestsetItem)
        .where(TestsetItem.testset_id == testset_id)
        .order_by(TestsetItem.seq)
    )
    total = len(db.execute(stmt).scalars().all())
    items = db.execute(stmt.offset((page - 1) * page_size).limit(page_size)).scalars().all()
    return TestsetItemPage(total=total, items=[_item_out(i) for i in items])


def _touch_testset_after_edit(testset_id: int, log_key: str, params: dict, db: Session) -> None:
    """Edits update the testset's completed_at and append a log entry."""
    ts = db.get(Testset, testset_id)
    if ts is None:
        return
    ts.completed_at = datetime.now(timezone.utc)
    entries = json.loads(ts.log_entries)
    entries.append({"time": ts.completed_at.isoformat(), "key": log_key, "params": params})
    ts.log_entries = json.dumps(entries, ensure_ascii=False)


@router.put("/{testset_id}/items/{item_id}", response_model=TestsetItemOut)
def update_item(
    testset_id: int, item_id: int, payload: TestsetItemUpdateIn, db: Session = Depends(get_db)
) -> TestsetItemOut:
    _get_or_404(testset_id, db)
    item = db.get(TestsetItem, item_id)
    if item is None or item.testset_id != testset_id:
        raise HTTPException(status_code=404, detail="Item not found")
    if not item.edited:
        item.original_user_input = item.user_input
        item.original_reference = item.reference
    item.user_input = payload.user_input
    item.reference = payload.reference
    item.edited = 1
    item.updated_at = datetime.now(timezone.utc)
    _touch_testset_after_edit(testset_id, "edited", {"seq": item.seq}, db)
    db.commit()
    db.refresh(item)
    return _item_out(item)


@router.post("/{testset_id}/items/{item_id}/restore", response_model=TestsetItemOut)
def restore_item(testset_id: int, item_id: int, db: Session = Depends(get_db)) -> TestsetItemOut:
    _get_or_404(testset_id, db)
    item = db.get(TestsetItem, item_id)
    if item is None or item.testset_id != testset_id:
        raise HTTPException(status_code=404, detail="Item not found")
    if item.original_user_input is None:
        raise HTTPException(status_code=400, detail="No original value to restore")
    item.user_input = item.original_user_input
    item.reference = item.original_reference or ""
    item.original_user_input = None
    item.original_reference = None
    item.edited = 0
    item.updated_at = datetime.now(timezone.utc)
    _touch_testset_after_edit(testset_id, "restored", {"seq": item.seq}, db)
    db.commit()
    db.refresh(item)
    return _item_out(item)


@router.delete("/{testset_id}/items/{item_id}", status_code=204)
def delete_item(testset_id: int, item_id: int, db: Session = Depends(get_db)) -> None:
    _get_or_404(testset_id, db)
    item = db.get(TestsetItem, item_id)
    if item is None or item.testset_id != testset_id:
        raise HTTPException(status_code=404, detail="Item not found")
    _touch_testset_after_edit(testset_id, "itemDeleted", {"seq": item.seq}, db)
    db.delete(item)
    db.commit()
