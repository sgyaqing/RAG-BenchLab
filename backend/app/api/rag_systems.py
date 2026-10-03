import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.model_configs import get_db
from app.db.models import ModelConfig, RagSystemConfig
from app.schemas.rag_system import (
    RagSystemCreateIn,
    RagSystemLogOut,
    RagSystemOut,
    RagSystemPage,
    RagSystemUpdateIn,
)
from app.services import adapter_assist
from app.services.rag_client import QUESTION_PLACEHOLDER, normalize_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag-systems", tags=["rag-systems"])


def _get_or_404(config_id: int, db: Session) -> RagSystemConfig:
    record = db.get(RagSystemConfig, config_id)
    if record is None:
        raise HTTPException(status_code=404, detail="RAG system adapter not found")
    return record


def _name_available(name: str, db: Session, exclude_id: int | None = None) -> bool:
    stmt = select(RagSystemConfig).where(
        func.lower(RagSystemConfig.name) == name.strip().lower()
    )
    if exclude_id is not None:
        stmt = stmt.where(RagSystemConfig.id != exclude_id)
    return db.execute(stmt).scalar_one_or_none() is None


@router.get("", response_model=RagSystemPage)
def list_rag_systems(
    name: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> RagSystemPage:
    stmt = select(RagSystemConfig).order_by(RagSystemConfig.id.desc())
    if name:
        stmt = stmt.where(RagSystemConfig.name.contains(name))
    total = len(db.execute(stmt).scalars().all())
    items = db.execute(stmt.offset((page - 1) * page_size).limit(page_size)).scalars().all()
    return RagSystemPage(total=total, items=[RagSystemOut.model_validate(i) for i in items])


@router.get("/check-name")
def check_name(name: str, exclude_id: int | None = None, db: Session = Depends(get_db)) -> dict:
    return {"available": _name_available(name, db, exclude_id)}


@router.get("/{config_id}/log", response_model=RagSystemLogOut)
def get_log(config_id: int, db: Session = Depends(get_db)) -> RagSystemLogOut:
    record = _get_or_404(config_id, db)
    return RagSystemLogOut(entries=json.loads(record.log_entries), error=record.error)


@router.post("", response_model=RagSystemOut, status_code=201)
async def create_rag_system(
    payload: RagSystemCreateIn, db: Session = Depends(get_db)
) -> RagSystemConfig:
    if not _name_available(payload.name, db):
        raise HTTPException(status_code=409, detail="Adapter name already exists")

    if payload.mode == "smart":
        if not payload.platform_hint:
            raise HTTPException(status_code=400, detail="platform_hint is required")
        llm = db.get(ModelConfig, payload.llm_config_id) if payload.llm_config_id else None
        if llm is None or llm.type != "llm":
            raise HTTPException(status_code=400, detail="Smart-fill LLM config invalid")
        record = RagSystemConfig(
            name=payload.name.strip(),
            base_url=normalize_url(payload.base_url),
            api_key=payload.api_key,
            llm_config_id=llm.id,
            llm_name=llm.name,
            platform_hint=payload.platform_hint.strip(),
            lang=payload.lang or "",
            status=RagSystemConfig.STATUS_CONFIGURING,
            progress=0,
            stage="init",
        )
        task = adapter_assist.run_assist
    else:
        if not payload.body_template or not payload.answer_path:
            raise HTTPException(
                status_code=400, detail="body_template and answer_path are required"
            )
        if QUESTION_PLACEHOLDER not in payload.body_template:
            raise HTTPException(
                status_code=400, detail="body_template must contain {{question}}"
            )
        record = RagSystemConfig(
            name=payload.name.strip(),
            base_url=normalize_url(payload.base_url),
            api_key=payload.api_key,
            headers=payload.headers or "{}",
            body_template=payload.body_template,
            answer_path=payload.answer_path,
            contexts_path=payload.contexts_path or None,
            lang=payload.lang or "",
            status=RagSystemConfig.STATUS_CONFIGURING,
            progress=0,
            stage="init",
        )
        task = adapter_assist.run_simple_test

    db.add(record)
    db.commit()
    db.refresh(record)
    asyncio.create_task(task(record.id))
    return record


@router.put("/{config_id}", response_model=RagSystemOut)
async def update_rag_system(
    config_id: int, payload: RagSystemUpdateIn, db: Session = Depends(get_db)
) -> RagSystemConfig:
    record = _get_or_404(config_id, db)
    if record.status == RagSystemConfig.STATUS_CONFIGURING:
        raise HTTPException(status_code=409, detail="Cannot edit while configuring")
    if not _name_available(payload.name, db, exclude_id=config_id):
        raise HTTPException(status_code=409, detail="Adapter name already exists")
    if QUESTION_PLACEHOLDER not in payload.body_template:
        raise HTTPException(status_code=400, detail="body_template must contain {{question}}")
    updates = payload.model_dump()
    if updates.get("lang") is None:
        updates.pop("lang")  # keep the previous locale when not provided
    for field, value in updates.items():
        setattr(record, field, value)
    record.base_url = normalize_url(record.base_url)
    record.status = RagSystemConfig.STATUS_CONFIGURING
    record.progress = 0
    # "testing" rather than "init": it records that the operation in flight is a
    # manual verification, which is what restart-resume must re-run. An edit of
    # a smart-created adapter keeps its llm_config_id forever, so the resume path
    # cannot tell the two apart by that field and would re-run the agent over the
    # user's own templates.
    record.stage = "testing"
    record.error = None
    db.commit()
    db.refresh(record)
    adapter_assist._log(record.id, "edited", {})
    asyncio.create_task(
        adapter_assist.run_simple_test(record.id, fill_blank_contexts=False)
    )
    return record


@router.delete("/{config_id}", status_code=204)
def delete_rag_system(config_id: int, db: Session = Depends(get_db)) -> None:
    record = _get_or_404(config_id, db)
    if record.status == RagSystemConfig.STATUS_CONFIGURING:
        raise HTTPException(status_code=409, detail="Cannot delete while configuring")
    db.delete(record)
    db.commit()
