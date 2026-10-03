import json
from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ModelConfig
from app.db.session import get_session_factory
from app.schemas.model_config import (
    ConnectivityTestIn,
    ConnectivityTestOut,
    ListModelsIn,
    ListModelsOut,
    ModelConfigIn,
    ModelConfigOut,
    ModelConfigPage,
)
from app.services import model_connect, thinking_probe

router = APIRouter(prefix="/api/model-configs", tags=["model-configs"])


def get_db() -> Generator[Session, None, None]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def _name_available(name: str, db: Session, exclude_id: int | None = None) -> bool:
    """Case-insensitive, whitespace-trimmed, and the same rule the other four
    resources apply — a name is how a testset or an evaluation refers to the
    model afterwards, so two configs sharing one is not just ugly."""
    stmt = select(ModelConfig).where(func.lower(ModelConfig.name) == name.strip().lower())
    if exclude_id is not None:
        stmt = stmt.where(ModelConfig.id != exclude_id)
    return db.execute(stmt).scalar_one_or_none() is None


@router.get("/check-name")
def check_name(
    name: str, exclude_id: int | None = None, db: Session = Depends(get_db)
) -> dict:
    return {"available": _name_available(name, db, exclude_id)}


@router.get("", response_model=ModelConfigPage)
def list_model_configs(
    name: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> ModelConfigPage:
    stmt = select(ModelConfig).order_by(ModelConfig.id)
    if name:
        stmt = stmt.where(ModelConfig.name.contains(name))
    total = len(db.execute(stmt).scalars().all())
    items = db.execute(stmt.offset((page - 1) * page_size).limit(page_size)).scalars().all()
    return ModelConfigPage(
        total=total, items=[ModelConfigOut.model_validate(i, from_attributes=True) for i in items]
    )


async def _apply_thinking_probe(record: ModelConfig, payload: ModelConfigIn) -> None:
    """Record what this endpoint does about thinking, at save time.

    The answer cannot be derived from the config: the field that turns thinking
    off has a different name on every platform, and one it does not recognise is
    ignored in silence, so the only way to know is to call the endpoint and look
    at what comes back. Unticking the box clears the finding rather than leaving
    an old one behind, so that what the list shows and what the runtime sends
    stay the same thing.
    """
    record.thinking_state = None
    record.thinking_param = None
    if payload.type != "llm" or not payload.auto_disable_thinking:
        return
    state, param = await thinking_probe.probe(
        payload.api_format, payload.base_url, payload.api_key, payload.model
    )
    record.thinking_state = state
    record.thinking_param = json.dumps(param, ensure_ascii=False) if param else None


@router.post("", response_model=ModelConfigOut, status_code=201)
async def create_model_config(
    payload: ModelConfigIn, db: Session = Depends(get_db)
) -> ModelConfig:
    if not _name_available(payload.name, db):
        raise HTTPException(status_code=409, detail="Model config name already exists")
    record = ModelConfig(**payload.model_dump(exclude={"auto_disable_thinking"}))
    await _apply_thinking_probe(record, payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.put("/{config_id}", response_model=ModelConfigOut)
async def update_model_config(
    config_id: int, payload: ModelConfigIn, db: Session = Depends(get_db)
) -> ModelConfig:
    record = db.get(ModelConfig, config_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model config not found")
    if not _name_available(payload.name, db, exclude_id=config_id):
        raise HTTPException(status_code=409, detail="Model config name already exists")
    for field, value in payload.model_dump(exclude={"auto_disable_thinking"}).items():
        setattr(record, field, value)
    # Re-probed on every save: the model name or the address may have changed,
    # and the old finding belongs to whatever was there before.
    await _apply_thinking_probe(record, payload)
    db.commit()
    db.refresh(record)
    return record


@router.delete("/{config_id}", status_code=204)
def delete_model_config(config_id: int, db: Session = Depends(get_db)) -> None:
    record = db.get(ModelConfig, config_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model config not found")
    db.delete(record)
    db.commit()


@router.post("/test", response_model=ConnectivityTestOut)
async def test_model_connectivity(payload: ConnectivityTestIn) -> ConnectivityTestOut:
    success, message, duration_ms = await model_connect.test_connectivity(
        payload.type, payload.api_format, payload.base_url, payload.api_key, payload.model
    )
    return ConnectivityTestOut(success=success, message=message, duration_ms=duration_ms)


@router.post("/available-models", response_model=ListModelsOut)
async def list_available_models(payload: ListModelsIn) -> ListModelsOut:
    try:
        models = await model_connect.list_models(
            payload.api_format, payload.base_url, payload.api_key
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return ListModelsOut(models=models)
