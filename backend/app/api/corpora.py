import asyncio
import json
import logging
import os
import shutil
import tempfile
import zipfile
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from app.api.model_configs import get_db
from app.db.models import Corpus, Testset
from app.schemas.corpus import (
    CorpusCreateIn,
    CorpusLogOut,
    CorpusOut,
    CorpusPage,
    NameCheckOut,
)
from app.services import corpus_convert, testset_gen

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/corpora", tags=["corpora"])

ALLOWED_UPLOAD_EXTENSIONS = corpus_convert.SUPPORTED_EXTENSIONS | {corpus_convert.ZIP_EXTENSION}


def _get_or_404(corpus_id: int, db: Session) -> Corpus:
    corpus = db.get(Corpus, corpus_id)
    if corpus is None:
        raise HTTPException(status_code=404, detail="Corpus not found")
    return corpus


@router.get("", response_model=CorpusPage)
def list_corpora(
    name: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> CorpusPage:
    stmt = select(Corpus).order_by(Corpus.id.desc())
    if name:
        stmt = stmt.where(Corpus.name.contains(name))
    total = len(db.execute(stmt).scalars().all())
    items = db.execute(stmt.offset((page - 1) * page_size).limit(page_size)).scalars().all()
    out = []
    for i in items:
        o = CorpusOut.model_validate(i)
        o.has_kg = testset_gen.kg_path_for(i.name).exists()
        out.append(o)
    return CorpusPage(total=total, items=out)


@router.get("/check-name", response_model=NameCheckOut)
def check_name(name: str, db: Session = Depends(get_db)) -> NameCheckOut:
    existing = db.execute(
        select(Corpus).where(func.lower(Corpus.name) == name.strip().lower())
    ).scalar_one_or_none()
    return NameCheckOut(available=existing is None)


@router.post("", response_model=CorpusOut, status_code=201)
def create_corpus(payload: CorpusCreateIn, db: Session = Depends(get_db)) -> Corpus:
    if not check_name(payload.name, db).available:
        raise HTTPException(status_code=409, detail="Corpus name already exists")
    corpus = Corpus(name=payload.name)
    db.add(corpus)
    db.commit()
    db.refresh(corpus)
    corpus_convert.upload_dir_for(corpus.name).mkdir(parents=True, exist_ok=True)
    return corpus


def _unusable_name(name: str) -> str:
    """The 400 shown when a file cannot be written under the data directory.

    The name is included, truncated: it is the only thing that tells the
    customer which file to rename, and a pathological name runs to hundreds of
    characters, which the notification cannot show anyway.
    """
    shown = name if len(name) <= 60 else name[:57] + "..."
    return (
        "A file has a name this system cannot save — too long for the "
        f"filesystem, or characters it does not allow: {shown}"
    )


@router.post("/{corpus_id}/files", status_code=201)
async def upload_file(
    corpus_id: int,
    file: UploadFile,
    path: str = Form(...),
    db: Session = Depends(get_db),
) -> dict:
    corpus = _get_or_404(corpus_id, db)
    upload_dir = corpus_convert.upload_dir_for(corpus.name)

    target = corpus_convert.safe_join(upload_dir, path)
    if target is None:
        raise HTTPException(status_code=400, detail="Invalid file path")
    suffix = target.suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {target.suffix}")
    # Zip is only allowed as a standalone top-level file; no nested unzip.
    if suffix == corpus_convert.ZIP_EXTENSION and target.parent != upload_dir:
        raise HTTPException(status_code=400, detail="Zip files are only allowed at the top level")

    # Sending one file at a time means a name this filesystem refuses reaches
    # us here too, not only through a zip: the whole path, upload directory
    # included, can be over Windows' limit. Answer with the name rather than
    # letting the OSError come back as a 500.
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        target.write_bytes(content)
    except OSError as e:
        raise HTTPException(status_code=400, detail=_unusable_name(path)) from e
    return {"saved": path}


@router.post("/{corpus_id}/process", response_model=CorpusOut)
async def start_processing(corpus_id: int, db: Session = Depends(get_db)) -> Corpus:
    corpus = _get_or_404(corpus_id, db)
    if corpus.status != Corpus.STATUS_CONVERTING or corpus.total_files > 0:
        raise HTTPException(status_code=409, detail="Corpus is already being processed")

    upload_dir = corpus_convert.upload_dir_for(corpus.name)

    def discard() -> None:
        """A zip we cannot use takes the corpus with it — there is nothing
        left to convert, and the customer's next try starts from an empty
        corpus rather than a half-extracted one."""
        shutil.rmtree(upload_dir, ignore_errors=True)
        db.delete(corpus)
        db.commit()

    try:
        corpus_convert.extract_zips(upload_dir)
    except zipfile.BadZipFile as e:
        discard()
        raise HTTPException(status_code=400, detail="Invalid zip file") from e
    except corpus_convert.ArchivePathError as e:
        discard()
        raise HTTPException(status_code=400, detail=_unusable_name(e.member)) from e

    total = len(corpus_convert.scan_supported_files(upload_dir))
    if total == 0:
        shutil.rmtree(upload_dir, ignore_errors=True)
        db.delete(corpus)
        db.commit()
        raise HTTPException(status_code=400, detail="No supported files found")

    corpus.total_files = total
    db.commit()
    db.refresh(corpus)
    asyncio.create_task(corpus_convert.run_conversion(corpus.id))
    return corpus


@router.get("/{corpus_id}/detect-language")
def detect_corpus_language(corpus_id: int, db: Session = Depends(get_db)) -> dict:
    """Detect zh/en from corpus docs (for the pre-generation confirmation)."""
    corpus = _get_or_404(corpus_id, db)
    return {"language": testset_gen.detect_language(corpus.name)}


@router.get("/{corpus_id}/log", response_model=CorpusLogOut)
def get_corpus_log(corpus_id: int, db: Session = Depends(get_db)) -> CorpusLogOut:
    corpus = _get_or_404(corpus_id, db)
    return CorpusLogOut(
        entries=json.loads(corpus.log_entries or "[]"),
        total_files=corpus.total_files,
        success_files=corpus.success_files,
        failed_files=json.loads(corpus.failed_files),
        convert_seconds=corpus.convert_seconds,
        completed_at=corpus.completed_at,
    )


@router.get("/{corpus_id}/export")
def export_corpus(corpus_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """Download a completed corpus as a zip of its converted .md files,
    preserving the directory hierarchy."""
    corpus = _get_or_404(corpus_id, db)
    if corpus.status != Corpus.STATUS_COMPLETED:
        raise HTTPException(status_code=409, detail="Only completed corpora can be exported")
    src = corpus_convert.corpus_dir_for(corpus.name)
    if not src.is_dir():
        raise HTTPException(status_code=404, detail="Corpus files not found")
    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Only the converted .md files belong to the corpus; kg.json /
            # kg_fingerprint.json / seeds.json are testset-generation
            # internals and must not leak into the export.
            for p in sorted(src.rglob("*.md")):
                if p.is_file():
                    zf.write(p, p.relative_to(src))
    except Exception:
        os.unlink(tmp_path)
        raise
    filename = quote(f"{corpus.name}.zip")
    logger.info("Exported corpus %s (%d files)", corpus.name, corpus.success_files)
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        background=BackgroundTask(os.unlink, tmp_path),
    )


@router.delete("/{corpus_id}", status_code=204)
def delete_corpus(corpus_id: int, db: Session = Depends(get_db)) -> None:
    corpus = _get_or_404(corpus_id, db)
    # Only block deletion once conversion has actually started; corpora still
    # in the upload phase (total_files == 0) can be removed, which also lets
    # the frontend roll back after a failed upload.
    if corpus.status == Corpus.STATUS_CONVERTING and corpus.total_files > 0:
        raise HTTPException(status_code=409, detail="Cannot delete a corpus while converting")
    # The corpus directory holds the .md files, the knowledge graph and the seed
    # list that a running generation reads and rewrites. Deleting it under one
    # fails that run with FileNotFoundError after its tokens are already spent.
    busy = db.execute(
        select(Testset).where(Testset.corpus_id == corpus.id,
                              Testset.status == Testset.STATUS_GENERATING)
    ).scalars().first()
    if busy is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a corpus while testset '{busy.name}' is generating",
        )
    shutil.rmtree(corpus_convert.corpus_dir_for(corpus.name), ignore_errors=True)
    shutil.rmtree(corpus_convert.upload_dir_for(corpus.name), ignore_errors=True)
    db.delete(corpus)
    db.commit()
