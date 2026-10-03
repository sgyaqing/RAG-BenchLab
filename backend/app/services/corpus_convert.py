"""Convert uploaded corpus files to Markdown using MarkItDown."""

import asyncio
import json
import logging
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.db.models import Corpus
from app.db.session import get_session_factory

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx", ".xls"}
ZIP_EXTENSION = ".zip"

_markitdown = None
_running: set[int] = set()


def _get_markitdown():
    global _markitdown
    if _markitdown is None:
        from markitdown import MarkItDown

        _markitdown = MarkItDown()
    return _markitdown


def upload_dir_for(name: str) -> Path:
    return get_settings().data_dir / "upload" / name


def corpus_dir_for(name: str) -> Path:
    return get_settings().data_dir / "corpus" / name


def scan_supported_files(upload_dir: Path) -> list[Path]:
    """All supported files under upload_dir (zip files are never converted)."""
    if not upload_dir.is_dir():
        return []
    return sorted(
        p for p in upload_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def safe_join(base: Path, relative: str) -> Path | None:
    """Resolve a user-supplied relative path under base; None if it escapes."""
    target = (base / relative).resolve()
    if target == base or base.resolve() not in target.parents:
        return None
    return target


class ArchivePathError(Exception):
    """A member of the archive cannot be written on this filesystem.

    Windows rejects names macOS accepts, nearly always because the path is
    past its 260-character limit — reserved device names (CON, NUL, COM1) and
    the characters it refuses in a name (`:`, `?`, `*`) land here too. Raised
    so the endpoint can answer 400 with the name that caused it; the raw
    OSError used to escape as a 500, telling the customer only that something
    went wrong.
    """

    def __init__(self, member: str, cause: OSError) -> None:
        super().__init__(f"{member}: {cause}")
        self.member = member
        self.cause = cause


def _member_name(info: zipfile.ZipInfo) -> str:
    """The member's real name.

    A zip written by a Windows tool often stores UTF-8 names without setting
    the flag that announces them (`flag_bits` bit 11), and `zipfile` follows
    the spec literally: no flag means cp437. Those names arrive as mojibake —
    a customer's `荣盛发展` came out as `Θò¬µ▮fΦ»üσê¬` — and mojibake is
    longer than what it encodes, three characters per Chinese character, which
    is how one upload was pushed past Windows' path limit and answered with a
    500 while the same archive extracted fine on macOS, into garbage names
    nobody was warned about.

    So the bytes are recovered and read again: UTF-8 first, which is what a
    writer meant by writing UTF-8 without saying so, then GBK, which is what a
    Windows tool writing in the local ANSI codepage meant.
    """
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        # Not cp437 after all, so something else decoded it; leave it alone.
        return info.filename
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return info.filename


def extract_zips(upload_dir: Path) -> None:
    """Extract top-level zip files in place and remove them.

    Zips nested in subdirectories (e.g. inside an uploaded zip) are left
    untouched — no recursive extraction, per design.
    """
    for zip_path in sorted(upload_dir.glob(f"*{ZIP_EXTENSION}")):
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                member = _member_name(info)
                try:
                    target = safe_join(upload_dir, member)
                    if target is None:
                        logger.warning("Skipping unsafe zip member %s in %s", member, zip_path)
                        continue
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        # Opened by ZipInfo, not by name: the name in the
                        # archive is the cp437 reading we just corrected.
                        with zf.open(info) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                except OSError as e:
                    raise ArchivePathError(member, e) from e
        zip_path.unlink()
        logger.info("Extracted and removed %s", zip_path)


def _convert_one(src: Path, upload_dir: Path, corpus_dir: Path) -> str:
    """Convert one file to Markdown. Returns the relative path. Raises on failure."""
    rel = src.relative_to(upload_dir)
    out_path = (corpus_dir / rel).with_suffix(".md")
    # Two sources can share a stem ("报告.pdf" and "报告.docx") and a resumed run
    # can land on the output of a file it already converted and deleted. Writing
    # anyway would silently drop the earlier document while both counted as
    # successes, so the second one keeps its original extension instead.
    if out_path.exists():
        out_path = out_path.with_suffix(f"{src.suffix.lower()}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = _get_markitdown().convert(str(src))
    out_path.write_text(result.text_content, encoding="utf-8")
    src.unlink()
    return str(rel)


def _since(corpus: Corpus) -> float:
    """Seconds since the corpus row was created — upload plus conversion so far.

    created_at is stored as naive UTC, which is what the rest of the codebase
    assumes it means.
    """
    created = corpus.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - created).total_seconds(), 1)


def _log(corpus: Corpus, key: str, params: dict | None = None) -> None:
    """Append one timestamped entry — the shape the other three logs use.

    Conversion ran to a summary instead because there was nowhere to put
    events: the row had counters but no log_entries. A list of files that
    failed is a result; the same thing with times on it is a log, and four
    pages showing four shapes of "log" is what that looked like from outside.
    """
    entries = json.loads(corpus.log_entries or "[]")
    entries.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "params": params or {},
    })
    corpus.log_entries = json.dumps(entries, ensure_ascii=False)


async def run_conversion(corpus_id: int) -> None:
    """Background task: convert all supported files of a corpus to Markdown.

    Never leaves the corpus stuck in 'converting': on an unexpected crash the
    remaining files are recorded as failed and the corpus is finalized as
    completed, so the UI always reaches a terminal state.
    """
    if corpus_id in _running:
        return
    _running.add(corpus_id)
    started = time.monotonic()
    db = get_session_factory()()
    corpus: Corpus | None = None
    files: list[Path] = []
    upload_dir: Path | None = None
    try:
        try:
            corpus = db.get(Corpus, corpus_id)
            if corpus is None:
                return
            upload_dir = upload_dir_for(corpus.name)
            corpus_dir = corpus_dir_for(corpus.name)
            corpus_dir.mkdir(parents=True, exist_ok=True)

            files = scan_supported_files(upload_dir)
            # The upload dir holds only what is LEFT to convert (each success
            # deletes its source), while success_files/processed_files carry over
            # from before a restart. Taking len(files) as the total therefore
            # reported 10 converted out of 5 files — a 200% progress bar.
            corpus.total_files = (
                corpus.success_files + len(json.loads(corpus.failed_files)) + len(files)
            )
            # The corpus row is created before the first file is uploaded, so its
            # age by the time conversion starts is how long the upload took —
            # the backend never sees those requests, one per file.
            _log(corpus, "uploaded", {
                "files": len(files),
                "seconds": _since(corpus),
            })
            db.commit()
            logger.info("Corpus %s: converting %d files", corpus.name, len(files))

            for src in files:
                try:
                    await asyncio.to_thread(_convert_one, src, upload_dir, corpus_dir)
                    corpus.success_files += 1
                except Exception as e:
                    logger.warning("Corpus %s: failed to convert %s: %s", corpus.name, src, e)
                    _log(corpus, "fileFailed", {
                        "path": str(src.relative_to(upload_dir)),
                        "error": str(e)[:200],
                    })
                    corpus.failed_files = json.dumps(
                        json.loads(corpus.failed_files) + [str(src.relative_to(upload_dir))]
                    )
                corpus.processed_files += 1
                db.commit()
        except Exception:
            logger.exception(
                "Corpus %s: conversion task crashed", corpus.name if corpus else corpus_id
            )
            db.rollback()
            corpus = db.get(Corpus, corpus_id)
            if corpus is not None and upload_dir is not None:
                # Record everything not yet processed as failed.
                remaining = [
                    str(f.relative_to(upload_dir)) for f in files[corpus.processed_files :]
                ]
                if remaining:
                    corpus.failed_files = json.dumps(json.loads(corpus.failed_files) + remaining)
                corpus.processed_files = corpus.total_files

        if corpus is not None:
            corpus.status = Corpus.STATUS_COMPLETED
            corpus.completed_at = datetime.now(timezone.utc)
            corpus.convert_seconds = round(time.monotonic() - started, 2)
            _log(corpus, "done", {
                "success": corpus.success_files,
                "failed": len(json.loads(corpus.failed_files)),
                "seconds": corpus.convert_seconds,
            })
            _log(corpus, "finished", {"seconds": _since(corpus)})
            db.commit()
            logger.info(
                "Corpus %s: done, %d/%d succeeded in %.2fs",
                corpus.name,
                corpus.success_files,
                corpus.total_files,
                corpus.convert_seconds,
            )
            # The upload directory must be fully removed once conversion ends.
            upload_dir = upload_dir_for(corpus.name)
            if upload_dir.exists():
                shutil.rmtree(upload_dir, ignore_errors=True)
    finally:
        db.close()
        _running.discard(corpus_id)


async def resume_interrupted() -> None:
    """On startup, handle corpora still marked as converting.

    - Conversion started (total_files > 0) and upload dir intact: resume.
    - Conversion started but upload dir gone: finalize with current stats.
    - Never started (upload interrupted, total_files == 0): delete the corpus
      and its partial files — a half-uploaded corpus cannot be resumed.
    """
    db = get_session_factory()()
    try:
        corpora = db.query(Corpus).filter(Corpus.status == Corpus.STATUS_CONVERTING).all()
        for corpus in corpora:
            if corpus.total_files == 0:
                logger.info("Discarding never-started corpus %s", corpus.name)
                shutil.rmtree(upload_dir_for(corpus.name), ignore_errors=True)
                shutil.rmtree(corpus_dir_for(corpus.name), ignore_errors=True)
                db.delete(corpus)
            elif upload_dir_for(corpus.name).exists():
                logger.info("Resuming conversion for corpus %s", corpus.name)
                asyncio.create_task(run_conversion(corpus.id))
            else:
                corpus.status = Corpus.STATUS_COMPLETED
                corpus.completed_at = datetime.now(timezone.utc)
                logger.info("Finalized interrupted corpus %s", corpus.name)
        db.commit()
    finally:
        db.close()
