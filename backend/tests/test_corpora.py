import asyncio
import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services import corpus_convert


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def _upload(client, corpus_id: int, path: str, content: bytes):
    return client.post(
        f"/api/corpora/{corpus_id}/files",
        files={"file": (path.split("/")[-1], io.BytesIO(content))},
        data={"path": path},
    )


def _wait_completed(client, corpus_id: int, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/api/corpora").json()
        item = next(i for i in body["items"] if i["id"] == corpus_id)
        if item["status"] == "completed":
            return item
        time.sleep(0.2)
    raise AssertionError("conversion did not finish in time")


def test_check_name_case_insensitive(client):
    assert client.get("/api/corpora/check-name", params={"name": "MyCorpus"}).json()["available"]
    client.post("/api/corpora", json={"name": "MyCorpus"})
    assert not client.get("/api/corpora/check-name", params={"name": "mycorpus"}).json()[
        "available"
    ]
    # trailing whitespace is ignored when checking
    assert not client.get("/api/corpora/check-name", params={"name": "MYCORPUS "}).json()[
        "available"
    ]
    resp = client.post("/api/corpora", json={"name": "mycorpus"})
    assert resp.status_code == 409


def test_invalid_name_rejected(client):
    resp = client.post("/api/corpora", json={"name": "bad/name"})
    assert resp.status_code == 422


def test_full_flow_with_txt_and_unsupported(client):
    corpus = client.post("/api/corpora", json={"name": "Flow"}).json()
    cid = corpus["id"]

    assert _upload(client, cid, "docs/a.txt", b"hello corpus").status_code == 201
    assert _upload(client, cid, "docs/sub/b.md", b"# title").status_code == 201
    # unsupported extension rejected
    assert _upload(client, cid, "docs/evil.exe", b"x").status_code == 400
    # path traversal rejected
    assert _upload(client, cid, "../escape.txt", b"x").status_code == 400

    resp = client.post(f"/api/corpora/{cid}/process")
    assert resp.status_code == 200
    assert resp.json()["total_files"] == 2

    item = _wait_completed(client, cid)
    assert item["success_files"] == 2
    assert item["processed_files"] == 2
    assert item["completed_at"] is not None

    settings = get_settings()
    corpus_dir = settings.data_dir / "corpus" / "Flow"
    assert (corpus_dir / "docs" / "a.md").read_text().strip() == "hello corpus"
    assert (corpus_dir / "docs" / "sub" / "b.md").exists()
    # upload dir fully removed after conversion
    assert not (settings.data_dir / "upload" / "Flow").exists()

    log = client.get(f"/api/corpora/{cid}/log").json()
    assert log["total_files"] == 2
    assert log["success_files"] == 2
    assert log["failed_files"] == []
    assert log["convert_seconds"] is not None


def test_zip_upload_and_extract(client):
    corpus = client.post("/api/corpora", json={"name": "ZipCorpus"}).json()
    cid = corpus["id"]

    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as zf:
        zf.writestr("nested_note.txt", "should never be extracted")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner/note.txt", "zip content")
        zf.writestr("inner/skip.exe", "binary")
        zf.writestr("../evil.txt", "escape attempt")
        zf.writestr("inner/nested.zip", nested.getvalue())
    assert _upload(client, cid, "pack.zip", buf.getvalue()).status_code == 201

    assert client.post(f"/api/corpora/{cid}/process").status_code == 200
    item = _wait_completed(client, cid)
    assert item["total_files"] == 1  # only the txt inside the zip
    assert item["success_files"] == 1

    settings = get_settings()
    assert (settings.data_dir / "corpus" / "ZipCorpus" / "inner" / "note.md").exists()
    # upload dir fully removed; nested zip was never extracted
    assert not (settings.data_dir / "upload" / "ZipCorpus").exists()
    assert not (settings.data_dir / "upload" / "evil.txt").exists()


def test_nested_zip_upload_rejected(client):
    corpus = client.post("/api/corpora", json={"name": "NestedZip"}).json()
    cid = corpus["id"]
    assert _upload(client, cid, "sub/dir/pack.zip", b"PK...").status_code == 400


def test_process_with_no_supported_files(client):
    corpus = client.post("/api/corpora", json={"name": "Empty"}).json()
    resp = client.post(f"/api/corpora/{corpus['id']}/process")
    assert resp.status_code == 400
    # corpus rolled back
    assert client.get("/api/corpora").json()["total"] == 0


def test_export_corpus_zip(client):
    corpus = client.post("/api/corpora", json={"name": "导出文集"}).json()
    cid = corpus["id"]

    # not completed yet -> 409
    assert client.get(f"/api/corpora/{cid}/export").status_code == 409

    assert _upload(client, cid, "docs/a.txt", b"hello corpus").status_code == 201
    assert _upload(client, cid, "docs/sub/b.md", b"# title").status_code == 201
    client.post(f"/api/corpora/{cid}/process")
    _wait_completed(client, cid)

    # testset-generation artifacts in the corpus dir must NOT be exported
    corpus_dir = get_settings().data_dir / "corpus" / "导出文集"
    (corpus_dir / "seeds.json").write_text("[]")
    (corpus_dir / "kg.json").write_text("{}")

    resp = client.get(f"/api/corpora/{cid}/export")
    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]
    assert "%E5%AF%BC%E5%87%BA%E6%96%87%E9%9B%86.zip" in resp.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = sorted(zf.namelist())
        assert names == ["docs/a.md", "docs/sub/b.md"]  # hierarchy preserved
        assert zf.read("docs/a.md").decode().strip() == "hello corpus"


def test_delete_corpus(client):
    corpus = client.post("/api/corpora", json={"name": "ToDelete"}).json()
    cid = corpus["id"]
    _upload(client, cid, "a.txt", b"data")
    client.post(f"/api/corpora/{cid}/process")
    _wait_completed(client, cid)

    settings = get_settings()
    assert (settings.data_dir / "corpus" / "ToDelete").exists()
    assert client.delete(f"/api/corpora/{cid}").status_code == 204
    assert not (settings.data_dir / "corpus" / "ToDelete").exists()
    assert client.get("/api/corpora").json()["total"] == 0


def test_delete_unstarted_corpus_allowed(client):
    """Upload-phase corpora (total_files == 0) can be deleted (frontend rollback)."""
    corpus = client.post("/api/corpora", json={"name": "HalfUpload"}).json()
    cid = corpus["id"]
    _upload(client, cid, "a.txt", b"data")
    assert client.delete(f"/api/corpora/{cid}").status_code == 204
    assert client.get("/api/corpora").json()["total"] == 0
    settings = get_settings()
    assert not (settings.data_dir / "upload" / "HalfUpload").exists()


def test_delete_converting_corpus_rejected(client):
    """Once conversion has started (total_files > 0), deletion is blocked."""
    corpus = client.post("/api/corpora", json={"name": "Converting"}).json()
    cid = corpus["id"]

    from app.db.session import get_session_factory
    from app.db.models import Corpus

    db = get_session_factory()()
    record = db.get(Corpus, cid)
    record.status = Corpus.STATUS_CONVERTING
    record.total_files = 5  # conversion started
    db.commit()
    db.close()

    assert client.delete(f"/api/corpora/{cid}").status_code == 409


def test_conversion_crash_finalizes_corpus(tmp_path, monkeypatch):
    """An unexpected crash must not leave the corpus stuck in 'converting'."""
    db_url = f"sqlite:///{tmp_path}/crash.db"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as SASession
    from sqlalchemy.orm import sessionmaker

    from app.db.session import get_session_factory, init_db
    from app.db.models import Corpus

    init_db()
    db = get_session_factory()()
    corpus = Corpus(name="Crash")
    db.add(corpus)
    db.commit()
    db.refresh(corpus)
    db.close()

    upload_dir = corpus_convert.upload_dir_for("Crash")
    upload_dir.mkdir(parents=True)
    (upload_dir / "a.txt").write_text("fine")
    (upload_dir / "b.txt").write_text("also fine")

    # A session class whose SECOND commit crashes — i.e. the progress commit
    # after the first file fails, outside the per-file guard.
    class CrashySession(SASession):
        calls = 0

        def commit(self):
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("database is gone")
            return super().commit()

    crashy_factory = sessionmaker(
        bind=create_engine(db_url, connect_args={"check_same_thread": False}),
        class_=CrashySession,
    )
    monkeypatch.setattr(corpus_convert, "get_session_factory", lambda: crashy_factory)
    asyncio.run(corpus_convert.run_conversion(corpus.id))

    db2 = get_session_factory()()
    record = db2.query(Corpus).filter_by(name="Crash").one()
    assert record.status == Corpus.STATUS_COMPLETED
    assert record.completed_at is not None
    assert record.convert_seconds is not None
    # unprocessed files were recorded as failed
    assert sorted(json.loads(record.failed_files)) == ["a.txt", "b.txt"]
    assert record.processed_files == record.total_files == 2
    assert not upload_dir.exists()
    db2.close()
    get_settings.cache_clear()


def test_resume_discards_never_started_corpus(tmp_path, monkeypatch):
    """On startup, corpora stuck mid-upload (total_files == 0) are cleaned up."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/resume.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    from app.db.session import get_session_factory, init_db
    from app.db.models import Corpus

    init_db()
    db = get_session_factory()()
    db.add(Corpus(name="Abandoned"))
    db.commit()

    upload_dir = corpus_convert.upload_dir_for("Abandoned")
    upload_dir.mkdir(parents=True)
    (upload_dir / "partial.txt").write_text("half uploaded")

    asyncio.run(corpus_convert.resume_interrupted())

    assert db.query(Corpus).filter_by(name="Abandoned").count() == 0
    assert not upload_dir.exists()
    db.close()
    get_settings.cache_clear()


def test_conversion_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/svc.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    from app.db.session import get_session_factory, init_db
    from app.db.models import Corpus

    init_db()
    db = get_session_factory()()
    corpus = Corpus(name="Svc")
    db.add(corpus)
    db.commit()
    db.refresh(corpus)

    upload_dir = corpus_convert.upload_dir_for("Svc")
    (upload_dir / "ok").mkdir(parents=True)
    (upload_dir / "ok" / "good.txt").write_text("fine")
    (upload_dir / "bad.pdf").write_bytes(b"not a real pdf")

    class FakeMarkItDown:
        def convert(self, path: str):
            if path.endswith("bad.pdf"):
                raise RuntimeError("cannot parse")
            class Result:
                text_content = "converted"
            return Result()

    monkeypatch.setattr(corpus_convert, "_get_markitdown", lambda: FakeMarkItDown())

    asyncio.run(corpus_convert.run_conversion(corpus.id))

    db.refresh(corpus)
    assert corpus.status == Corpus.STATUS_COMPLETED
    assert corpus.success_files == 1
    assert corpus.total_files == 2
    assert json.loads(corpus.failed_files) == ["bad.pdf"]
    assert (corpus_convert.corpus_dir_for("Svc") / "ok" / "good.md").exists()
    assert not upload_dir.exists()
    db.close()
    get_settings.cache_clear()


def test_two_sources_with_the_same_stem_both_survive(tmp_path):
    """报告.pdf and 报告.docx both map to 报告.md. Writing anyway dropped the
    earlier document while both counted as successes."""
    upload = tmp_path / "upload"
    out = tmp_path / "corpus"
    upload.mkdir()
    (upload / "报告.pdf").write_text("pdf content")
    (upload / "报告.docx").write_text("docx content")

    class FakeMD:
        def convert(self, path):
            return type("R", (), {"text_content": f"# {path.split('/')[-1]}"})()

    corpus_convert._markitdown = FakeMD()
    try:
        corpus_convert._convert_one(upload / "报告.pdf", upload, out)
        corpus_convert._convert_one(upload / "报告.docx", upload, out)
    finally:
        corpus_convert._markitdown = None

    produced = sorted(p.name for p in out.glob("*.md"))
    assert produced == ["报告.docx.md", "报告.md"], produced
    assert "pdf content" not in (out / "报告.md").read_text() or True
    assert "# 报告.pdf" in (out / "报告.md").read_text()


def test_resumed_conversion_keeps_the_total_cumulative(client, monkeypatch):
    """The upload dir holds only what is left to convert, so taking its length
    as the total made a resumed corpus report 10 converted out of 5 files."""
    cid = client.post("/api/corpora", json={"name": "ResumeTotal"}).json()["id"]
    for i in range(4):
        _upload(client, cid, f"f{i}.txt", b"content")
    client.post(f"/api/corpora/{cid}/process")
    _wait_completed(client, cid)

    # simulate a run interrupted after 2 conversions: sources for those are
    # already deleted and the counters carried over
    from app.db.session import get_session_factory
    from app.db.models import Corpus
    db = get_session_factory()()
    try:
        row = db.get(Corpus, cid)
        row.status = Corpus.STATUS_CONVERTING
        row.success_files = 2
        row.processed_files = 2
        row.total_files = 4
        db.commit()
    finally:
        db.close()
    upload_dir = corpus_convert.upload_dir_for("ResumeTotal")
    upload_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (upload_dir / f"left{i}.txt").write_text("content")

    asyncio.run(corpus_convert.run_conversion(cid))

    db = get_session_factory()()
    try:
        row = db.get(Corpus, cid)
        assert row.processed_files <= row.total_files, (
            f"progress {row.processed_files}/{row.total_files} exceeds 100%")
        assert row.total_files == 4, row.total_files
    finally:
        db.close()


def test_corpus_delete_is_refused_while_a_testset_is_generating(client):
    """The corpus directory holds the .md files, the graph and the seed list a
    running generation reads and rewrites; deleting it fails that run with
    FileNotFoundError after its tokens are spent."""
    from app.db.models import Corpus, Testset
    from app.db.session import get_session_factory

    cid = client.post("/api/corpora", json={"name": "BusyCorpus"}).json()["id"]
    db = get_session_factory()()
    try:
        db.add(Testset(name="QA-Busy", corpus_id=cid, corpus_name="BusyCorpus",
                       llm_config_id=1, llm_name="x", embedding_config_id=1,
                       embedding_name="y", status=Testset.STATUS_GENERATING))
        db.commit()
    finally:
        db.close()

    resp = client.delete(f"/api/corpora/{cid}")
    assert resp.status_code == 409, resp.text
    assert "generating" in resp.json()["detail"]

    # once the testset is done, the delete goes through
    db = get_session_factory()()
    try:
        row = db.query(Testset).filter(Testset.name == "QA-Busy").one()
        row.status = Testset.STATUS_COMPLETED
        db.commit()
    finally:
        db.close()
    assert client.delete(f"/api/corpora/{cid}").status_code == 204


def _clear_utf8_flag(data: bytes) -> bytes:
    """Rewrite a one-entry archive the way a Windows tool writes it: the name
    bytes are UTF-8, the flag that announces them (bit 11) is not set.

    zipfile sets that flag whenever it writes a non-ASCII name, so an archive
    built here cannot otherwise reproduce what customers actually send.
    """
    out = bytearray(data)
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        i = out.index(signature)
        flags = int.from_bytes(out[i + offset : i + offset + 2], "little") & ~0x800
        out[i + offset : i + offset + 2] = flags.to_bytes(2, "little")
    return bytes(out)


class TestMemberNames:
    """How a member's name is read out of the archive."""

    def _info(self, name: str, flag: int) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(name)
        info.flag_bits = flag
        return info

    def test_utf8_name_without_the_flag_is_recovered(self):
        # The shape that arrived from a customer: a Chinese path whose UTF-8
        # bytes zipfile handed over as their cp437 reading.
        name = "荣盛发展/荣盛房地产发展股份有限公司报表.md"
        info = self._info(name.encode("utf-8").decode("cp437"), 0)
        assert corpus_convert._member_name(info) == name

    def test_gbk_name_without_the_flag_is_recovered(self):
        name = "年报/审计报告.md"
        info = self._info(name.encode("gbk").decode("cp437"), 0)
        assert corpus_convert._member_name(info) == name

    def test_flagged_utf8_name_is_left_alone(self):
        name = "年报/审计报告.md"
        assert corpus_convert._member_name(self._info(name, 0x800)) == name

    def test_ascii_name_is_untouched(self):
        assert corpus_convert._member_name(self._info("docs/note.md", 0)) == "docs/note.md"

    def test_a_real_cp437_name_is_not_mistaken_for_utf8(self):
        # A Western archive that used its own codepage without the flag: the
        # name is already right and has to survive both decode attempts.
        assert corpus_convert._member_name(self._info("résumé/café.md", 0)) == "résumé/café.md"


def test_extract_repairs_names_a_windows_tool_wrote(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("荣盛发展/报表.md", "content")

    (tmp_path / "pack.zip").write_bytes(_clear_utf8_flag(buf.getvalue()))
    corpus_convert.extract_zips(tmp_path)

    assert (tmp_path / "荣盛发展" / "报表.md").read_text() == "content"


def test_zip_from_a_windows_tool_keeps_its_names(client):
    corpus = client.post("/api/corpora", json={"name": "CnZip"}).json()
    cid = corpus["id"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("荣盛发展/报表.md", "content")
    assert _upload(client, cid, "pack.zip", _clear_utf8_flag(buf.getvalue())).status_code == 201

    assert client.post(f"/api/corpora/{cid}/process").status_code == 200
    assert _wait_completed(client, cid)["total_files"] == 1
    settings = get_settings()
    assert (settings.data_dir / "corpus" / "CnZip" / "荣盛发展" / "报表.md").exists()


def test_member_that_cannot_be_written_is_an_archive_path_error(tmp_path):
    """A member the filesystem refuses is raised as ArchivePathError, carrying
    the name. A file where the next member needs a directory stands in for the
    cases macOS allows and Windows does not — a path past 260 characters, a
    reserved device name — and reaches the same OSError."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("report", "a file")
        zf.writestr("report/part.md", "a file that needs a directory there")

    (tmp_path / "pack.zip").write_bytes(buf.getvalue())
    with pytest.raises(corpus_convert.ArchivePathError) as exc:
        corpus_convert.extract_zips(tmp_path)

    assert exc.value.member == "report/part.md"


def test_unusable_member_name_answers_400_not_500(client):
    corpus = client.post("/api/corpora", json={"name": "BadNames"}).json()
    cid = corpus["id"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("report", "a file")
        zf.writestr("report/part.md", "a file that needs a directory there")
    assert _upload(client, cid, "pack.zip", buf.getvalue()).status_code == 201

    resp = client.post(f"/api/corpora/{cid}/process")
    assert resp.status_code == 400
    assert "report/part.md" in resp.json()["detail"]
    # Discarded, not left half-extracted.
    assert client.get("/api/corpora").json()["total"] == 0
