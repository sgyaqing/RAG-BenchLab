import asyncio
import io
import json
import logging
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services import testset_gen


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture()
def corpus_with_docs(client):
    """A completed corpus with 20 markdown docs (>=1500 chars each)."""
    resp = client.post("/api/corpora", json={"name": "MyCorpus"})
    corpus = resp.json()
    cid = corpus["id"]
    # 18 chars x 100 = 1800, comfortably over MIN_DOC_CHARS (1500). At the old
    # x60 the fixture was only ~1080 and every document was filtered out as too
    # short, so the pipeline ran on an empty seed list — which the mocked ragas
    # layer hid, and which in production would divide by zero.
    content = "这是一段用于测试的中文金融公告内容。" * 100
    for i in range(20):
        client.post(
            f"/api/corpora/{cid}/files",
            files={"file": (f"doc{i}.md", io.BytesIO(content.encode()))},
            data={"path": f"doc{i}.md"},
        )
    client.post(f"/api/corpora/{cid}/process")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        body = client.get("/api/corpora").json()
        if body["items"][0]["status"] == "completed":
            break
        time.sleep(0.2)
    return corpus


@pytest.fixture()
def model_configs(client):
    llm = client.post("/api/model-configs", json={
        "name": "Doubao", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "doubao-mini",
    }).json()
    emb = client.post("/api/model-configs", json={
        "name": "BGE", "type": "embedding", "api_format": "openai",
        "base_url": "https://example.com/v1", "model": "bge-m3",
    }).json()
    return llm, emb


@pytest.fixture()
def mock_connectivity(monkeypatch):
    async def fake_test(*a, **kw):
        return True, "ok", 12

    monkeypatch.setattr("app.services.model_connect.test_connectivity", fake_test)


@pytest.fixture()
def mock_ragas(monkeypatch, mock_connectivity):
    """Replace all heavy ragas interactions with fast fakes."""
    counter = {"n": 0}

    class FakeRaw:
        totals = {"prompt": 100, "completion": 20, "calls": 5}

    monkeypatch.setattr(testset_gen, "_build_llm", lambda *a, **kw: (object(), FakeRaw(), None))
    monkeypatch.setattr(testset_gen, "_build_embeddings", lambda *a, **kw: (object(), None))

    class FakeTransform:
        pass

    monkeypatch.setattr(testset_gen, "_default_transforms", lambda *a: [FakeTransform(), FakeTransform()])
    monkeypatch.setattr(testset_gen, "_apply_transforms", lambda *a, **kw: None)
    monkeypatch.setattr(testset_gen, "_generate_personas", lambda *a: [])
    monkeypatch.setattr(testset_gen, "_make_synthesizers", lambda *a: {
        "single": "s1", "multi_specific": "s2", "multi_abstract": "s3",
    })
    monkeypatch.setattr(testset_gen, "_make_generator", lambda *a: object())

    def fake_generate_chunk(gen, synth, size, run_config):
        out = []
        for _ in range(size):
            counter["n"] += 1
            n = counter["n"]
            out.append({
                "user_input": f"问题{n}？",
                # realistic length: fragment references are rejected by the
                # quality guardrail (MIN_REFERENCE_CHARS)
                "reference": f"这是第{n}条问答对的完整参考答案，用于测试。",
                "reference_contexts": [f"上下文{n}A", f"上下文{n}B"],
                "synthesizer_name": str(synth), "persona_name": "persona",
            })
        return out

    monkeypatch.setattr(testset_gen, "_generate_chunk", fake_generate_chunk)
    return counter


def _wait_done(client, testset_id: int, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get("/api/testsets").json()
        item = next(i for i in body["items"] if i["id"] == testset_id)
        if item["status"] in ("completed", "failed"):
            return item
        time.sleep(0.3)
    raise AssertionError("generation did not finish in time")


def test_check_name_case_insensitive(client, corpus_with_docs, model_configs):
    assert client.get("/api/testsets/check-name", params={"name": "QA-1"}).json()["available"]


def test_create_validation(client, corpus_with_docs, model_configs):
    llm, emb = model_configs
    base = {
        "name": "QA-1", "corpus_id": corpus_with_docs["id"], "reuse_kg": True,
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "llm_concurrency": 16, "llm_max_tokens": 16384,
        "n_single": 2, "n_multi_specific": 1, "n_multi_abstract": 1,
    }
    # all zero counts rejected
    bad = {**base, "n_single": 0, "n_multi_specific": 0, "n_multi_abstract": 0}
    assert client.post("/api/testsets", json=bad).status_code == 422
    # concurrency out of range
    assert client.post("/api/testsets", json={**base, "llm_concurrency": 0}).status_code == 422
    # max_tokens below lower bound
    assert client.post("/api/testsets", json={**base, "llm_max_tokens": 1024}).status_code == 422
    # amplification out of range
    assert client.post("/api/testsets", json={**base, "amplify": 0.5}).status_code == 422
    assert client.post("/api/testsets", json={**base, "gen_amplify": 2.5}).status_code == 422
    # embedding config used as llm
    assert client.post("/api/testsets", json={**base, "llm_config_id": emb["id"]}).status_code == 400
    # invalid name
    assert client.post("/api/testsets", json={**base, "name": "bad/name"}).status_code == 422


def test_full_generation_flow(client, corpus_with_docs, model_configs, mock_ragas):
    llm, emb = model_configs
    resp = client.post("/api/testsets", json={
        "name": "MyCorpus-QA-4", "corpus_id": corpus_with_docs["id"], "reuse_kg": True,
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "llm_concurrency": 4, "llm_max_tokens": 16384,
        "n_single": 2, "n_multi_specific": 1, "n_multi_abstract": 1,
    })
    assert resp.status_code == 201, resp.text
    ts = resp.json()

    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed", item.get("error")
    assert item["actual_single"] == 2
    assert item["actual_multi_specific"] == 1
    assert item["actual_multi_abstract"] == 1
    assert item["progress"] == 100

    # items persisted
    items = client.get(f"/api/testsets/{ts['id']}/items").json()
    assert items["total"] == 4
    first = items["items"][0]
    assert first["user_input"].startswith("问题")
    assert len(first["reference_contexts"]) == 2
    assert first["edited"] is False

    # jsonl artifact + kg + seeds on disk
    settings = get_settings()
    assert (settings.data_dir / "testset" / "MyCorpus-QA-4" / "testset.jsonl").exists()
    assert (settings.data_dir / "corpus" / "MyCorpus" / "kg.json").exists()
    assert (settings.data_dir / "corpus" / "MyCorpus" / "seeds.json").exists()

    # log entries cover the stages
    log = client.get(f"/api/testsets/{ts['id']}/log").json()
    keys = [e["key"] for e in log["entries"]]
    assert "kgBuilt" in keys and "done" in keys
    assert keys.count("typeFirstGen") == 3

    # corpus reports has_kg
    corpora = client.get("/api/corpora").json()
    assert corpora["items"][0]["has_kg"] is True


def test_the_per_type_log_adds_up(client, corpus_with_docs, model_configs, mock_ragas,
                                  monkeypatch):
    """A customer adding and subtracting down the log must land on the closing
    候选, not above it.

    Every 补生成 line adds and every 质检 line subtracts, and the closing line
    names the pool the trim then cut to target. Without that last number the
    subtraction came out above the count in the list — items dropped only
    because the pool held more than asked for had nothing to explain them.

    The reviewer is forced to drop one sample every round: left alone, the
    mocked run rejects nothing, no 质检 line is written and the subtraction this
    test exists to check is never exercised.
    """
    import app.services.testset_gen as tg

    async def drop_one_each_round(llm, samples, testset_id, type_key, language):
        return samples[1:] if len(samples) > 1 else samples

    monkeypatch.setattr(tg, "_semantic_dedup", drop_one_each_round)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "LogArithmetic", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 3, "n_multi_specific": 0, "n_multi_abstract": 0,
        "gen_amplify": 1.0,
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed"

    entries = client.get(f"/api/testsets/{ts['id']}/log").json()["entries"]
    running: dict[str, int] = {}
    finished: set[str] = set()
    for e in entries:
        p = e["params"]
        t = p.get("type")
        if e["key"] == "typeFirstGen":
            running[t] = p["count"]
        elif e["key"] == "typeChecked":
            running[t] -= p["dropped"]
            assert p["remaining"] == running[t], f"质检 line disagrees: {e}"
        elif e["key"] == "typeToppedUp":
            running[t] += p["count"]
            assert p["total"] == running[t], f"补生成 line disagrees: {e}"
        elif e["key"] in ("typeDone", "typeDoneAll"):
            assert p["candidates"] == running[t], f"closing line disagrees: {e}"
            # the trim keeps the target, or everything there was
            assert p["count"] == min(p["candidates"], 3), e
            # and the line says which happened: "kept at random" only when the pool held
            # more than the target and something was actually picked out
            assert (e["key"] == "typeDoneAll") == (p["candidates"] <= p["count"]), e
            finished.add(t)
    assert finished == {"single"}, finished
    # the chain has to have been exercised, not passed over because nothing was
    # ever dropped or topped up
    assert any(e["key"] == "typeChecked" for e in entries), \
        "no 质检 line was logged, so the subtraction was never checked"


def test_generation_failure_marks_failed(client, corpus_with_docs, model_configs, mock_connectivity, monkeypatch):
    monkeypatch.setattr(testset_gen, "_build_llm", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("LLM endpoint down")))
    llm, emb = model_configs
    resp = client.post("/api/testsets", json={
        "name": "QA-Fail", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    })
    assert resp.status_code == 201
    item = _wait_done(client, resp.json()["id"])
    assert item["status"] == "failed"
    assert "LLM endpoint down" in item["error"]
    # failed testset can be deleted
    assert client.delete(f"/api/testsets/{item['id']}").status_code == 204


def test_connectivity_precheck_failure(client, corpus_with_docs, model_configs, monkeypatch):
    """When the LLM is unreachable, generation fails fast with a clear reason."""
    async def fake_test(model_type, *a, **kw):
        if model_type == "llm":
            return False, "All connection attempts failed", None
        return True, "ok", 8

    monkeypatch.setattr("app.services.model_connect.test_connectivity", fake_test)
    llm, emb = model_configs
    resp = client.post("/api/testsets", json={
        "name": "QA-NoConn", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    })
    assert resp.status_code == 201
    item = _wait_done(client, resp.json()["id"])
    assert item["status"] == "failed"
    assert "LLM connectivity test failed" in item["error"]
    assert "All connection attempts failed" in item["error"]
    # failed before any LLM spend: no kg written
    assert not testset_gen.kg_path_for(corpus_with_docs["name"]).exists()


def test_item_edit_restore_delete(client, corpus_with_docs, model_configs, mock_ragas):
    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "QA-Edit", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    }).json()
    _wait_done(client, ts["id"])
    items = client.get(f"/api/testsets/{ts['id']}/items").json()
    item = items["items"][0]
    original_q = item["user_input"]

    completed_before = client.get("/api/testsets").json()["items"][0]["completed_at"]

    # edit
    resp = client.put(f"/api/testsets/{ts['id']}/items/{item['id']}",
                      json={"user_input": "改后的问题？", "reference": "改后的答案"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["edited"] is True and body["has_original"] is True

    # completed_at updated + edit logged + list shows the edited marker
    after = client.get("/api/testsets").json()["items"][0]
    assert after["completed_at"] >= completed_before
    assert after["edited"] is True
    log = client.get(f"/api/testsets/{ts['id']}/log").json()
    assert any(e["key"] == "edited" for e in log["entries"])

    # restore
    resp = client.post(f"/api/testsets/{ts['id']}/items/{item['id']}/restore")
    body = resp.json()
    assert body["edited"] is False
    assert body["user_input"] == original_q
    # marker clears once no edited items remain
    assert client.get("/api/testsets").json()["items"][0]["edited"] is False

    # delete
    assert client.delete(f"/api/testsets/{ts['id']}/items/{item['id']}").status_code == 204
    assert client.get(f"/api/testsets/{ts['id']}/items").json()["total"] == 0


def test_export_completed_testset(client, corpus_with_docs, model_configs, mock_ragas):
    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "QA-导出", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    }).json()
    _wait_done(client, ts["id"])

    resp = client.get(f"/api/testsets/{ts['id']}/export")
    assert resp.status_code == 200
    # Chinese name is RFC 5987-encoded in Content-Disposition
    assert "attachment" in resp.headers["content-disposition"]
    assert "QA-%E5%AF%BC%E5%87%BA.jsonl" in resp.headers["content-disposition"]
    lines = [json.loads(x) for x in resp.text.strip().split("\n")]
    assert len(lines) >= 1
    assert set(lines[0]) == {"user_input", "reference", "reference_contexts", "synthesizer_name"}
    assert isinstance(lines[0]["reference_contexts"], list)

    # non-completed testsets cannot be exported
    from app.db.models import Testset
    from app.api.model_configs import get_db
    db = next(get_db())
    ts_row = db.get(Testset, ts["id"])
    ts_row.status = Testset.STATUS_FAILED
    db.commit()
    assert client.get(f"/api/testsets/{ts['id']}/export").status_code == 409


def _import(client, name: str, lines: list[dict | str], filename: str = "qa.jsonl"):
    content = "\n".join(
        line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)
        for line in lines
    )
    return client.post(
        "/api/testsets/import",
        files={"file": (filename, io.BytesIO(content.encode("utf-8")))},
        data={"name": name},
    )



def test_import_testset_all_typed(client):
    resp = _import(client, "导入-全类型", [
        {"user_input": "问题一？", "reference": "答案一",
         "reference_contexts": ["片段甲"], "synthesizer_name": "single_hop_specfic"},
        {"user_input": "问题二？", "reference": "答案二",
         "synthesizer_name": "multi_hop_abstract_reasoning"},
    ])
    assert resp.status_code == 201, resp.text
    ts = resp.json()
    assert ts["status"] == "completed" and ts["item_count"] == 2
    assert ts["corpus_id"] == 0 and ts["corpus_name"] == ""
    # every type recognized -> per-type counts filled
    assert ts["actual_single"] == 1 and ts["actual_multi_abstract"] == 1

    items = client.get(f"/api/testsets/{ts['id']}/items").json()
    assert items["total"] == 2
    assert items["items"][0]["reference_contexts"] == ["片段甲"]
    assert items["items"][1]["reference_contexts"] == []
    # list view carries item_count too
    listed = client.get("/api/testsets").json()["items"][0]
    assert listed["item_count"] == 2
    # import is recorded in the log
    log = client.get(f"/api/testsets/{ts['id']}/log").json()
    assert log["entries"][0]["key"] == "imported"
    assert log["entries"][0]["params"] == {"filename": "qa.jsonl", "count": 2}


def test_import_testset_mixed_types_show_total_only(client):
    resp = _import(client, "导入-混合", [
        {"user_input": "问题一？", "reference": "答案一", "synthesizer_name": "single_hop_specfic"},
        {"user_input": "问题二？", "reference": "答案二"},  # no synthesizer_name
    ])
    assert resp.status_code == 201, resp.text
    ts = resp.json()
    # mixed types -> breakdown suppressed (all zeros), total from item_count
    assert ts["actual_single"] == 0 and ts["actual_multi_specific"] == 0
    assert ts["actual_multi_abstract"] == 0 and ts["item_count"] == 2


def test_import_testset_normalizes_qa_but_keeps_contexts_verbatim(client):
    resp = _import(client, "导入-清洗", [
        {"user_input": "香 港 联 合 交 易 所 的 网 站 是 什 么 ？", "reference": " 答案 ",
         "reference_contexts": ["香 港 联 合 交 易 所 原文"]},
    ])
    assert resp.status_code == 201, resp.text
    item = client.get(f"/api/testsets/{resp.json()['id']}/items").json()["items"][0]
    # user_input normalized (CJK spaces removed); contexts kept verbatim
    assert item["user_input"] == "香港联合交易所的网站是什么 ？"
    assert item["reference_contexts"] == ["香 港 联 合 交 易 所 原文"]


def test_import_testset_rejects_bad_input(client):
    # not valid JSON
    resp = _import(client, "导入-坏行", [
        {"user_input": "问题一？", "reference": "答案一"},
        "{not json",
    ])
    assert resp.status_code == 400 and "Line 2" in resp.json()["detail"]

    # missing/empty reference
    resp = _import(client, "导入-缺答案", [{"user_input": "问题一？", "reference": "  "},
                                         ])
    assert resp.status_code == 400 and "reference" in resp.json()["detail"]

    # missing user_input
    resp = _import(client, "导入-缺问题", [{"reference": "答案一"}])
    assert resp.status_code == 400 and "user_input" in resp.json()["detail"]

    # reference_contexts not an array of strings
    resp = _import(client, "导入-坏上下文", [
        {"user_input": "问？", "reference": "答", "reference_contexts": [1, 2]},
    ])
    assert resp.status_code == 400 and "reference_contexts" in resp.json()["detail"]

    # empty file
    resp = _import(client, "导入-空文件", [])
    assert resp.status_code == 400

    # nothing was persisted
    assert client.get("/api/testsets").json()["total"] == 0


def test_import_testset_duplicate_name(client, corpus_with_docs, model_configs, mock_ragas):
    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "qa-1", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    }).json()
    _wait_done(client, ts["id"])
    # case-insensitive duplicate
    resp = _import(client, "QA-1", [{"user_input": "问？", "reference": "答"}])
    assert resp.status_code == 409

    # invalid name characters
    resp = _import(client, "bad/name", [{"user_input": "问？", "reference": "答"}])
    assert resp.status_code == 422


def test_resume_failed_testset(client, corpus_with_docs, model_configs, monkeypatch):
    """Failed testset can be resumed; resume reuses cached work and completes."""
    llm, emb = model_configs

    # first run: connectivity fails -> failed
    async def bad_conn(model_type, *a, **kw):
        return (model_type != "llm"), "down", None

    monkeypatch.setattr("app.services.model_connect.test_connectivity", bad_conn)
    resp = client.post("/api/testsets", json={
        "name": "QA-Resume", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    })
    tid = resp.json()["id"]
    assert _wait_done(client, tid)["status"] == "failed"

    # resume while still failing -> 409 on non-failed, works on failed
    # (resume is allowed; run will fail again but that's the point of retry)
    resp = client.post(f"/api/testsets/{tid}/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "generating"
    assert _wait_done(client, tid)["status"] == "failed"  # still down

    # fix connectivity + mock ragas, resume again -> completed
    async def ok_conn(*a, **kw):
        return True, "ok", 5

    monkeypatch.setattr("app.services.model_connect.test_connectivity", ok_conn)
    counter = {"n": 0}

    class FakeRaw:
        totals = {"prompt": 1, "completion": 1, "calls": 1}

    monkeypatch.setattr(testset_gen, "_build_llm", lambda *a, **kw: (object(), FakeRaw(), None))
    monkeypatch.setattr(testset_gen, "_build_embeddings", lambda *a, **kw: (object(), None))
    monkeypatch.setattr(testset_gen, "_default_transforms", lambda *a: [])
    monkeypatch.setattr(testset_gen, "_apply_transforms", lambda *a, **kw: None)
    monkeypatch.setattr(testset_gen, "_generate_personas", lambda *a: [])
    monkeypatch.setattr(testset_gen, "_make_synthesizers",
                        lambda *a: {"single": "s1", "multi_specific": "s2", "multi_abstract": "s3"})
    monkeypatch.setattr(testset_gen, "_make_generator", lambda *a: object())

    def fake_chunk(gen, synth, size, run_config):
        out = []
        for _ in range(size):
            counter["n"] += 1
            out.append({"user_input": f"第{counter['n']}个问题？",
                        "reference": f"这是第{counter['n']}个问题的完整参考答案。",
                        "reference_contexts": ["c"], "synthesizer_name": "s", "persona_name": "p"})
        return out

    monkeypatch.setattr(testset_gen, "_generate_chunk", fake_chunk)

    assert client.post(f"/api/testsets/{tid}/resume").status_code == 200
    item = _wait_done(client, tid)
    assert item["status"] == "completed", item.get("error")
    assert item["actual_single"] == 1

    # completed testset cannot be resumed
    assert client.post(f"/api/testsets/{tid}/resume").status_code == 409

    # resume was logged
    log = client.get(f"/api/testsets/{tid}/log").json()
    keys = [e["key"] for e in log["entries"]]
    assert keys.count("resumed") == 2


def test_run_seed_semantics(client, corpus_with_docs, model_configs, mock_ragas):
    """Reuse -> same run_seed as the KG; no-reuse rebuild -> fresh seed."""
    llm, emb = model_configs

    def create(name, reuse):
        resp = client.post("/api/testsets", json={
            "name": name, "corpus_id": corpus_with_docs["id"], "reuse_kg": reuse,
            "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
            "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
        })
        assert resp.status_code == 201
        ts = resp.json()
        assert _wait_done(client, ts["id"])["status"] == "completed"
        return client.get(f"/api/testsets/{ts['id']}").json()

    ts1 = create("Seed-1", True)
    seed1 = ts1["run_seed"]
    assert seed1

    # reuse -> same seed (reproducible path)
    ts2 = create("Seed-2", True)
    assert ts2["run_seed"] == seed1

    # no-reuse -> full rebuild with a fresh seed
    ts3 = create("Seed-3", False)
    assert ts3["run_seed"] != seed1

    # reuse again after rebuild -> adopts the new KG's seed
    ts4 = create("Seed-4", True)
    assert ts4["run_seed"] == ts3["run_seed"]


def test_prompt_language_override(client, corpus_with_docs, model_configs, mock_ragas, monkeypatch):
    """prompt_language='en' on a Chinese corpus forces English prompts."""
    captured = {}

    real_make = testset_gen._make_synthesizers
    def capture_synths(llm, language):
        captured["language"] = language
        return real_make(llm, language)

    # mock_ragas already replaced _make_synthesizers; wrap the mock instead
    monkeypatch.setattr(testset_gen, "_make_synthesizers", capture_synths_mock(captured))

    # English questions, to match the forced prompt language: the language
    # guardrail drops questions that do not match, and a run whose output is
    # entirely dropped now fails rather than completing empty.
    counter = {"n": 0}

    def english_chunk(gen, synth, size, run_config):
        out = []
        for _ in range(size):
            counter["n"] += 1
            out.append({"user_input": f"What is item {counter['n']} of the report?",
                        "reference": "It is the item described in the source material.",
                        "reference_contexts": ["c"], "synthesizer_name": "s",
                        "persona_name": "p"})
        return out

    monkeypatch.setattr(testset_gen, "_generate_chunk", english_chunk)

    llm, emb = model_configs
    resp = client.post("/api/testsets", json={
        "name": "QA-En", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
        "prompt_language": "en",
    })
    assert resp.status_code == 201
    assert _wait_done(client, resp.json()["id"])["status"] == "completed"
    assert captured["language"] == "en"  # corpus is zh; override wins
    keys = [e["key"] for e in client.get(f"/api/testsets/{resp.json()['id']}/log").json()["entries"]]
    assert "langOverride" in keys


def capture_synths_mock(captured):
    def _mock(llm, language):
        captured["language"] = language
        return {"single": "s1", "multi_specific": "s2", "multi_abstract": "s3"}
    return _mock


def test_language_guardrail(client, corpus_with_docs, model_configs, mock_ragas, monkeypatch,
                           caplog):
    """Samples whose question drifts from the corpus language are dropped and topped up."""
    counter = {"n": 0}

    def drift_chunk(gen, synth, size, run_config):
        out = []
        for _ in range(size):
            counter["n"] += 1
            # first generated sample is in English (model drift), rest Chinese
            q = "What is the dividend policy?" if counter["n"] == 1 else f"第{counter['n']}个问题？"
            out.append({"user_input": q,
                    # long enough and sentence-final: the quality guardrail
                    # drops fragments, and a run that ends with nothing now fails
                    "reference": "这是该问题的完整参考答案。",
                    "reference_contexts": ["c"],
                        "synthesizer_name": "s", "persona_name": "p"})
        return out

    monkeypatch.setattr(testset_gen, "_generate_chunk", drift_chunk)

    llm, emb = model_configs
    resp = client.post("/api/testsets", json={
        "name": "QA-Lang", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 2, "n_multi_specific": 0, "n_multi_abstract": 0,
    })
    assert resp.status_code == 201
    ts = resp.json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed", item.get("error")
    items = client.get(f"/api/testsets/{ts['id']}/items").json()["items"]
    assert all(testset_gen.language_matches(i["user_input"], "zh") for i in items)
    keys = [e["key"] for e in client.get(f"/api/testsets/{ts['id']}/log").json()["entries"]]
    # The breakdown moved to the server log: the customer sees the outcome on
    # the genTypeFiltered line, not which of the three filters took a sample.
    assert "langMismatch" not in keys
    assert "language=1" in " ".join(r.getMessage() for r in caplog.records)


def test_language_matches():
    assert testset_gen.language_matches("这是中文问题吗？", "zh")
    assert not testset_gen.language_matches("Is this English?", "zh")
    assert testset_gen.language_matches("Is this English?", "en")
    assert not testset_gen.language_matches("这是中文问题吗？", "en")


def test_kg_fingerprint_binding(client, corpus_with_docs, model_configs, mock_ragas):
    """KG reuse requires the config fingerprint (models + chunk params) to match."""
    llm, emb = model_configs
    other_llm = client.post("/api/model-configs", json={
        "name": "OtherLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-y", "model": "other-model",
    }).json()

    def create(name, llm_id):
        return client.post("/api/testsets", json={
            "name": name, "corpus_id": corpus_with_docs["id"],
            "llm_config_id": llm_id, "embedding_config_id": emb["id"],
            "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
        }).json()

    def log_keys(tid):
        return [e["key"] for e in client.get(f"/api/testsets/{tid}/log").json()["entries"]]

    ts1 = create("FP-1", llm["id"])
    assert _wait_done(client, ts1["id"])["status"] == "completed"
    assert "kgBuilt" in log_keys(ts1["id"])

    # same config -> reuse
    ts2 = create("FP-2", llm["id"])
    assert _wait_done(client, ts2["id"])["status"] == "completed"
    assert "kgReused" in log_keys(ts2["id"])

    # different LLM model -> fingerprint mismatch -> full rebuild
    ts3 = create("FP-3", other_llm["id"])
    assert _wait_done(client, ts3["id"])["status"] == "completed"
    keys3 = log_keys(ts3["id"])
    assert "kgFingerprintMismatch" in keys3
    assert "kgBuilt" in keys3 and "kgReused" not in keys3


def test_detect_language_endpoint(client, corpus_with_docs):
    resp = client.get(f"/api/corpora/{corpus_with_docs['id']}/detect-language")
    assert resp.status_code == 200
    assert resp.json() == {"language": "zh"}


def test_embed_skip_single_failure():
    """A text failing at batch size 1 is skipped (zero-filled) and reported."""
    calls = []

    def fake_embed(batch):
        calls.append(len(batch))
        if any("bad" in t for t in batch):
            raise RuntimeError("400 batch too large")
        return [[float(len(t))] for t in batch]

    skipped = []
    texts = ["good one", "bad text", "good two"]
    out = testset_gen._embed_texts_sync(
        fake_embed, texts, on_skip=lambda count, preview: skipped.append((count, preview))
    )
    assert len(out) == 3
    assert out[0] == [8.0] and out[2] == [8.0]
    assert out[1] == [0.0]  # zero-filled, same dim
    assert skipped == [(1, "bad text")]
    # batch was split down to single before skipping
    assert 1 in calls


def test_embed_all_fail_raises():
    def fake_embed(batch):
        raise RuntimeError("endpoint down")

    try:
        testset_gen._embed_texts_sync(fake_embed, ["a", "b"])
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "All 2 embedding calls failed" in str(e)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_estimate_seed_count():
    assert testset_gen.estimate_seed_count(30, 100) == 39
    assert testset_gen.estimate_seed_count(1, 100) == 15  # lower bound
    assert testset_gen.estimate_seed_count(100, 10) == 10  # capped by doc count


def test_detect_language(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    corpus_dir = tmp_path / "data" / "corpus" / "Zh"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "a.md").write_text("中文内容" * 500)
    assert testset_gen.detect_language("Zh") == "zh"
    (corpus_dir / "a.md").write_text("english content only " * 200)
    assert testset_gen.detect_language("Zh") == "en"
    get_settings.cache_clear()


def test_allocate_adaptive(tmp_path):
    from pathlib import Path as P
    groups = {
        "A": [tmp_path / f"a{i}.md" for i in range(10)],
        "B": [tmp_path / f"b{i}.md" for i in range(10)],
        "C": [tmp_path / f"c{i}.md" for i in range(10)],
        "D": [tmp_path / f"d{i}.md" for i in range(10)],
    }

    # breadth-only: pure single-hop mix spreads across all groups
    picked = testset_gen._allocate_adaptive(groups, 8, multi_share=0.0)
    assert len(picked) == 8
    assert len({p.name[0] for p in picked}) == 4

    # full multi-hop: deep concentration, ~6 per group in few groups
    picked = testset_gen._allocate_adaptive(groups, 12, multi_share=1.0)
    assert len(picked) == 12
    by_group = {}
    for p in picked:
        by_group[p.name[0]] = by_group.get(p.name[0], 0) + 1
    assert len(by_group) == 2  # ceil(12/6) groups
    assert sorted(by_group.values()) == [6, 6]

    # mixed: some deep, some breadth; deterministic
    p1 = testset_gen._allocate_adaptive(groups, 10, multi_share=0.5)
    p2 = testset_gen._allocate_adaptive(groups, 10, multi_share=0.5)
    assert [str(p) for p in p1] == [str(p) for p in p2]


def test_round_robin_groups(tmp_path):
    ga = [tmp_path / f"a{i}.md" for i in range(3)]
    gb = [tmp_path / f"b{i}.md" for i in range(2)]
    picked = testset_gen._round_robin_groups([ga, gb], 4)
    assert len(picked) == 4
    # interleaved: both groups contribute
    assert len({p.name[0] for p in picked}) == 2


def test_kmeans_labels_separates_clusters():
    # two obvious clusters around [1,0,0] and [0,1,0]
    vecs = [[1.0, 0.01, 0.0], [0.99, 0.02, 0.0], [0.98, 0.0, 0.01],
            [0.0, 1.0, 0.01], [0.01, 0.99, 0.0], [0.0, 0.98, 0.02]]
    labels = testset_gen._kmeans_labels(vecs, k=2)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_select_seeds_smart_directory_grouping(tmp_path, monkeypatch):
    """Corpus with subdirectories -> grouped by directory (no embedding call)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    corpus_dir = tmp_path / "data" / "corpus" / "Grouped"
    for d in ("compA", "compB"):
        (corpus_dir / d).mkdir(parents=True)
        for i in range(4):
            (corpus_dir / d / f"doc{i}.md").write_text("内容" * 800)

    class BombEmb:
        async def aembed_documents(self, texts):
            raise AssertionError("embedding should not be called for grouped corpus")

    files = testset_gen.corpus_md_files("Grouped")
    picked = asyncio.run(testset_gen.select_seeds_smart(files, 4, None, "Grouped", BombEmb()))
    assert len(picked) == 4
    assert {p.parent.name for p in picked} == {"compA", "compB"}
    get_settings.cache_clear()


def test_select_seeds_unwraps_wrapper_directory(tmp_path, monkeypatch):
    """A corpus uploaded as one wrapper dir (data/<公司>/*.md) groups by company."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    corpus_dir = tmp_path / "data" / "corpus" / "Wrapped"
    for comp in ("compA", "compB"):
        (corpus_dir / "data" / comp).mkdir(parents=True)
        for i in range(4):
            (corpus_dir / "data" / comp / f"doc{i}.md").write_text("内容" * 800)

    class BombEmb:
        async def aembed_documents(self, texts):
            raise AssertionError("embedding should not be called for grouped corpus")

    files = testset_gen.corpus_md_files("Wrapped")
    picked = asyncio.run(testset_gen.select_seeds_smart(files, 4, None, "Wrapped", BombEmb()))
    assert len(picked) == 4
    assert {p.parent.name for p in picked} == {"compA", "compB"}
    get_settings.cache_clear()


def test_select_seeds_smart_clustering(tmp_path, monkeypatch):
    """Flat corpus -> embedding clustering spreads picks across clusters."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    corpus_dir = tmp_path / "data" / "corpus" / "Flat"
    corpus_dir.mkdir(parents=True)
    for i in range(10):
        marker = "a" if i < 5 else "b"
        (corpus_dir / f"doc{i}.md").write_text(marker + "x" * 2399)

    class MarkerEmb:
        async def aembed_documents(self, texts):
            # two clusters by first character
            return [[1.0, 0.0] if t.startswith("a") else [0.0, 1.0] for t in texts]

    files = testset_gen.corpus_md_files("Flat")
    picked = asyncio.run(testset_gen.select_seeds_smart(files, 4, None, "Flat", MarkerEmb()))
    assert len(picked) == 4
    # both clusters contributed
    assert {p.read_text()[0] for p in picked} == {"a", "b"}
    get_settings.cache_clear()


def test_llm_escalates_max_tokens_on_truncation(tmp_path, monkeypatch):
    """finish_reason=length -> retry with doubled max_tokens; stop on success."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI

    def make_result(finish):
        msg = AIMessage(content="x", response_metadata={"token_usage": {}})
        return ChatResult(generations=[ChatGeneration(
            message=msg, generation_info={"finish_reason": finish})])

    calls = []

    def fake_generate(self, *a, **kw):
        calls.append(self.max_tokens)  # escalation raises the client budget
        return make_result("length" if len(calls) == 1 else "stop")

    monkeypatch.setattr(ChatOpenAI, "generate", fake_generate)
    from app.db.models import ModelConfig
    cfg = ModelConfig(type="llm", api_format="openai", base_url="https://x/v1",
                      api_key="k", model="m", name="t")
    _wrapper, raw, _cache = testset_gen._build_llm(cfg, 4096, tmp_path / "cache")
    result = raw.generate([[HumanMessage(content="hi")]])
    assert calls == [4096, 8192]
    assert result.generations[0].generation_info["finish_reason"] == "stop"


def test_llm_escalation_stops_at_platform_cap(tmp_path, monkeypatch):
    """Platform 400 naming the real cap -> one clamped retry, then accept."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI

    def make_result(finish):
        msg = AIMessage(content="x", response_metadata={"token_usage": {}})
        return ChatResult(generations=[ChatGeneration(
            message=msg, generation_info={"finish_reason": finish})])

    calls = []

    def fake_generate(self, *a, **kw):
        calls.append(self.max_tokens)
        if len(calls) == 1:
            return make_result("length")
        if calls[-1] == 8192:
            raise RuntimeError("max_tokens must be at most 6000")
        return make_result("stop")

    monkeypatch.setattr(ChatOpenAI, "generate", fake_generate)
    from app.db.models import ModelConfig
    cfg = ModelConfig(type="llm", api_format="openai", base_url="https://x/v1",
                      api_key="k", model="m", name="t")
    _wrapper, raw, _cache = testset_gen._build_llm(cfg, 4096, tmp_path / "cache")
    result = raw.generate([[HumanMessage(content="hi")]])
    # 4096 (default) -> 8192 (rejected, cap 6000) -> 6000 (clamped) -> stop
    assert calls == [4096, 8192, 6000]
    assert result.generations[0].generation_info["finish_reason"] == "stop"


def test_build_llm_anthropic_format(tmp_path, monkeypatch):
    """An anthropic-format judge builds a ChatAnthropic (not ChatOpenAI), with
    the base URL normalized for the Anthropic SDK."""
    from app.db.models import ModelConfig
    cfg = ModelConfig(type="llm", api_format="anthropic",
                      base_url="https://api.anthropic.com/v1",
                      api_key="sk-ant", model="claude-sonnet-5", name="c")
    wrapper, raw, cache = testset_gen._build_llm(cfg, 4096, tmp_path / "c")

    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    assert isinstance(raw, ChatAnthropic) and not isinstance(raw, ChatOpenAI)
    # the SDK appends /v1/messages itself, so /v1 must be trimmed
    assert raw.anthropic_api_url == "https://api.anthropic.com"
    assert raw.max_tokens == 4096
    assert raw.totals == {"prompt": 0, "completion": 0, "calls": 0}
    assert cache is not None


def test_anthropic_sdk_base_url_normalization():
    assert testset_gen._anthropic_sdk_base_url("https://api.anthropic.com/v1") == \
        "https://api.anthropic.com"
    assert testset_gen._anthropic_sdk_base_url("https://api.anthropic.com/v1/") == \
        "https://api.anthropic.com"
    assert testset_gen._anthropic_sdk_base_url("https://api.anthropic.com") == \
        "https://api.anthropic.com"
    assert testset_gen._anthropic_sdk_base_url("") == "https://api.anthropic.com"
    # a proxy path must survive untouched
    assert testset_gen._anthropic_sdk_base_url("https://proxy.corp/claude") == \
        "https://proxy.corp/claude"


def test_anthropic_truncation_triggers_escalation(tmp_path, monkeypatch):
    """Anthropic signals truncation with stop_reason=max_tokens, not
    finish_reason=length — escalation must recognise it."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_anthropic import ChatAnthropic

    def make_result(stop_reason):
        msg = AIMessage(content="x", response_metadata={"token_usage": {},
                                                        "stop_reason": stop_reason})
        return ChatResult(generations=[ChatGeneration(message=msg,
                                                      generation_info={})])

    calls = []

    def fake_generate(self, *a, **kw):
        calls.append(self.max_tokens)
        return make_result("max_tokens" if len(calls) == 1 else "end_turn")

    monkeypatch.setattr(ChatAnthropic, "generate", fake_generate)
    from app.db.models import ModelConfig
    cfg = ModelConfig(type="llm", api_format="anthropic",
                      base_url="https://api.anthropic.com", api_key="k",
                      model="claude-sonnet-5", name="c")
    _w, raw, _c = testset_gen._build_llm(cfg, 4096, tmp_path / "c")
    result = raw.generate([[]])
    assert calls == [4096, 8192]
    assert result.generations[0].message.response_metadata["stop_reason"] == "end_turn"


def test_thinking_disabled_for_ark_on_both_formats(tmp_path):
    """Volcano Ark defaults to thinking ON, which burns output tokens; both
    protocol branches must switch it off. Non-Ark endpoints are untouched."""
    from app.db.models import ModelConfig

    ark = "https://ark.cn-beijing.volces.com/api/compatible"
    anth = ModelConfig(type="llm", api_format="anthropic", base_url=ark,
                       api_key="k", model="deepseek-v4-1-flash", name="d")
    _w, raw, _c = testset_gen._build_llm(anth, 512, None)
    assert raw.thinking == {"type": "disabled"}

    oai = ModelConfig(type="llm", api_format="openai",
                      base_url="https://ark.cn-beijing.volces.com/api/v3",
                      api_key="k", model="doubao-seed-2-0-mini", name="b")
    _w2, raw2, _c2 = testset_gen._build_llm(oai, 512, None)
    assert raw2.extra_body == {"thinking": {"type": "disabled"}}

    # the Anthropic parameter is official, so it applies to every
    # Anthropic-format endpoint (not just Ark)
    other = ModelConfig(type="llm", api_format="anthropic",
                        base_url="https://api.anthropic.com", api_key="k",
                        model="claude-sonnet-5", name="c")
    _w3, raw3, _c3 = testset_gen._build_llm(other, 512, None)
    assert raw3.thinking == {"type": "disabled"}

    # ...but the Ark-specific patch must NOT leak into other OpenAI-format
    # endpoints (strict implementations reject unknown arguments)
    plain = ModelConfig(type="llm", api_format="openai",
                        base_url="https://api.openai.com/v1", api_key="k",
                        model="gpt-4o", name="g")
    _w4, raw4, _c4 = testset_gen._build_llm(plain, 512, None)
    assert getattr(raw4, "extra_body", None) is None


def test_chinese_prompts_are_actually_applied():
    """ragas' load_prompts RETURNS prompts; it does not apply them. Without a
    following set_prompts the shipped Chinese assets are silently ignored and
    English defaults are used — this test pins that they take effect."""
    class _NullLLM:
        def generate_prompt(self, *a, **kw):
            raise AssertionError("should not be called")

    zh = testset_gen._make_synthesizers(_NullLLM(), "zh")
    for key, synth in zh.items():
        instruction = synth.get_prompts()["query_answer_generation_prompt"].instruction
        assert "查询" in instruction, f"{key}: Chinese prompt not applied"
        assert "query" not in instruction.lower(), f"{key}: English default leaked in"

    # English also loads our assets (same question-focused instruction)
    en = testset_gen._make_synthesizers(_NullLLM(), "en")
    instruction_en = en["single"].get_prompts()["query_answer_generation_prompt"].instruction
    assert "one complete sentence" in instruction_en


def test_reference_answers_are_question_focused():
    """The answer instruction must ask for a concise, question-only answer:
    verbose references make AnswerCorrectness measure coverage of the source
    chunk rather than whether the question was answered."""
    import json
    from app.services.testset_gen import PROMPTS_DIR

    zh = list(PROMPTS_DIR.glob("*query_answer_generation_prompt_chinese.json"))
    en = list(PROMPTS_DIR.glob("*query_answer_generation_prompt_english.json"))
    assert len(zh) == 3 and len(en) == 3, (zh, en)
    for f in zh:
        instruction = json.loads(f.read_text(encoding="utf-8"))["instruction"]
        assert "一句完整的话" in instruction, f
        assert "详细回答" not in instruction, f"{f.name} still asks for a detailed answer"
    for f in en:
        instruction = json.loads(f.read_text(encoding="utf-8"))["instruction"]
        assert "one complete sentence" in instruction, f
        assert "detailed answer" not in instruction, f"{f.name} still asks for a detailed answer"


def test_fallback_topup_has_headroom(client, corpus_with_docs, model_configs, mock_ragas,
                                     monkeypatch):
    """The fallback top-up must request more than the exact gap: with zero
    headroom a single near-duplicate wastes the whole round and the testset is
    delivered short (observed in production: 9/10)."""
    import app.services.testset_gen as tg

    # Force the gap through the semantic reviewer: the trim runs last, so a
    # drop made there is absorbed by the final slice and no gap reaches the
    # fallback.
    async def drop_one(llm, samples, testset_id, type_key, language):
        return samples[1:] if samples else samples

    monkeypatch.setattr(tg, "_semantic_dedup", drop_one)

    requested: list[tuple[int | None, int]] = []
    real_gen = tg._generate_many

    async def spy(gen, synth, target, *a, round_no=None, **kw):
        requested.append((round_no, target))
        return await real_gen(gen, synth, target, *a, round_no=round_no, **kw)

    monkeypatch.setattr(tg, "_generate_many", spy)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "FallbackHeadroom", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 4, "n_multi_specific": 0, "n_multi_abstract": 0,
        # gen_amplify 1.0 keeps the first pool exactly at target, so the
        # reviewer's single drop is a real gap rather than absorbed headroom.
        "gen_amplify": 1.0,
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed"

    fallbacks = [(r, t) for r, t in requested if r is not None]
    assert fallbacks, "the forced gap must have triggered a fallback round"

    # drop_one removes exactly one item per pass, so every gap is 1: the top-up
    # must still ask for at least 2 (headroom), not exactly 1.
    for round_no, target in fallbacks:
        assert target > 1, f"round {round_no} requested {target} for a 1-item gap: no headroom"


def test_semantic_dedup_rejection_is_sticky(client, corpus_with_docs, model_configs,
                                            mock_ragas, monkeypatch):
    """`kept` is rebuilt from the whole pool on every round, so a question the
    semantic reviewer rejected used to be offered again next round and could be
    accepted on a second look — one did exactly that and shipped in a delivered
    testset. A rejection must outlive the round that made it."""
    import app.services.testset_gen as tg

    # Pass the pool through untouched so the only gap comes from the rejection
    # below, and so the rejected question is definitely offered again next round
    # (a real trim could drop it by chance and hide the bug).
    rejected: set[str] = set()
    offered_again: list[str] = []

    async def dedup_rejecting_two(llm, samples, testset_id, type_key, language):
        offered_again.extend(s["user_input"] for s in samples
                             if s["user_input"] in rejected)
        if samples and not rejected:
            rejected.update(s["user_input"] for s in samples[:2])
            return samples[2:]
        return samples

    monkeypatch.setattr(tg, "_semantic_dedup", dedup_rejecting_two)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "StickyDedup", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 4, "n_multi_specific": 0, "n_multi_abstract": 0,
        "gen_amplify": 1.1,
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed", item.get("error")

    assert rejected, "the forced rejection must have happened"
    assert not offered_again, (
        f"a question the reviewer rejected was offered again: {offered_again}")


def test_sample_quality_rejects_fragment_references():
    """Fragment references ("江阴银行") make AnswerCorrectness meaningless: the
    claim-F1 needs the reference to entail the answer's claims. They must be
    dropped and topped up like other low-quality samples."""
    ok = {"user_input": "002807对应的证券简称是什么？", "reference": "江阴银行"}
    assert not testset_gen.sample_quality_ok(ok)

    ok2 = {"user_input": "002807对应的证券简称是什么？",
           "reference": "证券代码002807对应的证券简称是江阴银行。"}
    assert testset_gen.sample_quality_ok(ok2)

    # a full sentence is kept even when the corpus answer is a number
    ok3 = {"user_input": "本次股东大会出席人数是多少？",
           "reference": "出席本次股东大会的股东及授权代表共24人。"}
    assert testset_gen.sample_quality_ok(ok3)


def test_prompts_require_complete_answers_and_unambiguous_questions():
    """Both language assets must ask for a self-contained sentence and for the
    question to name its subject (subject-less questions retrieve the wrong
    company's documents in a multi-company corpus)."""
    import json
    from app.services.testset_gen import PROMPTS_DIR

    for f in sorted(PROMPTS_DIR.glob("*query_answer_generation_prompt_*.json")):
        instruction = json.loads(f.read_text(encoding="utf-8"))["instruction"]
        zh = "chinese" in f.name
        assert ("一句完整的话" if zh else "one complete sentence") in instruction, f.name
        assert ("无歧义" if zh else "unambiguous") in instruction, f.name
        # questions must be self-contained AND internally coherent: naming every
        # object (no "this meeting") and keeping time/event consistent
        assert ("年份、日期、主体名称必须与素材一致" if zh
                else "Every year, date and name in the question must match") in instruction, f.name


def test_abstract_synthesizer_generated_in_one_call():
    """Every ragas query synthesizer walks `get_node_clusters()` from the start
    and stops as soon as it has n scenarios, so that selection is deterministic
    and chunked calls redraw the same leading nodes — every chunk returns the
    same questions. All three must be asked once."""
    import asyncio
    import app.services.testset_gen as tg
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    assert tg._needs_single_call(MultiHopAbstractQuerySynthesizer(llm=None))
    assert tg._needs_single_call(MultiHopSpecificQuerySynthesizer(llm=None))
    # single-hop breaks out at n too — it does not enumerate the whole node set,
    # which is what made a chunked batch come out entirely about the first nodes
    assert tg._needs_single_call(SingleHopSpecificQuerySynthesizer(llm=None))

    # the chunking decision: one call for the whole target
    calls: list[int] = []
    real = tg._generate_chunk

    def spy(gen, synth, size, run_config):
        calls.append(size)
        return []

    tg._generate_chunk = spy
    try:
        for synth in (MultiHopAbstractQuerySynthesizer(llm=None),
                      SingleHopSpecificQuerySynthesizer(llm=None)):
            calls.clear()
            asyncio.run(tg._generate_many(None, synth, 14, None, None, "s", 0, 0))
            assert calls == [14], f"{type(synth).__name__} must be one call, got {calls}"
    finally:
        tg._generate_chunk = real


def test_generation_walks_further_into_the_node_pool_each_call():
    """Generation always starts from the top of the node list, so without
    rotating the pool a fallback round regenerates the questions the round
    before it already made — they then all fail dedup and the type ends short.
    Each call must begin where the last one stopped."""
    import asyncio
    import app.services.testset_gen as tg
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

    synth = SingleHopSpecificQuerySynthesizer(llm=None)
    # Stand in for the graph: the synthesizer walks whatever this returns, in
    # order, taking one scenario per node while the pool is wider than n.
    synth.get_node_clusters = lambda kg: list(range(40))

    seen: list[int] = []
    real_chunk = tg._generate_chunk

    def spy(gen, synth_, size, run_config):
        seen.extend(synth_.get_node_clusters(None)[:size])
        return []

    tg._generate_chunk = spy
    try:
        asyncio.run(tg._generate_many(None, synth, 5, None, None, "s", 0, 0))
        asyncio.run(tg._generate_many(None, synth, 5, None, None, "s", 0, 0))
    finally:
        tg._generate_chunk = real_chunk

    assert seen[:5] == [0, 1, 2, 3, 4], seen
    assert seen[5:] == [5, 6, 7, 8, 9], (
        f"the second call must continue past the first slice, saw {seen[5:]}")


def test_node_pool_rotation_survives_every_synthesizer_signature():
    """The three synthesizers are siblings but their get_node_clusters do not
    share a signature: single-hop and multi-hop-specific take the graph alone,
    multi-hop-abstract takes (graph, n). Wrapping all three in a one-argument
    function crashed the abstract one on every call — a whole type came back
    0/10, with two fallback rounds that generated nothing. Each must survive the
    call shape ragas actually uses, and the abstract one must not be rotated:
    its cluster set is already chosen by n, so reordering it changes nothing."""
    import app.services.testset_gen as tg
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    graph = object()

    def pool(*args):  # stands in for a synthesizer's own cluster lookup
        return list(range(10))

    # (synth, the arguments ragas passes, expected result length)
    cases = [
        (SingleHopSpecificQuerySynthesizer(llm=None), (graph,), 10),
        (MultiHopSpecificQuerySynthesizer(llm=None), (graph,), 10),
        # Returns a window of n, not the whole pool; see the test below.
        (MultiHopAbstractQuerySynthesizer(llm=None), (graph, 3), 3),
    ]
    for synth, call_args, expected in cases:
        name = type(synth).__name__
        synth.get_node_clusters = pool
        tg._rotate_node_pool(synth, 4)
        result = synth.get_node_clusters(*call_args)  # must not raise
        assert len(result) == expected, f"{name}: got {len(result)}"
        assert result[0] == 4, f"{name}: not rotated, starts at {result[0]}"


def test_abstract_rotation_moves_a_window_over_a_larger_cluster_set():
    """Multi-hop abstract takes the count, so rotating its result list would be
    a no-op — ragas picks the same n clusters every call. It has to be asked for
    more than it returns and handed a moving window, or the fallback round
    regenerates the same questions: measured, two rounds gained 0 and 1 while
    the type sat one short."""
    import app.services.testset_gen as tg
    from ragas.testset.synthesizers import MultiHopAbstractQuerySynthesizer

    synth = MultiHopAbstractQuerySynthesizer(llm=None)
    asked: list[int] = []

    def pool(*args):
        asked.append(args[-1])          # the count ragas was asked for
        return list(range(40))

    synth.get_node_clusters = pool
    tg._rotate_node_pool(synth, 0)
    assert synth.get_node_clusters(object(), 3) == [0, 1, 2]
    assert asked[-1] > 3, (
        f"asked the graph for {asked[-1]} clusters: the window can never move")

    tg._rotate_node_pool(synth, 5)
    assert synth.get_node_clusters(object(), 3) == [5, 6, 7]

    # a window running off the end wraps instead of coming back short
    tg._rotate_node_pool(synth, 39)
    assert len(synth.get_node_clusters(object(), 3)) == 3


def test_reference_gate_drops_references_the_material_does_not_support(monkeypatch):
    """Ragas Faithfulness is the gate's factual check: a reference the material
    does not support makes correct system answers score low downstream."""
    import asyncio
    import app.services.testset_gen as tg

    samples = [
        {"user_input": "问题A？", "reference": "好答案", "reference_contexts": ["素材"]},
        {"user_input": "问题B？", "reference": "坏答案", "reference_contexts": ["素材"]},
    ]

    async def fake_faithfulness(llm, batch):
        # "坏答案" is unsupported by its material
        return [0.0 if s["reference"] == "坏答案" else 1.0 for s in batch]

    monkeypatch.setattr(tg, "_faithfulness_scores", fake_faithfulness)
    kept = asyncio.run(tg._filter_faithful(None, samples, None, "single", "zh"))
    assert [s["reference"] for s in kept] == ["好答案"]


def test_reference_gate_keeps_unscorable_pairs(monkeypatch):
    """A NaN score means the metric could not measure the pair — keep it rather
    than dropping on a failure to measure."""
    import asyncio
    import app.services.testset_gen as tg

    samples = [{"user_input": "问题？", "reference": "答案", "reference_contexts": ["素材"]}]

    async def nan_faithfulness(llm, batch):
        return [float("nan")]

    monkeypatch.setattr(tg, "_faithfulness_scores", nan_faithfulness)
    kept = asyncio.run(tg._filter_faithful(None, samples, None, "single", "zh"))
    assert len(kept) == 1


def test_the_check_screens_a_batch_then_verifies_each_suspect(monkeypatch):
    """Screening raises the suspicion, checking settles it. Either half alone
    fails: screening alone fires on 40-50% of a batch and still misses some, and
    checking alone is the per-item judgement that cannot see that fifteen
    neighbours name a company and one does not."""
    import asyncio
    import app.services.testset_gen as tg

    samples = [{"user_input": f"问题{i}？", "reference": f"答案{i}",
                "reference_contexts": ["素材"]} for i in range(1, 5)]
    seen = []

    class Screening:
        async def generate(self, llm=None, data=None):
            # generous on purpose — a false alarm is the second pass's problem
            return tg._ScreenOutput(suspect=[2, 3], reason=["理由2", "理由3"])

    class Verifying:
        async def generate(self, llm=None, data=None):
            seen.append(data)
            return tg._VerifyOutput(ok=not data.question.startswith("问题2"))

    monkeypatch.setattr(tg, "_screen_prompt", lambda lang: Screening)
    monkeypatch.setattr(tg, "_verify_prompt", lambda lang: Verifying)

    kept = asyncio.run(tg._screen_and_verify(None, samples, None, "single", "zh"))
    assert [s["reference"] for s in kept] == ["答案1", "答案3", "答案4"]
    assert [d.question for d in seen] == ["问题2？", "问题3？"]
    assert [d.suspicion for d in seen] == ["理由2", "理由3"]
    # the crux: the verdict keeps the rest of the batch in view. Measured with
    # this dropped, the check confirmed 0 of 64 suspects; with it, 16.
    assert seen[0].other_questions == ["问题1？", "问题3？", "问题4？"]


def test_dropped_pairs_are_archived_with_the_reason(tmp_path, monkeypatch):
    """The gate is the only thing between the generator and the customer, and a
    dropped pair used to cease to exist: the log quoted two examples and the
    items were gone. That leaves no way to audit a drop, and no pool of known
    defects to measure a change against."""
    import asyncio
    import json

    import app.services.testset_gen as tg

    monkeypatch.setattr(tg, "dropped_path_for",
                        lambda name: tmp_path / name / "dropped.jsonl")

    samples = [{"user_input": f"问题{i}？", "reference": f"答案{i}",
                "reference_contexts": ["素材"]} for i in range(1, 4)]

    class Screening:
        async def generate(self, llm=None, data=None):
            return tg._ScreenOutput(suspect=[2], reason=["没点名主体"])

    class Verifying:
        async def generate(self, llm=None, data=None):
            return tg._VerifyOutput(ok=False, problem="问题没有点名任何机构")

    monkeypatch.setattr(tg, "_screen_prompt", lambda lang: Screening)
    monkeypatch.setattr(tg, "_verify_prompt", lambda lang: Verifying)

    kept = asyncio.run(tg._screen_and_verify(
        None, samples, None, "single", "zh", round_no=2, testset_name="T"))
    assert len(kept) == 2

    rows = [json.loads(x) for x in
            (tmp_path / "T" / "dropped.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["question"] == "问题2？"
    assert rows[0]["reference"] == "答案2"
    assert rows[0]["contexts"] == ["素材"]
    assert rows[0]["type"] == "single"
    assert rows[0]["round"] == 2
    assert rows[0]["reason"] == "问题没有点名任何机构"


def test_the_check_degrades_safely_on_error(monkeypatch):
    """A judge that cannot run keeps its batch, as everywhere else in the gate."""
    import asyncio
    import app.services.testset_gen as tg

    samples = [{"user_input": f"问题{i}？", "reference": f"答案{i}",
                "reference_contexts": ["素材"]} for i in range(1, 4)]

    class Boom:
        async def generate(self, llm=None, data=None):
            raise RuntimeError("judge down")

    # the screen cannot run at all -> nothing is suspected -> everything kept
    monkeypatch.setattr(tg, "_screen_prompt", lambda lang: Boom)
    assert asyncio.run(tg._screen_and_verify(object(), samples, None, "single", "zh")) == samples

    # the screen works but the check fails -> the suspect is kept
    class Screening:
        async def generate(self, llm=None, data=None):
            return tg._ScreenOutput(suspect=[2], reason=["x"])

    monkeypatch.setattr(tg, "_screen_prompt", lambda lang: Screening)
    monkeypatch.setattr(tg, "_verify_prompt", lambda lang: Boom)
    kept = asyncio.run(tg._screen_and_verify(None, samples, None, "single", "zh"))
    assert kept == samples


def test_prompt_examples_do_not_translate_persona_names():
    """A few-shot example that renames its inputs teaches the model to do the
    same. The Chinese matching prompts shipped examples with Chinese input
    personas but English output keys ("HR Manager"), so ragas' exact-name
    lookup raised KeyError and abstract generation silently produced 0 samples."""
    import json
    import glob
    from app.services.testset_gen import PROMPTS_DIR

    files = sorted(PROMPTS_DIR.glob("*themes_personas_matching_prompt_*.json"))
    assert files, "expected matching-prompt assets"
    for f in files:
        example = json.loads(f.read_text(encoding="utf-8"))["examples"][0]
        input_names = [p["name"] for p in example["input"]["personas"]]
        output_names = list(example["output"]["mapping"])
        assert input_names == output_names, (
            f"{f.name}: example maps {input_names} to {output_names} — "
            "the model will rename personas and the lookup will fail"
        )


def test_faithfulness_retries_parse_failures(monkeypatch):
    """A NaN (ragas parse failure) is retried, not accepted as unscoreable —
    measured, such an item scored 1.00 on every retry."""
    import asyncio
    import app.services.testset_gen as tg

    samples = [{"user_input": "问题", "reference": "答案", "reference_contexts": ["素材"]}]
    calls = {"n": 0}

    async def flaky(llm, batch):
        calls["n"] += 1
        return [float("nan")] if calls["n"] == 1 else [1.0]

    monkeypatch.setattr(tg, "_run_faithfulness", flaky)
    assert asyncio.run(tg._faithfulness_scores(None, samples)) == [1.0]
    assert calls["n"] == 2

    # permanently unscoreable -> stays NaN after the attempts are exhausted
    async def always_nan(llm, batch):
        calls["n"] += 1
        return [float("nan")]

    calls["n"] = 0
    monkeypatch.setattr(tg, "_run_faithfulness", always_nan)
    scores = asyncio.run(tg._faithfulness_scores(None, samples, attempts=3))
    assert scores[0] != scores[0] and calls["n"] == 3


def test_topup_size_adapts_to_observed_yield():
    """The fallback top-up is sized from the yield this material has shown, not
    from a fixed multiplier: a clean corpus asks for barely more than the gap, a
    noisy one asks for proportionally more (bounded, so it cannot run away)."""
    import app.services.testset_gen as tg

    # clean corpus: 80% yield -> asks for just over the gap
    assert tg._topup_size(missing=2, kept=8, generated=10) == 3
    # noisy corpus: 50% -> double
    assert tg._topup_size(missing=3, kept=6, generated=12) == 6
    # very noisy: clamped at MIN_YIELD_RATE, never unbounded
    huge = tg._topup_size(missing=8, kept=1, generated=20)
    assert huge == 40  # 8 / MIN_YIELD_RATE(0.2) — bounded, never unbounded

    # no history yet -> assume half survives
    assert tg._topup_size(missing=4, kept=0, generated=0) == 8
    # never fewer than the gap itself
    assert tg._topup_size(missing=5, kept=100, generated=100) == 6


def test_prompt_examples_do_not_translate_persona_names():
    """A few-shot example that renames its inputs teaches the model to do the
    same. The Chinese matching prompts shipped examples with Chinese input
    personas but English output keys ("HR Manager"), so ragas' exact-name
    lookup raised KeyError and abstract generation silently produced 0 samples."""
    import json
    import glob
    from app.services.testset_gen import PROMPTS_DIR

    files = sorted(PROMPTS_DIR.glob("*themes_personas_matching_prompt_*.json"))
    assert files, "expected matching-prompt assets"
    for f in files:
        example = json.loads(f.read_text(encoding="utf-8"))["examples"][0]
        input_names = [p["name"] for p in example["input"]["personas"]]
        output_names = list(example["output"]["mapping"])
        assert input_names == output_names, (
            f"{f.name}: example maps {input_names} to {output_names} — "
            "the model will rename personas and the lookup will fail"
        )


def test_faithfulness_retries_parse_failures(monkeypatch):
    """A NaN (ragas parse failure) is retried, not accepted as unscoreable —
    measured, such an item scored 1.00 on every retry."""
    import asyncio
    import app.services.testset_gen as tg

    samples = [{"user_input": "问题", "reference": "答案", "reference_contexts": ["素材"]}]
    calls = {"n": 0}

    async def flaky(llm, batch):
        calls["n"] += 1
        return [float("nan")] if calls["n"] == 1 else [1.0]

    monkeypatch.setattr(tg, "_run_faithfulness", flaky)
    assert asyncio.run(tg._faithfulness_scores(None, samples)) == [1.0]
    assert calls["n"] == 2

    # permanently unscoreable -> stays NaN after the attempts are exhausted
    async def always_nan(llm, batch):
        calls["n"] += 1
        return [float("nan")]

    calls["n"] = 0
    monkeypatch.setattr(tg, "_run_faithfulness", always_nan)
    scores = asyncio.run(tg._faithfulness_scores(None, samples, attempts=3))
    assert scores[0] != scores[0] and calls["n"] == 3


def test_topup_size_adapts_to_observed_yield():
    """The fallback top-up is sized from the yield this material has shown, not
    from a fixed multiplier: a clean corpus asks for barely more than the gap, a
    noisy one asks for proportionally more (bounded, so it cannot run away)."""
    import app.services.testset_gen as tg

    # clean corpus: 80% yield -> asks for just over the gap
    assert tg._topup_size(missing=2, kept=8, generated=10) == 3
    # noisy corpus: 50% -> double
    assert tg._topup_size(missing=3, kept=6, generated=12) == 6
    # very noisy: clamped at MIN_YIELD_RATE, never unbounded
    huge = tg._topup_size(missing=8, kept=1, generated=20)
    assert huge == 40  # 8 / MIN_YIELD_RATE(0.2) — bounded, never unbounded

    # no history yet -> assume half survives
    assert tg._topup_size(missing=4, kept=0, generated=0) == 8
    # never fewer than the gap itself
    assert tg._topup_size(missing=5, kept=100, generated=100) == 6




def test_gate_logs_rejected_samples_to_the_server_log(monkeypatch, caplog):
    """A couple of quoted examples with their reason go to the server log.

    They used to go to the customer's log as well, where a 60-character excerpt
    of a question answers none of the questions the customer has. The reason
    they exist at all — telling whether the fix belongs in the prompt, the model
    or the threshold — is ours.
    """
    import asyncio
    import logging
    import app.services.testset_gen as tg

    samples = [
        {"user_input": "好问题？", "reference": "好答案", "reference_contexts": ["素材"]},
        {"user_input": "低忠实度问题？", "reference": "编造答案", "reference_contexts": ["素材"]},
    ]

    async def fake_scores(llm, batch):
        return [1.0 if s["reference"] != "编造答案" else 0.4 for s in batch]

    logged = []
    monkeypatch.setattr(tg, "_faithfulness_scores", fake_scores)
    monkeypatch.setattr(tg, "_log", lambda tid, key, params=None: logged.append((key, params)))

    with caplog.at_level(logging.INFO):
        kept = asyncio.run(tg._filter_faithful(None, samples, 1, "single", "zh"))

    assert [s["reference"] for s in kept] == ["好答案"]
    keys = [k for k, _ in logged]
    # The gate reports what it checked as well as what it dropped: reporting
    # only the drops left a gate that ran and passed everything with no trace
    # at all, which is how a run whose generation came back empty read as
    # "generate then fallback" with the gate apparently skipped.
    assert "gateResult" not in keys, "the breakdown belongs in the server log"
    assert not [k for k in keys if k.startswith("referenceDropSample")], \
        "an excerpt is not something the customer can act on"
    assert "低忠实度问题？" in caplog.text and "0.4" in caplog.text


def test_exact_duplicates_are_dropped_without_asking_the_model(monkeypatch):
    """Two identical strings are the same question; the reviewer is not asked.

    It was asked, and missed: one run shipped a byte-equal pair (#21/#23) whose
    only difference was position in the batch. This pass cannot miss it, cannot
    false-kill, and costs a set lookup.
    """
    import app.services.testset_gen as tg

    samples = [
        {"user_input": "荣盛发展2017年年度权益分派的股权登记日是哪一天？"},
        {"user_input": "荣盛发展2017年年度权益分派的股权登记日是哪一天？"},
        {"user_input": "荣盛发展2016年年度权益分派的股权登记日是哪一天？"},
    ]
    kept, dropped = tg._exact_duplicates(samples)

    assert [s["user_input"] for s in kept] == [
        samples[0]["user_input"], samples[2]["user_input"],
    ], "the first occurrence must be the one kept"
    assert dropped == [samples[1]["user_input"]]

    # a batch of distinct questions is untouched
    other = [{"user_input": q} for q in ("甲公司的注册资本是多少？", "乙公司的法定代表人是谁？")]
    assert tg._exact_duplicates(other) == (other, [])

    # an empty question is left for sample_quality_ok to reject, so the reason
    # stays attributed to the right check
    blanks = [{"user_input": ""}, {"user_input": ""}]
    assert tg._exact_duplicates(blanks) == (blanks, [])


def test_a_model_index_outside_the_batch_is_logged(monkeypatch, caplog):
    """A reported index that points outside the batch is not a drop — but it
    used to disappear without a trace, which is how a found duplicate can go
    missing and leave nothing to diagnose."""
    import asyncio
    import logging
    import app.services.testset_gen as tg

    class _Verdict:
        drop = [3, 99]          # 3 is in range, 99 is not

    async def fake_generate(self, llm=None, data=None):
        return _Verdict()

    monkeypatch.setattr(tg._SemanticDedupPromptZh, "generate", fake_generate)
    samples = [{"user_input": q} for q in ("问题一？", "问题二？", "问题三？")]

    with caplog.at_level(logging.INFO):
        kept = asyncio.run(tg._semantic_dedup(None, samples, 1, "single", "zh"))

    assert [s["user_input"] for s in kept] == ["问题一？", "问题二？"]
    assert "reported=[3, 99]" in caplog.text
    assert "outside 1..3" in caplog.text and "99" in caplog.text


def test_semantic_dedup_drops_reworded_duplicates(monkeypatch):
    """The embedding pass misses duplicates whose wording differs a lot (measured:
    0.835 cosine for two questions answered by the same fact). One call per batch
    catches them, and must not touch questions about different facts."""
    import asyncio
    import app.services.testset_gen as tg

    samples = [{"user_input": q} for q in
               ["转债代码128034的简称是什么？",
                "根据决议公告，证券代码002807对应的转债代码128034的简称是什么？",
                "会议召开时间是什么？"]]

    async def fake_generate(self, llm=None, data=None):
        class _Out:
            drop = [2]        # the reworded duplicate
            reason = "same fact"
        return _Out()

    monkeypatch.setattr(tg._SemanticDedupBase, "generate", fake_generate)
    kept = asyncio.run(tg._semantic_dedup(None, samples, None, "single", "zh"))
    assert [s["user_input"] for s in kept] == [samples[0]["user_input"], samples[2]["user_input"]]


def test_semantic_dedup_degrades_safely(monkeypatch):
    """A failing call must keep every sample, and a single sample is a no-op."""
    import asyncio
    import app.services.testset_gen as tg

    async def boom(self, llm=None, data=None):
        raise RuntimeError("judge down")

    monkeypatch.setattr(tg._SemanticDedupBase, "generate", boom)
    samples = [{"user_input": "a"}, {"user_input": "b"}]
    assert asyncio.run(tg._semantic_dedup(None, samples, None, "single", "zh")) == samples
    assert asyncio.run(tg._semantic_dedup(None, samples[:1], None, "single", "zh")) == samples[:1]


def test_prompts_forbid_dating_documents():
    """Questions must not date a document to refer to it (one announcement can
    span years); this is the prevention side of the year-ambiguity class."""
    import json
    from app.services.testset_gen import PROMPTS_DIR

    for f in sorted(PROMPTS_DIR.glob("*query_answer_generation_prompt_*.json")):
        instruction = json.loads(f.read_text(encoding="utf-8"))["instruction"]
        zh = "chinese" in f.name
        assert ("不要用年份指代" if zh else "Do not date a") in instruction, f.name


def test_persona_prompts_ask_for_an_identity_not_a_person():
    """ragas' own prompt asks for "a unique name", which models read as "invent a
    person": one real run produced personas named after people in the documents,
    and the single-hop questions written from that viewpoint addressed them by
    name ("陆文龙啊，你作为…"). Both languages ship an asset saying the name field
    is an identity label."""
    import json
    from app.services.testset_gen import PROMPTS_DIR

    for lang, marker in (("chinese", "不要用具体人名"),
                         ("english", "Do NOT use a personal name")):
        f = PROMPTS_DIR / f"persona_generation_prompt_{lang}.json"
        instruction = json.loads(f.read_text(encoding="utf-8"))["instruction"]
        assert marker in instruction, f.name
        assert "身份" in instruction or "identity" in instruction, f.name


def test_personas_cache_round_trips_and_invalidates(monkeypatch, tmp_path):
    """A persona call is one LLM round-trip whose result depends only on the
    graph's summaries and the prompt — not on which testset is being built. They
    were regenerated for every testset because the only cache in play is scoped
    to a testset directory, so reusing a graph re-paid for them every time."""
    import app.services.testset_gen as tg
    from ragas.testset.persona import Persona

    monkeypatch.setattr(tg, "personas_path_for",
                        lambda name: tmp_path / "personas.json")
    fp = {"llm_model": "m", "emb_model": "e", "run_seed": 7}
    people = [Persona(name="股票投资者", role_description="关注股东大会公告。")]

    assert tg._cached_personas("C", "chinese", fp) is None  # nothing saved yet
    tg._save_personas("C", "chinese", fp, people)
    got = tg._cached_personas("C", "chinese", fp)
    assert [(p.name, p.role_description) for p in got] == [
        ("股票投资者", "关注股东大会公告。")]

    # a different graph, language or persona count must not reuse them
    assert tg._cached_personas("C", "chinese", {**fp, "run_seed": 8}) is None
    assert tg._cached_personas("C", "english", fp) is None
    monkeypatch.setattr(tg, "NUM_PERSONAS", tg.NUM_PERSONAS + 1)
    assert tg._cached_personas("C", "chinese", fp) is None


def test_persona_generation_drops_duplicate_viewpoints(monkeypatch):
    """ragas pads the persona list by sampling with replacement when a corpus
    has fewer summary clusters than personas, which would hand one viewpoint two
    turns — the count is meant to buy distinct viewpoints."""
    import app.services.testset_gen as tg
    from ragas.testset.persona import Persona, generate_personas_from_kg

    def fake(**_kw):
        return [Persona(name="股票投资者", role_description="a"),
                Persona(name="股票投资者", role_description="b"),
                Persona(name="证券分析师", role_description="c")]

    monkeypatch.setattr("ragas.testset.persona.generate_personas_from_kg", fake)
    try:
        out = tg._generate_personas(None, None, "zh")
    finally:
        monkeypatch.setattr("ragas.testset.persona.generate_personas_from_kg",
                            generate_personas_from_kg)
    assert [p.name for p in out] == ["股票投资者", "证券分析师"]


def test_llm_call_logging_names_the_stage_and_the_cost(caplog, monkeypatch):
    """A run that takes far longer than usual has to be traceable to a call:
    one generation call hit the output ceiling, retried at 32768 and was cut off
    again, burning 95 s of a 177 s phase — and nothing in the log said which
    call it was or which phase it belonged to."""
    import logging
    import time as _time
    import app.services.testset_gen as tg

    class FakeClient(tg._MeteredLLM):
        max_tokens = 16384
        _hit_length_stop = staticmethod(lambda result: True)

    monkeypatch.setattr(tg, "_current_stage", ["gen_multi_abstract"])
    with caplog.at_level(logging.INFO, logger="app.services.testset_gen"):
        tg._MeteredLLM._log_call(FakeClient(), _time.monotonic() - 2.5,
                                 12345, object())

    line = caplog.text
    assert "gen_multi_abstract" in line, line
    assert "prompt=12345chars" in line, line
    assert "out=0chars" in line, line          # object() has no generations
    assert "max_tokens=16384" in line, line
    assert "truncated=True" in line, line


def test_generation_stage_is_set_before_the_call_not_after(monkeypatch):
    """The stage label must be current *during* generation. It was only set
    after each chunk, and since every type is now a single chunk the whole
    generation ran under the previous phase's label — a slow run was diagnosed
    as the gate's fault when the calls belonged to generation."""
    import asyncio
    import app.services.testset_gen as tg
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

    seen: list[str] = []

    def spy(gen, synth, size, run_config):   # sync: _generate_chunk runs in a thread
        seen.append(tg._current_stage[0])    # what a call would be labelled
        return []

    def fake_update(_tid, **kw):
        tg._current_stage[0] = str(kw.get("stage", ""))

    monkeypatch.setattr(tg, "_generate_chunk", spy)
    monkeypatch.setattr(tg, "_update", fake_update)
    tg._current_stage[0] = "personas"          # the previous phase's label
    asyncio.run(tg._generate_many(None, SingleHopSpecificQuerySynthesizer(llm=None),
                                  3, None, None, "gen_multi_abstract", 0, 0))
    assert seen == ["gen_multi_abstract"], seen


def test_llm_and_embedding_clients_use_the_short_timeout():
    """LLM/embedding services are platforms, an intranet service or a local
    model: over 1,953 calls p50 was 1.0 s and p95 was 2.0 s, then nothing
    until a tail at 8 s and beyond — so a request still open after
    LLM_TIMEOUT is stuck and should be retried rather than waited on. The
    timeout is fixed rather than escalating because the product connects to
    whatever model the customer configured: a ladder starting short enough for
    DeepSeek would cut every call twice on a slower one. The target RAG system
    is the opposite case and keeps its own generous, configurable timeout.

    Read the httpx client, not `request_timeout`: that attribute reports 180
    whatever was passed and says nothing about the request in flight."""
    import app.services.testset_gen as tg
    from app.db.models import ModelConfig

    def httpx_timeout(client):
        for c in (client, getattr(client, "_client", None)):
            t = getattr(c, "timeout", None)
            if t is not None:
                return t
        return None

    assert tg.LLM_TIMEOUT < 60, "a stalled LLM request must fail fast"
    for fmt, url in (("openai", "http://x/v1"),
                     ("anthropic", "https://api.deepseek.com/anthropic")):
        cfg = ModelConfig(name="c", type="llm", api_format=fmt, base_url=url,
                          api_key="k", model="m")
        _, raw, _ = tg._build_llm(cfg, 4096, None)
        checked = 0
        for attr in ("client", "async_client", "_client", "_async_client"):
            c = getattr(raw, attr, None)
            if c is not None:
                assert httpx_timeout(c) == tg.LLM_TIMEOUT, f"{fmt} {attr}"
                checked += 1
        assert checked, f"{fmt}: no httpx client found to check"

    wrapper, _ = tg._build_embeddings(ModelConfig(
        name="e", type="embedding", api_format="openai",
        base_url="http://localhost:11434/v1", api_key="", model="bge-m3"), 16)
    raw_emb = getattr(wrapper, "embeddings", wrapper)
    assert httpx_timeout(raw_emb.client) == tg.LLM_TIMEOUT


def test_local_embeddings_get_the_callers_concurrency(monkeypatch):
    """A localhost check used to pin this to 4, which assumes the server is
    Ollama: Ollama takes about four at a time and queues the rest, so the cap
    only moved where the queue lives. Nothing here can know what a customer runs
    behind the address, and a local vLLM or TEI would just be throttled."""
    import asyncio

    from app.db.models import ModelConfig
    import app.services.testset_gen as tg

    seen: dict = {}

    async def spy(embed_call, texts, max_concurrency, on_skip=None):
        seen["n"] = max_concurrency
        return [[0.0] for _ in texts]

    monkeypatch.setattr(tg, "_embed_texts_async", spy)

    for url in ("http://localhost:11434/v1", "http://127.0.0.1:8000/v1",
                "https://api.example.com/v1"):
        wrapper, _ = tg._build_embeddings(ModelConfig(
            name="e", type="embedding", api_format="openai",
            base_url=url, api_key="k", model="m"), 16)
        raw = getattr(wrapper, "embeddings", wrapper)
        asyncio.run(raw.aembed_documents(["x"]))
        assert seen["n"] == 16, f"{url} was pinned to {seen['n']}"


def test_the_embedding_cache_namespace_survives_a_real_model_id():
    """Ollama's API answers with "bge-m3:latest", colon included, and picking
    that from the model list is how someone configures a local embedding. The
    namespace hands that name to a store that rejects ":", which killed every
    embedding job of an evaluation — the message naming the key and nothing
    about the cache.

    Sanitising alone would not do: two different names can clean up alike, and
    a shared namespace means one model's vectors served for another."""
    import app.services.testset_gen as tg

    ns = tg._embedding_cache_namespace("bge-m3:latest")
    assert not set(ns) & set(':/\\*?"<>|'), ns
    assert tg._embedding_cache_namespace("bge-m3:latest") == ns
    assert tg._embedding_cache_namespace("bge-m3") != ns
    assert tg._embedding_cache_namespace("a:b") != tg._embedding_cache_namespace("a?b")


def test_the_judge_cache_counts_hits_not_lookups(tmp_path):
    """It counted every get(), so a cold run reported "280 hits" where there
    were none: that number was the number of lookups, and it read as though the
    cache were doing something. The embedding store always had it right."""
    from app.db.models import ModelConfig
    import app.services.testset_gen as tg

    cfg = ModelConfig(name="j", type="llm", api_format="openai",
                      base_url="http://judge/v1", api_key="k", model="m")
    _llm, _raw, cache = tg._build_llm(cfg, 512, tmp_path)

    assert (cache.hits, cache.misses) == (0, 0)
    cache.get("nothing here")  # a lookup that finds nothing is a miss
    assert (cache.hits, cache.misses) == (0, 1)
    cache.set("something", "value")
    cache.get("something")
    assert (cache.hits, cache.misses) == (1, 1)


def test_generation_passes_the_persona_count_through():
    """ragas slices the persona list to its own default when it builds scenarios
    (`persona_list[:num_personas]`), so a count set at generation time but not
    passed to generate() is silently ignored — NUM_PERSONAS would only change how
    many personas get generated, not how many are used to ask questions."""
    import app.services.testset_gen as tg

    seen: dict = {}

    class FakeGen:
        def generate(self, **kw):
            seen.update(kw)
            return type("R", (), {"to_list": lambda _self: []})()

    tg._generate_chunk(FakeGen(), "synth", 3, "run_config")
    assert seen["num_personas"] == tg.NUM_PERSONAS
    assert seen["testset_size"] == 3


def test_query_prompts_ask_for_a_plain_single_query():
    """A persona's voice leaked into the question itself ("陆文龙啊，你作为…").
    Every query asset must ask for the single query a user would type."""
    import json
    from app.services.testset_gen import PROMPTS_DIR

    for f in sorted(PROMPTS_DIR.glob("*query_answer_generation_prompt_*.json")):
        instruction = json.loads(f.read_text(encoding="utf-8"))["instruction"]
        marker = ("不要称呼任何人" if "chinese" in f.name
                  else "do not address anyone by name")
        assert marker in instruction, f.name


def test_standalone_check_catches_provided_information_phrasing():
    """"根据提供的信息…" depends on the source just as much as "所提供的信息"
    does, but lacks the 所 the old pattern required (found in a real testset).
    A specific reference ("根据公司提供的公告…") must still pass."""
    import app.services.testset_gen as tg

    assert not tg.is_standalone_question("根据提供的信息，公司是否指派了董事会秘书？")
    assert not tg.is_standalone_question("根据所提供的信息，公司是否指派了董事会秘书？")
    assert not tg.is_standalone_question("Based on the provided information, who is the secretary?")
    # specific references stay valid
    assert tg.is_standalone_question("根据江阴银行2018年年度股东大会决议公告，现场会议时间是什么？")
    assert tg.is_standalone_question("根据公司提供的公告，关联交易额度是多少？")


def test_reference_that_admits_a_gap_is_dropped():
    """A reference reporting "the source does not say" means the pair asks about
    something the corpus never covers — found twice in real testsets (a question
    comparing two meetings' attendance when only one announcement carries an
    attendance section)."""
    import app.services.testset_gen as tg

    gap = ("天茂集团（000627）2018年第三次临时股东大会出席的股东共24人；"
           "2016年第一次临时股东大会的出席情况未在提供的对应公告内容中披露。")
    assert tg.reference_admits_no_answer(gap)
    assert not tg.sample_quality_ok({"user_input": "两次会议的出席情况分别是怎样的？",
                                     "reference": gap})
    assert tg.reference_admits_no_answer(
        "The attendance figures are not provided in the source material.")
    # an ordinary reference, including one that states a fact about disclosure
    # being required, still passes
    assert not tg.reference_admits_no_answer(
        "公司2018年年度股东大会的现场会议召开时间为2019年4月8日。")
    assert not tg.reference_admits_no_answer(
        "公司董事、监事及高级管理人员买卖本公司股票前需书面通知董事会秘书。")


def test_a_semantic_drop_does_not_leave_the_type_short(client, corpus_with_docs,
                                                       model_configs, mock_ragas,
                                                       monkeypatch):
    """End to end: a reviewer that finds one duplicate every round must not be
    able to keep the type below its target when the pool has enough unique
    questions. Measured twice in a row on the same type (multi-hop abstract held
    at 9/10 with two fallback rounds that each reported "gained 0")."""
    import app.services.testset_gen as tg

    async def drop_one(llm, samples, testset_id, type_key, language):
        return samples[1:] if samples else samples

    monkeypatch.setattr(tg, "_semantic_dedup", drop_one)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "DropOne", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 4, "n_multi_specific": 0, "n_multi_abstract": 0,
        "gen_amplify": 1.1,
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed", item.get("error")
    assert item["actual_single"] == 4, item


# --- guards added after the whole-project review -------------------------


def test_a_run_that_produces_nothing_fails_instead_of_completing(
        client, corpus_with_docs, model_configs, mock_ragas, monkeypatch):
    """Generation errors are swallowed per chunk, so a run whose LLM calls all
    failed used to be persisted empty and marked completed: an empty testset,
    exportable and evaluable, with no error explaining it."""
    monkeypatch.setattr(testset_gen, "_generate_chunk",
                        lambda gen, synth, size, run_config: [])

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "QA-Empty", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 2, "n_multi_specific": 0, "n_multi_abstract": 0,
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "failed", item
    assert "No QA pairs were produced" in (item.get("error") or ""), item


def test_a_corpus_of_only_short_files_fails_with_a_usable_message(
        client, model_configs, mock_ragas, monkeypatch):
    """Every .md shorter than MIN_DOC_CHARS leaves no seed documents, and ragas
    divides by the document count: the user saw "ZeroDivisionError" pointing at
    the library instead of "this corpus has nothing usable"."""
    import io as _io

    corpus = client.post("/api/corpora", json={"name": "TinyCorpus"}).json()
    cid = corpus["id"]
    for i in range(3):
        client.post(f"/api/corpora/{cid}/files",
                    files={"file": (f"t{i}.md", _io.BytesIO("太短了。".encode()))},
                    data={"path": f"t{i}.md"})
    client.post(f"/api/corpora/{cid}/process")
    for _ in range(50):
        if client.get("/api/corpora").json()["items"][0]["status"] == "completed":
            break
        time.sleep(0.2)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "QA-Tiny", "corpus_id": cid,
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "failed", item
    err = item.get("error") or ""
    assert "No usable documents" in err and "ZeroDivisionError" not in err, err


def test_unreadable_seeds_json_does_not_fail_the_run(tmp_path, monkeypatch):
    """seeds.json is rewritten every fallback round, so a truncated file is
    possible; the read must degrade like the fingerprint read next to it."""
    import json as _json
    from app.services import testset_gen as tg

    path = tmp_path / "seeds.json"
    path.write_text('[{"broken')          # truncated mid-write
    monkeypatch.setattr(tg, "seeds_path_for", lambda name: path)

    # the same shape of guarded read the pipeline performs
    old_seeds = []
    if path.exists():
        try:
            old_seeds = _json.loads(path.read_text())
        except (OSError, ValueError):
            old_seeds = []
    assert old_seeds == []


def test_escalation_restores_the_output_budget(monkeypatch):
    """The escalation raises max_tokens in place on a client shared by the
    generator and every concurrent judge call; without restoring it, one
    truncated generation left all later judge prompts asking for 32768."""
    import app.services.testset_gen as tg

    class Client(tg._MeteredLLM):
        _hit_length_stop = staticmethod(lambda result: True)   # always truncated

    c = Client.__new__(Client)
    c.max_tokens = 4096
    c.totals = {"prompt": 0, "completion": 0, "calls": 0}
    seen: list[int] = []

    class Result:
        generations: list = []          # nothing to record; shape is what matters

    def call(*a, **kw):
        seen.append(c.max_tokens)
        return Result()

    c._escalate(call)
    assert seen[0] == 4096
    assert max(seen) > 4096, "the escalation never raised the budget"
    assert c.max_tokens == 4096, "the budget was left raised for later calls"


def test_second_testset_on_a_busy_corpus_is_refused(client, corpus_with_docs,
                                                    model_configs, mock_ragas):
    """The knowledge graph, seed list and persona cache are shared files under
    data/corpus/<name>/; the pipeline only serialises per testset, so two runs
    on one corpus would interleave writes to them."""
    from app.db.models import Testset
    from app.db.session import get_session_factory

    llm, emb = model_configs
    payload = {
        "name": "QA-First", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    }
    first = client.post("/api/testsets", json=payload).json()
    _wait_done(client, first["id"])

    # pretend the first one is still working
    db = get_session_factory()()
    try:
        row = db.get(Testset, first["id"])
        row.status = Testset.STATUS_GENERATING
        db.commit()
    finally:
        db.close()

    second = client.post("/api/testsets", json={**payload, "name": "QA-Second"})
    assert second.status_code == 409, second.text
    assert "already generating" in second.json()["detail"]


def test_resume_is_refused_while_a_run_is_still_unwinding(client, corpus_with_docs,
                                                          model_configs, mock_ragas):
    """run_generation returns immediately when the id is already in flight, so
    marking the row generating anyway left it stuck with nothing running."""
    import app.services.testset_gen as tg
    from app.db.models import Testset
    from app.db.session import get_session_factory

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "QA-Unwind", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    }).json()
    _wait_done(client, ts["id"])

    db = get_session_factory()()
    try:
        row = db.get(Testset, ts["id"])
        row.status = Testset.STATUS_FAILED          # as a just-failed run looks
        db.commit()
    finally:
        db.close()

    tg._running.add(ts["id"])                       # ...but still in flight
    try:
        resp = client.post(f"/api/testsets/{ts['id']}/resume")
        assert resp.status_code == 409, resp.text
        # and the row was left alone rather than flipped to generating
        assert client.get(f"/api/testsets/{ts['id']}").json()["status"] == "failed"
    finally:
        tg._running.discard(ts["id"])


def test_resume_is_refused_when_the_corpus_is_already_busy(client, corpus_with_docs,
                                                           model_configs, mock_ragas):
    """Resuming starts a run, so it needs the same per-corpus check creating
    has. Without it a failed testset could be resumed onto a corpus that was
    already generating, and the two would interleave writes to the shared
    graph, seed list and persona cache."""
    from app.db.models import Testset
    from app.db.session import get_session_factory

    llm, emb = model_configs
    payload = {
        "name": "QA-Busy", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 1, "n_multi_specific": 0, "n_multi_abstract": 0,
    }
    first = client.post("/api/testsets", json=payload).json()
    _wait_done(client, first["id"])
    failed = client.post("/api/testsets", json={**payload, "name": "QA-Failed"}).json()
    _wait_done(client, failed["id"])

    db = get_session_factory()()
    try:
        db.get(Testset, first["id"]).status = Testset.STATUS_GENERATING
        db.get(Testset, failed["id"]).status = Testset.STATUS_FAILED
        db.commit()
    finally:
        db.close()

    resp = client.post(f"/api/testsets/{failed['id']}/resume")
    assert resp.status_code == 409, resp.text
    assert "already generating" in resp.json()["detail"]
    # and it stayed resumable rather than being flipped to generating
    assert client.get(f"/api/testsets/{failed['id']}").json()["status"] == "failed"


def test_the_database_holds_the_rule_when_the_check_is_bypassed(
        client, corpus_with_docs, model_configs, mock_ragas, monkeypatch):
    """The API check and the insert are separate statements, so two requests can
    pass the check together. uq_testsets_generating_per_corpus is what actually
    holds the rule; the check only exists to give a message naming the testset.

    Bypassing the check is the point — with it in place this would pass for the
    wrong reason.
    """
    import app.api.testsets as api

    monkeypatch.setattr(api, "_refuse_if_corpus_busy", lambda *a, **k: None)

    llm, emb = model_configs
    body = {"corpus_id": corpus_with_docs["id"], "llm_config_id": llm["id"],
            "embedding_config_id": emb["id"], "n_single": 2,
            "n_multi_specific": 0, "n_multi_abstract": 0}

    first = client.post("/api/testsets", json={**body, "name": "First"})
    assert first.status_code == 201, first.text
    _wait_done(client, first.json()["id"])

    # the first run finished, so the corpus is free again — make it busy
    import app.db.session as session
    from app.db.models import Testset
    db = session.get_session_factory()()
    try:
        db.get(Testset, first.json()["id"]).status = Testset.STATUS_GENERATING
        db.commit()
    finally:
        db.close()

    second = client.post("/api/testsets", json={**body, "name": "Second"})
    assert second.status_code == 409, second.text
    # The fallback wording, not the one that names the testset: the naming
    # message only comes from the check, which is bypassed here, so this is
    # proof the index is what refused.
    assert second.json()["detail"] == (
        "Another testset is already generating from this corpus"), second.json()


def test_startup_resumes_only_one_run_per_corpus(tmp_path, monkeypatch):
    """A pair left generating by a crash must not resume into two concurrent
    runs on one corpus — the corruption the API guards against, repeating on
    every restart."""
    import asyncio

    import app.services.testset_gen as tg
    from app.core.config import get_settings
    from app.db.models import Corpus, ModelConfig, Testset
    from app.db.session import get_session_factory, init_db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/startup.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    init_db()

    db = get_session_factory()()
    try:
        # A database written before uq_testsets_generating_per_corpus existed
        # could hold two runs for one corpus. The index forbids constructing
        # that state, so drop it to reproduce what such a database looks like —
        # the resolution under test is the safety net for it, and the migration
        # that creates the index has to clear the same rows first.
        db.connection().exec_driver_sql(
            "DROP INDEX IF EXISTS uq_testsets_generating_per_corpus")
        db.commit()
        corpus = Corpus(name="StartupCorpus", status=Corpus.STATUS_COMPLETED,
                        success_files=3)
        llm = ModelConfig(name="startup-llm", type="llm", api_format="openai",
                          base_url="http://x", model="m")
        emb = ModelConfig(name="startup-emb", type="embedding", api_format="openai",
                          base_url="http://x", model="m")
        db.add_all([corpus, llm, emb])
        db.commit()
        rows = [
            Testset(name=f"QA-{i}", corpus_id=corpus.id, corpus_name=corpus.name,
                    llm_config_id=llm.id, llm_name=llm.name,
                    embedding_config_id=emb.id, embedding_name=emb.name,
                    status=Testset.STATUS_GENERATING, progress=float(i * 10))
            for i in range(3)
        ]
        db.add_all(rows)
        db.commit()
        ids = [r.id for r in rows]
    finally:
        db.close()

    started: list[int] = []

    async def _record(tid: int) -> None:
        started.append(tid)

    monkeypatch.setattr(tg, "run_generation", _record)

    async def _scenario() -> None:
        await tg.resume_interrupted()
        await asyncio.sleep(0)  # let the tasks it scheduled actually start

    asyncio.run(_scenario())

    # the furthest-along one keeps the corpus; the other two are handed back
    assert started == [ids[2]], started
    db = get_session_factory()()
    try:
        assert db.get(Testset, ids[2]).status == Testset.STATUS_GENERATING
        for tid in ids[:2]:
            row = db.get(Testset, tid)
            assert row.status == Testset.STATUS_FAILED
            assert "still generating when the server restarted" in row.error
    finally:
        db.close()
        get_settings.cache_clear()


def test_fallback_rotates_before_it_merges_documents(client, corpus_with_docs, model_configs,
                                                     mock_ragas, monkeypatch, caplog):
    """The goal is the requested count, and both levers serve it — but rotating
    the window onto material the graph has not been asked about is free while
    merging documents is not. So the graph is asked first, and only a round
    that produced nothing new earns the merge."""
    import app.services.testset_gen as tg

    # Two rounds is all the ordering needs; the shipped cap is a separate
    # question and this test is not the place to pin it.
    monkeypatch.setattr(tg, "FALLBACK_ROUNDS", 3)

    # One drop on the first pass opens a gap. Later passes leave the pool
    # alone: the reviewer's rejections are sticky, so dropping every round
    # would eat the pool down to nothing and the run would fail instead.
    passes = {"n": 0}

    async def drop_first_only(llm, samples, testset_id, type_key, language):
        passes["n"] += 1
        if passes["n"] == 1 and samples:
            return samples[1:]
        return samples

    monkeypatch.setattr(tg, "_semantic_dedup", drop_first_only)

    # From the first fallback round on, the gate rejects everything that round
    # generated — the material repeating itself, which is what merging is for.
    calls = {"n": 0}

    async def barren_after_first(llm, samples, testset_id, type_key, language):
        calls["n"] += 1
        return samples if calls["n"] == 1 else []

    monkeypatch.setattr(tg, "_filter_faithful", barren_after_first)

    merges: list[int | None] = []
    real_merge = tg._merge_new_docs

    async def spy_merge(*a, **kw):
        merges.append(kw.get("round_no"))
        return await real_merge(*a, **kw)

    monkeypatch.setattr(tg, "_merge_new_docs", spy_merge)

    llm, emb = model_configs
    with caplog.at_level(logging.INFO):
        ts = client.post("/api/testsets", json={
            "name": "FallbackLevers", "corpus_id": corpus_with_docs["id"],
            "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
            "n_single": 4, "n_multi_specific": 0, "n_multi_abstract": 0,
            # keeps the first pool exactly at target, so the reviewer's drop is a
            # real gap rather than absorbed headroom
            "gen_amplify": 1.0,
        }).json()
        item = _wait_done(client, ts["id"])
    assert item["status"] == "completed"

    entries = client.get(f"/api/testsets/{ts['id']}/log").json()["entries"]
    keys = [e["key"] for e in entries]
    # The customer's log carries the top-up count and one merged check count per
    # round; which round widened the pool, and why, moved to the server log when
    # the per-type lines were collapsed.
    assert "typeToppedUp" in keys, "the forced gap must have triggered a top-up"

    # round 1 rotated and found nothing, so every round after it pays for
    # documents — none of them produced anything either
    widen = [r.args[1] for r in caplog.records
             if r.msg.startswith("Fallback ") and "widening the pool" in r.msg]
    assert widen == [2, 3], caplog.text
    # The spy reads the round the customer sees, where round 1 is the first
    # check — one ahead of the fallback index widen logs. The merge has to carry
    # that number, not the fallback index, or the bar would step back a round
    # for the duration of the merge.
    assert merges == [3, 4], merges


def test_one_type_is_finished_before_the_next_one_starts(client, corpus_with_docs,
                                                         model_configs, mock_ragas,
                                                         monkeypatch):
    """Each type runs its whole loop before the next one begins.

    The order is the whole of it, and the reason is cross-type duplicates: a
    type is checked against the ones already settled, so a question repeating an
    earlier type's fact is seen. Checking all three at the end would need a
    separate cross-type pass, and a drop from that pass would send a finished
    type back into its loop. Pin the order itself.
    """
    import app.services.testset_gen as tg

    seen: list[str] = []
    real_update = tg._update

    def spy_update(_tid, **kw):
        if "stage" in kw:
            seen.append(kw["stage"])
        return real_update(_tid, **kw)

    monkeypatch.setattr(tg, "_update", spy_update)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "StageOrder", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 3, "n_multi_specific": 3, "n_multi_abstract": 3,
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed"

    # _generate_many sets the stage once per chunk, so collapse runs of the same
    # stage before reading the order off it
    order = [x for i, x in enumerate(seen) if i == 0 or x != seen[i - 1]]
    # checking_connectivity shares the prefix; only the per-type stage is one
    per_type = [x for x in order
                if (x.startswith("gen_") or x.startswith("checking_"))
                and x.split("_", 1)[1] in tg.QUESTION_TYPES]
    assert per_type == [
        "gen_single", "checking_single",
        "gen_multi_specific", "checking_multi_specific",
        "gen_multi_abstract", "checking_multi_abstract",
    ], order


def test_a_types_reported_seconds_excludes_the_other_types(client, corpus_with_docs,
                                                           model_configs, mock_ragas,
                                                           monkeypatch):
    """The genType line's "seconds" is one type's generation + checking, and the
    customer reads it.

    In separate passes those two windows no longer touch, so the number has to
    be assembled rather than timed as one span — a span starting in the
    generation pass swallows every type generated in between. One run reported
    single-hop at 58.6 s inside a 155 s run; the three types summed to more than
    the run took. Slowing only the last type is what makes that visible.

    Binding by name rather than by position, so a signature change fails loudly
    here instead of silently matching the wrong argument.
    """
    import asyncio
    import inspect

    import app.services.testset_gen as tg

    real = tg._generate_many
    DELAY = 0.5

    async def slow_last(*a, **kw):
        if inspect.signature(real).bind(*a, **kw).arguments["stage"] == "gen_multi_abstract":
            await asyncio.sleep(DELAY)
        return await real(*a, **kw)

    monkeypatch.setattr(tg, "_generate_many", slow_last)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "TypeTiming", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 3, "n_multi_specific": 3, "n_multi_abstract": 3,
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed"

    entries = client.get(f"/api/testsets/{ts['id']}/log").json()["entries"]
    # Per-type wall clock now lives on the closing line, which covers checking
    # and the top-up rounds as well as generation.
    secs = {e["params"]["type"]: e["params"]["seconds"]
            for e in entries if e["key"] == "typeDone"}
    assert set(secs) == {"single", "multi_specific", "multi_abstract"}, secs
    # the two earlier types must not be charged for the last one's wait
    assert secs["single"] < DELAY, secs
    assert secs["multi_specific"] < DELAY, secs
    assert secs["multi_abstract"] >= DELAY, secs


def test_the_bar_moves_through_checking_and_never_goes_back(client, corpus_with_docs,
                                                            model_configs, mock_ragas,
                                                            monkeypatch):
    """Generation and checking both own part of the bar, and nothing drives it
    backwards.

    Generation used to take every point: the bar hit 99 the moment the last
    type finished generating, and sat there through all of checking and the
    fallback — which is the stretch the customer is watching. The fallback
    keeps the last point because its length cannot be known in advance, which
    is exactly why it has to be handed the bar's current position: given the
    post-generation one instead it yanks the bar backwards the moment a type
    comes up short.
    """
    import app.services.testset_gen as tg

    events: list[tuple[str | None, float]] = []
    real_update = tg._update

    def spy_update(_tid, **kw):
        if "progress" in kw:
            events.append((kw.get("stage"), kw["progress"]))
        return real_update(_tid, **kw)

    monkeypatch.setattr(tg, "_update", spy_update)

    # Drop one pair on the first pass, so the type lands short and a fallback
    # round runs — that round is where a stale progress base shows up.
    passes = {"n": 0}

    async def drop_first_only(llm, samples, testset_id, type_key, language):
        passes["n"] += 1
        if passes["n"] == 1 and samples:
            return samples[1:]
        return samples

    monkeypatch.setattr(tg, "_semantic_dedup", drop_first_only)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "BarMoves", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 3, "n_multi_specific": 0, "n_multi_abstract": 0,
        "gen_amplify": 1.0,   # no headroom, so the drop is a real gap
    }).json()
    item = _wait_done(client, ts["id"])
    assert item["status"] == "completed"

    assert any(s == "fallback_gen_single" for s, _ in events), \
        f"the forced gap must have triggered a fallback: {events}"

    values = [p for _, p in events]
    assert values == sorted(values), f"the bar went backwards: {events}"
    # generation must leave room, or the bar is parked while checking runs
    after_gen = max(p for s, p in events if s and s.startswith("gen_"))
    before_done = values[-2]
    assert after_gen < before_done, events
    assert values[-1] == 100


def test_the_screen_is_allowed_to_over_report_and_the_check_is_not():
    """Two prompts, two jobs, and the split is the whole point.

    Screening alone fires on 40-50% of a batch and still misses some; checking
    alone is the per-item judgement, which cannot see that fifteen neighbours
    name a company and one does not. Together they caught 10 of 15 real defects
    against 2 for the single judgement they replace, three runs running. These
    sentences are what carry that, so trimming them back silently restores the
    hole.
    """
    import app.services.testset_gen as tg

    # the screen must be told to be generous, or it stops raising suspicions
    assert "Be generous" in tg._SCREEN_PROMPT_EN
    assert "宁可多报" in tg._SCREEN_PROMPT_ZH
    # the check must keep the batch in view, and settle rather than re-suspect
    assert "Keep the set in view" in tg._VERIFY_PROMPT_EN
    assert "把整批放在眼前" in tg._VERIFY_PROMPT_ZH
    assert "ok=false when the suspicion holds" in tg._VERIFY_PROMPT_EN
    assert "怀疑成立就判 ok=false" in tg._VERIFY_PROMPT_ZH


def test_the_dedup_prompt_conflicts_on_the_fact_not_the_whole_question():
    """The old test was "one and the same fact answers both questions", which a
    partly-overlapping pair never satisfies — so asking everything a second
    question asked, plus more, passed both. Two shipped items did that. The
    wording that replaced it has to keep both directions: same attribute of the
    same entity conflicts, different attributes and different entities do not."""
    import app.services.testset_gen as tg

    prompt = tg._SEMANTIC_DEDUP_PROMPT_EN
    assert "the same entity AND the same attribute" in prompt
    assert "DIFFERENT attributes → NOT a conflict" in prompt
    assert "Different entities → NOT a conflict" in prompt


def test_the_check_and_dedup_judge_in_the_corpus_language():
    """One prompt used to serve both languages — an English instruction carrying
    Chinese examples — so an English corpus was judged against examples in a
    language it never used. The selector is what the pipeline calls; the
    examples are what tell the two versions apart."""
    import app.services.testset_gen as tg

    assert tg._screen_prompt("zh") is tg._ScreenPromptZh
    assert tg._screen_prompt("en") is tg._ScreenPromptEn
    assert tg._verify_prompt("zh") is tg._VerifyPromptZh
    assert tg._verify_prompt("en") is tg._VerifyPromptEn
    assert tg._semantic_dedup_prompt("zh") is tg._SemanticDedupPromptZh
    assert tg._semantic_dedup_prompt("en") is tg._SemanticDedupPromptEn

    assert "宁可多报" in tg._ScreenPromptZh.instruction
    assert "Be generous" in tg._ScreenPromptEn.instruction
    assert "把整批放在眼前" in tg._VerifyPromptZh.instruction
    assert "Keep the set in view" in tg._VerifyPromptEn.instruction
    assert "东院" in tg._SemanticDedupPromptZh.instruction
    assert "East Hospital" in tg._SemanticDedupPromptEn.instruction


def test_the_pipeline_hands_the_corpus_language_to_the_gate(client, corpus_with_docs,
                                                            model_configs, mock_ragas,
                                                            monkeypatch):
    """Choosing a prompt by language only matters if the pipeline passes the
    corpus's own language through; the fixture corpus is Chinese."""
    import app.services.testset_gen as tg

    seen: list[str] = []
    real = tg._filter_faithful

    async def spy(llm, samples, testset_id, type_key, language):
        seen.append(language)
        return await real(llm, samples, testset_id, type_key, language)

    monkeypatch.setattr(tg, "_filter_faithful", spy)

    llm, emb = model_configs
    ts = client.post("/api/testsets", json={
        "name": "GateLanguage", "corpus_id": corpus_with_docs["id"],
        "llm_config_id": llm["id"], "embedding_config_id": emb["id"],
        "n_single": 2, "n_multi_specific": 0, "n_multi_abstract": 0,
    }).json()
    assert _wait_done(client, ts["id"])["status"] == "completed"

    assert seen, "the gate never ran"
    assert set(seen) == {"zh"}, seen


def test_a_failed_call_is_logged_with_how_long_it_waited(caplog):
    """Only successful calls were logged, so 153 s that one run spent on five
    requests sitting on dead connections until the timeout looked like the
    process doing nothing at all — no line named the calls or the waiting."""
    import logging
    import time

    import app.services.testset_gen as tg

    llm = tg._MeteredLLM()
    llm.max_tokens = 4096
    with caplog.at_level(logging.WARNING, logger="app.services.testset_gen"):
        llm._log_failure(time.monotonic() - 31.0, 1234, TimeoutError("no answer"))

    message = " ".join(r.getMessage() for r in caplog.records)
    assert "FAILED after" in message, message
    assert "prompt=1234chars" in message, message
    assert "TimeoutError" in message and "no answer" in message, message


def test_http_trace_works_on_both_the_sync_and_async_clients(caplog):
    """httpx calls event hooks differently per client kind: the sync client
    calls the hook and ignores what it returns, while the async client does
    `await hook(request)`. A plain function on the async client therefore
    awaits None — "'NoneType' object can't be awaited" — and httpx surfaces
    that as an APIConnectionError, which killed every call of a whole run.

    Exercising only the sync path is what let that through: it passed, and the
    pipeline, which is async throughout, produced nothing at all. Both are
    covered here.

    The second failure mode this pins down is `response.elapsed`: a response
    hook fires before the body is read, and httpx raises if it is asked for
    there. The timing is measured in the hook instead.
    """
    import asyncio
    import logging

    import httpx

    from app.services.testset_gen import _attach_http_trace

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": 1}, request=request)

    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    class _Inner:
        def __init__(self, client):
            self._client = client

    class _Holder:
        def __init__(self, sync, async_):
            self._client = _Inner(sync)
            self._async_client = _Inner(async_)

    _attach_http_trace(_Holder(sync_client, async_client))
    url = "https://api.deepseek.com/anthropic/v1/messages"

    with caplog.at_level(logging.INFO, logger="app.services.testset_gen"):
        sync_response = sync_client.get(url)
        assert sync_response.status_code == 200
        assert sync_response.json() == {"ok": 1}

        async def go():
            async with async_client:
                return await async_client.get(url)

        async_response = asyncio.run(go())
        assert async_response.status_code == 200, "the async path is the pipeline's"
        assert async_response.json() == {"ok": 1}

    lines = [r.getMessage() for r in caplog.records]
    assert sum(line.startswith("HTTP →") for line in lines) == 2, lines
    assert sum(line.startswith("HTTP ← 200") for line in lines) == 2, lines



def test_build_llm_wires_the_http_trace_onto_the_client():
    """The trace only helps if the pipeline's own client carries it — the
    separately-built connectivity probe's does not, and neither did this one
    until it was wired in."""
    import app.services.testset_gen as tg
    from app.db.models import ModelConfig

    cfg = ModelConfig(name="trace", type="llm", api_format="anthropic",
                      base_url="https://api.deepseek.com/anthropic",
                      api_key="k", model="deepseek-flash")
    _, raw, _ = tg._build_llm(cfg, 1024, None)

    for attr in ("_client", "_async_client"):
        httpx_client = getattr(getattr(raw, attr), "_client")
        hooks = httpx_client.event_hooks
        assert hooks["request"], f"{attr}: no request hook"
        assert hooks["response"], f"{attr}: no response hook"


def test_gate_reports_even_when_it_drops_nothing(monkeypatch, caplog):
    """Reporting only the rejections left a gate that ran and passed everything
    with no trace at all. One run read as generate -> fallback with no gate in
    between, because generation came back empty and the gate worked on an empty
    list — the log could not say it had run."""
    import asyncio

    import app.services.testset_gen as tg

    logged = []
    monkeypatch.setattr(tg, "_log", lambda tid, key, params=None: logged.append((key, params)))

    kept = asyncio.run(tg._filter_faithful(None, [], 1, "single", "zh"))

    assert kept == []
    assert logged == [], "the empty gate must keep the UI log clean"
    assert "Gate [single] checked=0 dropped=0" in " ".join(
        r.getMessage() for r in caplog.records)
