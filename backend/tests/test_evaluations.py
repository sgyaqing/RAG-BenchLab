"""Tests for RAG system adapters (lifecycle, copy, smart fill) and the
evaluation pipeline."""

import asyncio
import json
from types import SimpleNamespace
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import init_db
from app.main import create_app
from app.services import adapter_assist, eval_run, rag_client


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """A database of its own: tables and all, but no rows.

    The tests that ask for this drive eval_run directly, with no app and no
    request to hang the setup off. Without it they query data/rag_benchlab.db —
    which has tables only on a machine where the app has been started at least
    once, so they passed on the developer's box and failed on a fresh clone and
    inside the container, both times as "no such table: eval_runs".
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/isolated.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    init_db()
    yield
    get_settings.cache_clear()


RAG_PAYLOAD = {
    "mode": "manual",
    "name": "MyRAG",
    "base_url": "http://rag.local/query",
    "headers": "{}",
    "body_template": '{"question": "{{question}}"}',
    "answer_path": "data.answer",
    "contexts_path": "data.contexts[].text",
}

FAKE_RESPONSE = {"data": {"answer": "这是回答", "contexts": [{"text": "片段1"}]}}


@pytest.fixture()
def fake_probe(monkeypatch):
    """Manual create/edit ends in a probe; fake the target system."""

    async def _probe(cfg, question, url=None):
        return FAKE_RESPONSE, 0.05

    monkeypatch.setattr(adapter_assist, "_probe", _probe)


def _wait_adapter(client, config_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        items = client.get("/api/rag-systems").json()["items"]
        row = [i for i in items if i["id"] == config_id][0]
        if row["status"] in ("completed", "failed"):
            return row
        time.sleep(0.1)
    raise TimeoutError("adapter did not reach a terminal state")


def _create_adapter(client, payload=None) -> dict:
    resp = client.post("/api/rag-systems", json=payload or RAG_PAYLOAD)
    assert resp.status_code == 201, resp.json()
    return _wait_adapter(client, resp.json()["id"])


# ---------------------------------------------------------------------------
# rag_client unit tests
# ---------------------------------------------------------------------------


def test_metrics_endpoint_lists_every_selectable_key(client):
    """A caller has to see what it can ask for, and which of those need an
    adapter with a contexts path — the create endpoint rejects those otherwise,
    and a 400 is a late way to find out."""
    from app.services.eval_run import CONTEXT_METRICS, METRIC_KEYS

    body = client.get("/api/evaluations/metrics").json()

    assert [m["key"] for m in body["metrics"]] == list(METRIC_KEYS)
    assert ({m["key"] for m in body["metrics"] if m["requires_contexts"]}
            == set(CONTEXT_METRICS))


def test_literal_routes_are_not_shadowed_by_the_id_route(client):
    """`/metrics` and `/check-name` are literal paths declared above `/{run_id}`.

    Declared the other way round the parameterised route matches first and the
    literal one answers 422 — measured while adding `/metrics`, by putting it on
    the wrong side of the list.
    """
    assert client.get("/api/evaluations/metrics").status_code == 200
    assert client.get("/api/evaluations/check-name?name=x").status_code == 200


def test_every_metric_key_builds_and_relevancy_draws_once():
    """Every selectable metric must survive instantiation, and response
    relevancy must draw one synthetic question rather than three.

    Fixed twice over: a key handled outside the class mapping was filtered out
    entirely — the metric silently stopped being measured — and ragas averages
    `strictness` draws, where the default of 3 costs one request on an
    OpenAI-format judge (n=3 in a single call) but three on an Anthropic-format
    one, which has no `n` for ragas to set.
    """
    from app.services.eval_run import METRIC_KEYS, _RAGAS_COLUMN, _build_metric_instances

    built = _build_metric_instances(list(METRIC_KEYS))
    # _RAGAS_COLUMN lists only the keys ragas renames; the rest keep their own
    assert [m.name for m in built] == [_RAGAS_COLUMN.get(k, k) for k in METRIC_KEYS]

    relevancy = next(m for m in built if m.name == "answer_relevancy")
    assert relevancy.strictness == 1


def test_hit_at_k_reads_the_curve_off_the_verdicts():
    """One sequence answers three questions: did it find anything, is the hit
    deep, and how much would a smaller budget cost."""
    from app.api.evaluations import _hit_at_k_from

    # first item's opening chunk is useful, second item's third, third found nothing
    curve = _hit_at_k_from([[1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

    assert curve == {"1": 0.3333, "2": 0.3333, "3": 0.6667,
                     "4": 0.6667, "5": 0.6667, "6": 0.6667}
    values = [curve[str(k)] for k in range(1, 7)]
    assert values == sorted(values), "the curve cannot fall as k grows"
    # Hit@1 < Hit@3 says a hit sits deep (rerank territory); Hit@3 == Hit@6 says
    # a smaller budget loses nothing. The item that found nothing keeps the
    # ceiling below 1 — otherwise the run would read as fully covered.
    assert curve["1"] < curve["3"] and curve["3"] == curve["6"] < 1


def test_hit_at_k_reads_short_items_at_all_they_returned():
    """The published rule is "the first k chunks it gave, or everything it gave".

    Items that returned fewer chunks are read at their own length rather than
    dropped or padded, so one curve can cover a run whose items differ.
    """
    from app.api.evaluations import _hit_at_k_from

    curve = _hit_at_k_from([[0, 1], [0, 0, 0, 1]])
    assert curve == {"1": 0.0, "2": 0.5, "3": 0.5, "4": 1.0}
    assert _hit_at_k_from([]) == {}


def test_render_body_escapes_question():
    body = rag_client.render_body(
        '{"q": "{{question}}"}', '什么是"可转债"？\n第二行')
    assert body == {"q": '什么是"可转债"？\n第二行'}


def test_extract_path_variants():
    data = {
        "data": {
            "answer": "42",
            "reference": {"chunks": [{"content": "a"}, {"content": "b"}]},
        },
        "choices": [{"message": {"content": "hi"}}],
    }
    assert rag_client.extract_answer(data, "data.answer") == "42"
    assert rag_client.extract_contexts(data, "data.reference.chunks[].content") == ["a", "b"]
    assert rag_client.extract_path(data, "choices[0].message.content") == "hi"
    assert rag_client.extract_answer(data, "data.missing") is None
    assert rag_client.extract_contexts(data, "") == []
    assert rag_client.extract_contexts(data, None) == []


def test_find_contexts_path_skips_prompt():
    """The scanner must pick reference chunks, never the prompt/answer."""
    data = {
        "data": {
            "answer": "The answer you are looking for is not found in the dataset!",
            "prompt": "You are an assistant. Question: ... 3000 chars of instructions",
            "reference": {
                "chunks": [
                    {"content": "片段1", "similarity": 0.9},
                    {"content": "片段2", "similarity": 0.8},
                ],
                # IDs under a reference-ish key must NOT win over content
                "doc_aggs": [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}],
            },
        }
    }
    assert adapter_assist._find_contexts_path(data) == "data.reference.chunks[].content"
    # no context-ish list -> None (never falls back to prompt)
    assert adapter_assist._find_contexts_path({"data": {"answer": "x", "prompt": "y"}}) is None


def test_find_contexts_path_rejects_pathish_values():
    """Lists of file paths/URLs are never contexts, even under context-ish keys."""
    data = {"data": {"chunks": ["dataset/x/公告A.md", "dataset/x/公告B.md"]}}
    assert adapter_assist._find_contexts_path(data) is None
    # CJK prose has no spaces but has punctuation — must pass
    assert adapter_assist._contexts_look_real(["本公司于2015年10月9日召开现场会议。"])
    assert not adapter_assist._contexts_look_real(["dataset/x/a.md", "dataset/x/b.md"])
    # majority rules: half real, half pathish -> accepted
    assert adapter_assist._contexts_look_real(["dataset/x/a.md", "本公司召开会议，内容如下。"])


def test_find_contexts_path_skips_camelcase_ids():
    """FastGPT's quoteList[].sourceId is a file path, not chunk text: camelCase
    ID keys must be excluded even though 'source' is a context-ish word."""
    data = {
        "responseData": [{
            "moduleType": "datasetSearchNode",
            "quoteList": [
                {"q": "正文片段一", "sourceId": "dataset/abc/文件A.md", "score": 0.9},
                {"q": "正文片段二", "sourceId": "dataset/abc/文件B.md", "score": 0.8},
            ],
        }],
        "choices": [{"message": {"content": "回答"}}],
    }
    assert adapter_assist._find_contexts_path(data) == "responseData[].quoteList[].q"


def test_ask_llm_anthropic_sends_the_system_prompt(monkeypatch):
    """Anthropic takes the system prompt as a top-level parameter, not as a
    message. It used to be dropped (only messages[1:] was sent), which left the
    model holding a bare example config and no instructions — the Dify adapter
    answered "you haven't included a specific question to ask the RAG system"
    and failed on a missing field."""
    sent: dict = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": "{}"}], "usage": {}}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            sent["url"] = url
            sent["json"] = json
            return FakeResponse()

    monkeypatch.setattr(adapter_assist.httpx, "AsyncClient", FakeClient)
    cfg = SimpleNamespace(
        api_format="anthropic",
        base_url="https://api.deepseek.com/anthropic",
        api_key="sk-test",
        model="deepseek-v4-flash",
        thinking_param=None,
    )

    content, usage = asyncio.run(adapter_assist._ask_llm(cfg, "SYSTEM PROMPT", "USER MSG"))

    # The /v1 is what these endpoints expect; it was missing here once, and
    # this assertion had been written to match the missing one.
    assert sent["url"] == "https://api.deepseek.com/anthropic/v1/messages"
    assert sent["json"]["system"] == "SYSTEM PROMPT"
    assert sent["json"]["messages"] == [{"role": "user", "content": "USER MSG"}]
    assert content == "{}"
    assert usage == {}


def test_render_headers_api_key_placeholder():
    headers = {"Authorization": "Bearer {{api_key}}", "X-App": "rb"}
    assert rag_client.render_headers(headers, "sk-1") == {
        "Authorization": "Bearer sk-1", "X-App": "rb"}
    # no key -> the auth header is dropped entirely
    assert rag_client.render_headers(headers, None) == {"X-App": "rb"}


# ---------------------------------------------------------------------------
# adapter lifecycle
# ---------------------------------------------------------------------------


def test_manual_create_runs_simple_test(client, fake_probe):
    row = _create_adapter(client)
    assert row["status"] == "completed"
    assert row["completed_at"] is not None
    log = client.get(f"/api/rag-systems/{row['id']}/log").json()
    assert any(e["key"] == "probeResult" for e in log["entries"])
    assert log["entries"][-1]["key"] == "done"


def test_simple_test_discovers_contexts_path(client, fake_probe):
    """Manual create with blank contexts path -> auto-discovered from response."""
    row = _create_adapter(client, {**RAG_PAYLOAD, "name": "AutoCtx",
                                   "contexts_path": None})
    assert row["status"] == "completed"
    # FAKE_RESPONSE has data.contexts[].text
    assert row["contexts_path"] == "data.contexts[].text"


def test_manual_create_fails_when_probe_fails(client, monkeypatch):
    async def _probe_fail(cfg, question, url=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(adapter_assist, "_probe", _probe_fail)
    row = _create_adapter(client)
    assert row["status"] == "failed"
    assert "refused" in row["error"]


def test_failed_log_carries_actionable_code(client, monkeypatch):
    """A 404 probe failure is classified so the UI can show targeted guidance
    instead of a raw HTTPStatusError stack."""
    import httpx

    async def _probe_404(cfg, question, url=None):
        req = httpx.Request("POST", url or cfg.base_url)
        raise httpx.HTTPStatusError(
            "404", request=req, response=httpx.Response(404, request=req))

    monkeypatch.setattr(adapter_assist, "_probe", _probe_404)
    row = _create_adapter(client)
    assert row["status"] == "failed"
    log = client.get(f"/api/rag-systems/{row['id']}/log").json()
    failed = [e for e in log["entries"] if e["key"] == "failed"]
    assert failed and failed[0]["params"]["code"] == "notFound"
    # the failed address is recorded for debugging
    assert failed[0]["params"]["urls"] == RAG_PAYLOAD["base_url"]


def test_probe_question_follows_ui_locale(client, fake_probe):
    row = _create_adapter(client, {**RAG_PAYLOAD, "name": "EnProbe", "lang": "en"})
    assert row["status"] == "completed"
    log = client.get(f"/api/rag-systems/{row['id']}/log").json()
    probe = [e for e in log["entries"] if e["key"] == "probeResult"][0]
    assert probe["params"]["question"] == adapter_assist.DEFAULT_PROBE_QUESTION_EN
    # default (no lang) stays Chinese
    row2 = _create_adapter(client, {**RAG_PAYLOAD, "name": "ZhProbe"})
    log2 = client.get(f"/api/rag-systems/{row2['id']}/log").json()
    probe2 = [e for e in log2["entries"] if e["key"] == "probeResult"][0]
    assert probe2["params"]["question"] == adapter_assist.DEFAULT_PROBE_QUESTION


def test_ragas_column_aliases_cover_all_metrics():
    """ragas 0.4.x names result columns after metric classes, not our keys.
    Pin the alias map so a silent column-name drift can't drop scores again."""
    for k in eval_run.METRIC_KEYS:
        assert eval_run._RAGAS_COLUMN.get(k, k)  # resolves to a column name
    assert eval_run._RAGAS_COLUMN["response_relevancy"] == "answer_relevancy"
    assert eval_run._RAGAS_COLUMN["context_precision"] == "llm_context_precision_with_reference"
    # faithfulness / context_recall pass through (ragas column == our key)
    assert "faithfulness" not in eval_run._RAGAS_COLUMN
    assert "context_recall" not in eval_run._RAGAS_COLUMN


def test_manual_create_validates_template(client):
    resp = client.post("/api/rag-systems", json={
        **RAG_PAYLOAD, "body_template": '{"q": "no placeholder"}',
    })
    assert resp.status_code == 400


def test_url_scheme_normalized(client, fake_probe):
    assert rag_client.normalize_url("localhost:8080/query") == "http://localhost:8080/query"
    assert rag_client.normalize_url("https://a.b/c") == "https://a.b/c"


def test_endpoint_and_key_not_persisted(client, fake_probe):
    """Adapters persist only templates/paths; endpoint+key are cleared once
    the configuration reaches a terminal state."""
    row = _create_adapter(client)
    assert row["status"] == "completed"
    assert row["base_url"] == "" and row["api_key"] is None


def test_name_uniqueness_case_insensitive(client, fake_probe):
    _create_adapter(client)
    resp = client.post("/api/rag-systems", json={**RAG_PAYLOAD, "name": "myrag"})
    assert resp.status_code == 409


def test_edit_keeps_deliberately_cleared_contexts_path(client, fake_probe):
    """Editing with a blank contexts path must NOT auto-discover one — a
    cleared field on edit is a deliberate choice (create still discovers)."""
    row = _create_adapter(client)  # discovers data.contexts[].text at create
    assert row["contexts_path"] == "data.contexts[].text"
    resp = client.put(f"/api/rag-systems/{row['id']}", json={
        "name": row["name"], "base_url": RAG_PAYLOAD["base_url"],
        "headers": "{}", "body_template": RAG_PAYLOAD["body_template"],
        "answer_path": "data.answer", "contexts_path": None,
    })
    assert resp.status_code == 200
    updated = _wait_adapter(client, row["id"])
    assert updated["status"] == "completed"
    assert updated["contexts_path"] is None  # stayed cleared


def test_edit_never_rewrites_contexts_path(client, fake_probe):
    """Edit probes the config as given: even an implausible contexts path is
    NOT auto-relocated (rewrites belong to create/smart-fill only)."""
    row = _create_adapter(client)
    resp = client.put(f"/api/rag-systems/{row['id']}", json={
        "name": row["name"], "base_url": RAG_PAYLOAD["base_url"],
        "headers": "{}", "body_template": RAG_PAYLOAD["body_template"],
        "answer_path": "data.answer", "contexts_path": "data.answer",
    })
    assert resp.status_code == 200
    updated = _wait_adapter(client, row["id"])
    # "data.answer" extracts a single non-context value; create-time logic
    # would relocate it, edit must leave it untouched
    assert updated["contexts_path"] == "data.answer"


def test_edit_triggers_simple_test(client, fake_probe):
    row = _create_adapter(client)
    resp = client.put(f"/api/rag-systems/{row['id']}", json={
        "name": "MyRAG", "base_url": "http://rag.local/v2/query",
        "headers": "{}", "body_template": '{"q": "{{question}}"}',
        "answer_path": "data.answer", "contexts_path": None,
    })
    assert resp.status_code == 200
    updated = _wait_adapter(client, row["id"])
    assert updated["status"] == "completed"
    assert updated["base_url"] == ""  # endpoint not persisted
    log = client.get(f"/api/rag-systems/{row['id']}/log").json()
    assert any(e["key"] == "edited" for e in log["entries"])


def test_smart_fill_flow(client, monkeypatch):
    """Smart fill: fake LLM writes the config, fake probe verifies it."""
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    async def fake_fetch(url):
        return "API docs: POST /v1/chat ..."

    async def fake_ask(cfg, system, user):
        return json.dumps({
            "endpoint": "/v1/chat",
            "headers": {"Authorization": "Bearer {{api_key}}"},
            "body_template": '{"query": "{{question}}"}',
            "answer_path": "data.answer",
            "contexts_path": "data.contexts[].text",
            "platform": "custom",
        }), {"prompt_tokens": 100, "completion_tokens": 50}

    probed = []

    async def fake_probe(cfg, question, url=None):
        probed.append(url)
        return FAKE_RESPONSE, 0.05

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(adapter_assist, "fetch_doc_text", fake_fetch)
    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "SmartOne", "base_url": "http://rag.local",
        "api_key": "k", "llm_config_id": llm["id"], "platform_hint": "ExampleRAG",
    })
    assert resp.status_code == 201
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    assert probed == ["http://rag.local/v1/chat"]  # bare host -> generated path
    assert row["answer_path"] == "data.answer"
    log = client.get(f"/api/rag-systems/{row['id']}/log").json()
    keys = [e["key"] for e in log["entries"]]
    assert keys[0] == "connectivityOk"
    assert "internalKnowledge" in keys and "probeResult" in keys
    # This config has a contexts path, so no warning. It once got one anyway:
    # the pipeline's config object predates the generated config being written,
    # so its contexts_path was still None while the probe was extracting chunks.
    assert "noContexts" not in keys


def test_an_adapter_with_no_contexts_says_so_in_its_log(client, monkeypatch):
    """A system that never returns the retrieved chunks is a valid adapter, not
    an error — but it silently scores fewer metrics, and the customer should
    hear that where they can still fix it (by hand, in the adapter)."""
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM-for-contexts", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    async def fake_ask(cfg, system, user):
        return json.dumps({
            "endpoint": "/v1/chat",
            "headers": {},
            "body_template": '{"query": "{{question}}"}',
            "answer_path": "data.answer",
            "contexts_path": None,
            "platform": "custom",
        }), {}

    async def fake_probe(cfg, question, url=None):
        # No contexts in the response, so none can be discovered either.
        return {"data": {"answer": "这是回答"}}, 0.05

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "NoContexts", "base_url": "http://rag.local",
        "api_key": "k", "llm_config_id": llm["id"], "platform_hint": "ExampleRAG",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    assert not row["contexts_path"]

    entries = client.get(f"/api/rag-systems/{row['id']}/log").json()["entries"]
    keys = [e["key"] for e in entries]
    assert "noContexts" in keys
    # It reads as a note about the result, so it sits next to the completion,
    # not among the steps that produced it.
    assert keys[-2:] == ["noContexts", "done"]


def test_smart_fill_prefers_user_url_with_path(client, monkeypatch):
    """When the user's address already carries a path, it is probed first and
    kept as-is (no path completion)."""
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    async def fake_ask(cfg, system, user):
        return json.dumps({
            "endpoint": "/v1/chat",
            "headers": {},
            "body_template": '{"query": "{{question}}"}',
            "answer_path": "data.answer",
            "contexts_path": None,
            "platform": "custom",
        }), {}

    probed_urls = []

    async def fake_probe(cfg, question, url=None):
        probed_urls.append(url)
        return FAKE_RESPONSE, 0.05

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "FullURL",
        "base_url": "http://rag.local/api/v1/chats/abc/completions",
        "llm_config_id": llm["id"], "platform_hint": "ExampleRAG",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    # user URL kept verbatim; generated /v1/chat never probed
    assert probed_urls == ["http://rag.local/api/v1/chats/abc/completions"]
    assert row["base_url"] == ""  # endpoint not persisted after completion


def test_smart_fill_bare_host_failure_logs_guide(client, monkeypatch):
    """Bare host + all probes fail -> failed, with an actionable guide entry."""
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    async def fake_ask(cfg, system, user):
        return json.dumps({
            "endpoint": "/api/v1/chats/<chat_id>/completions",
            "headers": {}, "body_template": '{"question": "{{question}}"}',
            "answer_path": "data.answer", "contexts_path": None, "platform": "ragflow",
        }), {}

    async def fake_probe(cfg, question, url=None):
        raise RuntimeError("404 Client Error")

    async def fake_chat_url(config_id, host, api_key):
        return None

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)
    # RAGFlow's chat_id discovery is a real GET to {host}/api/v1/chats and is
    # not reached through _probe, so mocking _probe alone left this test making
    # an actual request to 192.168.1.10. On a laptop that address fails fast
    # (unreachable private host) and the test passed; in a container with no
    # route it hung until the 15 s client timeout, past _wait_adapter's 10 s.
    monkeypatch.setattr(adapter_assist, "_ragflow_chat_url", fake_chat_url)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "BareHost", "base_url": "http://192.168.1.10",
        "llm_config_id": llm["id"], "platform_hint": "RAGFlow",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "failed"
    log = client.get(f"/api/rag-systems/{row['id']}/log").json()
    keys = [e["key"] for e in log["entries"]]
    assert "guideFullUrl" in keys
    assert keys[-1] == "failed"


def _smart_fill_prelude(client, monkeypatch):
    """Shared fixture for smart-fill rescue tests: fill-LLM config + fake
    connectivity check. Returns the created LLM config dict."""
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    return llm


def test_docs_rescue_after_probe_failure(client, monkeypatch):
    """Probe fails with internal knowledge -> rescue asks the LLM for the
    official docs URL, fetches it, regenerates and succeeds."""
    llm = _smart_fill_prelude(client, monkeypatch)

    config_v1 = json.dumps({
        "endpoint": "/v1/chat", "headers": {},
        "body_template": '{"question": "{{question}}"}',
        "answer_path": "data.answer", "contexts_path": None, "platform": "custom",
    })
    config_v2 = json.dumps({
        "endpoint": "/api/v1/chat/completions", "headers": {},
        "body_template": '{"question": "{{question}}"}',
        "answer_path": "data.answer", "contexts_path": "data.contexts[].text",
        "platform": "custom",
    })
    config_calls = []

    async def fake_ask(cfg, system, user):
        if "official documentation" in system:  # docs-URL prompt
            return "https://docs.example.com/api/chat", {}
        config_calls.append(user)
        return (config_v1 if len(config_calls) == 1 else config_v2), {}

    fetched = []

    async def fake_fetch(url):
        fetched.append(url)
        return "SomeRAG API docs: POST /api/v1/chat/completions"

    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "fetch_doc_text", fake_fetch)

    probed = []

    async def fake_probe(cfg, question, url=None):
        probed.append(url)
        if "/api/v1/chat/completions" in (url or cfg.base_url):
            return FAKE_RESPONSE, 0.05
        raise RuntimeError("404 Client Error")

    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "Rescue", "base_url": "http://rag.local",
        "api_key": "k", "llm_config_id": llm["id"], "platform_hint": "SomeRAG",
    })
    assert resp.status_code == 201
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    # the LLM named the docs URL only in the rescue round
    assert fetched == ["https://docs.example.com/api/chat"]
    assert "http://rag.local/api/v1/chat/completions" in probed


def test_body_fix_on_http_400_validation_error(client, monkeypatch):
    """A 400 with a machine-readable payload (FastGPT-style zodError: body had
    an empty appId) triggers the LLM body-fix round; the fixed body probes OK."""
    llm = _smart_fill_prelude(client, monkeypatch)

    config = json.dumps({
        "endpoint": "/api/v1/chat/completions",
        "headers": {"Authorization": "Bearer {{api_key}}"},
        "body_template": '{"appId": "", "messages": [{"role": "user", "content": "{{question}}"}], "stream": false}',
        "answer_path": "data.answer", "contexts_path": "data.contexts[].text",
        "platform": "custom",
    })
    fixed_body = '{"messages": [{"role": "user", "content": "{{question}}"}], "stream": false}'

    async def fake_ask(cfg, system, user):
        if "chat endpoint failed" in system:  # _BODY_FIX_PROMPT
            return fixed_body, {}
        return config, {}

    async def fake_probe(cfg, question, url=None):
        if "appId" in cfg.body_template:
            raise adapter_assist.ProbeHTTPError(
                400, '{"zodError": [{"path": ["appId"], "code": "invalid_format"}]}')
        return FAKE_RESPONSE, 0.05

    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "Fix400", "base_url": "http://rag.local",
        "api_key": "k", "llm_config_id": llm["id"], "platform_hint": "SomeRAG",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    assert row["body_template"] == fixed_body


def test_docs_rescue_corrective_round_on_wrong_url(client, monkeypatch):
    """The LLM first names a WRONG docs site (e.g. FastGPT -> fastai): the
    fetched page does not mention the platform, so the rescue feeds that back
    and asks again; the second URL is accepted."""
    llm = _smart_fill_prelude(client, monkeypatch)

    config_v1 = json.dumps({
        "endpoint": "/v1/chat", "headers": {},
        "body_template": '{"question": "{{question}}"}',
        "answer_path": "data.answer", "contexts_path": None, "platform": "custom",
    })
    config_v2 = json.dumps({
        "endpoint": "/api/v1/chat/completions", "headers": {},
        "body_template": '{"question": "{{question}}", "detail": true}',
        "answer_path": "data.answer", "contexts_path": "data.contexts[].text",
        "platform": "custom",
    })
    docs_asks = []
    config_calls = []

    async def fake_ask(cfg, system, user):
        if "official documentation" in system:
            docs_asks.append(user)
            if len(docs_asks) == 1:
                return "https://docs.wrong-site.example.com/api", {}
            return "https://docs.example.com/api/chat", {}
        config_calls.append(user)
        return (config_v1 if len(config_calls) == 1 else config_v2), {}

    fetched = []

    async def fake_fetch(url):
        fetched.append(url)
        if "wrong-site" in url:
            return "fastai: a deep learning library for PyTorch"  # wrong platform
        return "SomeRAG API docs: POST /api/v1/chat/completions"

    async def fake_probe(cfg, question, url=None):
        if "/api/v1/chat/completions" in (url or cfg.base_url):
            return FAKE_RESPONSE, 0.05
        raise RuntimeError("404 Client Error")

    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "fetch_doc_text", fake_fetch)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "Corrective", "base_url": "http://rag.local",
        "api_key": "k", "llm_config_id": llm["id"], "platform_hint": "SomeRAG",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    # the wrong URL was rejected by content validation; feedback went into the
    # second ask; the second URL was used
    assert fetched == ["https://docs.wrong-site.example.com/api",
                       "https://docs.example.com/api/chat"]
    assert "not the official API documentation" in docs_asks[1]


def test_docs_rescue_when_contexts_missing(client, monkeypatch):
    """Probe works but no contexts -> rescue regenerates with the detail flag
    and the contexts path gets filled."""
    llm = _smart_fill_prelude(client, monkeypatch)
    no_ctx = {"data": {"answer": "这是回答"}}  # no context-ish keys anywhere

    config_v1 = json.dumps({
        "endpoint": "/v1/chat", "headers": {},
        "body_template": '{"question": "{{question}}"}',
        "answer_path": "data.answer", "contexts_path": None, "platform": "custom",
    })
    config_v2 = json.dumps({
        "endpoint": "/v1/chat", "headers": {},
        "body_template": '{"question": "{{question}}", "detail": true}',
        "answer_path": "data.answer", "contexts_path": "data.contexts[].text",
        "platform": "custom",
    })
    config_calls = []

    async def fake_ask(cfg, system, user):
        if "official documentation" in system:
            return "https://docs.example.com/api/chat", {}
        config_calls.append(user)
        return (config_v1 if len(config_calls) == 1 else config_v2), {}

    async def fake_fetch(url):
        return "SomeRAG API docs: pass detail=true to get reference chunks"

    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "fetch_doc_text", fake_fetch)

    async def fake_probe(cfg, question, url=None):
        if "detail" in cfg.body_template:
            return FAKE_RESPONSE, 0.05
        return no_ctx, 0.05

    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "NoCtx", "base_url": "http://rag.local",
        "api_key": "k", "llm_config_id": llm["id"], "platform_hint": "SomeRAG",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    assert row["contexts_path"] == "data.contexts[].text"
    assert "detail" in row["body_template"]


def test_docs_rescue_keeps_working_config_when_not_better(client, monkeypatch):
    """Contexts missing, rescue round still yields no contexts -> the original
    working config is restored (rescue must be strictly better)."""
    llm = _smart_fill_prelude(client, monkeypatch)
    no_ctx = {"data": {"answer": "这是回答"}}

    config_v1 = json.dumps({
        "endpoint": "/v1/chat", "headers": {},
        "body_template": '{"question": "{{question}}"}',
        "answer_path": "data.answer", "contexts_path": None, "platform": "custom",
    })
    config_v2 = json.dumps({
        "endpoint": "/v1/chat", "headers": {},
        "body_template": '{"question": "{{question}}", "detail": true}',
        "answer_path": "data.answer", "contexts_path": "data.missing[].text",
        "platform": "custom",
    })
    config_calls = []

    async def fake_ask(cfg, system, user):
        if "official documentation" in system:
            return "https://docs.example.com/api/chat", {}
        config_calls.append(user)
        return (config_v1 if len(config_calls) == 1 else config_v2), {}

    async def fake_fetch(url):
        return "SomeRAG docs"

    async def fake_probe(cfg, question, url=None):
        return no_ctx, 0.05  # never any contexts, even with detail=true

    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "fetch_doc_text", fake_fetch)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "NotBetter", "base_url": "http://rag.local",
        "api_key": "k", "llm_config_id": llm["id"], "platform_hint": "SomeRAG",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    # rescue not accepted: original body restored, contexts stay empty
    assert row["body_template"] == '{"question": "{{question}}"}'
    assert not row["contexts_path"]


def test_docs_rescue_unknown_url_falls_back(client, monkeypatch):
    """LLM does not know the docs URL -> rescue is skipped silently; the
    working-but-contextless config is kept."""
    llm = _smart_fill_prelude(client, monkeypatch)
    no_ctx = {"data": {"answer": "这是回答"}}

    async def fake_ask(cfg, system, user):
        if "official documentation" in system:
            return "UNKNOWN", {}
        return json.dumps({
            "endpoint": "/v1/chat", "headers": {},
            "body_template": '{"question": "{{question}}"}',
            "answer_path": "data.answer", "contexts_path": None, "platform": "custom",
        }), {}

    fetch_calls = []

    async def fake_fetch(url):
        fetch_calls.append(url)
        return "docs"

    async def fake_probe(cfg, question, url=None):
        return no_ctx, 0.05

    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "fetch_doc_text", fake_fetch)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "NoDocs", "base_url": "http://rag.local",
        "api_key": "k", "llm_config_id": llm["id"], "platform_hint": "ObscureRAG",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    assert not row["contexts_path"]
    assert fetch_calls == []  # never fetched anything


def test_smart_fill_recovers_from_sse(client, monkeypatch):
    """SSE answer -> agent injects stream:false into the template and retries."""
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    async def fake_ask(cfg, system, user):
        # LLM forgets "stream": false in the template (the real-world failure)
        return json.dumps({
            "endpoint": "/api/v1/chats/x/completions",
            "headers": {}, "body_template": '{"question": "{{question}}"}',
            "answer_path": "data.answer", "contexts_path": None, "platform": "ragflow",
        }), {}

    calls = {"n": 0}

    async def fake_probe(cfg, question, url=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise adapter_assist.StreamResponseError(url)
        return FAKE_RESPONSE, 0.05

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "SSE",
        "base_url": "http://rag.local/api/v1/chats/x/completions",
        "llm_config_id": llm["id"], "platform_hint": "RAGFlow",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    assert calls["n"] == 2  # first attempt streamed, retried with stream off
    assert json.loads(row["body_template"])["stream"] is False
    log = client.get(f"/api/rag-systems/{row['id']}/log").json()
    assert any(e["key"] == "probeResult" for e in log["entries"])


def test_smart_fill_recovers_from_error_payload(client, monkeypatch):
    """HTTP 200 with an error payload -> body-fix round using the API error."""
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    asks = {"n": 0}

    async def fake_ask(cfg, system, user):
        asks["n"] += 1
        if asks["n"] == 1:
            # generation: wrong field name for RAGFlow
            return json.dumps({
                "endpoint": "/api/v1/chats/x/completions",
                "headers": {}, "body_template": '{"query": "{{question}}", "stream": false}',
                "answer_path": "data.answer", "contexts_path": None, "platform": "ragflow",
            }), {}
        # body-fix call
        return '{"question": "{{question}}", "stream": false}', {}

    probes = {"n": 0}

    async def fake_probe(cfg, question, url=None):
        # The real _probe turns error payloads into ApiPayloadError.
        probes["n"] += 1
        if probes["n"] == 1:
            raise adapter_assist.ApiPayloadError("required argument are missing: question")
        return FAKE_RESPONSE, 0.05

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "ErrPayload",
        "base_url": "http://rag.local/api/v1/chats/x/completions",
        "llm_config_id": llm["id"], "platform_hint": "RAGFlow",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    assert json.loads(row["body_template"])["question"] == "{{question}}"
    log = client.get(f"/api/rag-systems/{row['id']}/log").json()
    keys = [e["key"] for e in log["entries"]]
    assert "done" in keys


def test_smart_fill_discovers_ragflow_chat_id(client, monkeypatch):
    """Bare host + RAGFlow hint + exactly one assistant -> URL auto-completed."""
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    async def fake_ask(cfg, system, user):
        return json.dumps({
            "endpoint": "/api/v1/chats/<chat_id>/completions",
            "headers": {}, "body_template": '{"question": "{{question}}", "stream": false}',
            "answer_path": "data.answer", "contexts_path": None, "platform": "ragflow",
        }), {}

    async def fake_chat_url(config_id, host, api_key):
        return "http://192.168.1.10/api/v1/chats/abc/completions"

    probed = []

    async def fake_probe(cfg, question, url=None):
        probed.append(url)
        return FAKE_RESPONSE, 0.05

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    monkeypatch.setattr(adapter_assist, "_ragflow_chat_url", fake_chat_url)
    monkeypatch.setattr(adapter_assist, "_probe", fake_probe)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "BareRAGFlow", "base_url": "http://192.168.1.10",
        "llm_config_id": llm["id"], "platform_hint": "RAGFlow",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "completed"
    assert probed == ["http://192.168.1.10/api/v1/chats/abc/completions"]


def test_smart_fill_unsupported_system(client, monkeypatch):
    llm = client.post("/api/model-configs", json={
        "name": "FillLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "m",
    }).json()

    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    async def fake_ask(cfg, system, user):
        return json.dumps({"unsupported": "async polling API"}), {}

    monkeypatch.setattr(adapter_assist.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)

    resp = client.post("/api/rag-systems", json={
        "mode": "smart", "name": "AsyncRAG", "base_url": "http://rag.local",
        "llm_config_id": llm["id"], "platform_hint": "AsyncRAG",
    })
    row = _wait_adapter(client, resp.json()["id"])
    assert row["status"] == "failed"
    assert "async polling" in row["error"]


# ---------------------------------------------------------------------------
# evaluation pipeline
# ---------------------------------------------------------------------------


@pytest.fixture()
def judge_configs(client):
    llm = client.post("/api/model-configs", json={
        "name": "JudgeLLM", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-x", "model": "judge",
    }).json()
    emb = client.post("/api/model-configs", json={
        "name": "JudgeEmb", "type": "embedding", "api_format": "openai",
        "base_url": "https://example.com/v1", "model": "bge-m3",
    }).json()
    return llm, emb


@pytest.fixture()
def testset_with_items(client, judge_configs):
    """A completed testset with 3 items, inserted directly."""
    from app.db.models import Testset, TestsetItem
    from app.db.session import get_session_factory

    llm, emb = judge_configs
    db = get_session_factory()()
    ts = Testset(name="TS", corpus_id=1, corpus_name="C", llm_config_id=llm["id"],
                 llm_name="L", embedding_config_id=emb["id"], embedding_name="E",
                 status="completed", progress=100, stage="done")
    db.add(ts)
    db.commit()
    ts_id = ts.id
    for i in range(1, 4):
        db.add(TestsetItem(testset_id=ts_id, seq=i, user_input=f"问题{i}？",
                           reference=f"答案{i}",
                           reference_contexts=json.dumps([f"证据{i}"])))
    db.commit()
    db.close()
    return ts_id


@pytest.fixture()
def mock_pipeline(monkeypatch):
    async def fake_connectivity(*a, **kw):
        return True, "ok", 5

    async def fake_call(**kw):
        q = kw["question"]
        if "问题2" in q:
            raise RuntimeError("target exploded")
        return f"回答:{q}", ["检索片段"], 0.02

    def fake_evaluate(run_id, run_items, metric_keys, llm, emb, concurrency):
        return ({r.seq: {k: 0.9 for k in metric_keys} for r in run_items}, {})

    monkeypatch.setattr("app.services.model_connect.test_connectivity", fake_connectivity)
    monkeypatch.setattr(eval_run.model_connect, "test_connectivity", fake_connectivity)
    monkeypatch.setattr(eval_run.rag_client, "call_rag_system", fake_call)
    monkeypatch.setattr(eval_run, "_evaluate", fake_evaluate)
    class _FakeRaw:
        totals = {"prompt": 0, "completion": 0, "calls": 0}

    monkeypatch.setattr(eval_run.testset_gen, "_build_llm",
                        lambda *a, **kw: (object(), _FakeRaw(), None))
    monkeypatch.setattr(eval_run.testset_gen, "_build_embeddings", lambda *a, **kw: (object(), None))


def _wait_run(client, run_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/evaluations/{run_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.2)
    raise TimeoutError("evaluation did not finish")


def _mk_run(client, testset_id, rag, llm, emb, name, metrics):
    resp = client.post("/api/evaluations", json={
        "name": name, "testset_id": testset_id,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "concurrency": 2,
        "metrics": metrics, "base_url": "http://rag.local/query",
    })
    assert resp.status_code == 201, resp.json()
    return resp.json()


def test_evaluation_full_flow(client, judge_configs, testset_with_items,
                              mock_pipeline, fake_probe):
    rag = _create_adapter(client)
    llm, emb = judge_configs
    run = _mk_run(client, testset_with_items, rag, llm, emb, "Run1",
                  ["faithfulness", "response_relevancy"])
    run = _wait_run(client, run["id"])
    assert run["status"] == "completed"
    assert run["progress"] == 100
    assert run["total_items"] == 3
    assert run["failed_items"] == 1  # 问题2 exploded in fake_call
    summary = json.loads(run["summary"])
    assert summary == {"faithfulness": 0.9, "response_relevancy": 0.9}

    items = client.get(f"/api/evaluations/{run['id']}/items?page_size=10").json()
    assert items["total"] == 3
    item2 = [i for i in items["items"] if i["seq"] == 2][0]
    assert item2["answer"] is None and "target exploded" in item2["error"]
    item1 = [i for i in items["items"] if i["seq"] == 1][0]
    assert json.loads(item1["scores"])["faithfulness"] == 0.9

    log = client.get(f"/api/evaluations/{run['id']}/log").json()
    keys = [e["key"] for e in log["entries"]]
    assert "connectivityOk" in keys and "queryDone" in keys and "evalDone" in keys
    assert "itemFailed" in keys and "done" in keys


def test_verdicts_and_the_k_curve_reach_the_api(
        client, judge_configs, testset_with_items, mock_pipeline, fake_probe, monkeypatch):
    """The judge's per-chunk verdicts are kept, and Hit@k is read off them.

    Verdicts are the raw reading and the curve is one view of it, so the curve
    is derived on read rather than stored beside them. An item whose verdicts
    were never computed reports an empty list instead of a missing key: a
    consumer handles one case, not two.
    """
    rag = _create_adapter(client)
    llm, emb = judge_configs

    def fake_evaluate(run_id, run_items, metric_keys, llm, emb, concurrency):
        scores = {r.seq: {k: 0.9 for k in metric_keys} for r in run_items}
        return scores, {1: [1, 0, 1], 3: [0, 0, 1]}

    monkeypatch.setattr(eval_run, "_evaluate", fake_evaluate)
    run = _mk_run(client, testset_with_items, rag, llm, emb, "Verdicts",
                  ["context_precision"])
    run = _wait_run(client, run["id"])
    assert run["status"] == "completed"

    # seq 1 hits at k=1, seq 3 only at k=3
    assert json.loads(run["hit_at_k"]) == {
        "items": 2, "curve": {"1": 0.5, "2": 0.5, "3": 1.0}}

    items = client.get(f"/api/evaluations/{run['id']}/items?page_size=10").json()
    by_seq = {i["seq"]: i for i in items["items"]}
    assert json.loads(by_seq[1]["context_verdicts"]) == [1, 0, 1]
    assert json.loads(by_seq[3]["context_verdicts"]) == [0, 0, 1]
    # seq 2's target call failed, so it never earned a verdict
    assert json.loads(by_seq[2]["context_verdicts"]) == []


def test_rescoring_one_item_replaces_a_missing_score(
        client, judge_configs, testset_with_items, mock_pipeline, fake_probe, monkeypatch):
    """An item the judge left unscored can be rescored on its own.

    Nothing else could: the run is finished, and `resume` only takes a failed
    run. The rescore has to read around the judge cache — the unusable reply is
    cached, so a normal retry would replay the very answer that failed.
    """
    rag = _create_adapter(client)
    llm, emb = judge_configs
    seen: dict = {"calls": 0, "read_cache": None, "asked": []}

    def fake_evaluate(run_id, run_items, metric_keys, llm, emb, concurrency,
                      report_progress=True):
        seen["calls"] += 1
        seen["asked"].append(list(metric_keys))
        first_pass = seen["calls"] == 1
        out = {}
        for r in run_items:
            if first_pass and r.seq == 1:
                # Judged for one metric only — the other reply came back
                # unparsable and left no score. (Item 2 is the one whose target
                # call failed: it has nothing to score, and is not what this
                # endpoint is for.)
                out[r.seq] = {"answer_correctness": 0.5}
            else:
                out[r.seq] = {k: (0.9 if first_pass else 0.0) for k in metric_keys}
        return out, {}

    monkeypatch.setattr(eval_run, "_evaluate", fake_evaluate)

    class _Raw:
        totals = {"prompt": 0, "completion": 0, "calls": 0}

    def spy_build(*_a, **kw):
        seen["read_cache"] = kw.get("read_cache")
        return object(), _Raw(), None

    monkeypatch.setattr(eval_run.testset_gen, "_build_llm", spy_build)

    run = _mk_run(client, testset_with_items, rag, llm, emb, "Rescore",
                  ["faithfulness", "answer_correctness"])
    run = _wait_run(client, run["id"])
    assert run["status"] == "completed"
    # item 1 has one of its two metrics; item 3 has both; item 2's call failed
    assert json.loads(run["summary"]) == {"faithfulness": 0.9, "answer_correctness": 0.7}

    resp = client.post(f"/api/evaluations/{run['id']}/items/1/recompute")
    assert resp.status_code == 200, resp.json()
    # the missing metric is filled in, and the one that was already there keeps
    # its stored value — re-running it would swap a good number for a fresh
    # sample of the judge's noise and move the mean for no reason
    assert json.loads(resp.json()["scores"]) == {"answer_correctness": 0.5,
                                                 "faithfulness": 0.0}
    assert seen["asked"][-1] == ["faithfulness"], "only the unscored metric is asked for"
    assert seen["read_cache"] is False, "a rescore must not read the cached reply"

    # the summary moved with the row: faithfulness is now the mean over two
    # items (0.0 rescored, 0.9 the other answered one); answer_correctness is
    # untouched at (0.5 + 0.9) / 2
    after = client.get(f"/api/evaluations/{run['id']}").json()
    assert json.loads(after["summary"]) == {"faithfulness": 0.45,
                                            "answer_correctness": 0.7}


def test_rescoring_is_refused_on_an_unfinished_run(
        client, judge_configs, testset_with_items, mock_pipeline, fake_probe):
    """A failed run is resumed whole; rescoring one item of it would leave the
    rest of the failure in place."""
    rag = _create_adapter(client)
    llm, emb = judge_configs
    run = _mk_run(client, testset_with_items, rag, llm, emb, "RescoreGuard",
                  ["faithfulness"])
    run = _wait_run(client, run["id"])
    # A finished run with an unknown seq is a 404, not a silent no-op.
    assert client.post(
        f"/api/evaluations/{run['id']}/items/999/recompute").status_code == 404


def test_k_curve_is_absent_when_context_precision_did_not_run(
        client, judge_configs, testset_with_items, mock_pipeline, fake_probe):
    """No verdicts means no curve — and the field is empty, not missing."""
    rag = _create_adapter(client)
    llm, emb = judge_configs
    run = _mk_run(client, testset_with_items, rag, llm, emb, "NoContexts",
                  ["faithfulness"])
    run = _wait_run(client, run["id"])
    assert run["status"] == "completed"
    assert run["hit_at_k"] == '{"items": 0, "curve": {}}'
    items = client.get(f"/api/evaluations/{run['id']}/items?page_size=10").json()
    assert all(i["context_verdicts"] == "[]" for i in items["items"])


def test_evaluation_rejects_unready_adapter(
        client, judge_configs, testset_with_items, mock_pipeline, monkeypatch):
    async def _probe_fail(cfg, question, url=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(adapter_assist, "_probe", _probe_fail)
    rag = _create_adapter(client)  # probe fails -> adapter failed
    assert rag["status"] == "failed"
    llm, emb = judge_configs
    resp = client.post("/api/evaluations", json={
        "name": "RunX", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://rag.local/query",
    })
    assert resp.status_code == 400


def test_evaluation_rejects_context_metrics_without_contexts_path(
        client, judge_configs, testset_with_items, mock_pipeline, monkeypatch):
    # Probe response WITHOUT contexts so auto-discovery finds nothing.
    async def _probe_no_ctx(cfg, question, url=None):
        return {"data": {"answer": "x"}}, 0.05

    monkeypatch.setattr(adapter_assist, "_probe", _probe_no_ctx)
    rag = _create_adapter(client, {**RAG_PAYLOAD, "name": "NoCtx", "contexts_path": None})
    llm, emb = judge_configs
    resp = client.post("/api/evaluations", json={
        "name": "RunX", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"],
        "metrics": ["faithfulness"], "base_url": "http://rag.local/query",
    })
    assert resp.status_code == 400
    run = _mk_run(client, testset_with_items, rag, llm, emb, "RunY",
                  ["response_relevancy"])
    run = _wait_run(client, run["id"])
    assert run["status"] == "completed"


def test_evaluation_endpoint_snapshot(client, judge_configs, testset_with_items,
                                      mock_pipeline, fake_probe, monkeypatch):
    """The run snapshots the endpoint/key given at creation; the pipeline uses
    the snapshot (adapters carry no endpoint anymore)."""
    rag = _create_adapter(client)
    llm, emb = judge_configs

    seen = {}

    async def fake_call(**kw):
        seen["base_url"] = kw["base_url"]
        seen["api_key"] = kw["api_key"]
        return "回答", ["片段"], 0.01

    monkeypatch.setattr(eval_run.rag_client, "call_rag_system", fake_call)
    resp = client.post("/api/evaluations", json={
        "name": "RunOverride2", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://other-host:9000/api/v1/chats/xyz/completions",
        "api_key": "other-key",
    })
    assert resp.status_code == 201
    run2 = _wait_run(client, resp.json()["id"])
    assert run2["run_base_url"] == "http://other-host:9000/api/v1/chats/xyz/completions"
    assert seen["base_url"] == "http://other-host:9000/api/v1/chats/xyz/completions"
    assert seen["api_key"] == "other-key"


def test_evaluation_judge_concurrency(client, judge_configs, testset_with_items,
                                      mock_pipeline, fake_probe, monkeypatch):
    """judge_concurrency is stored per run and drives the ragas phase, while
    target calls keep using concurrency."""
    rag = _create_adapter(client)
    llm, emb = judge_configs

    seen = {}

    async def fake_call(**kw):
        return "回答", ["片段"], 0.01

    def fake_build_embeddings(cfg, concurrency, **kw):
        seen["emb_concurrency"] = concurrency
        return object(), None

    def fake_evaluate(run_id, run_items, metric_keys, llm, emb, concurrency):
        seen["ragas_concurrency"] = concurrency
        return ({r.seq: {k: 0.9 for k in metric_keys} for r in run_items}, {})

    monkeypatch.setattr(eval_run.rag_client, "call_rag_system", fake_call)
    monkeypatch.setattr(eval_run.testset_gen, "_build_embeddings", fake_build_embeddings)
    monkeypatch.setattr(eval_run, "_evaluate", fake_evaluate)

    resp = client.post("/api/evaluations", json={
        "name": "JudgeConc", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://rag.local/query",
        "concurrency": 3, "judge_concurrency": 9,
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["concurrency"] == 3 and body["judge_concurrency"] == 9
    run = _wait_run(client, body["id"])
    assert run["status"] == "completed"
    assert seen["emb_concurrency"] == 9 and seen["ragas_concurrency"] == 9
    # default when not provided
    resp2 = client.post("/api/evaluations", json={
        "name": "JudgeConcDefault", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://rag.local/query",
    })
    assert resp2.status_code == 201
    assert resp2.json()["judge_concurrency"] == 16


def test_evaluation_timeout_snapshot(client, judge_configs, testset_with_items,
                                     mock_pipeline, fake_probe, monkeypatch):
    """Per-run timeout is snapshotted and drives the target calls."""
    rag = _create_adapter(client)
    llm, emb = judge_configs
    resp = client.post("/api/evaluations", json={
        "name": "RunTimeout", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://rag.local/query", "timeout": 300,
    })
    assert resp.status_code == 201
    run = _wait_run(client, resp.json()["id"])
    assert run["status"] == "completed"
    assert run["timeout"] == 300
    # default when omitted
    run2 = _mk_run(client, testset_with_items, rag, llm, emb, "RunTimeoutDefault",
                   ["response_relevancy"])
    assert run2["timeout"] == 120


def test_evaluation_precheck_retries_busy_embedding(
        client, judge_configs, testset_with_items, monkeypatch,
        mock_pipeline, fake_probe):
    """Judge embedding busy (ReadTimeout twice, then ok) -> precheck retries
    and the run proceeds instead of failing at the checking stage."""
    rag = _create_adapter(client)
    llm, emb = judge_configs
    monkeypatch.setattr(eval_run, "_PRECHECK_BACKOFF", (0, 0))

    calls = {"embedding": 0}

    async def flaky_connectivity(model_type, *a, **kw):
        if model_type == "embedding":
            calls["embedding"] += 1
            if calls["embedding"] <= 2:
                return False, "ReadTimeout: ", None
        return True, "ok", 5

    monkeypatch.setattr(eval_run.model_connect, "test_connectivity", flaky_connectivity)

    resp = client.post("/api/evaluations", json={
        "name": "RetryPrecheck", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://rag.local/query",
    })
    assert resp.status_code == 201
    run = _wait_run(client, resp.json()["id"])
    assert run["status"] == "completed"
    assert calls["embedding"] == 3

    # permanently busy -> still fails, after exactly _PRECHECK_ATTEMPTS tries
    async def always_busy(model_type, *a, **kw):
        if model_type == "embedding":
            calls["embedding"] += 1
            return False, "ReadTimeout: ", None
        return True, "ok", 5

    monkeypatch.setattr(eval_run.model_connect, "test_connectivity", always_busy)
    resp = client.post("/api/evaluations", json={
        "name": "BusyPrecheck", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://rag.local/query",
    })
    run2 = _wait_run(client, resp.json()["id"])
    assert run2["status"] == "failed"
    assert "Embedding=ReadTimeout" in run2["error"]


def test_evaluation_all_calls_fail_marks_failed(
        client, judge_configs, testset_with_items, monkeypatch,
        mock_pipeline, fake_probe):
    rag = _create_adapter(client)
    llm, emb = judge_configs

    async def always_fail(**kw):
        raise ConnectionError("down")

    monkeypatch.setattr(eval_run.rag_client, "call_rag_system", always_fail)
    run = _mk_run(client, testset_with_items, rag, llm, emb, "RunFail",
                  ["response_relevancy"])
    run = _wait_run(client, run["id"])
    assert run["status"] == "failed"
    assert "unreachable" in run["error"]  # precheck hits the same fake

    # resume is only for failed runs; it re-runs and fails again (still down)
    resp = client.post(f"/api/evaluations/{run['id']}/resume")
    assert resp.status_code == 200
    run = _wait_run(client, run["id"])
    assert run["status"] == "failed"


def test_judge_cache_dir_shared_per_model(client):
    """Judge cache is shared across runs but partitioned by judge model."""
    from app.db.models import ModelConfig
    a1 = ModelConfig(type="llm", api_format="openai", base_url="https://a/v1",
                     api_key="k1", model="m", name="A1")
    a2 = ModelConfig(type="llm", api_format="openai", base_url="https://a/v1",
                     api_key="k2", model="m", name="A2")  # same model, other key
    b = ModelConfig(type="llm", api_format="openai", base_url="https://a/v1",
                    api_key="k1", model="m2", name="B")  # different model
    d1 = eval_run._judge_cache_dir(a1)
    assert "_judge_cache" in str(d1)
    assert d1 == eval_run._judge_cache_dir(a2)  # same model+endpoint -> shared
    assert d1 != eval_run._judge_cache_dir(b)   # different model -> isolated
    # a different variant (e.g. output budget, or llm vs emb) must not share
    assert d1 != eval_run._judge_cache_dir(a1, "llm16384")
    assert eval_run._judge_cache_dir(a1, "emb") != eval_run._judge_cache_dir(a1, "llm8192")


def test_evaluation_use_judge_cache_toggle(client, judge_configs, testset_with_items,
                                           mock_pipeline, fake_probe, monkeypatch):
    """The judge cache always writes; use_judge_cache only gates reading.
    use_judge_cache=False -> read_cache=False (fresh compute, results stored);
    default True -> reads allowed."""
    rag = _create_adapter(client)
    llm, emb = judge_configs

    seen = {}

    def fake_build_llm(cfg, max_tokens, cache_dir, read_cache=True):
        seen["llm_cache_dir"] = cache_dir
        seen["llm_read_cache"] = read_cache
        class R: totals = {"prompt": 0, "completion": 0, "calls": 0}
        return object(), R(), None

    def fake_build_embeddings(cfg, concurrency, **kw):
        seen["emb_cache_dir"] = kw.get("cache_dir")
        seen["emb_read_cache"] = kw.get("read_cache")
        return object(), None

    monkeypatch.setattr(eval_run.testset_gen, "_build_llm", fake_build_llm)
    monkeypatch.setattr(eval_run.testset_gen, "_build_embeddings", fake_build_embeddings)

    def payload(name, **extra):
        return {
            "name": name, "testset_id": testset_with_items,
            "rag_system_id": rag["id"], "llm_config_id": llm["id"],
            "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
            "base_url": "http://rag.local/query", **extra,
        }

    # default: reads allowed, shared per-model dir
    resp = client.post("/api/evaluations", json=payload("CacheOn"))
    assert resp.status_code == 201 and resp.json()["use_judge_cache"] is True
    run = _wait_run(client, resp.json()["id"])
    assert run["status"] == "completed"
    assert seen["llm_cache_dir"] is not None and "_judge_cache" in str(seen["llm_cache_dir"])
    assert seen["emb_cache_dir"] is not None
    assert seen["llm_read_cache"] is True and seen["emb_read_cache"] is True

    # off: still cached (writes), but reads bypassed
    resp = client.post("/api/evaluations", json=payload("CacheOff", use_judge_cache=False))
    assert resp.status_code == 201
    run = _wait_run(client, resp.json()["id"])
    assert run["status"] == "completed"
    assert seen["llm_cache_dir"] is not None and seen["emb_cache_dir"] is not None
    assert seen["llm_read_cache"] is False and seen["emb_read_cache"] is False

    # the cache stats entry reports whether reads were enabled
    log = client.get(f"/api/evaluations/{run['id']}/log").json()
    jc = [e for e in log["entries"] if e["key"] == "judgeCache"]
    assert jc and jc[0]["params"]["readEnabled"] == 0


def test_delete_run_removes_run_dir_but_keeps_shared_cache(
        client, judge_configs, testset_with_items, mock_pipeline, fake_probe):
    """Deleting a run removes its own artifacts; the shared judge cache stays."""
    from app.core.config import get_settings
    rag = _create_adapter(client)
    llm, emb = judge_configs
    resp = client.post("/api/evaluations", json={
        "name": "ToDelete", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://rag.local/query",
    })
    run = _wait_run(client, resp.json()["id"])
    assert run["status"] == "completed"

    run_dir = get_settings().data_dir / "evaluation" / "ToDelete"
    assert run_dir.is_dir()
    from app.db.models import ModelConfig as _MC
    shared = eval_run._judge_cache_dir(_MC(
        type="llm", api_format="openai", base_url=llm["base_url"],
        api_key="k", model=llm["model"], name="j"))
    assert shared.is_dir()

    assert client.delete(f"/api/evaluations/{run['id']}").status_code == 204
    assert not run_dir.exists()      # run artifacts cleaned
    assert shared.is_dir()           # shared cache untouched


def test_judge_cache_never_serves_a_different_prompt(tmp_path, monkeypatch, isolated_db):
    """The decisive cache-safety property: a cached entry is only ever reused
    for byte-identical input. Verifies (a) distinct prompts never share a key,
    and (b) re-running the same items produces the same keys (real hits)."""
    import ragas.cache as rc
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI

    seen: dict[str, str] = {}
    collisions: list[str] = []
    real = rc._generate_cache_key

    def spy(func, args, kwargs):
        key = real(func, args, kwargs)
        payload = json.dumps(
            {"f": func.__qualname__,
             "a": rc._make_hashable(args),
             "k": rc._make_hashable(
                 {k: v for k, v in kwargs.items() if k not in rc.EXCLUDE_KEYS})},
            sort_keys=True, default=str)
        if key in seen and seen[key] != payload:
            collisions.append(key)
        seen[key] = payload
        return key

    monkeypatch.setattr(rc, "_generate_cache_key", spy)

    def _canned(self, *a, **kw):
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content='{"statements": [], "verdicts": [], "reason": "ok"}'),
            generation_info={"finish_reason": "stop"})])

    monkeypatch.setattr(ChatOpenAI, "generate", _canned)
    monkeypatch.setattr(ChatOpenAI, "agenerate", _canned)

    from app.db.models import ModelConfig
    cfg = ModelConfig(type="llm", api_format="openai", base_url="https://judge/v1",
                      api_key="k", model="judge-model", name="j")
    cache_dir = tmp_path / "llm"
    from app.services import testset_gen as _tg
    llm, _raw, _cache = _tg._build_llm(cfg, 4096, cache_dir)

    def item(seq, answer):
        return SimpleNamespace(seq=seq, user_input=f"问题{seq}", reference=f"答案{seq}",
                               answer=answer, contexts=json.dumps(["片段A"]))

    metrics = ["faithfulness", "context_precision"]
    items_a = [item(1, "回答一"), item(2, "回答二")]

    keys_run1, verdicts_run1 = eval_run._evaluate(0, items_a, metrics, llm, None, 2)
    seen_after_1 = set(seen)
    assert seen_after_1, "expected the judge LLM to be called"
    assert collisions == [], "two different prompts hashed to the same cache key"

    # identical run -> every key is already known (genuine cache hits)
    eval_run._evaluate(0, items_a, metrics, llm, None, 2)
    assert set(seen) == seen_after_1, "same input produced new cache keys"

    # change ONE answer -> new prompt -> new keys, and the old entries are not
    # reused for it (a false hit would leave the key set unchanged)
    items_b = [item(1, "回答一"), item(2, "回答二（已修改）")]
    eval_run._evaluate(0, items_b, metrics, llm, None, 2)
    assert set(seen) > seen_after_1, "changed answer reused a stale cache entry"
    assert collisions == []
    assert isinstance(keys_run1, dict)
    # This test's canned reply is the faithfulness shape, so context_precision
    # cannot parse it and no verdict is recorded. An item with no verdict
    # contributes no curve — the safe direction: a parse failure must not read
    # as "nothing was relevant".
    assert verdicts_run1 == {}


def test_answer_correctness_needs_no_contexts(client, judge_configs, testset_with_items,
                                              mock_pipeline, fake_probe):
    """AnswerCorrectness only needs question/answer/reference, so an adapter
    without a contexts path must accept it (unlike the context metrics)."""
    # Create-time discovery fills a blank contexts path, so clear it via an
    # edit (edits never rewrite the config) to get a contexts-less adapter.
    created = _create_adapter(client, {**RAG_PAYLOAD, "name": "NoCtx"})
    assert created["contexts_path"]
    resp = client.put(f"/api/rag-systems/{created['id']}", json={
        "name": created["name"], "base_url": RAG_PAYLOAD["base_url"],
        "headers": "{}", "body_template": RAG_PAYLOAD["body_template"],
        "answer_path": "data.answer", "contexts_path": None,
    })
    assert resp.status_code == 200
    rag = _wait_adapter(client, created["id"])
    assert rag["contexts_path"] is None
    llm, emb = judge_configs

    def payload(metrics):
        return {"name": f"AC-{'-'.join(metrics)}", "testset_id": testset_with_items,
                "rag_system_id": rag["id"], "llm_config_id": llm["id"],
                "embedding_config_id": emb["id"], "metrics": metrics,
                "base_url": "http://rag.local/query"}

    # accepted without contexts
    resp = client.post("/api/evaluations", json=payload(["answer_correctness"]))
    assert resp.status_code == 201, resp.json()
    run = _wait_run(client, resp.json()["id"])
    assert run["status"] == "completed"
    assert "answer_correctness" in run["summary"]

    # the context metrics are still rejected in that situation
    assert client.post("/api/evaluations",
                       json=payload(["faithfulness"])).status_code == 400


def test_metric_registry_consistency():
    """AnswerCorrectness is offered, is not a context metric, and maps to a
    ragas class whose result column equals our key (no alias needed)."""
    assert "answer_correctness" in eval_run.METRIC_KEYS
    assert "answer_correctness" not in eval_run.CONTEXT_METRICS
    assert "answer_correctness" not in eval_run._RAGAS_COLUMN
    instances = eval_run._build_metric_instances(["answer_correctness"])
    assert len(instances) == 1 and instances[0].name == "answer_correctness"
    assert eval_run.METRIC_KEYS[-1] == "answer_correctness"  # displayed last


def test_latency_logged_per_run(client, judge_configs, testset_with_items,
                                mock_pipeline, fake_probe, monkeypatch):
    """The target-system latency line reports avg/median/max of successful calls."""
    rag = _create_adapter(client)
    llm, emb = judge_configs

    delays = iter([0.10, 0.20, 0.30] * 20)

    async def fake_call(**kw):
        return "回答", ["片段"], next(delays)

    monkeypatch.setattr(eval_run.rag_client, "call_rag_system", fake_call)
    resp = client.post("/api/evaluations", json={
        "name": "Latency", "testset_id": testset_with_items,
        "rag_system_id": rag["id"], "llm_config_id": llm["id"],
        "embedding_config_id": emb["id"], "metrics": ["response_relevancy"],
        "base_url": "http://rag.local/query",
    })
    run = _wait_run(client, resp.json()["id"])
    assert run["status"] == "completed"
    log = client.get(f"/api/evaluations/{run['id']}/log").json()
    lat = [e for e in log["entries"] if e["key"] == "latency"]
    assert lat, "expected a latency log entry"
    p = lat[0]["params"]
    assert p["avg"] > 0 and p["median"] > 0 and p["max"] >= p["median"]


def test_evaluation_retries_unscored_items(monkeypatch, isolated_db):
    """A judge parse failure yields NaN, which would silently drop the item from
    that metric's average. The metric is retried for just those items."""
    import pandas as pd
    import ragas
    from app.db.models import EvalRunItem

    calls = {"n": 0}

    class _Result:
        def __init__(self, vals):
            self._df = pd.DataFrame({"faithfulness": vals})

        def to_pandas(self):
            return self._df

    def fake_evaluate(dataset, metrics, llm, embeddings, run_config, show_progress):
        calls["n"] += 1
        n = len(dataset.samples)
        # first pass fails to score anything; the retry scores it properly
        vals = [float("nan")] * n if calls["n"] == 1 else [0.75] * n
        return _Result(vals)

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)

    items = [
        EvalRunItem(run_id=1, seq=i, user_input=f"问{i}", reference="答",
                    answer="系统回答", contexts='["片段"]')
        for i in (1, 2)
    ]
    scores, verdicts = eval_run._evaluate(0, items, ["faithfulness"], None, None, 2)
    # every item ends up scored, and the failure was retried once
    assert scores == {1: {"faithfulness": 0.75}, 2: {"faithfulness": 0.75}}
    # faithfulness has no per-chunk verdict, so there is no curve to read
    assert verdicts == {}
    assert calls["n"] == 2


def test_malformed_extraction_path_returns_none_instead_of_raising():
    """The module promises extraction never raises; `int(idx)` broke that, and
    a user-typed path like "chunks[abc]" surfaced as an unexplained per-item
    failure rather than "that path is not valid"."""
    data = {"a": {"b": [{"c": "x"}]}}
    for bad in ("a.b[abc].c", "a[b]", "a.b[]c", "a.b[-]"):
        assert rag_client.extract_path(data, bad) in (None, [])
    assert rag_client.extract_answer(data, "a[b]") is None
    assert rag_client.extract_contexts(data, "a[b]") == []
    # valid paths are unaffected
    assert rag_client.extract_path(data, "a.b[].c") == "x"


def test_partial_metric_coverage_is_reported_not_hidden():
    """A mean is taken over the items that scored, so 25 of 30 reads exactly like
    30 of 30 — and a metric that scored nothing drops out of the summary and
    reads like one never requested. A run against RAGFlow finished with fifteen
    such gaps across its items and reported success."""
    from app.services import eval_run

    scores = {
        1: {"a": 1.0, "b": 1.0},
        2: {"a": 0.5, "b": float("nan")},
        3: {"a": 0.0},
    }
    assert eval_run._coverage(scores, ["a", "b", "c"], 3) == [("b", 1), ("c", 0)]
    assert eval_run._coverage({1: {"a": 1.0}}, ["a"], 1) == []


def test_item_scores_are_written_as_json_a_browser_can_parse():
    """json.dumps writes a metric that scored nothing as a bare NaN, and that is
    not JSON. One of them made JSON.parse throw on the whole string, so the
    report page showed "-" under every metric, the scoring ones included."""
    import json

    from app.services import eval_run

    stored = json.dumps(eval_run._finite(
        {"a": 1.0, "b": float("nan"), "c": float("inf")}))
    assert "NaN" not in stored and "Infinity" not in stored
    assert json.loads(stored) == {"a": 1.0, "b": None, "c": None}


def test_resume_runs_the_operation_that_was_interrupted(client, monkeypatch):
    """A smart-created adapter keeps its llm_config_id forever, so dispatching
    resume on that field re-ran the agent for an interrupted manual EDIT and let
    it overwrite the user's own templates."""
    from app.db.models import RagSystemConfig
    from app.db.session import get_session_factory

    db = get_session_factory()()
    try:
        rows = {
            "smart_created": ("init", 7),        # create via smart fill
            "manual_created": ("init", None),    # create by hand
            "manual_edit": ("testing", 7),       # EDIT of a smart adapter
            "mid_assist": ("probing", 7),
        }
        for name, (stage, llm_id) in rows.items():
            db.add(RagSystemConfig(
                name=name, base_url="http://x/query", headers="{}",
                body_template='{"q":"{{question}}"}', answer_path="a",
                status=RagSystemConfig.STATUS_CONFIGURING, stage=stage,
                llm_config_id=llm_id))
        db.commit()
    finally:
        db.close()

    called: list[tuple[str, int, bool | None]] = []

    async def fake_assist(cid):
        called.append(("assist", cid, None))

    async def fake_simple(cid, fill_blank_contexts=True):
        called.append(("simple", cid, fill_blank_contexts))

    monkeypatch.setattr(adapter_assist, "run_assist", fake_assist)
    monkeypatch.setattr(adapter_assist, "run_simple_test", fake_simple)
    monkeypatch.setattr(adapter_assist, "_log", lambda *a, **kw: None)

    asyncio.run(adapter_assist.resume_interrupted())

    db = get_session_factory()()
    try:
        by_name = {r.name: r.id for r in db.query(RagSystemConfig).all()}
    finally:
        db.close()
    modes = {name: None for name in by_name}
    for mode, cid, flag in called:
        for name, rid in by_name.items():
            if rid == cid:
                modes[name] = (mode, flag)

    assert modes["smart_created"] == ("assist", None)
    assert modes["manual_created"] == ("simple", True)
    # the interrupted edit must be verified, never re-written, and must not have
    # its deliberately cleared contexts path auto-filled
    assert modes["manual_edit"] == ("simple", False)
    assert modes["mid_assist"] == ("assist", None)
