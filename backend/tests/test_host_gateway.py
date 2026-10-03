"""Loopback addresses mean the user's machine, not the container.

The image runs the app and nothing else, so a customer whose RAG system or
embedding server sits on the same box will type the address they always type.
The image declares the host gateway (RBL_HOST_GATEWAY, set in the Dockerfile,
paired with --add-host for Linux) and every request that leaves the app
translates a loopback host into it.

These tests pin the translation itself, and then the points where it has to
happen: the two URL builders every model-config request goes through, the
target RAG call, and the judge LLM.
"""

import asyncio

import pytest

from app.core.host import host_gateway
from app.services import llm_http, rag_client, testset_gen


@pytest.fixture()
def gateway(monkeypatch):
    monkeypatch.setenv("RBL_HOST_GATEWAY", "host.docker.internal")


def test_without_a_gateway_nothing_moves(monkeypatch):
    """start.sh runs the app directly, where localhost is already right."""
    monkeypatch.delenv("RBL_HOST_GATEWAY", raising=False)
    assert host_gateway("http://localhost:3000/query") == "http://localhost:3000/query"


def test_an_empty_gateway_is_off_too(monkeypatch):
    """Blank is how a deployment turns the translation off without a code path."""
    monkeypatch.setenv("RBL_HOST_GATEWAY", "   ")
    assert host_gateway("http://localhost:3000/query") == "http://localhost:3000/query"


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:3000/query", "http://host.docker.internal:3000/query"),
    ("http://127.0.0.1:3000/query", "http://host.docker.internal:3000/query"),
    ("http://[::1]:3000/query", "http://host.docker.internal:3000/query"),
    ("http://localhost", "http://host.docker.internal"),
    ("https://LOCALHOST:8080/a/b?c=d", "https://host.docker.internal:8080/a/b?c=d"),
])
def test_loopback_becomes_the_gateway(url, expected, gateway):
    assert host_gateway(url) == expected


def test_everything_else_is_left_alone(gateway):
    for url in ("https://api.deepseek.com/v1", "http://192.168.1.10:3000/x",
                "http://host.docker.internal:3000/x", "localhost-not-really/v1"):
        assert host_gateway(url) == url


def test_a_pasted_url_without_a_scheme_keeps_its_shape(gateway):
    """Users paste "localhost:3000/v1", and urlsplit reads the host as the
    scheme when there is none — so the rewrite has to hand back the same shape
    it was given, not a "//" the caller never wrote."""
    assert host_gateway("localhost:3000/v1") == "host.docker.internal:3000/v1"
    assert host_gateway("example.com/v1") == "example.com/v1"


def test_the_model_config_urls_translate(gateway):
    """Every model call and connectivity check goes through these two."""
    assert llm_http._openai_url("http://localhost:11434/v1", "chat/completions") \
        == "http://host.docker.internal:11434/v1/chat/completions"
    assert llm_http._anthropic_url("http://localhost:11434", "messages") \
        == "http://host.docker.internal:11434/v1/messages"


def test_the_target_rag_call_posts_to_the_gateway(monkeypatch, gateway):
    seen: dict = {}

    class _Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"answer": "ok", "chunks": ["c1"]}

    class _Client:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def post(self, url, headers=None, json=None):
            seen["url"] = url
            return _Response()

    monkeypatch.setattr(rag_client.httpx, "AsyncClient", _Client)

    answer, contexts, _ = asyncio.run(rag_client.call_rag_system(
        base_url="http://localhost:8080/query", api_key=None, headers={},
        body_template='{"q": "{{question}}"}', answer_path="answer",
        contexts_path="chunks", timeout=5, question="q",
    ))
    assert seen["url"] == "http://host.docker.internal:8080/query"
    assert (answer, contexts) == ("ok", ["c1"])


def test_the_judge_llm_points_at_the_gateway(gateway, tmp_path):
    """A local embedding or judge server is the likeliest thing to sit on the
    customer's own machine, so the client the pipeline builds must carry the
    translated address, not the one that was typed."""
    from app.db.models import ModelConfig

    cfg = ModelConfig(type="llm", api_format="openai", base_url="http://localhost:11434/v1",
                      api_key="k", model="judge-model", name="judge")
    _llm, raw, _cache = testset_gen._build_llm(cfg, 4096, None)
    assert "host.docker.internal" in str(raw.openai_api_base)
