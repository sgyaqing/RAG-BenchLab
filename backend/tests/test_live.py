"""Live tests that call real LLM/Embedding endpoints — they cost tokens.

Excluded from the default run (pytest.ini: -m "not live"). To run them:

    RUN_LIVE_TESTS=1 \\
    LIVE_LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3 \\
    LIVE_LLM_API_KEY=...  LIVE_LLM_MODEL=doubao-... \\
    LIVE_EMB_BASE_URL=http://localhost:11434/v1 LIVE_EMB_MODEL=bge-m3 \\
    python -m pytest -m live -q

LIVE_LLM_API_FORMAT defaults to "openai"; set it to "anthropic" for Anthropic
endpoints. Embedding API key is optional (e.g. local Ollama).
"""

import asyncio
import os

import pytest

from app.services import model_connect

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1",
        reason="live tests disabled (set RUN_LIVE_TESTS=1)",
    ),
]


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set")
    return value


def test_live_llm_connectivity():
    ok, msg, ms = asyncio.run(
        model_connect.test_connectivity(
            "llm",
            os.environ.get("LIVE_LLM_API_FORMAT", "openai"),
            _required("LIVE_LLM_BASE_URL"),
            os.environ.get("LIVE_LLM_API_KEY"),
            _required("LIVE_LLM_MODEL"),
        )
    )
    assert ok, f"LLM connectivity failed: {msg}"
    assert ms is not None and ms > 0


def test_live_embedding_connectivity():
    ok, msg, _ = asyncio.run(
        model_connect.test_connectivity(
            "embedding",
            "openai",
            _required("LIVE_EMB_BASE_URL"),
            os.environ.get("LIVE_EMB_API_KEY"),
            _required("LIVE_EMB_MODEL"),
        )
    )
    assert ok, f"Embedding connectivity failed: {msg}"
