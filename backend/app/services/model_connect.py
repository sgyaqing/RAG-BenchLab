"""Connectivity checks and model listing for LLM/Embedding providers.

The HTTP itself — what path a Base URL needs, what headers a protocol wants,
the chat call — lives in `llm_http`, so that this test and the calls the rest
of the app makes cannot drift apart. This module is the config page's use of
it, plus the model list.
"""

import time

import httpx

from app.services.llm_http import (
    _anthropic_headers,
    _anthropic_url,
    _openai_headers,
    _openai_url,
    chat_completion,
)

TIMEOUT = httpx.Timeout(15.0, connect=10.0)


async def test_connectivity(
    model_type: str, api_format: str, base_url: str, api_key: str | None, model: str
) -> tuple[bool, str, int | None]:
    """Send a minimal real request to verify the model endpoint is reachable.

    Returns (success, message, duration_ms). duration_ms is set on success only.
    """
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if api_format == "anthropic" and model_type == "embedding":
                return False, "Anthropic does not provide embedding models", None
            if model_type == "embedding":
                resp = await client.post(
                    _openai_url(base_url, "embeddings"),
                    headers=_openai_headers(api_key),
                    json={"model": model, "input": "ping"},
                )
            else:
                # max_tokens=1: the question is "can I reach you", not "what do
                # you think" — and one token is all an answer needs to prove it.
                # Nothing else is sent: this asks whether the endpoint answers,
                # so it neither needs nor should rehearse how the runtime calls
                # the model. A model that thinks first still answers in time —
                # measured 1.5 s on Ark against this check's 15 s timeout.
                reply = await chat_completion(
                    client,
                    api_format=api_format,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                resp = reply.response
    except httpx.HTTPError as e:
        # Some httpx exceptions (e.g. ReadTimeout) have an empty str().
        return False, f"{type(e).__name__}: {e}", None

    duration_ms = int((time.perf_counter() - started) * 1000)
    if resp.status_code < 400:
        return True, "ok", duration_ms
    return False, f"HTTP {resp.status_code}: {resp.text[:200]}", None


async def list_models(api_format: str, base_url: str, api_key: str | None) -> list[str]:
    """Fetch the available model IDs from the provider."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if api_format == "openai":
            resp = await client.get(
                _openai_url(base_url, "models"), headers=_openai_headers(api_key)
            )
        else:
            resp = await client.get(
                _anthropic_url(base_url, "models"), headers=_anthropic_headers(api_key)
            )
    resp.raise_for_status()
    data = resp.json()
    return sorted(item["id"] for item in data.get("data", []) if "id" in item)
