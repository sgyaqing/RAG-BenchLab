"""Connectivity checks and model listing for LLM/Embedding providers.

The HTTP itself — what path a Base URL needs, what headers a protocol wants,
the chat call — lives in `llm_http`, so that this test and the calls the rest
of the app makes cannot drift apart. This module is the config page's use of
it, plus the model list.
"""

import asyncio
import random
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

# What usually kills one of these calls is a silent window, not a broken
# endpoint: measured on a customer run (the story behind testset_gen's retry
# count), eight concurrent calls fell into one ~10 s window together and the
# endpoint was answering again 3 s after the first failure — one attempt lands
# inside such a window, a retry walks past it. So the busy class of answers
# and transport errors are retried here too, with the same count and the same
# exclusions: billing and auth answers (401/402/403) are not retried, so a
# rejected key still fails on the first attempt rather than after a minute of
# spinner, and a genuinely dead endpoint still costs only its timeouts.
MAX_RETRIES = 3
RETRYABLE_STATUSES = frozenset({408, 429, *range(500, 600)})
_BACKOFF_CAP = 8.0
# The retries must not turn a dead endpoint into a minute of spinner: the
# sweep hands each attempt only the budget the deadline has left, so the
# whole thing answers within TOTAL_DEADLINE seconds no matter what. The
# first attempt always keeps its full 15 s read / 10 s connect — only later
# ones are trimmed — and httpx applies a per-request timeout wholesale,
# replacing the client's (read off 0.28.1's build_request, where anything
# that is not USE_CLIENT_DEFAULT is turned into the request's own Timeout).
# Same principle as testset_gen's retry count: walk past a silent window,
# never wait longer on a hopeless endpoint.
TOTAL_DEADLINE = 30.0
_MIN_ATTEMPT = 0.05  # floor, so a spent budget still makes one real call
_now = time.monotonic  # a seam the tests drive; nothing else calls this


async def _with_retries(call):
    """Await `call(per_attempt_timeout)`, retrying silent-window failures.

    `call` is handed the httpx.Timeout for its attempt and returns whatever
    its caller needs back — the only thing read here is `status_code`, which
    httpx.Response and ChatReply both carry. The last result is returned
    as-is whatever its status, so each caller's own "HTTP 401: ..." reporting
    keeps working; a transport error on the last attempt raises through to
    the caller unchanged.
    """
    deadline = _now() + TOTAL_DEADLINE
    for attempt in range(MAX_RETRIES + 1):
        remaining = deadline - _now()
        budget = max(remaining, _MIN_ATTEMPT)
        try:
            result = await call(
                httpx.Timeout(min(TIMEOUT.read, budget), connect=min(TIMEOUT.connect, budget))
            )
        except httpx.TransportError:
            if attempt == MAX_RETRIES:
                raise
        else:
            if attempt == MAX_RETRIES or result.status_code not in RETRYABLE_STATUSES:
                return result
        # Exponential with jitter and a cap — 0.5 s, 1 s, 2 s give or take
        # half — so that retries of concurrent calls (an eval precheck's LLM
        # and embedding pings, say) do not line up into a second burst that
        # lands inside the same window the first one died of. A spent budget
        # skips the wait: the remaining attempts fail in milliseconds, and
        # the caller hears about it now rather than one backoff later.
        if remaining > 0:
            await asyncio.sleep(
                min(0.5 * 2**attempt, _BACKOFF_CAP) * random.uniform(0.5, 1.5)
            )


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
                resp = await _with_retries(
                    lambda t: client.post(
                        _openai_url(base_url, "embeddings"),
                        headers=_openai_headers(api_key),
                        json={"model": model, "input": "ping"},
                        timeout=t,
                    )
                )
            else:
                # max_tokens=1: the question is "can I reach you", not "what do
                # you think" — and one token is all an answer needs to prove it.
                # Nothing else is sent: this asks whether the endpoint answers,
                # so it neither needs nor should rehearse how the runtime calls
                # the model. A model that thinks first still answers in time —
                # measured 1.5 s on Ark against this check's 15 s timeout.
                reply = await _with_retries(
                    lambda t: chat_completion(
                        client,
                        api_format=api_format,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                        timeout=t,
                    )
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
            resp = await _with_retries(
                lambda t: client.get(
                    _openai_url(base_url, "models"), headers=_openai_headers(api_key), timeout=t
                )
            )
        else:
            resp = await _with_retries(
                lambda t: client.get(
                    _anthropic_url(base_url, "models"), headers=_anthropic_headers(api_key), timeout=t
                )
            )
    resp.raise_for_status()
    data = resp.json()
    return sorted(item["id"] for item in data.get("data", []) if "id" in item)
