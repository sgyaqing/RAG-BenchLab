"""The one place that turns a Base URL into a request.

The bug this guards: the smart fill hand-built `base_url + "/messages"` while
the probe and the connectivity test went through the shared helper, which adds
the `/v1` these endpoints expect. The same key against the same endpoint
produced two addresses — one reachable, one 401 — and the 401 read as a bad
credential.
"""

import asyncio
import itertools
from types import SimpleNamespace

import httpx
import pytest

from app.services import adapter_assist, model_connect
from app.services.llm_http import _anthropic_url, chat_completion

ARK = "https://ark.cn-beijing.volces.com/api/compatible"


class Recorder:
    """An httpx.AsyncClient that keeps the last request instead of sending it."""

    def __init__(self, *args, **kwargs):
        self.sent: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        self.sent = {"url": url, "headers": headers or {}, "json": json or {}}
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [{"message": {"content": "ok"}}],
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )


@pytest.fixture()
def cfg():
    return SimpleNamespace(
        api_format="openai", base_url="https://api.deepseek.com", api_key="sk-x",
        model="m", thinking_param=None,
    )


def _empty_reply_client(calls: list):
    """An AsyncClient whose every reply carries no content — the shape that
    reached the customer as "LLM did not return a JSON object: ''"."""

    class Empty:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            calls.append(url)
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"choices": [{"message": {"content": ""}}], "usage": {}},
            )

    return Empty


def test_an_empty_reply_is_named_as_such(monkeypatch, cfg):
    """Not as "did not return a JSON object: ''", which says nothing about the
    cause and is what the customer used to see."""
    monkeypatch.setattr(httpx, "AsyncClient", _empty_reply_client([]))
    with pytest.raises(adapter_assist.AssistError) as caught:
        asyncio.run(adapter_assist._ask_llm(cfg, "SYS", "USER"))
    assert caught.value.code == "noOutput"


def test_a_probe_failure_is_named_rather_than_dumped():
    """A platform that reports its errors inside a 200 ({"code": 102,
    "message": "Session not found"}) used to reach the customer as
    "Probe failed: ApiPayloadError: Session not found" — the platform's own
    words, wrapped in ours until they read as neither."""
    platform = adapter_assist.ApiPayloadError("Session not found")
    assert str(platform) == "Session not found"  # the platform's words, unwrapped
    assert adapter_assist._classify_error(platform) == "platformError"

    streaming = adapter_assist.StreamResponseError("http://rag.local/x")
    assert adapter_assist._classify_error(streaming) == "streamResponse"


def test_a_reply_that_is_not_a_config_carries_its_own_code():
    """The parse failures used to reach the customer as the raw English
    "AssistError: LLM did not return a JSON object: '...'" under the generic
    fallback message. Every one of them names its category now, so the log can
    say what to do instead."""
    for text in ("I cannot determine this platform's API.", "{not json", "[1, 2]"):
        with pytest.raises(adapter_assist.AssistError) as caught:
            adapter_assist._parse_json_object(text)
        assert caught.value.code == "noConfig", text


def test_a_model_with_no_config_to_give_is_not_a_parse_error(monkeypatch, cfg):
    """The prompt's {"unsupported": ...} escape (async, streaming-only, signed
    APIs — and, in practice, a hint with no documentation behind it) used to
    reach the customer as the model's raw English reason under a generic
    "configuration failed"."""
    async def fake_ask(cfg, system, user):
        return '{"unsupported": "The documentation contains no API specification."}', {}

    monkeypatch.setattr(adapter_assist, "_ask_llm", fake_ask)
    with pytest.raises(adapter_assist.AssistError) as caught:
        asyncio.run(
            adapter_assist._generate_config(cfg, "http://rag.local", "Hint", "", True)
        )
    assert caught.value.code == "noDocs"


def test_an_empty_reply_does_not_trigger_the_json_repair(monkeypatch, cfg):
    """The repair round works by showing the model its own answer and asking
    for clean JSON. With an empty answer there is nothing to show, and a model
    asked to correct nothing answers nothing again — measured as a second empty
    reply 18 s later. One call, not two."""
    calls: list = []
    monkeypatch.setattr(httpx, "AsyncClient", _empty_reply_client(calls))
    with pytest.raises(adapter_assist.AssistError):
        asyncio.run(
            adapter_assist._generate_config(cfg, "http://rag.local", "Hint", "", True)
        )
    assert len(calls) == 1


@pytest.fixture()
def recorder(monkeypatch) -> Recorder:
    fake = Recorder()
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: fake)
    return fake


def test_one_base_url_gives_every_caller_the_same_address(recorder):
    """The address the connectivity test proves reachable is the address the
    smart fill then calls — they are the same code path now."""
    cfg = SimpleNamespace(
        api_format="anthropic", base_url=ARK, api_key="sk-x", model="doubao",
        thinking_param=None,
    )
    asyncio.run(adapter_assist._ask_llm(cfg, "SYS", "USER"))
    from_the_fill = recorder.sent["url"]

    asyncio.run(
        model_connect.test_connectivity("llm", "anthropic", cfg.base_url, cfg.api_key, cfg.model)
    )
    assert recorder.sent["url"] == from_the_fill
    assert from_the_fill == f"{ARK}/v1/messages"


def test_the_version_segment_is_added_once():
    assert (
        _anthropic_url("https://api.deepseek.com/anthropic", "messages")
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    # A base that already carries it must not grow a second one.
    assert _anthropic_url("https://gateway/v1", "messages") == "https://gateway/v1/messages"


def test_the_system_prompt_goes_where_each_protocol_keeps_it(recorder):
    user = [{"role": "user", "content": "U"}]
    common = dict(
        api_key="sk-x", model="m", messages=user, system="SYS", max_tokens=8
    )

    asyncio.run(chat_completion(recorder, api_format="anthropic", base_url=ARK, **common))
    assert recorder.sent["json"]["system"] == "SYS"
    assert recorder.sent["json"]["messages"] == user

    asyncio.run(
        chat_completion(recorder, api_format="openai", base_url="https://api.deepseek.com", **common)
    )
    assert "system" not in recorder.sent["json"]
    assert recorder.sent["json"]["messages"] == [{"role": "system", "content": "SYS"}, *user]


def test_the_adapter_calls_the_model_the_way_its_config_says(recorder):
    """The probe measured how to stop this model thinking; the config stored it;
    the smart fill has to send it. Thinking tokens come out of the same
    max_tokens budget as the answer, so a model left thinking can spend the
    whole budget and return no content at all — measured as an empty reply
    after 18 s, against 0.8 s and 99 tokens with the setting applied."""
    cfg = SimpleNamespace(
        api_format="openai", base_url="https://api.deepseek.com", api_key="sk-x",
        model="deepseek-flash", thinking_param='{"thinking": {"type": "disabled"}}',
    )
    asyncio.run(adapter_assist._ask_llm(cfg, "SYS", "USER"))
    assert recorder.sent["json"]["thinking"] == {"type": "disabled"}

    # A model that was never probed sends nothing extra rather than a guess.
    cfg.thinking_param = None
    asyncio.run(adapter_assist._ask_llm(cfg, "SYS", "USER"))
    assert "thinking" not in recorder.sent["json"]


class FlakyClient:
    """An AsyncClient whose calls walk a scripted list of outcomes — an
    Exception entry is raised, anything else is returned — counting the
    attempts and the per-attempt timeout each one was handed."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0
        self.timeouts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def _next(self, outcome, timeout):
        self.calls += 1
        self.timeouts.append(timeout)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def post(self, url, headers=None, json=None, timeout=None):
        return await self._next(self._outcomes[self.calls], timeout)

    async def get(self, url, headers=None, timeout=None):
        return await self._next(self._outcomes[self.calls], timeout)


def _chat(status=200, body=None):
    url = "https://api.deepseek.com/chat/completions"
    return httpx.Response(
        status, request=httpx.Request("POST", url),
        json=body if body is not None else {"choices": [{"message": {"content": "ok"}}]},
    )


@pytest.fixture()
def no_backoff(monkeypatch) -> list[float]:
    """Record the retry backoffs instead of sleeping them out."""
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(model_connect.asyncio, "sleep", record)
    return sleeps


def _flaky(monkeypatch, outcomes) -> FlakyClient:
    fake = FlakyClient(outcomes)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: fake)
    return fake


def test_a_silent_window_is_walked_past_by_the_retry(monkeypatch, cfg, no_backoff):
    """The config page's morning: the endpoint sits silent past the timeout
    once, then answers. The retry carries the save instead of failing it."""
    fake = _flaky(monkeypatch, [httpx.ReadTimeout("timed out"), _chat()])
    ok, message, ms = asyncio.run(
        model_connect.test_connectivity("llm", "openai", cfg.base_url, cfg.api_key, cfg.model)
    )
    assert (ok, fake.calls, len(no_backoff)) == (True, 2, 1)
    assert ms is not None


def test_a_window_that_never_lifts_still_names_the_exception(monkeypatch, cfg, no_backoff):
    fake = _flaky(monkeypatch, [httpx.ReadTimeout("timed out")] * 4)
    ok, message, ms = asyncio.run(
        model_connect.test_connectivity("llm", "openai", cfg.base_url, cfg.api_key, cfg.model)
    )
    assert (ok, ms) == (False, None)
    assert fake.calls == model_connect.MAX_RETRIES + 1
    assert len(no_backoff) == model_connect.MAX_RETRIES
    assert "ReadTimeout" in message


def test_an_auth_failure_is_not_retried(monkeypatch, cfg, no_backoff):
    """A rejected key must fail on the first attempt — retrying it buys a
    minute of spinner and never a different answer."""
    fake = _flaky(monkeypatch, [_chat(401, {"error": "bad key"})])
    ok, message, _ = asyncio.run(
        model_connect.test_connectivity("llm", "openai", cfg.base_url, cfg.api_key, cfg.model)
    )
    assert (ok, fake.calls, no_backoff) == (False, 1, [])
    assert message.startswith("HTTP 401")


def test_a_busy_answer_is_retried_like_a_silent_one(monkeypatch, cfg, no_backoff):
    fake = _flaky(monkeypatch, [_chat(503, {"error": "overloaded"}), _chat()])
    ok, _, _ = asyncio.run(
        model_connect.test_connectivity("llm", "openai", cfg.base_url, cfg.api_key, cfg.model)
    )
    assert (ok, fake.calls) == (True, 2)


def test_the_embedding_ping_retries_the_same_way(monkeypatch, cfg, no_backoff):
    url = "https://api.deepseek.com/embeddings"
    fake = _flaky(monkeypatch, [
        httpx.Response(503, request=httpx.Request("POST", url), json={}),
        httpx.Response(200, request=httpx.Request("POST", url), json={}),
    ])
    ok, _, _ = asyncio.run(
        model_connect.test_connectivity(
            "embedding", "openai", cfg.base_url, cfg.api_key, cfg.model
        )
    )
    assert (ok, fake.calls) == (True, 2)


def test_the_model_list_retries_the_same_way(monkeypatch, cfg, no_backoff):
    url = "https://api.deepseek.com/models"
    fake = _flaky(monkeypatch, [
        httpx.ConnectError("refused"),
        httpx.Response(
            200, request=httpx.Request("GET", url), json={"data": [{"id": "deepseek-flash"}]},
        ),
    ])
    models = asyncio.run(model_connect.list_models("openai", cfg.base_url, cfg.api_key))
    assert (models, fake.calls) == (["deepseek-flash"], 2)


def test_the_sweep_is_bounded_by_a_total_deadline(monkeypatch, cfg, no_backoff):
    """Retries must not turn a dead endpoint into a minute of spinner: each
    attempt after the first gets only the budget the deadline has left, and
    one with nothing left still fires, so the caller hears a real exception
    rather than a fabricated one."""
    clock = itertools.count(step=8)
    monkeypatch.setattr(model_connect, "_now", lambda: next(clock))
    fake = _flaky(monkeypatch, [httpx.ReadTimeout("timed out")] * 4)
    ok, message, _ = asyncio.run(
        model_connect.test_connectivity("llm", "openai", cfg.base_url, cfg.api_key, cfg.model)
    )
    assert ok is False
    assert "ReadTimeout" in message
    # The deadline is set at t=0; the four attempts then read 22, 14, 6 and
    # -2 seconds off the clock — capped at the ordinary 15 s read / 10 s
    # connect, floored so the spent attempt still makes a real call.
    assert [(t.read, t.connect) for t in fake.timeouts] == [
        (15.0, 10.0), (14.0, 10.0), (6.0, 6.0), (0.05, 0.05),
    ]
    # The attempt whose budget was already gone skipped its backoff wait.
    assert len(no_backoff) == 3
