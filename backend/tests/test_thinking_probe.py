"""The thinking probe: what it concludes, and what it does when the endpoint
will not play along.

The wiring on the runtime side (the stored field actually reaching the request
body) is checked against a live endpoint rather than mocked here: the builder
sits under a LangChain wrapper, and a test that has to fake its way through that
would be asserting on the fake.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.api import model_configs
from app.main import create_app
from app.services import thinking_probe


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def _openai_payload(*, tokens: int, text: str, reasoning: bool) -> dict:
    message = {"content": text}
    if reasoning:
        message["reasoning_content"] = "…"
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"completion_tokens": tokens},
    }


# What the two real endpoints answered to the probe question (measured): a model
# that thinks spends 54 tokens on "3/10"; one that does not spends 8.
THINKS = _openai_payload(tokens=54, text="3/10", reasoning=True)
STOPPED = _openai_payload(tokens=8, text="3/10", reasoning=False)
NEVER_THOUGHT = _openai_payload(tokens=5, text="3/10", reasoning=False)


class FakeClient:
    """Replies per body field. `behaviour` maps a field name to a canned reply;
    "__baseline__" is the reply when no candidate field is present."""

    def __init__(self, behaviour: dict):
        self.behaviour = behaviour
        self.seen: list[list[str]] = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None):
        body = json or {}
        self.seen.append(sorted(set(body) - {"model", "max_tokens", "messages"}))

        class Response:
            status_code = 200

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        for field, reply in self.behaviour.items():
            if field in body:
                if reply is None:
                    Response.status_code = 400
                    return Response({})
                return Response(reply)
        return Response(self.behaviour.get("__baseline__", NEVER_THOUGHT))


def _probe(monkeypatch, behaviour) -> tuple[str | None, dict | None]:
    fake = FakeClient(behaviour)
    monkeypatch.setattr(thinking_probe.httpx, "AsyncClient", fake)
    state, param = asyncio.run(
        thinking_probe.probe("openai", "https://example.com/v1", "sk-x", "some-model")
    )
    return state, param, fake


def test_a_model_that_does_not_think_needs_nothing(monkeypatch):
    state, param, fake = _probe(monkeypatch, {})
    assert state == thinking_probe.STATE_ALREADY_OFF
    assert param is None
    assert len(fake.seen) == 1, "one call is enough to know it does not think"


def test_the_first_field_that_works_is_the_one_kept(monkeypatch):
    state, param, fake = _probe(
        monkeypatch, {"__baseline__": THINKS, "thinking": STOPPED, "reasoning_effort": THINKS}
    )
    assert state == thinking_probe.STATE_DISABLED
    assert param == {"thinking": {"type": "disabled"}}
    assert fake.seen[-1] == ["thinking"], "and it stops there"


def test_a_rejected_field_does_not_end_the_sweep(monkeypatch):
    # `thinking` is refused outright; the next candidate is what works.
    state, param, fake = _probe(
        monkeypatch, {"__baseline__": THINKS, "thinking": None, "enable_thinking": STOPPED}
    )
    assert state == thinking_probe.STATE_DISABLED
    assert param == {"enable_thinking": False}


def test_a_model_that_keeps_thinking_is_reported_not_hidden(monkeypatch):
    state, param, fake = _probe(monkeypatch, {"__baseline__": THINKS})
    assert state == thinking_probe.STATE_UNSUPPORTED
    assert param is None
    assert len(fake.seen) == 1 + len(thinking_probe.CANDIDATES["openai"])


def test_a_chatty_answer_is_not_mistaken_for_thinking(monkeypatch):
    """Plenty of tokens, plenty of answer: that is verbosity, not thinking."""
    state, _, _ = _probe(
        monkeypatch, {"__baseline__": _openai_payload(tokens=90, text="x" * 400, reasoning=False)}
    )
    assert state == thinking_probe.STATE_ALREADY_OFF


def test_an_unreachable_endpoint_gets_no_verdict(monkeypatch):
    """Not the same as "it thinks": a blank cell, not a 否."""
    class Broken(FakeClient):
        async def post(self, url, headers=None, json=None):
            raise thinking_probe.httpx.ConnectError("no route")

    monkeypatch.setattr(thinking_probe.httpx, "AsyncClient", Broken({}))
    state, param = asyncio.run(
        thinking_probe.probe("openai", "https://example.com/v1", "sk-x", "m")
    )
    assert state is None
    assert param is None


def test_saving_records_the_finding_and_unticking_clears_it(client, monkeypatch):
    async def fake_probe(*args):
        return thinking_probe.STATE_DISABLED, {"thinking": {"type": "disabled"}}

    monkeypatch.setattr(model_configs.thinking_probe, "probe", fake_probe)
    payload = {
        "name": "Thinker", "type": "llm", "api_format": "openai",
        "base_url": "https://example.com/v1", "api_key": "sk-test",
        "model": "some-model", "auto_disable_thinking": True,
    }
    created = client.post("/api/model-configs", json=payload).json()
    assert created["thinking_state"] == "disabled"

    # Unticking the box clears the finding: what the list shows and what the
    # runtime sends have to be the same thing.
    updated = client.put(
        f"/api/model-configs/{created['id']}", json={**payload, "auto_disable_thinking": False}
    ).json()
    assert updated["thinking_state"] is None


def test_an_embedding_config_is_never_probed(client, monkeypatch):
    async def boom(*args):
        raise AssertionError("an embedding config must not be probed")

    monkeypatch.setattr(model_configs.thinking_probe, "probe", boom)
    created = client.post("/api/model-configs", json={
        "name": "BGE", "type": "embedding", "api_format": "openai",
        "base_url": "http://localhost:11434/v1", "model": "bge-m3",
        "auto_disable_thinking": True,
    }).json()
    assert created["thinking_state"] is None
