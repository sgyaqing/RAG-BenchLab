import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


VALID_PAYLOAD = {
    "name": "Doubao 2.0 mini",
    "type": "llm",
    "api_format": "openai",
    "base_url": "https://example.com/v1",
    "api_key": "sk-test",
    "model": "doubao-seed-2-0-mini",
}


def test_crud_flow(client):
    # create
    resp = client.post("/api/model-configs", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    created = resp.json()
    assert created["name"] == VALID_PAYLOAD["name"]
    config_id = created["id"]

    # list
    resp = client.get("/api/model-configs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == config_id

    # search by name
    assert client.get("/api/model-configs", params={"name": "Doubao"}).json()["total"] == 1
    assert client.get("/api/model-configs", params={"name": "bge"}).json()["total"] == 0

    # update
    resp = client.put(f"/api/model-configs/{config_id}", json={**VALID_PAYLOAD, "name": "BGE-M3"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "BGE-M3"

    # delete
    assert client.delete(f"/api/model-configs/{config_id}").status_code == 204
    assert client.get("/api/model-configs").json()["total"] == 0
    assert client.delete(f"/api/model-configs/{config_id}").status_code == 404


def test_check_name_case_insensitive(client):
    assert client.get(
        "/api/model-configs/check-name", params={"name": VALID_PAYLOAD["name"]}
    ).json()["available"]
    assert client.post("/api/model-configs", json=VALID_PAYLOAD).status_code == 201

    assert not client.get(
        "/api/model-configs/check-name", params={"name": "doubao 2.0 MINI"}
    ).json()["available"]
    # trailing whitespace is ignored when checking
    assert not client.get(
        "/api/model-configs/check-name", params={"name": " Doubao 2.0 mini "}
    ).json()["available"]

    resp = client.post("/api/model-configs", json={**VALID_PAYLOAD, "name": "doubao 2.0 mini"})
    assert resp.status_code == 409


def test_check_name_excludes_the_record_being_edited(client):
    config_id = client.post("/api/model-configs", json=VALID_PAYLOAD).json()["id"]
    other = {**VALID_PAYLOAD, "name": "Other"}
    other_id = client.post("/api/model-configs", json=other).json()["id"]

    # a record's own name is still available to it while editing...
    assert client.get(
        "/api/model-configs/check-name",
        params={"name": VALID_PAYLOAD["name"], "exclude_id": config_id},
    ).json()["available"]
    # ...but not to a different record
    assert not client.get(
        "/api/model-configs/check-name",
        params={"name": VALID_PAYLOAD["name"], "exclude_id": other_id},
    ).json()["available"]

    # saving a record under another's name is refused; its own name is not
    assert client.put(
        f"/api/model-configs/{other_id}", json={**other, "name": VALID_PAYLOAD["name"]}
    ).status_code == 409
    assert client.put(f"/api/model-configs/{config_id}", json=VALID_PAYLOAD).status_code == 200


def test_embedding_allows_missing_api_key(client):
    payload = {
        **VALID_PAYLOAD,
        "name": "BGE-M3",
        "type": "embedding",
        "api_key": None,
        "model": "bge-m3",
    }
    assert client.post("/api/model-configs", json=payload).status_code == 201


def test_embedding_rejects_anthropic(client):
    payload = {**VALID_PAYLOAD, "type": "embedding", "api_format": "anthropic"}
    assert client.post("/api/model-configs", json=payload).status_code == 422


def test_an_llm_config_may_omit_the_api_key(client):
    """A key is optional for every type.

    Ollama, vLLM and the on-prem endpoints customers actually run take none, and
    requiring one only forced a placeholder that then travelled with the config.
    An endpoint that does need a key is caught by the connectivity test the save
    button runs first, which reports the 401 instead of a form rule guessing at
    it."""
    payload = {**VALID_PAYLOAD, "api_key": None}
    assert client.post("/api/model-configs", json=payload).status_code == 201


def test_connectivity_test_endpoint(client, monkeypatch):
    async def fake_test(*args):
        return True, "ok", 123

    monkeypatch.setattr("app.services.model_connect.test_connectivity", fake_test)
    resp = client.post(
        "/api/model-configs/test",
        json={
            "type": "llm",
            "api_format": "openai",
            "base_url": "https://example.com/v1",
            "api_key": "sk-test",
            "model": "gpt-4o-mini",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"success": True, "message": "ok", "duration_ms": 123}


def test_available_models_endpoint(client, monkeypatch):
    async def fake_list(*args):
        return ["gpt-4o-mini", "gpt-4o"]

    monkeypatch.setattr("app.services.model_connect.list_models", fake_list)
    resp = client.post(
        "/api/model-configs/available-models",
        json={"api_format": "openai", "base_url": "https://example.com/v1", "api_key": "sk"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"models": ["gpt-4o-mini", "gpt-4o"]}
