from fastapi.testclient import TestClient

from app.main import create_app


def test_health():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "RAG-BenchLab"}
