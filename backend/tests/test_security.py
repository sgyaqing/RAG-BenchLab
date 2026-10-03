"""Guards against paths that escape the directory they are meant to stay in.

Two were measured against a running server before being fixed:
  * GET /../../data/rag_benchlab.db returned the whole database (every stored
    API key in it) and /../../logs/app.log returned the application logs.
  * An evaluation named "../../logs" resolved its artifact directory to
    <repo>/logs, which the delete endpoint then rmtree'd — "../../data" would
    have removed the entire data directory.
"""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services import eval_run


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def _spa_endpoint(app):
    """The catch-all route that serves the built frontend."""
    for route in app.routes:
        if getattr(route, "path", "") == "/{full_path:path}":
            return route.endpoint
    return None


def test_spa_route_refuses_to_serve_outside_the_build_directory():
    """A URL path is attacker-controlled; `dist / full_path` follows ".." out of
    the build directory unless the result is checked."""
    app = create_app()
    endpoint = _spa_endpoint(app)
    if endpoint is None:
        pytest.skip("frontend build not present, so the SPA route is not mounted")

    dist = get_settings().frontend_dist
    inside = next((p for p in (dist / "assets").glob("*.js")), None)

    # a real asset is still served
    if inside is not None:
        served = asyncio.run(endpoint(f"assets/{inside.name}"))
        assert Path(served.path) == inside

    # escaping paths fall back to index.html instead of the target file
    for escape in ("../../data/rag_benchlab.db",
                   "../../logs/app.log",
                   "../../../../../../etc/hosts",
                   "../main.py"):
        served = asyncio.run(endpoint(escape))
        assert Path(served.path).name == "index.html", (
            f"{escape} was served from {served.path}")
        assert Path(served.path).resolve().is_relative_to(dist.resolve())


def test_evaluation_name_cannot_escape_its_directory(client):
    """The name becomes a directory that delete rmtrees, so a traversing name
    would delete something else entirely."""
    resp = client.post("/api/evaluations", json={
        "name": "../../logs", "testset_id": 1, "rag_system_id": 1,
        "llm_config_id": 1, "embedding_config_id": 1,
        "metrics": ["faithfulness"], "base_url": "http://x/query",
    })
    assert resp.status_code == 422, resp.text


def test_eval_path_for_refuses_to_leave_its_base():
    """Defence in depth at the sink: the schema validates new requests, but a
    row written earlier (or by any future caller) must not be able to point the
    delete path somewhere else."""
    base = get_settings().data_dir / "evaluation"
    assert eval_run.eval_path_for("普通名字") == base / "普通名字"
    for bad in ("../../logs", "../data", "../../../etc", ".."):
        with pytest.raises(ValueError):
            eval_run.eval_path_for(bad)
