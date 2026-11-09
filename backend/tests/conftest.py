import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{_tmp}/test.db",
        "LOCAL_STORAGE_DIR": f"{_tmp}/storage",
        "JWT_SECRET": "test-secret",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "admin-pass-123",
        "HF_WEBHOOK_SECRET": "hook-secret",
        "PIPELINE_TOKEN": "pipe-token",
        "CORS_ORIGINS": "http://localhost:5173",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPG = b"\xff\xd8\xff\xe0" + b"0" * 64


@pytest.fixture()
def client():
    Base.metadata.drop_all(engine)
    with TestClient(app) as c:  # lifespan creates tables and seeds the admin
        yield c


@pytest.fixture()
def queued(monkeypatch):
    """Replaces the real broker call; records enqueued job ids."""
    calls: list[str] = []
    monkeypatch.setattr("app.routers.jobs.enqueue_job", lambda job_id: calls.append(job_id))
    return calls


def login(client, username="admin", password="admin-pass-123"):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def auth(client):
    return {"Authorization": f"Bearer {login(client)['access_token']}"}
