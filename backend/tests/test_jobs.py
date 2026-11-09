import pytest

from app.config import settings
from app.db import SessionLocal
from app.models import Job
from tests.conftest import JPG, PNG, login


def test_detect_creates_job_and_enqueues(client, auth, queued):
    r = client.post("/detect", files={"file": ("doc.png", PNG, "image/png")}, headers=auth)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"
    assert queued == [job_id]

    job = client.get(f"/jobs/{job_id}", headers=auth).json()
    assert job["status"] == "queued"
    assert job["job_type"] == "detect"
    assert job["result_json"] is None


def test_detect_accepts_jpeg(client, auth, queued):
    r = client.post("/detect", files={"file": ("doc.jpg", JPG, "image/jpeg")}, headers=auth)
    assert r.status_code == 202


def test_detect_requires_auth(client, queued):
    r = client.post("/detect", files={"file": ("doc.png", PNG, "image/png")})
    assert r.status_code == 401
    assert queued == []


def test_detect_rejects_fake_image_even_with_image_content_type(client, auth, queued):
    r = client.post("/detect", files={"file": ("evil.png", b"<script>alert(1)</script>", "image/png")}, headers=auth)
    assert r.status_code == 415
    assert queued == []


def test_detect_rejects_oversized_file(client, auth, queued, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    r = client.post("/detect", files={"file": ("doc.png", PNG, "image/png")}, headers=auth)
    assert r.status_code == 413


def test_compare_needs_both_images(client, auth, queued):
    r = client.post("/compare", files={"reference": ("a.png", PNG, "image/png")}, headers=auth)
    assert r.status_code == 422
    r = client.post(
        "/compare",
        files={"reference": ("a.png", PNG, "image/png"), "test": ("b.jpg", JPG, "image/jpeg")},
        headers=auth,
    )
    assert r.status_code == 202
    job = client.get(f"/jobs/{r.json()['job_id']}", headers=auth).json()
    assert job["job_type"] == "compare"


def test_idempotency_key_returns_same_job(client, auth, queued):
    headers = {**auth, "Idempotency-Key": "abc-123"}
    first = client.post("/detect", files={"file": ("d.png", PNG, "image/png")}, headers=headers).json()
    second = client.post("/detect", files={"file": ("d.png", PNG, "image/png")}, headers=headers).json()
    assert first["job_id"] == second["job_id"]
    assert len(queued) == 1  # not enqueued twice


def test_job_not_found(client, auth):
    assert client.get("/jobs/does-not-exist", headers=auth).status_code == 404


def test_users_cannot_see_each_others_jobs(client, auth, queued):
    job_id = client.post("/detect", files={"file": ("d.png", PNG, "image/png")}, headers=auth).json()["job_id"]
    client.post("/auth/register", json={"username": "bob", "password": "bob-password-1"}, headers=auth)
    bob = {"Authorization": f"Bearer {login(client, 'bob', 'bob-password-1')['access_token']}"}
    assert client.get(f"/jobs/{job_id}", headers=bob).status_code == 404
    assert client.get("/jobs", headers=bob).json() == []
    assert len(client.get("/jobs", headers=auth).json()) == 1


def test_queue_failure_marks_job_failed_and_returns_503(client, auth, monkeypatch):
    def boom(job_id):
        raise ConnectionError("redis down")

    monkeypatch.setattr("app.routers.jobs.enqueue_job", boom)
    r = client.post("/detect", files={"file": ("d.png", PNG, "image/png")}, headers=auth)
    assert r.status_code == 503
    with SessionLocal() as db:
        jobs = db.query(Job).all()
        assert len(jobs) == 1
        assert jobs[0].status == "failed"
        assert "Queue unavailable" in jobs[0].error


def test_list_jobs_status_filter(client, auth, queued):
    client.post("/detect", files={"file": ("d.png", PNG, "image/png")}, headers=auth)
    assert len(client.get("/jobs?status=queued", headers=auth).json()) == 1
    assert client.get("/jobs?status=done", headers=auth).json() == []
