from datetime import timedelta

from app.auth import create_token
from tests.conftest import login


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_login_success_returns_both_tokens(client):
    data = login(client)
    assert data["token_type"] == "bearer"
    assert data["access_token"] and data["refresh_token"]


def test_login_wrong_password_and_unknown_user(client):
    assert client.post("/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    assert client.post("/auth/login", json={"username": "ghost", "password": "nope"}).status_code == 401


def test_protected_route_requires_token(client):
    assert client.get("/jobs").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_expired_token_is_rejected(client):
    token = create_token("admin", "admin", "access", timedelta(seconds=-5))
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
    assert "expired" in r.json()["detail"].lower()


def test_refresh_flow_and_token_type_separation(client):
    tokens = login(client)
    # refresh token cannot be used as an access token
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['refresh_token']}"})
    assert r.status_code == 401
    # access token cannot be used to refresh
    assert client.post("/auth/refresh", json={"refresh_token": tokens["access_token"]}).status_code == 401
    # valid refresh works
    r = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 200
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).status_code == 200


def test_register_is_admin_only(client, auth):
    body = {"username": "bob", "password": "bob-password-1"}
    assert client.post("/auth/register", json=body).status_code == 401
    assert client.post("/auth/register", json=body, headers=auth).status_code == 201
    assert client.post("/auth/register", json=body, headers=auth).status_code == 409
    bob = login(client, "bob", "bob-password-1")
    r = client.post(
        "/auth/register",
        json={"username": "eve", "password": "eve-password-1"},
        headers={"Authorization": f"Bearer {bob['access_token']}"},
    )
    assert r.status_code == 403


def test_cors_preflight_allows_frontend_origin(client):
    r = client.options(
        "/auth/login",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r = client.options(
        "/auth/login",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") is None
