import pytest

from tests.conftest import login

HF_HEADERS = {"X-Webhook-Secret": "hook-secret"}


def hf_payload(sha="abc123", repo_type="dataset", action="update", scope="repo.content"):
    return {
        "event": {"action": action, "scope": scope},
        "repo": {"type": repo_type, "name": "team/signatures", "headSha": sha},
    }


@pytest.fixture()
def trigger_calls(monkeypatch):
    calls = []
    monkeypatch.setattr("app.routers.webhooks.trigger_training", lambda repo, sha: calls.append((repo, sha)) or "fake")
    return calls


def test_webhook_rejects_bad_or_missing_secret(client, trigger_calls):
    assert client.post("/webhooks/hf", json=hf_payload()).status_code == 401
    assert client.post("/webhooks/hf", json=hf_payload(), headers={"X-Webhook-Secret": "wrong"}).status_code == 401
    assert trigger_calls == []


def test_webhook_ignores_non_dataset_and_non_update_events(client, trigger_calls):
    for payload in (hf_payload(repo_type="model"), hf_payload(action="create"), hf_payload(scope="discussion")):
        r = client.post("/webhooks/hf", json=payload, headers=HF_HEADERS)
        assert r.status_code == 202
        assert r.json()["status"] == "ignored"
    assert trigger_calls == []


def test_webhook_triggers_training_once_per_commit(client, auth, trigger_calls):
    first = client.post("/webhooks/hf", json=hf_payload(sha="s1"), headers=HF_HEADERS).json()
    assert first["status"] == "triggered"
    dup = client.post("/webhooks/hf", json=hf_payload(sha="s1"), headers=HF_HEADERS).json()
    assert dup["status"] == "duplicate"
    client.post("/webhooks/hf", json=hf_payload(sha="s2"), headers=HF_HEADERS)
    assert trigger_calls == [("team/signatures", "s1"), ("team/signatures", "s2")]
    runs = client.get("/training/runs", headers=auth).json()
    assert len(runs) == 2


def test_webhook_survives_trigger_failure(client, auth, monkeypatch):
    def boom(repo, sha):
        raise RuntimeError("github down")

    monkeypatch.setattr("app.routers.webhooks.trigger_training", boom)
    r = client.post("/webhooks/hf", json=hf_payload(sha="s9"), headers=HF_HEADERS)
    assert r.status_code == 202
    assert r.json()["status"] == "trigger_failed"


def test_manual_trigger_is_admin_only(client, auth, trigger_calls):
    assert client.post("/training/trigger", json={"dataset_repo": "team/signatures"}).status_code == 401
    r = client.post("/training/trigger", json={"dataset_repo": "team/signatures"}, headers=auth)
    assert r.status_code == 202
    assert len(trigger_calls) == 1


PIPE = {"X-Pipeline-Token": "pipe-token"}


def promote(client, version, map50, headers=PIPE):
    return client.post(
        "/models/promote",
        json={"version": version, "url": f"s3://models/{version}.onnx", "metrics": {"map50": map50}},
        headers=headers,
    )


def test_promote_requires_pipeline_token_or_admin(client):
    assert promote(client, "v1", 0.8, headers={}).status_code == 401
    assert promote(client, "v1", 0.8, headers={"X-Pipeline-Token": "wrong"}).status_code == 401
    assert promote(client, "v1", 0.8).status_code == 200


def test_evaluation_gate_promotes_only_better_models(client, auth):
    r = promote(client, "v1", 0.80).json()
    assert r["promoted"] is True

    r = promote(client, "v2", 0.75).json()  # worse -> rejected
    assert r["promoted"] is False
    assert client.get("/models/current", headers=auth).json()["version"] == "v1"

    r = promote(client, "v3", 0.90).json()  # better -> promoted
    assert r["promoted"] is True
    assert client.get("/models/current", headers=auth).json()["version"] == "v3"
    assert len(client.get("/models", headers=auth).json()) == 3


def test_equal_map_is_not_promoted(client):
    promote(client, "v1", 0.80)
    assert promote(client, "v2", 0.80).json()["promoted"] is False


def test_promote_validation(client):
    r = client.post("/models/promote", json={"version": "v1", "url": "x", "metrics": {"precision": 0.9}}, headers=PIPE)
    assert r.status_code == 422
    promote(client, "v1", 0.8)
    assert promote(client, "v1", 0.9).status_code == 409


def test_rollback_restores_previous_model(client, auth):
    promote(client, "v1", 0.80)
    promote(client, "v2", 0.90)
    r = client.post("/models/rollback", headers=PIPE)
    assert r.status_code == 200
    assert r.json()["version"] == "v1"
    assert client.get("/models/current", headers=auth).json()["version"] == "v1"


def test_rollback_without_history_returns_409(client):
    assert client.post("/models/rollback", headers=PIPE).status_code == 404  # no model at all
    promote(client, "v1", 0.8)
    assert client.post("/models/rollback", headers=PIPE).status_code == 409


def test_current_model_404_before_first_promotion(client, auth):
    assert client.get("/models/current", headers=auth).status_code == 404
