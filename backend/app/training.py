"""Triggers the remote training job. Default backend: GitHub Actions workflow_dispatch."""
import httpx

from app.config import settings


class TrainingNotConfigured(Exception):
    pass


def trigger_training(dataset_repo: str, dataset_sha: str) -> str:
    if not (settings.github_repo and settings.github_token):
        raise TrainingNotConfigured("GITHUB_REPO / GITHUB_TOKEN not set; trigger training manually")

    url = (
        f"https://api.github.com/repos/{settings.github_repo}"
        f"/actions/workflows/{settings.github_workflow}/dispatches"
    )
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {settings.github_token}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "ref": settings.github_ref,
            "inputs": {"dataset_repo": dataset_repo, "dataset_sha": dataset_sha},
        },
        timeout=15,
    )
    if response.status_code != 204:
        raise RuntimeError(f"GitHub dispatch failed with status {response.status_code}")
    return "github_actions"
