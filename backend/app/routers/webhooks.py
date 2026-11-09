import hmac
import logging
import uuid

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import require_admin
from app.config import settings
from app.db import get_db
from app.models import TrainingRun, User
from app.schemas import TrainingRunOut, TrainTriggerRequest
from app.training import TrainingNotConfigured, trigger_training

logger = logging.getLogger("api.webhooks")
router = APIRouter(tags=["training"])


def start_training(db: Session, dataset_repo: str, dataset_sha: str) -> tuple[TrainingRun, bool]:
    """Record a training run (deduplicated by repo + sha) and trigger the remote job once."""
    run = TrainingRun(dataset_repo=dataset_repo, dataset_sha=dataset_sha, status="pending")
    db.add(run)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.scalar(
            select(TrainingRun).where(
                TrainingRun.dataset_repo == dataset_repo, TrainingRun.dataset_sha == dataset_sha
            )
        )
        return existing, False

    try:
        run.detail = trigger_training(dataset_repo, dataset_sha)
        run.status = "triggered"
    except TrainingNotConfigured as exc:
        run.status = "pending_manual"
        run.detail = str(exc)
    except Exception as exc:  # network or GitHub error: keep the record, do not crash the webhook
        logger.exception("training trigger failed")
        run.status = "trigger_failed"
        run.detail = str(exc)[:500]
    db.commit()
    return run, True


@router.post("/webhooks/hf", status_code=status.HTTP_202_ACCEPTED)
def hf_webhook(request: Request, payload: dict = Body(...), db: Session = Depends(get_db)):
    """Hugging Face webhook. HF sends the secret in the X-Webhook-Secret header."""
    if not settings.hf_webhook_secret:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Webhook secret not configured")
    received = request.headers.get("X-Webhook-Secret", "")
    if not hmac.compare_digest(received, settings.hf_webhook_secret):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid webhook secret")

    event = payload.get("event") or {}
    repo = payload.get("repo") or {}
    is_dataset_update = (
        repo.get("type") == "dataset"
        and event.get("action") == "update"
        and str(event.get("scope", "")).startswith("repo.content")
    )
    if not is_dataset_update:
        return {"status": "ignored"}

    dataset_repo = repo.get("name") or "unknown"
    dataset_sha = repo.get("headSha") or f"unknown-{uuid.uuid4()}"
    run, created = start_training(db, dataset_repo, dataset_sha)
    return {"status": run.status if created else "duplicate", "run_id": run.id}


@router.post("/training/trigger", response_model=TrainingRunOut, status_code=status.HTTP_202_ACCEPTED)
def manual_trigger(body: TrainTriggerRequest, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    """Manual backup trigger (use it if the HF webhook does not fire)."""
    sha = body.dataset_sha or f"manual-{uuid.uuid4()}"
    run, _created = start_training(db, body.dataset_repo, sha)
    return run


@router.get("/training/runs", response_model=list[TrainingRunOut])
def list_runs(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    return list(db.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(50)))
